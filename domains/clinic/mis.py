"""Тестовый стенд МИС «Регистратура» (DBI): REST-клиент и реестр поверх него.

Правила стенда, которые учитываем:
- пациент определяется фамилией, именем и датой рождения;
- даты — строки «ДД.ММ.ГГГГ» и «ДД.ММ.ГГГГ ЧЧ:ММ» в GMT+4 без смещения;
- стенд не хранит расписание и не проверяет занятость — свободные слоты считаем сами;
- лимит 10 запросов в минуту: справочник врачей и наши записи кэшируем, на 429 ждём Retry-After.
"""

from __future__ import annotations

import logging
import re
import time as _time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from core.resilience import breaker

from . import reminders as c10
from .schedule import Booking, Doctor, Patient, Schedule, Slot, SlotTakenError, locked
from .timetable import Timetable

log = logging.getLogger(__name__)

STAND_TZ = timezone(timedelta(hours=4))
TECH_PATIENT = ("Технический", "Пациент", date(2000, 1, 1))  # вымышленная карточка для заявок без идентификации
# Отметки обзвона в комментарии клиента (уточнение DBI 20.10): в строке подтверждения — время приёма
MIS_NOTES = {
    "CONFIRMED": lambda today, b: f"Подтверждена запись #{b.id} {b.slot.start:%d.%m.%Y %H:%M}",
    "OPT_OUT": lambda today, b: f"Отказ от обзвона {today:%d.%m.%Y}",
    "OPT_OUT_ALL": lambda today, b: f"Отказ от уведомлений {today:%d.%m.%Y}",
}
STATUS_PLANNED, STATUS_CANCELLED = "Запланирована", "Отменена"


def stand_today() -> date:
    return datetime.now(STAND_TZ).date()


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%d.%m.%Y %H:%M")


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%d.%m.%Y %H:%M")


class MisError(Exception):
    """Ошибка стенда: код и описание на русском."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class MisClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 10.0, max_wait: float = 15.0,
                 attempts: int = 3):
        self._http = httpx.Client(base_url=base_url.rstrip("/") + "/api/v1",
                                  headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
        self._max_wait = max_wait
        self._attempts = attempts  # в звонке ждём недолго (R-06 — перевод к регистратору), в пакетных скриптах — дольше
        self.waited = 0.0          # с: сколько всего ждали лимит стенда — замеры отделяют это от работы ассистента
        self.sent: list[float] = []  # время запросов за последнюю минуту (monotonic)

    def _request(self, method: str, path: str, **kwargs) -> tuple[int, Any]:
        """Запрос к стенду. Сеть или 5xx сбоят подряд — предохранитель «mis» размыкается, и следующие запросы
        сразу дают MisError: ассистент без ожидания таймаутов переводит к регистратору (R-06)."""
        guard = breaker("mis")
        if not guard.allow():
            raise MisError(503, "СТЕНД_НЕДОСТУПЕН", "Стенд временно недоступен")
        try:
            result = self._request_once(method, path, **kwargs)
        except MisError as err:
            guard.fail(err) if err.status >= 500 or err.status == 429 else guard.ok()
            raise
        except httpx.HTTPError as err:
            guard.fail(err)
            raise MisError(503, "СТЕНД_НЕДОСТУПЕН", f"Нет связи со стендом ({type(err).__name__})") from err
        guard.ok()
        return result

    def _request_once(self, method: str, path: str, **kwargs) -> tuple[int, Any]:
        for attempt in range(self._attempts):
            now = _time.monotonic()
            self.sent = [t for t in self.sent if now - t < 60] + [now]
            resp = self._http.request(method, path, **kwargs)
            if resp.status_code == 429 and attempt < self._attempts - 1:
                wait = min(float(resp.headers.get("Retry-After", "6")), self._max_wait)
                log.warning("Стенд: лимит запросов, ждём %.0f с", wait)
                _time.sleep(wait)
                self.waited += wait
                continue
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except ValueError:
                    body = {}
                raise MisError(resp.status_code, body.get("code", ""), body.get("message", resp.text[:200]))
            return resp.status_code, (resp.json() if resp.content else None)
        raise MisError(429, "ПРЕВЫШЕН_ЛИМИТ", "Стенд ограничил частоту запросов")

    def pace(self, budget: int, per_minute: int = 10) -> float:
        """Для пакетных прогонов: ждём, пока в минутном окне стенда освободится `budget` запросов, —
        следующий сценарий не упрётся в лимит посреди разговора. Возвращает, сколько ждали (с)."""
        budget = min(budget, per_minute)
        now = _time.monotonic()
        recent = sorted(t for t in self.sent if now - t < 60)
        excess = len(recent) + budget - per_minute
        if excess <= 0:
            return 0.0
        wait = 60 - (now - recent[excess - 1]) + 0.5
        _time.sleep(wait)
        return wait

    def _paged(self, path: str, params: dict | None = None) -> list[dict]:
        items, offset = [], 0
        while True:
            _, page = self._request("GET", path, params={**(params or {}), "limit": 50, "offset": offset})
            batch = page if isinstance(page, list) else page.get("items", [])
            items += batch
            if len(batch) < 50:
                return items
            offset += 50

    def doctors(self) -> list[dict]:
        return self._paged("/doctors")

    def clients(self) -> list[dict]:
        return self._paged("/clients")

    def find_clients(self, last_name: str, first_name: str, birth_date: date) -> list[dict]:
        return self._paged("/clients", {"last_name": last_name, "first_name": first_name,
                                        "birth_date": birth_date.strftime("%d.%m.%Y")})

    def create_client(self, last_name: str, first_name: str, birth_date: date) -> dict:
        _, body = self._request("POST", "/clients", json={
            "last_name": last_name, "first_name": first_name, "birth_date": birth_date.strftime("%d.%m.%Y")})
        return body  # 201 — создан, 200 — уже был

    def get_client(self, client_id: int) -> dict:
        _, body = self._request("GET", f"/clients/{client_id}")
        return body

    def update_client(self, client_id: int, **fields) -> dict:
        _, body = self._request("PATCH", f"/clients/{client_id}", json=fields)
        return body

    def visits(self, **filters) -> list[dict]:
        return self._paged("/visits", {k: v for k, v in filters.items() if v is not None})

    def create_visit(self, client_id: int, doctor_id: int, at: datetime) -> dict:
        _, body = self._request("POST", "/visits", json={
            "client_id": client_id, "doctor_id": doctor_id, "scheduled_at": fmt_dt(at)})
        return body

    def update_visit(self, visit_id: int, **fields) -> dict:
        _, body = self._request("PATCH", f"/visits/{visit_id}", json=fields)
        return body

    def get_escalation(self, escalation_id: int) -> dict:
        _, body = self._request("GET", f"/escalations/{escalation_id}")
        return body

    def create_escalation(self, client_id: int, description: str) -> dict:
        _, body = self._request("POST", "/escalations", json={
            "client_id": client_id, "description": description[:1000]})
        return body


class MisRegistry(Schedule):
    """Реестр клиники поверх стенда: врачи и записи — из МИС, свободные слоты — по расписанию."""

    def __init__(self, client: MisClient, today: date | None = None, horizon_days: int = 14,
                 timetable: Timetable | None = None, require_timetable: bool = False):
        self.client = client
        # Стенд не хранит расписание. Без расписания кейса свободное время не предлагаем — иначе это были бы
        # выдуманные слоты: поиск времени уходит в R-06 (перевод к регистратору)
        self.require_timetable = require_timetable and timetable is None
        doctors = [
            Doctor(id=str(d["id"]), name=d["full_name"], specialty=d["specialty"].strip().lower(),
                   is_active=d["is_active"])
            for d in client.doctors()
        ]
        super().__init__(doctors, today or stand_today, horizon_days=horizon_days, busy_share=0,
                         timetable=timetable)
        if timetable is not None:
            matched = sum(d.timetable_id is not None for d in self.doctors.values())
            log.info("Расписание кейса: сопоставлено %d из %d врачей стенда", matched, len(self.doctors))
        self.patients: dict[str, Patient] = {}
        self._known: dict[tuple[str, str, date], str] = {}  # ФИО и дата рождения → id карточки на стенде
        self._load_visits()

    def find(self, **kwargs) -> list:
        if self.require_timetable:
            raise MisError(503, "НЕТ_РАСПИСАНИЯ", "Расписание врачей не загружено — время проверить нельзя")
        return super().find(**kwargs)

    def _to_booking(self, v: dict, patient_name: str = "") -> Booking | None:
        doc = self.doctors.get(str(v["doctor"]["id"]))
        if doc is None:
            return None
        slot = Slot(doctor_id=doc.id, doctor_name=doc.name, specialty=doc.specialty, branch=doc.branch,
                    cabinet=doc.cabinet, start=parse_dt(v["scheduled_at"]))
        try:
            created = parse_dt(v["created_at"]) if v.get("created_at") else None
        except ValueError:
            created = None
        return Booking(id=str(v["id"]), slot=slot, patient_name=patient_name, patient_id=str(v["client_id"]),
                       created=created)

    def _load_visits(self) -> None:
        """Наши запланированные записи — один раз при старте, дальше держим кэш сами."""
        for v in self.client.visits(status=STATUS_PLANNED):
            if booking := self._to_booking(v):
                self._store(booking)
        self._load_reminder_marks()

    def _load_reminder_marks(self) -> None:
        """Отказы от обзвона и подтверждения живут в комментариях клиентов на стенде — восстанавливаем их при старте,
        чтобы после перезапуска не звонить отказавшимся и уже подтвердившим (С-10)."""
        try:
            people = self.client.clients()
        except MisError:
            log.warning("Отметки обзвона со стенда не прочитаны — правила С-10 по ним восстановятся при звонке")
            return
        for c in people:
            for line in (c.get("comment") or "").splitlines():
                if line.startswith("Отказ от уведомлений"):
                    self.notify_opt_out[str(c["id"])] = "OPT_OUT_ALL"
                elif line.startswith("Отказ от обзвона") and self.notify_opt_out.get(str(c["id"])) != "OPT_OUT_ALL":
                    self.notify_opt_out[str(c["id"])] = "OPT_OUT"
                elif m := re.match(r"Подтверждена запись #(\d+) (\d\d\.\d\d\.\d{4} \d\d:\d\d)", line):
                    self.reminder_final[m.group(1)] = ("CONFIRMED", parse_dt(m.group(2)))

    def identify(self, last_name: str, first_name: str, birth_date: date, *, create: bool) -> Patient | None:
        # Уже найденную карточку повторно не ищем: id на стенде не меняется, а лимит стенда — 10 запросов в минуту
        # (запись и жалоба одного пациента, повторное «да», технический пациент для экстренных заявок)
        key = (last_name.strip().lower(), first_name.strip().lower(), birth_date)
        if (known := self._known.get(key)) in self.patients:
            return self.patients[known]
        found = self.client.find_clients(last_name, first_name, birth_date)
        if found:
            c = found[0]
        elif create:
            c = self.client.create_client(last_name, first_name, birth_date)
        else:
            return None
        patient = Patient(id=str(c["id"]), last_name=c["last_name"], first_name=c["first_name"],
                          birth_date=datetime.strptime(c["birth_date"], "%d.%m.%Y").date(),
                          comment=c.get("comment") or "", middle_name=c.get("middle_name") or "")
        self.patients[patient.id] = patient
        self._known[key] = patient.id
        for b in self.bookings.values():  # имена для записей, загруженных при старте
            if b.patient_id == patient.id and not b.patient_name:
                b.patient_name = patient.full_name
        return patient

    def _taken_on_stand(self, slot: Slot) -> bool:
        """Стенд занятость не проверяет, а записи могли сделать после нашего старта (регистратор, другой канал):
        перед записью сверяем слот с актуальными визитами врача. Сверка не удалась — запись не создаём: кэш не доказывает,
        что время свободно, а стенд двойную запись не запрещает. MisError уходит выше — перевод к регистратору (R-06)."""
        try:
            visits = self.client.visits(doctor_id=int(slot.doctor_id), status=STATUS_PLANNED)
        except MisError:
            log.warning("Сверка слота со стендом не удалась — запись не создаём, перевод к регистратору")
            raise
        at = fmt_dt(slot.start)
        for v in visits:
            if str(v.get("doctor", {}).get("id")) == slot.doctor_id and v.get("scheduled_at") == at:
                if booking := self._to_booking(v):
                    self._store(booking)  # теперь кэш знает об этом визите
                return True
        return False

    @locked
    def book(self, slot: Slot, patient_name: str, patient_id: str | None = None) -> Booking:
        if patient_id is None:
            raise ValueError("Для записи в МИС нужен идентифицированный пациент")
        if self.is_busy(slot.doctor_id, slot.start) or self._taken_on_stand(slot):
            raise SlotTakenError(slot.spoken())
        v = self.client.create_visit(int(patient_id), int(slot.doctor_id), slot.start)
        booking = Booking(id=str(v["id"]), slot=slot, patient_name=patient_name, patient_id=patient_id,
                          created=c10.clinic_now())
        self._store(booking)
        return booking

    def cancel(self, booking_id: str) -> Booking:
        self.client.update_visit(int(booking_id), status=STATUS_CANCELLED)
        return super().cancel(booking_id)

    @locked
    def reschedule(self, booking_id: str, slot: Slot) -> Booking:
        if self.is_busy(slot.doctor_id, slot.start) or self._taken_on_stand(slot):
            raise SlotTakenError(slot.spoken())
        self.client.update_visit(int(booking_id), doctor_id=int(slot.doctor_id), scheduled_at=fmt_dt(slot.start))
        old = super().cancel(booking_id)
        moved = old.model_copy(update={"slot": slot})
        self._store(moved)
        return moved

    def visit_log(self, refresh: bool = False) -> list[dict]:
        """Журнал всех записей команды прямо со стенда: и активные, и отменённые, и завершённые.
        Два запроса (визиты и карточки) раз в минуту — лимит стенда 10 запросов в минуту."""
        cached = getattr(self, "_log_cache", None)
        if cached and not refresh and _time.monotonic() - cached[0] < 60:
            return cached[1]
        try:
            visits = self.client.visits()
            people = {str(c["id"]): c for c in self.client.clients()}
        except MisError:
            log.warning("Журнал записей: стенд недоступен, показываем кэш")
            return cached[1] if cached else super().visit_log()
        items = []
        for v in visits:
            c = people.get(str(v.get("client_id")), {})
            doctor = v.get("doctor") or {}
            try:
                at = parse_dt(v["scheduled_at"]).isoformat(timespec="minutes")
            except (KeyError, ValueError):
                at = v.get("scheduled_at", "")
            items.append({
                "id": str(v["id"]), "patient": f"{c.get('last_name', '')} {c.get('first_name', '')}".strip(),
                "birth_date": c.get("birth_date", ""), "doctor": doctor.get("full_name", ""),
                "specialty": (doctor.get("specialty") or "").lower(), "at": at, "status": v.get("status", ""),
                "updated": v.get("updated_at") or v.get("created_at") or "", "note": "",
            })
        self._log_cache = (_time.monotonic(), items)
        return items

    def patient_of(self, booking: Booking) -> Patient | None:
        """Пациент записи — для исходящего напоминания (карточку читаем со стенда один раз)."""
        if not booking.patient_id:
            return None
        if booking.patient_id not in self.patients:
            c = self.client.get_client(int(booking.patient_id))
            self.patients[booking.patient_id] = Patient(
                id=str(c["id"]), last_name=c["last_name"], first_name=c["first_name"],
                birth_date=datetime.strptime(c["birth_date"], "%d.%m.%Y").date(), comment=c.get("comment") or "",
                middle_name=c.get("middle_name") or "")
        return self.patients[booking.patient_id]

    def log_reminder(self, booking: Booking, patient: Patient | None, code: str) -> None:
        """Итог обзвона в комментарий карточки клиента — строками, которые проверяет заказчик (уточнение DBI 20.10):
        «Подтверждена запись #<id> ДД.ММ.ГГГГ ЧЧ:ММ» (время приёма), «Отказ от обзвона ДД.ММ.ГГГГ».
        Перенос и отмена уже отражены в самой записи. PATCH заменяет комментарий целиком — сначала читаем актуальный
        со стенда (его мог дописать регистратор) и дописываем строку."""
        super().log_reminder(booking, patient, code)
        if patient is None or code not in MIS_NOTES:
            return
        self._append_comment(patient, MIS_NOTES[code](self.today, booking))

    def set_opt_out(self, patient: Patient, code: str) -> None:
        super().set_opt_out(patient, code)
        self._append_comment(patient, MIS_NOTES[code](self.today, None))

    def _append_comment(self, patient: Patient, note: str) -> None:
        """Дописать строку в комментарий карточки. Стенд не ответил (лимит запросов, сеть) — разговор не обрываем:
        итог уже в журнале обзвона, отметку допишет регистратор по журналу."""
        try:
            current = self.client.get_client(int(patient.id)).get("comment") or ""
            if note in current.split("\n"):
                return  # уже отмечено (повторный звонок)
            patient.comment = "\n".join(x for x in (current, note) if x)[-1000:]
            self.client.update_client(int(patient.id), comment=patient.comment)
        except MisError:
            log.warning("Отметка обзвона не записана в карточку: стенд не ответил")

    def escalate(self, patient: Patient | None, description: str) -> str:
        self.escalations.append({"patient": patient.full_name if patient else None, "description": description})
        if patient is None:
            # Экстренный звонок без идентификации: стенд требует client_id — заявка на технического пациента
            # (разрешено заказчиком), регистратор видит, что личность не установлена. Стенд не ответил — перевод
            # звонка идёт и без заявки: рекомендация 103/112 не должна теряться из-за сбоя МИС
            try:
                patient = self.identify(*TECH_PATIENT, create=True)
                description = f"{description}\nПациент не идентифицирован (заявка на технического пациента)"
                return str(self.client.create_escalation(int(patient.id), description)["id"])
            except MisError:
                log.warning("Заявка без идентификации не создана: стенд не ответил")
                return ""
        return str(self.client.create_escalation(int(patient.id), description)["id"])
