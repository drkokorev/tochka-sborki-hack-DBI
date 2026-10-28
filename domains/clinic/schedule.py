"""Врачи, расписание и записи. После 13 окт сюда добавится адаптер под данные или API МИС от DBI."""

from __future__ import annotations

import functools
import threading

import zlib
from datetime import date, datetime, time, timedelta

from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel

from core.ru import WEEKDAYS_DAT_PL, cardinal, day_ordinal, hour_gen, join_and, on_date, time_words

from . import reminders as c10

if TYPE_CHECKING:
    from .timetable import Timetable

SPECIALTIES = [
    "терапевт", "стоматолог", "оториноларинголог", "офтальмолог",
    "невролог", "кардиолог", "дерматолог", "педиатр",
]
_SPECIALTY_ALIASES = {
    "лор": "оториноларинголог",
    "отоларинголог": "оториноларинголог",
    "ухо-горло-нос": "оториноларинголог",
    "окулист": "офтальмолог",
    "глазной": "офтальмолог",
    "зубной": "стоматолог",
    "дантист": "стоматолог",
    "детский врач": "педиатр",
    "врач общей практики": "терапевт",
}


def normalize_specialty(raw: str) -> str | None:
    key = raw.strip().lower().replace("ё", "е")
    if key in _SPECIALTY_ALIASES:
        return _SPECIALTY_ALIASES[key]
    # «к терапевту», «стоматологу» — сравниваем по основе слова
    return next((s for s in SPECIALTIES if key.startswith(s[:-1]) or s.startswith(key)), None)


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def _cabinet(number: str) -> str:
    return cardinal(int(number)) if number.isdigit() and int(number) < 1000 else number


def on_day(d: date) -> str:
    """«девятнадцатого октября» — без дня недели, для «не принимает до …»."""
    from core.ru import MONTHS_GEN
    return f"{day_ordinal(d.day)} {MONTHS_GEN[d.month - 1]}"


class Doctor(BaseModel):
    id: str
    name: str          # «Фамилия Имя Отчество»
    specialty: str
    branch: str = ""
    weekdays: list[int] = [0, 1, 2, 3, 4]
    start_hour: int = 9
    end_hour: int = 17
    slot_minutes: int = 30
    is_active: bool = True
    cabinet: str = ""
    hours: dict[int, tuple[int, int]] = {}   # из расписания кейса: день недели → (с, до)
    timetable_id: str | None = None          # врач найден в расписании — слоты берём оттуда

    @property
    def surname(self) -> str:
        return self.name.split()[0]

    def describe(self) -> str:
        where = f", {self.branch}" if self.branch else ""
        if self.cabinet:
            where += f", кабинет {_cabinet(self.cabinet)}"
        hours = self.hours or {d: (self.start_hour, self.end_hour) for d in self.weekdays}
        groups: dict[tuple[int, int], list[int]] = {}
        for day, span in sorted(hours.items()):
            groups.setdefault(span, []).append(day)
        parts = [f"по {join_and([WEEKDAYS_DAT_PL[d] for d in days])} с {hour_gen(a)} до {hour_gen(b)}"
                 for (a, b), days in groups.items()]
        return f"{self.specialty.capitalize()} {self.name} принимает {'; '.join(parts)}{where}"


class Slot(BaseModel):
    doctor_id: str
    doctor_name: str
    specialty: str
    branch: str = ""
    cabinet: str = ""
    start: datetime

    def spoken(self) -> str:
        where = f", {self.branch}" if self.branch else ""
        if self.cabinet:
            where += f", кабинет {_cabinet(self.cabinet)}"
        return f"{self.when()}, {self.specialty} {self.doctor_name}{where}"

    def when(self) -> str:
        """«в среду, седьмого октября, в девять тридцать» — для голоса."""
        return f"{on_date(self.start.date())}, в {time_words(self.start.time())}"


class Patient(BaseModel):
    id: str
    last_name: str
    first_name: str
    birth_date: date
    comment: str = ""
    middle_name: str = ""  # отчество: в исходящем звонке называем только имя и отчество

    @property
    def full_name(self) -> str:
        return f"{self.last_name} {self.first_name}"


class Booking(BaseModel):
    id: str
    slot: Slot
    patient_name: str
    patient_id: str | None = None
    created: datetime | None = None  # когда создана (время клиники): моложе 24 часов — не обзваниваем (С-10)


# Коды результата обзвона из регламента (С-10)
REMINDER_NOTES = {
    "CONFIRMED": "пациент подтвердил визит",
    "RESCHEDULED": "запись перенесена",
    "CANCELLED": "запись отменена",
    "WRONG_PERSON": "трубку взял другой человек",
    "UNSURE": "пациент не уверен, статус не менялся",
    "OPT_OUT": "отказ от обзвона",
    "OPT_OUT_ALL": "отказ от всех уведомлений (звонки и SMS)",
    "CALL_PLACED": "попытка дозвона",
    "ESCALATED": "переведён на регистратора",
    "NO_ANSWER": "не удалось поговорить",
    "NOT_REACHED": "не дозвонились — пациент не взял трубку",
}


class SlotTakenError(Exception):
    pass


DOCTORS = [
    Doctor(id="d1", name="Белова Анна Сергеевна", specialty="терапевт", branch="филиал на Лесной", weekdays=[0, 2, 4], start_hour=9, end_hour=15),
    Doctor(id="d2", name="Ковалёв Игорь Павлович", specialty="терапевт", branch="филиал на Садовой", weekdays=[1, 3, 5], start_hour=12, end_hour=20),
    Doctor(id="d3", name="Миронова Елена Викторовна", specialty="стоматолог", branch="филиал на Лесной", weekdays=[0, 1, 2, 3], start_hour=10, end_hour=18),
    Doctor(id="d4", name="Гусев Олег Андреевич", specialty="стоматолог", branch="филиал на Садовой", weekdays=[2, 4, 5], start_hour=9, end_hour=14),
    Doctor(id="d5", name="Яковлева Марина Олеговна", specialty="оториноларинголог", branch="филиал на Лесной", weekdays=[1, 3], start_hour=9, end_hour=13),
    Doctor(id="d6", name="Соколов Денис Игоревич", specialty="офтальмолог", branch="филиал на Садовой", weekdays=[0, 3], start_hour=14, end_hour=20),
    Doctor(id="d7", name="Лебедева Ольга Николаевна", specialty="невролог", branch="филиал на Лесной", weekdays=[1, 4], start_hour=10, end_hour=16),
    Doctor(id="d8", name="Орлов Сергей Михайлович", specialty="кардиолог", branch="филиал на Садовой", weekdays=[0, 2], start_hour=9, end_hour=13),
    Doctor(id="d9", name="Фёдорова Ирина Алексеевна", specialty="дерматолог", branch="филиал на Лесной", weekdays=[2, 4], start_hour=13, end_hour=19),
    Doctor(id="d10", name="Никитин Павел Романович", specialty="педиатр", branch="филиал на Садовой", weekdays=[0, 1, 2, 3, 4], start_hour=8, end_hour=14),
]


def locked(method):
    """Метод реестра под его блокировкой (RLock: вложенные вызовы, например reschedule → cancel, разрешены)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


class Schedule:
    def __init__(self, doctors: list[Doctor], today: "date | Callable[[], date]", horizon_days: int = 14,
                 busy_share: float = 0.45, timetable: Timetable | None = None):
        self.timetable = timetable
        if timetable is not None:
            doctors = [self._enrich(d) for d in doctors]
        self.doctors = {d.id: d for d in doctors}
        # «Сегодня» — дата или часы (date.today, stand_today): сервер работает неделями, дата не должна застывать
        self._today = today
        # Веб и телефония меняют реестр из разных потоков: проверка «слот свободен» и запись — под одной блокировкой
        self.lock = threading.RLock()
        self.horizon_days = horizon_days
        self.busy_share = busy_share
        self.bookings: dict[str, Booking] = {}
        self._by_slot: dict[tuple[str, datetime], str] = {}
        self._next_id = 1
        self.patients: dict[str, Patient] = {}
        self.escalations: list[dict] = []
        self.reminder_log: list[dict] = []
        self.notify_opt_out: dict[str, str] = {}   # пациент → OPT_OUT / OPT_OUT_ALL (со стенда — из комментария)
        self.reminder_final: dict[str, tuple[str, datetime]] = {}  # запись → (итог, время приёма) со стенда
        self.history: dict[str, dict] = {}  # журнал записей: id → последнее состояние (для демо без стенда)

    @property
    def today(self) -> date:
        return self._today() if callable(self._today) else self._today

    @today.setter
    def today(self, value: "date | Callable[[], date]") -> None:
        self._today = value

    def _enrich(self, doctor: Doctor) -> Doctor:
        assert self.timetable is not None
        tt = self.timetable.by_name(doctor.name)
        if tt is None:
            return doctor
        return doctor.model_copy(update={
            "cabinet": tt.cabinet, "hours": tt.hours, "timetable_id": tt.id, "slot_minutes": tt.slot_minutes,
            "weekdays": sorted(tt.hours), "branch": doctor.branch,
        })

    def absence(self, doctor: Doctor) -> tuple[str, date] | None:
        """Врач в отпуске или на больничном сегодня — до какой даты."""
        if self.timetable is None or doctor.timetable_id is None:
            return None
        return self.timetable.absent_until(self.timetable.doctors[doctor.timetable_id], self.today)

    def find_doctor(self, name: str) -> Doctor | None:
        """Ищет по фамилии в любом падеже и порядке слов: «к Беловой», «Анна Белова»."""
        for word in _norm(name).split():
            stem = word[:5]
            if len(stem) < 4:
                continue
            for d in self.doctors.values():
                if _norm(d.surname).startswith(stem):
                    return d
        return None

    def doctors_for(self, specialty: str | None) -> list[Doctor]:
        return [d for d in self.doctors.values() if specialty is None or d.specialty == specialty]

    def is_busy(self, doctor_id: str, start: datetime) -> bool:
        if (doctor_id, start) in self._by_slot:
            return True
        doc = self.doctors.get(doctor_id)
        if doc is not None and doc.timetable_id is not None:
            return False  # занятость уже учтена в расписании: там только свободные слоты
        # Детерминированная «занятость» мока: одинакова между запусками
        return zlib.crc32(f"{doctor_id}{start:%Y%m%d%H%M}".encode()) % 100 < self.busy_share * 100

    def find(
        self,
        *,
        specialty: str | None = None,
        doctor_id: str | None = None,
        day: date | None = None,
        time_from: time | None = None,
        time_to: time | None = None,
        limit: int = 3,
    ) -> list[Slot]:
        days = [day] if day else [self.today + timedelta(days=i) for i in range(1, self.horizon_days + 1)]
        slots: list[Slot] = []
        for d in days:
            if not self.today < d <= self.today + timedelta(days=self.horizon_days):
                continue
            for doc in self.doctors.values():
                if (doctor_id and doc.id != doctor_id) or (specialty and doc.specialty != specialty):
                    continue
                if not doc.is_active:
                    continue
                for t in self._starts(doc, d):
                    if ((time_from is None or t.time() >= time_from)
                            and (time_to is None or t.time() < time_to)
                            and not self.is_busy(doc.id, t)):
                        slots.append(Slot(doctor_id=doc.id, doctor_name=doc.name, specialty=doc.specialty,
                                          branch=doc.branch, cabinet=doc.cabinet, start=t))
        slots.sort(key=lambda s: s.start)
        return slots[:limit]

    def _starts(self, doc: Doctor, d: date) -> list[datetime]:
        """Начала слотов врача в этот день: из расписания кейса, иначе — по сетке рабочих часов."""
        if self.timetable is not None and doc.timetable_id is not None:
            return self.timetable.free_slots(doc.timetable_id, d, d)
        if d.weekday() not in doc.weekdays:
            return []
        t, end = datetime.combine(d, time(doc.start_hour)), datetime.combine(d, time(doc.end_hour))
        starts = []
        while t < end:
            starts.append(t)
            t += timedelta(minutes=doc.slot_minutes)
        return starts

    @locked
    def identify(self, last_name: str, first_name: str, birth_date: date, *, create: bool) -> Patient | None:
        """Пациент по фамилии, имени и дате рождения — как на стенде МИС."""
        key = (_norm(last_name), _norm(first_name), birth_date)
        for p in self.patients.values():
            if (_norm(p.last_name), _norm(p.first_name), p.birth_date) == key:
                return p
        if not create:
            return None
        patient = Patient(id=str(len(self.patients) + 1), last_name=last_name, first_name=first_name,
                          birth_date=birth_date)
        self.patients[patient.id] = patient
        return patient

    def bookings_of(self, patient: Patient) -> list[Booking]:
        return sorted((b for b in self.bookings.values() if b.patient_id == patient.id), key=lambda b: b.slot.start)

    # ---------- исходящие напоминания (С-10) ----------

    def reminder_day(self) -> date:
        """Звоним за день до приёма; приёмы в воскресенье не бывают — в субботу звоним про понедельник."""
        day = self.today + timedelta(days=1)
        return day + timedelta(days=1) if day.weekday() == 6 else day

    def reminder_targets(self) -> list[Booking]:
        day = self.reminder_day()
        return sorted((b for b in self.bookings.values() if b.slot.start.date() == day), key=lambda b: b.slot.start)

    def patient_of(self, booking: Booking) -> Patient | None:
        return self.patients.get(booking.patient_id or "")

    @locked
    def log_reminder(self, booking: Booking, patient: Patient | None, code: str) -> None:
        self.reminder_log.append({"booking": booking.id, "patient": patient.full_name if patient else booking.patient_name,
                                  "patient_id": booking.patient_id, "start": booking.slot.start.isoformat(),
                                  "code": code, "note": REMINDER_NOTES.get(code, code),
                                  "at": c10.clinic_now().isoformat(timespec="seconds")})

    def set_opt_out(self, patient: Patient, code: str) -> None:
        """Отказ от обзвона (OPT_OUT) или от всех уведомлений (OPT_OUT_ALL) — вне звонка-напоминания.
        Пишется и в журнал обзвона: по нему правила С-10 узнают отказ и после перезапуска."""
        self.notify_opt_out[patient.id] = code
        self.reminder_log.append({"booking": "", "patient": patient.full_name, "patient_id": patient.id, "code": code,
                                  "note": REMINDER_NOTES.get(code, code), "at": c10.clinic_now().isoformat(timespec="seconds")})

    def note_reminder_attempt(self, booking: Booking) -> None:
        """Звонок-напоминание ушёл — попытка учитывается в правилах С-10 (до 2 попыток, интервал от 2 часов)."""
        self.log_reminder(booking, None, c10.ATTEMPT)

    @locked
    def escalate(self, patient: Patient | None, description: str) -> str:
        """Заявка на соединение с регистратором. Возвращает её номер."""
        self.escalations.append({"patient": patient.full_name if patient else None, "description": description})
        return str(len(self.escalations))

    @locked
    def book(self, slot: Slot, patient_name: str, patient_id: str | None = None) -> Booking:
        if self.is_busy(slot.doctor_id, slot.start):
            raise SlotTakenError(slot.spoken())
        booking = Booking(id=f"B{self._next_id:04d}", slot=slot, patient_name=patient_name, patient_id=patient_id,
                          created=c10.clinic_now())
        self._next_id += 1
        self._store(booking)
        self._log(booking, "Запланирована")
        return booking

    @locked
    def cancel(self, booking_id: str) -> Booking:
        booking = self.bookings.pop(booking_id)
        del self._by_slot[(booking.slot.doctor_id, booking.slot.start)]
        self._log(booking, "Отменена")
        return booking

    @locked
    def reschedule(self, booking_id: str, slot: Slot) -> Booking:
        if self.is_busy(slot.doctor_id, slot.start):
            raise SlotTakenError(slot.spoken())
        old = self.cancel(booking_id)
        moved = old.model_copy(update={"slot": slot})
        self._store(moved)
        self._log(moved, "Запланирована", note="перенесена")
        return moved

    def _log(self, booking: Booking, status: str, note: str = "") -> None:
        patient = self.patients.get(booking.patient_id or "")
        self.history[booking.id] = {
            "id": booking.id, "patient": booking.patient_name or (patient.full_name if patient else ""),
            "birth_date": patient.birth_date.strftime("%d.%m.%Y") if patient else "",
            "doctor": booking.slot.doctor_name, "specialty": booking.slot.specialty,
            "at": booking.slot.start.isoformat(timespec="minutes"), "status": status,
            "updated": datetime.now().isoformat(timespec="minutes"),
            "note": note or self.history.get(booking.id, {}).get("note", ""),
        }

    def visit_log(self, refresh: bool = False) -> list[dict]:
        """Все записи: активные, отменённые, перенесённые. В демо — из истории реестра."""
        for b in self.bookings.values():  # записи, созданные до запуска журнала (демо-пациенты)
            if b.id not in self.history:
                self._log(b, "Запланирована")
        return list(self.history.values())

    def find_bookings(self, *, booking_id: str | None = None, patient_name: str | None = None) -> list[Booking]:
        if booking_id:
            found = [self.bookings[booking_id]] if booking_id in self.bookings else []
        elif patient_name:
            # «Иван Иванов» и «Иванов Иван» — один и тот же пациент
            query = set(_norm(patient_name).split())
            found = [b for b in self.bookings.values()
                     if query <= set(_norm(b.patient_name).split()) or set(_norm(b.patient_name).split()) <= query]
        else:
            found = []
        return sorted(found, key=lambda b: b.slot.start)

    def _store(self, booking: Booking) -> None:
        self.bookings[booking.id] = booking
        self._by_slot[(booking.slot.doctor_id, booking.slot.start)] = booking.id


# Фамилия Имя → (специальность, дата рождения)
DEMO_PATIENTS = {
    "Иванов Иван": ("терапевт", date(1960, 5, 12)),
    "Смирнова Анна": ("стоматолог", date(1985, 3, 17)),
    "Петров Сергей": ("кардиолог", date(1958, 11, 2)),
}


def seed_demo_bookings(schedule: Schedule) -> None:
    """Существующие записи, чтобы на демо можно было проверить статус, перенос и отмену."""
    for name, (specialty, birth_date) in DEMO_PATIENTS.items():
        last_name, first_name = name.split()
        patient = schedule.identify(last_name, first_name, birth_date, create=True)
        slots = schedule.find(specialty=specialty, limit=1)
        if slots and patient:
            booking = schedule.book(slots[0], patient.full_name, patient.id)
            booking.created = None  # записи «из прошлого» — правило 24 часов С-10 к ним не относится
            if hasattr(schedule, "_save_booking"):
                schedule._save_booking(booking)  # демо-реестр в SQLite — сохраняем и это
