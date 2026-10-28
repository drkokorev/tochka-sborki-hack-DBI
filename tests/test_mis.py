from datetime import date, datetime

from core.agent import Agent
from domains.clinic.domain import TRANSFER, ClinicDomain
from domains.clinic.mis import MisRegistry, fmt_dt, parse_dt
from tests.conftest import TODAY


class FakeMisClient:
    """Повторяет контракт стенда «Регистратура» в памяти и запоминает вызовы."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.clients_db: dict[int, dict] = {}
        self.visits_db: dict[int, dict] = {}
        self.doctors_db = [
            {"id": 3, "full_name": "Иванова Мария Петровна", "specialty": "терапевт", "is_active": True},
            {"id": 4, "full_name": "Сидоров Олег Ильич", "specialty": "терапевт", "is_active": False},
            {"id": 5, "full_name": "Кузнецова Ольга Юрьевна", "specialty": "Гинеколог", "is_active": True},
        ]

    def doctors(self):
        self.calls.append(("doctors",))
        return self.doctors_db

    def clients(self):
        self.calls.append(("clients",))
        return list(self.clients_db.values())

    def find_clients(self, last_name, first_name, birth_date):
        self.calls.append(("find_clients", last_name, first_name, birth_date))
        bd = birth_date.strftime("%d.%m.%Y")
        return [c for c in self.clients_db.values()
                if (c["last_name"], c["first_name"], c["birth_date"]) == (last_name, first_name, bd)]

    def create_client(self, last_name, first_name, birth_date):
        self.calls.append(("create_client", last_name, first_name, birth_date))
        cid = len(self.clients_db) + 1
        self.clients_db[cid] = {"id": cid, "last_name": last_name, "first_name": first_name,
                             "birth_date": birth_date.strftime("%d.%m.%Y")}
        return self.clients_db[cid]

    def get_client(self, client_id):
        self.calls.append(("get_client", client_id))
        return self.clients_db[client_id]

    def update_client(self, client_id, **fields):
        self.calls.append(("update_client", client_id, fields))
        self.clients_db[client_id].update(fields)
        return self.clients_db[client_id]

    def visits(self, **filters):
        self.calls.append(("visits", filters))
        return [v for v in self.visits_db.values() if v["status"] == filters.get("status", v["status"])]

    def create_visit(self, client_id, doctor_id, at):
        self.calls.append(("create_visit", client_id, doctor_id, fmt_dt(at)))
        vid = len(self.visits_db) + 1
        doctor = next(d for d in self.doctors_db if d["id"] == doctor_id)
        self.visits_db[vid] = {"id": vid, "client_id": client_id, "doctor": doctor,
                               "scheduled_at": fmt_dt(at), "status": "Запланирована"}
        return self.visits_db[vid]

    def update_visit(self, visit_id, **fields):
        self.calls.append(("update_visit", visit_id, fields))
        self.visits_db[visit_id].update(fields)
        return self.visits_db[visit_id]

    def create_escalation(self, client_id, description):
        self.calls.append(("create_escalation", client_id, description))
        return {"id": 77, "client_id": client_id, "description": description}


def make(client=None):
    client = client or FakeMisClient()
    domain = ClinicDomain(schedule=MisRegistry(client, today=TODAY))
    return Agent(domain, None, today=lambda: TODAY), domain, client


def talk(agent, *phrases):
    s = agent.new_session()
    return s, [agent.handle(s, p) for p in phrases]


def test_dates_in_stand_format():
    assert fmt_dt(datetime(2026, 10, 7, 9, 30)) == "07.10.2026 09:30"
    assert parse_dt("12.09.2026 10:30") == datetime(2026, 9, 12, 10, 30)


def test_doctors_come_from_stand_and_inactive_are_not_offered():
    agent, domain, _ = make()
    assert domain.specialties == ["гинеколог", "терапевт"]
    _, steps = talk(agent, "запишите к терапевту")
    assert {o.doctor_id for o in steps[0].payload} == {"3"}  # деактивированный Сидоров не предлагается


def test_booking_creates_client_and_visit_in_stand():
    agent, domain, client = make()
    _, steps = talk(agent, "запишите к терапевту на среду", "первый", "Петров Иван 12.03.1958", "да")
    assert steps[3].done
    assert ("create_client", "Петров", "Иван", date(1958, 3, 12)) in client.calls
    visit = next(c for c in client.calls if c[0] == "create_visit")
    assert visit[1:3] == (1, 3) and visit[3].startswith("07.10.2026")
    assert steps[3].payload.id == "1"


def test_cancel_and_reschedule_update_stand():
    client = FakeMisClient()
    agent, domain, _ = make(client)
    talk(agent, "запишите к терапевту", "первый", "Петров Иван 12.03.1958", "да")
    _, steps = talk(agent, "перенесите мою запись", "Петров Иван 12.03.1958", "второй", "да")
    assert steps[3].done
    assert any(c[0] == "update_visit" and "scheduled_at" in c[2] for c in client.calls)
    _, steps = talk(agent, "отмените запись", "Петров Иван 12.03.1958", "нет", "да")
    assert steps[3].done and client.visits_db[1]["status"] == "Отменена"


def test_escalation_is_recorded_in_stand():
    agent, _, client = make()
    _, steps = talk(agent, "Врач нагрубил, соедините с регистратором", "Петров Иван 12.03.1958")
    assert steps[1].reply.endswith(TRANSFER)
    esc = next(c for c in client.calls if c[0] == "create_escalation")
    # первая строка заявки — правило и приоритет регламента (уточнение DBI 20.10), дальше суть
    assert esc[1] == 1 and esc[2].startswith("ПЕРЕВОД R-03 P2\nЖалоба: ") and "нагрубил" in esc[2]


def test_mis_failure_goes_to_registrar():
    from domains.clinic.mis import MisError

    client = FakeMisClient()
    agent, _, _ = make(client)

    def broken(*args, **kwargs):
        raise MisError(503, "НЕДОСТУПЕН", "Стенд недоступен")

    client.create_visit = broken
    _, steps = talk(agent, "запишите к терапевту", "первый", "Петров Иван 12.03.1958", "да")
    assert steps[3].done and steps[3].reply.startswith("Извините, система не отвечает")


STAND_DOCTORS = [
    {"id": 1, "full_name": "Орлова Анна Игоревна", "specialty": "Терапевт", "is_active": True},
    {"id": 2, "full_name": "Лебедев Пётр Ильич", "specialty": "Лор (отоларинголог)", "is_active": True},
    {"id": 3, "full_name": "Зуева Ольга Павловна", "specialty": "Стоматолог-терапевт", "is_active": True},
    {"id": 4, "full_name": "Кротов Иван Сергеевич", "specialty": "Стоматолог-хирург", "is_active": True},
    {"id": 5, "full_name": "Кравцова Светлана Игоревна", "specialty": "Акушер-гинеколог", "is_active": True},
    {"id": 6, "full_name": "Морозов Олег Юрьевич", "specialty": "Невролог", "is_active": True},
    {"id": 7, "full_name": "Белкина Ирина Олеговна", "specialty": "Детский невролог", "is_active": True},
    {"id": 8, "full_name": "Ершов Денис Андреевич", "specialty": "Уролог", "is_active": True},
]


import pytest  # noqa: E402


@pytest.mark.parametrize("phrase, specialty", [
    ("запишите к лору", "лор (отоларинголог)"),
    ("болит зуб", "стоматолог-терапевт"),
    ("мне к гинекологу", "акушер-гинеколог"),
    ("к неврологу", "невролог"),
    ("ребёнка к детскому неврологу", "детский невролог"),
    ("нужен уролог", "уролог"),
])
def test_stand_specialty_names(phrase, specialty):
    client = FakeMisClient()
    client.doctors_db = STAND_DOCTORS
    agent, _, _ = make(client)
    s, _ = talk(agent, phrase)
    assert s.state.specialty == specialty


def test_reminder_outcome_is_written_to_client_card():
    from core.agent import Session

    client = FakeMisClient()
    agent, domain, _ = make(client)
    talk(agent, "запишите к терапевту", "первый", "Петров Иван 12.03.1958", "да")
    domain.schedule.patients.clear()  # как после перезапуска: карточку читаем со стенда
    booking = next(iter(domain.schedule.bookings.values()))
    state, hello = domain.start_reminder(booking)
    assert hello.endswith("это Иван?") and ("get_client", 1) in client.calls
    s = Session(state=state)
    agent.handle(s, "да")
    agent.handle(s, "12.03.1958")  # собеседник сам называет дату рождения
    agent.handle(s, "приду")
    # строка, которую проверяет заказчик: время приёма, а не дата звонка
    assert client.clients_db[1]["comment"] == f"Подтверждена запись #{booking.id} {booking.slot.start:%d.%m.%Y %H:%M}"


def test_visit_log_from_stand_includes_cancelled():
    client = FakeMisClient()
    agent, domain, _ = make(client)
    talk(agent, "запишите к терапевту", "первый", "Петров Иван 12.03.1958", "да")
    talk(agent, "отмените запись", "Петров Иван 12.03.1958", "нет", "да")
    log = domain.schedule.visit_log()
    assert [(v["patient"], v["status"]) for v in log] == [("Петров Иван", "Отменена")]
    assert log[0]["birth_date"] == "12.03.1958" and log[0]["specialty"] == "терапевт"
    calls = len(client.calls)
    domain.schedule.visit_log()
    assert len(client.calls) == calls  # кэш: стенд не опрашиваем чаще раза в минуту


def test_without_case_timetable_no_slots_are_invented():
    """Стенд не хранит расписание: без расписания кейса время не придумываем — перевод к регистратору (R-06)."""
    from core.agent import Agent
    from domains.clinic.domain import ClinicDomain
    client = FakeMisClient()
    domain = ClinicDomain(schedule=MisRegistry(client, today=TODAY, require_timetable=True))
    agent = Agent(domain, None, today=lambda: TODAY)
    s = agent.new_session()
    step = agent.handle(s, "Запишите к терапевту")
    assert s.state.escalation == "mis_error" and step.done and not step.payload


def test_failed_slot_recheck_does_not_create_visit():
    """Стенд не ответил на сверку занятости — визит не создаём (стенд двойную запись не запрещает), а переводим к
    регистратору (R-06): кэш не доказывает, что время не занял другой канал."""
    from domains.clinic.mis import MisError

    class Flaky(FakeMisClient):
        def visits(self, **filters):
            if "doctor_id" in filters:
                raise MisError(429, "ПРЕВЫШЕН_ЛИМИТ", "лимит")
            return super().visits(**filters)

    agent, domain, client = make(Flaky())
    _, steps = talk(agent, "запишите к терапевту", "первый", "Петров Иван 12.03.1958", "да")
    assert not any(c[0] == "create_visit" for c in client.calls)
    assert steps[3].done and steps[3].reply.startswith("Извините, система не отвечает")


def test_emergency_without_identity_is_recorded_on_technical_patient():
    agent, domain, client = make()
    _, steps = talk(agent, "У мужа давит в груди и отдаёт в руку")
    esc = next(c for c in client.calls if c[0] == "create_escalation")
    tech = client.clients_db[esc[1]]
    assert (tech["last_name"], tech["first_name"]) == ("Технический", "Пациент")
    assert esc[2].startswith("ПЕРЕВОД R-01 P1\n") and "не идентифицирован" in esc[2]


def test_opt_out_and_confirmation_are_appended_and_restored_after_restart():
    """Отметки — строками заказчика, дописываются к актуальному комментарию (чужие строки не затираются);
    после перезапуска реестр читает их со стенда и по этим записям больше не звонит."""
    from datetime import datetime as dt

    from domains.clinic import reminders
    client = FakeMisClient()
    agent, domain, _ = make(client)
    talk(agent, "запишите к терапевту", "первый", "Петров Иван 12.03.1958", "да")
    talk(agent, "запишите к гинекологу", "первый", "Петров Иван 12.03.1958", "да")
    first, second = list(domain.schedule.bookings.values())
    client.clients_db[1]["comment"] = "Аллергия на пенициллин"  # запись регистратора — не трогаем
    domain.schedule.log_reminder(first, domain.schedule.patients["1"], "CONFIRMED")
    domain.schedule.log_reminder(second, domain.schedule.patients["1"], "OPT_OUT")
    lines = client.clients_db[1]["comment"].split("\n")
    assert lines[0] == "Аллергия на пенициллин"
    assert lines[1] == f"Подтверждена запись #{first.id} {first.slot.start:%d.%m.%Y %H:%M}"
    assert lines[2] == f"Отказ от обзвона {TODAY:%d.%m.%Y}"
    restarted = MisRegistry(client, today=TODAY)  # перезапуск сервера
    assert restarted.notify_opt_out["1"] == "OPT_OUT"
    assert restarted.reminder_final[first.id] == ("CONFIRMED", first.slot.start)
    assert reminders.final_code(restarted, first.id) == "CONFIRMED"
    now = dt(TODAY.year, TODAY.month, TODAY.day, 12, 0)
    assert reminders.skip_reason(restarted, restarted.bookings[first.id], now) == "пациент отказался от обзвона"


def test_stand_failures_do_not_break_emergency_or_reminder_answers():
    """Стенд не отвечает: экстренная рекомендация 103/112 звучит, итог напоминания не роняет разговор."""
    from domains.clinic.mis import MisError

    class Down(FakeMisClient):
        def find_clients(self, *a, **k):
            raise MisError(503, "СТЕНД_НЕДОСТУПЕН", "нет связи")

    agent, domain, _ = make(Down())
    _, steps = talk(agent, "У мужа давит в груди и отдаёт в руку")
    assert steps[0].reply.startswith("Пожалуйста, немедленно звоните в скорую помощь")

    class Limited(FakeMisClient):
        def get_client(self, client_id):
            raise MisError(429, "ПРЕВЫШЕН_ЛИМИТ", "лимит")

    client = Limited()
    agent, domain, _ = make(client)
    talk(agent, "запишите к терапевту", "первый", "Петров Иван 12.03.1958", "да")
    booking = next(iter(domain.schedule.bookings.values()))
    assert booking.created is not None  # запись со стенда — правило 24 часов действует
    from core.agent import Session
    state, _ = domain.start_reminder(booking)
    s = Session(state=state)
    for text in ("да", "12.03.1958", "приду"):
        step = agent.handle(s, text)
    assert step.done and domain.schedule.reminder_log[-1]["code"] == "CONFIRMED"


def test_known_patient_and_technical_patient_are_not_searched_again():
    """Лимит стенда — 10 запросов в минуту: найденную карточку повторно не ищем (запись, потом жалоба того же
    пациента; технический пациент для каждой экстренной заявки)."""
    agent, domain, client = make()
    talk(agent, "запишите к терапевту", "первый", "Петров Иван, 12.03.1958", "да")
    searches = sum(c[0] == "find_clients" for c in client.calls)
    talk(agent, "хочу пожаловаться на врача", "Петров Иван, 12.03.1958")
    talk(agent, "у мужа давит в груди и отдаёт в руку")
    talk(agent, "у мужа давит в груди и отдаёт в руку")
    assert sum(c[0] == "find_clients" for c in client.calls) == searches + 1  # только технический пациент, один раз
    escalations = [c for c in client.calls if c[0] == "create_escalation"]
    assert len(escalations) == 3 and escalations[1][1] == escalations[2][1]

