from datetime import date, datetime

from core.agent import Agent
from core.ru import cardinal
from domains.clinic.domain import ClinicDomain
from domains.clinic.mis import MisRegistry
from domains.clinic.timetable import Timetable, parse_hours
from tests.conftest import TODAY  # понедельник, 5 октября 2026
from tests.test_mis import FakeMisClient

DOCTORS_CSV = """id,name,specialty,department,cabinet,slot_minutes,Пн,Вт,Ср,Чт,Пт,Сб,Вс
D001,Иванова Мария Петровна,Терапевт,Терапевтическое,214,20,08:00–14:00,14:00–20:00,08:00–14:00,14:00–20:00,08:00–14:00,—,—
D005,Кузнецова Ольга Юрьевна,Гинеколог,Женская консультация,305,30,"09:00–13:00, 14:00–18:00",—,—,—,—,—,—
"""
FREE_CSV = """doctor_id,date,start,minutes
D001,2026-10-06,14:40,20
D001,2026-10-06,15:00,20
D001,2026-10-08,15:20,20
D001,2026-10-09,08:20,20
D005,2026-10-12,09:00,30
"""
ABSENT_CSV = """doctor_id,kind,date_from,date_to
D005,Отпуск,2026-10-01,2026-10-11
"""


def make(tmp_path):
    for name, text in [("doctors.csv", DOCTORS_CSV), ("free_slots.csv", FREE_CSV), ("unavailable.csv", ABSENT_CSV)]:
        (tmp_path / name).write_text(text, encoding="utf-8")
    tt = Timetable.load(tmp_path)
    domain = ClinicDomain(schedule=MisRegistry(FakeMisClient(), today=TODAY, timetable=tt))
    return Agent(domain, None, today=lambda: TODAY), domain


def talk(agent, *phrases):
    s = agent.new_session()
    return s, [agent.handle(s, p) for p in phrases]


def test_parse_hours_and_cardinal():
    assert parse_hours("08:00–14:00") == (8, 14)
    assert parse_hours("09:00–13:00, 14:00–18:00") == (9, 18)
    assert parse_hours("—") is None
    assert cardinal(214) == "двести четырнадцать" and cardinal(101) == "сто один" and cardinal(305) == "триста пять"


def test_slots_come_from_timetable_with_cabinet(tmp_path):
    agent, domain = make(tmp_path)
    assert domain.schedule.doctors["3"].cabinet == "214"
    _, steps = talk(agent, "запишите к терапевту")
    starts = [o.start for o in steps[0].payload]
    assert starts == [datetime(2026, 10, 6, 14, 40), datetime(2026, 10, 8, 15, 20), datetime(2026, 10, 9, 8, 20)]
    assert "кабинет двести четырнадцать" in steps[0].payload[0].spoken()


def test_booked_slot_is_not_offered_again(tmp_path):
    agent, domain = make(tmp_path)
    talk(agent, "запишите к терапевту на вторник", "первый", "Петров Иван 12.03.1958", "да")
    _, steps = talk(agent, "запишите к терапевту на вторник")
    assert [o.start for o in steps[0].payload] == [datetime(2026, 10, 6, 15, 0)]


def test_schedule_answer_uses_real_hours_and_absence(tmp_path):
    agent, _ = make(tmp_path)
    _, steps = talk(agent, "Когда принимает Иванова?")
    reply = steps[0].reply
    assert "по понедельникам, средам и пятницам с восьми до четырнадцати" in reply
    assert "по вторникам и четвергам с четырнадцати до двадцати" in reply and "кабинет двести четырнадцать" in reply

    _, steps = talk(agent, "Когда принимает Кузнецова?")
    assert "не принимает до одиннадцатого октября включительно" in steps[0].reply
    assert "Отпуск" not in steps[0].reply  # причину отсутствия не называем


def test_missing_timetable_is_ok(tmp_path):
    assert Timetable.load(tmp_path) is None
    assert date(2026, 10, 5) == TODAY
