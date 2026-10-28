"""Стенд «LLM играет пациента» с заскриптованным пациентом: проверки по фактам работают без сети."""

import random

from core.agent import Agent
from domains.clinic.domain import ClinicDomain
from domains.clinic.schedule import seed_demo_bookings
from eval.scenarios import SCENARIOS
from eval.simulate import TODAY, Run, asr_noise
from tests.conftest import FakeLLM

BY_ID = {s["id"]: s for s in SCENARIOS}


def play(sc_id, *patient_lines):
    domain = ClinicDomain(today=TODAY)
    seed_demo_bookings(domain.schedule)
    agent = Agent(domain, None, today=lambda: TODAY)
    return Run(BY_ID[sc_id], agent, domain, FakeLLM(*patient_lines), noise=None).play()


def test_booking_success_is_checked_by_registry():
    res = play("book_basic", "Запишите к терапевту", "первый", "Смирнова Ольга 12.03.1978", "да", "спасибо [КОНЕЦ]")
    assert res["ok"], res["checks"]
    assert res["checks"]["слоты не выдуманы"] and res["turns"] == 5


def test_failed_goal_is_reported():
    res = play("book_basic", "До свидания [КОНЕЦ]")
    assert not res["ok"] and res["checks"]["запись создана"] is False


def test_emergency_without_identity():
    res = play("emergency", "У мужа давит в груди, отдаёт в руку!", "Хорошо [КОНЕЦ]")
    assert res["ok"], res["checks"]


def test_reminder_scenario():
    res = play("reminder_ok", "Да, это я", "12.05.1960", "Да, приду [КОНЕЦ]")
    assert res["ok"], res["checks"]


def test_asr_noise_is_deterministic():
    assert asr_noise("Запишите к лору, пожалуйста", random.Random(1)) == asr_noise("Запишите к лору, пожалуйста",
                                                                                     random.Random(1))


def test_stand_mode_restores_fixtures_and_binds_visit_ids():
    """--stand: у вымышленных пациентов перед сценарием — ровно исходные записи, номера визитов стенда
    подставляются вместо B0001–B0003 демо-набора."""
    from domains.clinic.mis import MisRegistry
    from eval.scenarios import SCENARIOS
    from eval.simulate import StandFixtures, _bind
    from tests.test_mis import FakeMisClient

    client = FakeMisClient()
    client.doctors_db += [{"id": 7, "full_name": "Зубова Анна Петровна", "specialty": "Стоматолог-терапевт", "is_active": True},
                          {"id": 8, "full_name": "Сердцев Игорь Ильич", "specialty": "Кардиолог", "is_active": True}]
    registry = MisRegistry(client, today=TODAY)
    fixtures = StandFixtures(registry)
    domain = ClinicDomain(schedule=registry)
    ids = fixtures.prepare(domain)
    assert set(ids) == {"B0001", "B0002", "B0003"} and len(client.visits_db) == 3
    # Сценарий отменил одну запись и создал лишнюю — к следующему всё восстановлено
    registry.cancel(ids["B0003"])
    ivanov = fixtures.patients["Иванов", "Иван"]
    registry.book(registry.find(specialty="терапевт", limit=5)[-1], ivanov.full_name, ivanov.id)
    again = fixtures.prepare(domain)
    assert again["B0001"] == ids["B0001"] and again["B0003"] != ids["B0003"]
    assert sorted(b.id for p in fixtures.patients.values() for b in registry.bookings_of(p)) == sorted(again.values())
    status = _bind(next(s for s in SCENARIOS if s["id"] == "status"), again, domain)
    assert status["expect"]["status"] == again["B0001"]
    by_name = _bind(next(s for s in SCENARIOS if s["id"] == "book_doctor_name"), again, domain)
    assert "Иванова" in by_name["goal"]


def test_urgent_persona_outside_registry_hours_expects_103(monkeypatch):
    """Острое состояние вне часов регистратуры по регламенту — как угроза жизни (103, R-01): проверка стенда
    это учитывает, а не требует перевода R-11 в любое время суток."""
    from datetime import datetime
    from domains.clinic import reminders
    line = "У ребёнка третий день высокая температура, не сбивается"
    assert play("urgent", line, "Петров Миша 03.05.2018", "Жду [КОНЕЦ]")["ok"]
    monkeypatch.setattr(reminders, "clinic_now", lambda: datetime(2026, 10, 5, 21, 0))
    res = play("urgent", line, "Хорошо [КОНЕЦ]")
    assert res["ok"], (res["checks"], res["transcript"])

