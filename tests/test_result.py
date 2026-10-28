"""Машиночитаемый итог реплики (clinic.turn/v1), уверенность понимания и карточка перевода (clinic.transfer/v1)."""

from apps.journal import Journal
from core.agent import Agent
from domains.clinic.domain import ClinicDomain
from domains.clinic.result import FIX_HINT, HINTS, turn_result
from domains.clinic.schedule import seed_demo_bookings
from tests.conftest import TODAY


def run(*phrases):
    domain = ClinicDomain(today=TODAY)
    seed_demo_bookings(domain.schedule)
    agent = Agent(domain, None, today=lambda: TODAY)
    s = agent.new_session()
    out = []
    for text in phrases:
        step = agent.handle(s, text)
        out.append(turn_result(s.state, s.last_update, step.reply, step.payload, source=s.last_nlu,
                               confidence=s.last_confidence))
    return s, out


def test_booking_turns_are_machine_readable():
    s, res = run("Запишите меня к терапевту", "первый", "Тестов Тест, 01.01.1990", "да")
    first = res[0]
    assert first["schema"] == "clinic.turn/v1" and first["intent"] == "book" and first["awaiting"] == "choice"
    assert len(first["offered"]) == 3 and {"n", "doctor", "specialty", "at"} <= set(first["offered"][0])
    assert first["understanding"]["source"] == "phrasebook" and first["understanding"]["level"] == "high"
    assert first["hint"] == HINTS["choice"]
    assert res[1]["awaiting"] == "identity" and res[2]["awaiting"] == "confirm"
    done = res[3]["action"]
    assert done["type"] == "booking_created" and done["booking"]["id"] == s.state.result.id
    assert done["booking"]["at"].startswith(s.state.result.slot.start.date().isoformat())


def test_cancel_status_and_transfer_actions():
    _, res = run("Отмените мою запись", "Иванов Иван, 12.05.1960", "нет, отмените", "да")
    assert res[-1]["action"]["type"] == "booking_cancelled"
    _, res = run("Когда у меня запись?", "Иванов Иван, 12.05.1960")
    assert res[-1]["action"]["type"] == "status_reported" and res[-1]["action"]["bookings"]
    _, res = run("Очень сильно болит зуб, опухла щека", "Смирнова Анна 17.03.1985")
    assert res[-1]["status"] == "transferred" and res[-1]["action"]["reason"] == "dental"


def test_unsure_understanding_comes_with_a_way_to_fix_it():
    domain = ClinicDomain(today=TODAY, use_phrasebook=False)
    agent = Agent(domain, None, today=lambda: TODAY)
    s = agent.new_session()
    step = agent.handle(s, "мне бы к лору попасть в среду")     # без фразника и LLM — правила
    r = turn_result(s.state, s.last_update, step.reply, step.payload, source=s.last_nlu, confidence=s.last_confidence)
    assert s.last_nlu == "rules" and r["understanding"]["level"] == "medium" and r["hint"] == FIX_HINT


def test_transfer_card_has_structured_fields():
    journal = Journal()
    domain = ClinicDomain(today=TODAY)
    seed_demo_bookings(domain.schedule)
    agent = Agent(domain, None, today=lambda: TODAY)
    s = agent.new_session()
    journal.start("c1", "веб", "Здравствуйте")
    for text in ["Хочу пожаловаться на врача", "Иванов Иван, 12.05.1960"]:
        step = agent.handle(s, text)
        journal.turn("c1", text, step, s.state, s.last_update, s.last_nlu, 1.0)
    card = journal.transfers()[0]
    assert card["schema"] == "clinic.transfer/v1" and card["reason_code"] == "complaint" and card["priority"] == "P2"
    assert card["patient"]["found_in_mis"] and card["patient"]["birth_date"] == "1960-05-12"
    assert card["escalation_id"]
