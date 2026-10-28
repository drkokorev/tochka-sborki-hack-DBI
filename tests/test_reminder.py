"""С-10: исходящий звонок-напоминание за день до приёма."""

from core.agent import Agent, Session
from domains.clinic.domain import REMIND_CONFIRMED, REMIND_UNSURE, REMIND_WRONG, ClinicDomain
from domains.clinic.schedule import seed_demo_bookings
from tests.conftest import TODAY, FakeLLM


def setup(*responses):
    domain = ClinicDomain(today=TODAY, use_phrasebook=not responses)
    seed_demo_bookings(domain.schedule)
    agent = Agent(domain, FakeLLM(*responses) if responses else None, today=lambda: TODAY)
    return agent, domain


ME = "да, это я, 12.05.1960"  # B0001 — Иванов Иван: после «это Иван?» собеседник сам называет дату рождения


def call(agent, domain, booking_id, *phrases):
    state, hello = domain.start_reminder(domain.schedule.bookings[booking_id])
    s = Session(state=state, history=[{"role": "assistant", "content": hello}])
    return hello, [agent.handle(s, p) for p in phrases]


def test_hello_does_not_disclose_visit_before_identity():
    agent, domain = setup()
    hello, steps = call(agent, domain, "B0001", "да, это я", "12.05.1960")
    assert hello.endswith("это Иван?") and "терапевт" not in hello
    # «да» ещё не проверка личности: дату рождения называет сам собеседник, только потом — врач и время
    assert "дату рождения" in steps[0].reply and "терапевт" not in steps[0].reply
    assert steps[1].reply.startswith("Напоминаю: у вас запись") and steps[1].reply.endswith("Вы придёте?")


def test_wrong_birth_date_does_not_disclose_visit():
    agent, domain = setup()
    _, steps = call(agent, domain, "B0001", "да, это я", "01.01.1950", "02.02.1951")
    assert steps[1].reply.startswith("Дата не совпала с карточкой.") and "терапевт" not in steps[1].reply
    assert steps[2].reply == REMIND_WRONG and domain.schedule.reminder_log[-1]["code"] == "WRONG_PERSON"


def test_confirmed():
    agent, domain = setup()
    _, steps = call(agent, domain, "B0001", ME, "да, приду")
    assert steps[1].reply == REMIND_CONFIRMED and steps[1].done
    assert domain.schedule.reminder_log[-1]["code"] == "CONFIRMED"
    assert "B0001" in domain.schedule.bookings


def test_cannot_come_offers_reschedule():
    agent, domain = setup()
    old = domain.schedule.bookings["B0001"].slot.start
    _, steps = call(agent, domain, "B0001", ME, "нет, завтра не получится", "первый", "да")
    assert steps[1].reply.startswith("Давайте перенесу.") and steps[1].payload
    assert all(o.start.date() != old.date() for o in steps[1].payload)
    assert steps[2].reply.startswith("Переношу запись:")
    assert steps[3].done and domain.schedule.bookings["B0001"].slot.start != old
    assert domain.schedule.reminder_log[-1]["code"] == "RESCHEDULED"


def test_llm_cancel_on_cannot_come_is_treated_as_reschedule():
    agent, domain = setup({"intent": "cancel"})
    _, steps = call(agent, domain, "B0001", ME, "не смогу прийти")
    assert steps[1].reply.startswith("Давайте перенесу.")


def test_cancel_from_reminder_offers_move_once():
    agent, domain = setup()
    _, steps = call(agent, domain, "B0001", ME, "отмените запись", "нет", "да")
    assert "Могу перенести её" in steps[1].reply
    assert steps[3].done and "B0001" not in domain.schedule.bookings
    assert domain.schedule.reminder_log[-1]["code"] == "CANCELLED"


def test_wrong_person_and_unsure_and_opt_out():
    agent, domain = setup()
    _, steps = call(agent, domain, "B0001", "нет, вы ошиблись")
    assert steps[0].reply == REMIND_WRONG and domain.schedule.reminder_log[-1]["code"] == "WRONG_PERSON"
    _, steps = call(agent, domain, "B0001", ME, "ну может быть")
    assert steps[1].reply == REMIND_UNSURE and domain.schedule.reminder_log[-1]["code"] == "UNSURE"
    _, steps = call(agent, domain, "B0001", ME, "не звоните мне больше")
    assert steps[1].reply.startswith("Хорошо, больше не буду звонить")
    assert domain.schedule.reminder_log[-1]["code"] == "OPT_OUT"


def test_red_flag_in_reminder_interrupts():
    agent, domain = setup()
    _, steps = call(agent, domain, "B0001", ME, "мне сейчас плохо, давит в груди")
    assert steps[1].reply.startswith("Пожалуйста, немедленно звоните")
    assert domain.schedule.reminder_log[-1]["code"] == "ESCALATED"


def test_reminder_targets_are_tomorrow():
    _, domain = setup()
    day = domain.schedule.reminder_day()
    assert all(b.slot.start.date() == day for b in domain.schedule.reminder_targets())


def test_llm_messages_do_not_start_with_assistant():
    agent, domain = setup({"confirm": True})
    _, steps = call(agent, domain, "B0001", "да")
    sent = agent.llm.calls[0]
    assert [m["role"] for m in sent] == ["system", "user"] and "это [ФИО]?" in sent[0]["content"]
    assert "Иван" not in str(sent)  # 152-ФЗ: имя пациента во внешнюю модель не уходит


def test_llm_tomorrow_in_cannot_come_is_not_new_day():
    agent, domain = setup({"intent": "reschedule", "day": "2026-10-06", "confirm": False})
    old = domain.schedule.bookings["B0001"].slot.start
    _, steps = call(agent, domain, "B0001", ME, "завтра не смогу, перенесите")
    assert steps[1].payload and all(o.start.date() != old.date() for o in steps[1].payload)
