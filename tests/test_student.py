"""Каскад «ученик → LLM»: уверенный ученик отвечает без LLM, сомнительные случаи уходят модели."""

from core.agent import Agent
from domains.clinic.acts import ACT_HINTS, ACTS, act_fields
from domains.clinic.domain import ClinicDomain
from domains.clinic.schedule import seed_demo_bookings
from tests.conftest import TODAY, FakeLLM


class FakeStudent:
    tau = 0.8

    def __init__(self, answers):
        self.answers = answers  # текст → (акт, вероятность)

    def predict(self, text):
        return self.answers.get(text, ("other", 0.99))


def make(answers, *llm):
    domain = ClinicDomain(today=TODAY, student=FakeStudent(answers), use_phrasebook=False)
    seed_demo_bookings(domain.schedule)
    return Agent(domain, FakeLLM(*llm) if llm else None, today=lambda: TODAY), domain


def test_every_act_has_hint_and_fields():
    assert set(ACTS) <= set(ACT_HINTS)
    assert act_fields("intent:book") == {"intent": "book"} and act_fields("faq:hours") == {"topic": "hours"}
    assert act_fields("yes") == {"confirm": True} and act_fields("slots") == {}


def test_confident_student_skips_llm_and_rules_fill_slots():
    agent, _ = make({"хочу к лору завтра": ("intent:book", 0.95)}, {"intent": "status"})
    s = agent.new_session()
    step = agent.handle(s, "хочу к лору завтра")
    assert s.last_nlu == "student" and agent.llm.calls == []
    assert s.state.specialty == "оториноларинголог" and step.payload


def test_low_confidence_and_other_go_to_llm():
    agent, _ = make({"ну как бы это": ("intent:book", 0.5)}, {"intent": "book", "specialty": "терапевт"})
    s = agent.new_session()
    agent.handle(s, "ну как бы это")
    assert s.last_nlu == "llm" and len(agent.llm.calls) == 1


def test_identity_without_names_goes_to_llm():
    agent, _ = make({"это я": ("identity", 0.99)}, {})
    s = agent.new_session()
    agent.handle(s, "это я")
    assert s.last_nlu != "student"


def test_red_flag_from_rules_survives_student():
    agent, _ = make({"запишите к врачу, давит в груди": ("intent:book", 0.97)})
    s = agent.new_session()
    agent.handle(s, "запишите к врачу, давит в груди")
    assert s.state.escalation == "emergency"


def test_faq_by_student_without_regex_match():
    agent, domain = make({"а в выходные вы открыты": ("faq:hours", 0.93)})
    s = agent.new_session()
    step = agent.handle(s, "а в выходные вы открыты")
    assert "в субботу с девяти" in step.reply and not domain.schedule.escalations


def test_yes_with_thanks_confirms_even_if_student_guesses_gratitude():
    """Стенд «LLM-пациент»: «Да, всё верно. Спасибо!» на «Верно?» ученик принимал за благодарность."""
    agent, domain = make({"Да, всё верно. Спасибо!": ("faq:gratitude", 0.97)})
    s = agent.new_session()
    for text in ["Запишите к терапевту", "первый", "Петров Иван 12.03.1958"]:
        agent.handle(s, text)
    before = len(domain.schedule.bookings)
    step = agent.handle(s, "Да, всё верно. Спасибо!")
    assert step.done and len(domain.schedule.bookings) == before + 1


def test_reminder_yes_i_will_come_is_not_cancel():
    from core.agent import Session
    agent, domain = make({"Да, обязательно приду. Спасибо!": ("intent:cancel", 0.95), "Да, это я.": ("yes", 0.99)})
    state, hello = domain.start_reminder(domain.schedule.bookings["B0001"])
    s = Session(state=state, history=[{"role": "assistant", "content": hello}])
    agent.handle(s, "Да, это я.")
    agent.handle(s, "12.05.1960")
    step = agent.handle(s, "Да, обязательно приду. Спасибо!")
    assert step.done and domain.schedule.reminder_log[-1]["code"] == "CONFIRMED"
