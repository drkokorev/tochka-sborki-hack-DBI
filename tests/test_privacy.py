"""152-ФЗ: ФИО и дата рождения пациента не уходят во внешнюю LLM, но запись всё равно оформляется."""

from core.agent import Agent
from domains.clinic.domain import ClinicDomain
from domains.clinic.privacy import BIRTH, NAME, mask
from tests.conftest import TODAY, FakeLLM


def test_mask_hides_patient_but_keeps_doctor_and_request():
    domain = ClinicDomain(today=TODAY)
    assert mask("Петров Иван Николаевич, 12.03.1958") == f"{NAME} {BIRTH}"
    assert mask("двенадцатое марта пятьдесят восьмого года") == BIRTH
    assert mask("меня зовут Анна Смирнова, хочу к терапевту") == f"меня зовут {NAME}, хочу к терапевту"
    assert mask("Запишите к Беловой на завтра", is_doctor=domain.schedule.find_doctor) == "Запишите к Беловой на завтра"
    assert mask("Сколько стоит приём? Это срочно") == "Сколько стоит приём? Это срочно"


def test_llm_never_sees_identity_but_booking_completes():
    domain = ClinicDomain(today=TODAY, use_phrasebook=False)
    llm = FakeLLM({"intent": "book", "specialty": "терапевт"}, {"choice": 1}, {}, {"confirm": True})
    agent = Agent(domain, llm, today=lambda: TODAY)
    s = agent.new_session()
    for text in ["Запишите к терапевту", "первый", "Петров Иван Николаевич, 12.03.1958", "да"]:
        step = agent.handle(s, text)
    sent = str(llm.calls)
    assert "Петров" not in sent and "1958" not in sent and "Иван," not in sent
    assert step.done and s.state.patient.last_name == "Петров"  # личность поняли правила на машине



def test_lowercase_inflected_names_numbers_are_masked():
    """Распознавание отдаёт строчные буквы, пациенты называют себя в падежах — ПДн всё равно не уходят в LLM."""
    from datetime import date

    from domains.clinic.privacy import mask
    from domains.clinic.schedule import DOCTORS, Schedule

    doctor = Schedule(DOCTORS, date(2026, 10, 19)).find_doctor
    for text, secret in [("запишите иванова ивана петровича к терапевту", "иван"), ("Запишите Иванова Ивана Петровича", "Иван"),
                         ("кузнецова ольга 12.03.1978", "кузнецов"), ("полис 1234567890123456 хочу к лору", "1234"),
                         ("мой телефон +7 912 345-67-89", "912"), ("снилс 123-456-789 01", "456")]:
        assert secret.lower() not in mask(text, (), doctor).lower(), text
    assert mask("у меня паралич лица", (), doctor) == "у меня паралич лица"           # не отчество
    assert mask("к терапевту Беловой на 10:30", (), doctor) == "к терапевту Беловой на 10:30"  # врач и время — не ПДн


def test_identity_turn_never_reaches_llm():
    """Пока ассистент ждёт ФИО и дату рождения, реплику разбирают только правила на машине."""
    from core.agent import Agent
    from domains.clinic.domain import ClinicDomain
    from tests.conftest import TODAY, FakeLLM

    llm = FakeLLM({"intent": "book", "specialty": "терапевт"}, {"choice": 1})
    agent = Agent(ClinicDomain(today=TODAY, use_phrasebook=False), llm, today=lambda: TODAY)
    s = agent.new_session()
    agent.handle(s, "Запишите к терапевту")
    agent.handle(s, "давайте первый")
    calls = len(llm.calls)
    agent.handle(s, "петров сергей ильич второе ноября пятьдесят восьмого года")
    assert len(llm.calls) == calls and s.state.patient is not None
    assert all("петров" not in m["content"].lower() for call in llm.calls for m in call)
