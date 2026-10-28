"""Поддакивания: на них ассистент не замолкает."""

from apps.voice import is_backchannel


def test_backchannels():
    for text in ["угу", "Ага.", "да-да", "ну да, понятно", "Хорошо", "мгм"]:
        assert is_backchannel(text), text
    for text in ["", "да, запишите", "нет", "соедините с оператором", "угу, а сколько стоит", "второй"]:
        assert not is_backchannel(text), text


def test_single_meaningless_word_over_reply_is_filler():
    """«Угу» распознаётся как «иду» — одно слово без смысла для диалога тоже не перебивание; стоп-слова — всегда."""
    from datetime import date

    from apps.voice import is_filler, meaningful_for
    from core.agent import Agent
    from domains.clinic.domain import ClinicDomain

    agent = Agent(ClinicDomain(today=date(2026, 10, 19)), None, today=lambda: date(2026, 10, 19))
    check = meaningful_for(agent, agent.new_session().state)
    for text in ["Иду.", "алло", "угу", "ну да"]:
        assert is_filler(text, check), text
    for text in ["Подождите", "стоп", "нет", "оператора", "терапевт", "отменить", "в субботу работаете?"]:
        assert not is_filler(text, check), text
    # Ассистент зачитывает варианты — «второй» поверх них это выбор, а не поддакивание
    s = agent.new_session()
    agent.handle(s, "Запишите к терапевту")
    assert not is_filler("второй", meaningful_for(agent, s.state))



def test_yes_is_an_answer_when_reply_asks_but_noise_word_is_still_ignored():
    from datetime import date

    from apps.voice import is_filler, meaningful_for
    from core.agent import Agent
    from domains.clinic.domain import ClinicDomain

    agent = Agent(ClinicDomain(today=date(2026, 10, 19)), None, today=lambda: date(2026, 10, 19))
    check = meaningful_for(agent, agent.new_session().state)
    assert not is_filler("угу", check, asks=True) and not is_filler("да", check, asks=True)
    assert is_filler("Иду.", check, asks=True)
