"""Основной сценарий кейса одной командой, без ключей и сети: `make demo` или `python tools/demo_dialog.py`.

Ассистент проходит разговоры пациента на демо-данных (фразник и правила, без LLM): запись, статус, перенос,
отмена, справка, перевод при острой боли и напоминание. После каждого разговора итог сверяется с реестром:
создана ли запись, изменилось ли время, отменена ли, создана ли заявка на перевод. Код выхода 0 — всё прошло.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent import Agent, Session  # noqa: E402
from domains.clinic.domain import ClinicDomain  # noqa: E402
from domains.clinic.schedule import seed_demo_bookings  # noqa: E402

TODAY = date(2026, 10, 19)
IVAN = "Иванов Иван, двенадцатое мая шестидесятого года"   # демо-пациент с записью к терапевту


def run() -> int:
    domain = ClinicDomain(today=TODAY)
    seed_demo_bookings(domain.schedule)
    agent = Agent(domain, None, today=lambda: TODAY)
    reg = domain.schedule
    failures = 0

    def dialog(title: str, phrases: list[str], check, session: Session | None = None) -> None:
        nonlocal failures
        s = session or agent.new_session()
        print(f"\n━━ {title}")
        if session is not None:
            print(f"  Ассистент: {session.history[0]['content']}")
        for text in phrases:
            step = agent.handle(s, text)
            print(f"  Пациент:   {text}\n  Ассистент: {step.reply}   [{s.last_nlu}]")
        ok, what = check(s)
        failures += not ok
        print(f"  {'✅' if ok else '❌'} {what}")

    before = set(reg.bookings)
    dialog("С-01 Запись на приём",
           ["Здравствуйте, мне к врачу", "к терапевту на среду после обеда", "да, подходит",
            "Петров Сергей, второе ноября пятьдесят восьмого года", "да"],
           lambda s: (bool(set(reg.bookings) - before), "запись создана в реестре"))
    dialog("С-05 Статус записи", ["Когда у меня запись?", IVAN],
           lambda s: ("номер записи" in s.history[-1]["content"], "назван номер записи"))
    target = next(b for b in reg.bookings.values() if b.patient_name.startswith("Иванов"))
    old = target.slot.start
    dialog("С-03 Перенос", ["Хочу перенести запись", IVAN, "второй", "да"],
           lambda s: (reg.bookings[target.id].slot.start != old, "время записи изменено, номер прежний"))
    dialog("С-04 Отмена", ["Отмените мою запись", IVAN, "нет, отмените", "да"],
           lambda s: (target.id not in reg.bookings, "запись отменена"))
    dialog("Справка", ["До скольки вы работаете в субботу?"],
           lambda s: ("в субботу" in s.history[-1]["content"], "ответ по регламенту, без перевода"))
    flags = len(reg.escalations)
    dialog("7.3 Острая боль → регистратор", ["Очень сильно болит зуб, опухла щека", "Смирнова Анна 17.03.1985"],
           lambda s: (len(reg.escalations) > flags and s.state.escalation == "dental",
                      "перевод с карточкой и заявкой, без попытки лечить"))
    booking = next(iter(reg.bookings.values()))
    state, hello = domain.start_reminder(booking)
    born = reg.patient_of(booking).birth_date  # собеседник сам называет дату рождения — сверка с карточкой
    dialog("С-10 Напоминание (исходящий звонок)", ["да, это я", f"{born:%d.%m.%Y}", "приду"],
           lambda s: (reg.reminder_log and reg.reminder_log[-1]["code"] == "CONFIRMED", "визит подтверждён"),
           Session(state=state, history=[{"role": "assistant", "content": hello}]))
    total = 7
    print(f"\nИтог: {total - failures} из {total} сценариев доведены до действия в реестре.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
