"""Сквозная проверка на настоящем стенде МИС «Регистратура»: сценарии разговора без заглушек.

Каждый сценарий — разговор с ассистентом текстом (тот же агент и каскад понимания, что в звонке), а итог проверяется
отдельным запросом к API стенда: создана ли запись, перенесена ли, отменена ли, записан ли итог напоминания
в карточку, создана ли заявка на перевод. Так жюри увидит те же действия в админ-панели стенда.

    set -a; . /opt/hackathon/ai.env; set +a
    .venv/bin/python tools/stand_e2e.py            # отчёт: eval/stand_e2e.md, eval/results/stand_e2e.json

Пациент вымышленный (на стенде запрещены реальные данные). Лимит стенда — 10 запросов в минуту: перед каждым
сценарием прогон ждёт, пока окно лимита освободится, а если стенд всё же ответил 429 — ждёт по Retry-After.
Время реплики в отчёте — работа ассистента; ожидание лимита стенда указано отдельно. Прогон занимает 10–15 минут.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core.version import commit  # noqa: E402

from core.agent import Agent, Session  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.llm import OpenAICompatLLM  # noqa: E402
from domains.clinic.domain import ClinicDomain  # noqa: E402
from domains.clinic.mis import STATUS_CANCELLED, STATUS_PLANNED, MisClient, MisRegistry, parse_dt, stand_today  # noqa: E402
from domains.clinic.student import Student  # noqa: E402
from domains.clinic.timetable import Timetable  # noqa: E402

LAST, FIRST, BORN = "Сквознов", "Тест", date(1990, 5, 5)
IDENTITY = f"{LAST} {FIRST}, {BORN:%d.%m.%Y}"
BORN_SAID = f"{BORN:%d.%m.%Y}"  # исходящий звонок: дату рождения собеседник называет сам


def main() -> None:
    load_dotenv()  # токен из .env, как в README (на ВМ — ещё и /opt/hackathon/ai.env через run.sh)
    token = os.getenv("DBI_API_TOKEN")
    if not token:
        sys.exit("Нужен DBI_API_TOKEN (на ВМ — в /opt/hackathon/ai.env)")
    client = MisClient(os.getenv("DBI_API_URL", ""), token,
                       max_wait=30, attempts=10)  # пакетный прогон: лимит 10 запросов в минуту — ждём сколько нужно
    registry = MisRegistry(client, timetable=Timetable.load(os.getenv("TIMETABLE_DIR", "data")), require_timetable=True)
    matched = sum(d.timetable_id is not None for d in registry.doctors.values())
    settings = get_settings()
    domain = ClinicDomain(schedule=registry, student=Student.load(os.getenv("STUDENT_DIR", "data/student")))
    agent = Agent(domain, OpenAICompatLLM(settings) if settings.llm_enabled else None, today=stand_today)

    # Чистый старт: у тестового пациента не должно быть активных записей с прошлых прогонов
    patient = registry.identify(LAST, FIRST, BORN, create=True)
    for b in list(registry.bookings_of(patient)):
        registry.cancel(b.id)

    results: list[dict] = []

    def talk(name: str, phrases: list[str], session: Session | None = None) -> tuple[Session, list[dict]]:
        client.pace(8)  # перед сценарием освобождаем окно лимита стенда, чтобы разговор не ждал посреди реплики
        s = session or agent.new_session()
        turns = []
        for text in phrases:
            t0, w0 = time.perf_counter(), client.waited
            step = agent.handle(s, text)
            waited = client.waited - w0  # ожидание по Retry-After стенда — не время ассистента, показываем отдельно
            turn = {"patient": text, "assistant": step.reply, "nlu": s.last_nlu,
                    "ms": round((time.perf_counter() - t0 - waited) * 1000)}
            if waited:
                turn["limit_wait_ms"] = round(waited * 1000)
            turns.append(turn)
        print(f"— {name}: {len(turns)} реплик", flush=True)
        return s, turns

    def visit(visit_id: str) -> dict | None:
        return next((v for v in client.visits(client_id=int(patient.id)) if str(v["id"]) == str(visit_id)), None)

    def escalation(s: Session) -> dict | None:
        """Заявка, созданная в разговоре, — прочитанная обратно со стенда."""
        number = (s.last.payload or {}).get("escalation") if s.last and isinstance(s.last.payload, dict) else None
        return client.get_escalation(int(number)) if number else None

    def header(esc: dict | None) -> str:
        return (esc or {}).get("description", "").split("\n")[0]

    def record(name: str, scenario: str, turns: list[dict], checks: dict[str, bool], evidence: dict) -> None:
        results.append({"id": name, "scenario": scenario, "ok": all(checks.values()), "checks": checks,
                        "evidence": evidence, "turns": turns})

    # С-01 Запись
    s, turns = talk("book", ["Запишите меня к терапевту", "давайте первый", IDENTITY, "да"])
    booked = s.state.result
    v = visit(booked.id) if booked else None
    record("book", "С-01 запись на приём", turns,
           {"запись создана в МИС": bool(v), "статус «Запланирована»": bool(v) and v["status"] == STATUS_PLANNED,
            "время совпадает с выбранным": bool(v) and parse_dt(v["scheduled_at"]) == booked.slot.start},
           {"visit_id": booked.id if booked else None, "scheduled_at": v["scheduled_at"] if v else None})

    if booked is None:  # без записи остальные сценарии не о чем проверять — отчёт с ошибкой
        print("Запись не создана:", turns[-1]["assistant"])
        sys.exit(1)

    # С-05 Статус записи
    _, turns = talk("status", ["Когда у меня запись?", IDENTITY])
    record("status", "С-05 статус записи", turns,
           {"номер записи назван": bool(booked) and booked.id in turns[-1]["assistant"]}, {"visit_id": booked.id if booked else None})

    # С-10 Напоминание: исходящий звонок, подтверждение — отметка в карточке пациента
    state, hello = domain.start_reminder(registry.bookings[booked.id])
    s = Session(state=state, history=[{"role": "assistant", "content": hello}])
    _, turns = talk("reminder", ["Да, это я.", BORN_SAID, "Да, обязательно приду."], s)
    comment = client.get_client(int(patient.id)).get("comment") or ""
    line = f"Подтверждена запись #{booked.id} {booked.slot.start:%d.%m.%Y %H:%M}"  # строка, которую проверяет заказчик
    record("reminder", "С-10 напоминание о визите", [{"assistant": hello}] + turns,
           {"до даты рождения врач и время не названы": "Напоминаю" not in turns[0]["assistant"],
            "в карточке строка «Подтверждена запись #id дата время приёма»": line in comment.split("\n")},
           {"comment_line": line if line in comment else comment[-120:]})

    # С-03 Перенос
    before = visit(booked.id)
    s, turns = talk("reschedule", ["Хочу перенести запись", IDENTITY, "давайте второй", "да"])
    after = visit(booked.id)
    record("reschedule", "С-03 перенос записи", turns,
           {"время изменено в МИС": bool(before and after) and after["scheduled_at"] != before["scheduled_at"],
            "запись осталась «Запланирована»": bool(after) and after["status"] == STATUS_PLANNED},
           {"visit_id": booked.id, "было": before and before["scheduled_at"], "стало": after and after["scheduled_at"]})

    # С-04 Отмена (сначала ассистент предлагает перенос)
    _, turns = talk("cancel", ["Отмените мою запись", IDENTITY, "нет, отмените", "да"])
    v = visit(booked.id)
    record("cancel", "С-04 отмена записи", turns,
           {"статус «Отменена» в МИС": bool(v) and v["status"] == STATUS_CANCELLED}, {"visit_id": booked.id})

    # 7.3 Жалоба — перевод на сотрудника с заявкой в МИС
    s, turns = talk("complaint", ["Хочу пожаловаться на регистратора, мне нагрубили", IDENTITY])
    esc = escalation(s)
    record("complaint", "7.3 жалоба → перевод с карточкой (R-03)", turns,
           {"заявка в МИС «ПЕРЕВОД R-03 P2»": header(esc) == "ПЕРЕВОД R-03 P2", "перевод, а не запись": s.state.intent == "escalate"},
           {"escalation_id": esc and esc["id"], "первая строка": header(esc)})

    # С-07 Расписание врача — часы и ближайшее время по расписанию кейса, без изменений в МИС
    s, turns = talk("schedule", ["Когда принимает терапевт?"])
    record("schedule", "С-07 расписание врача", turns,
           {"названо ближайшее свободное время": "Ближайшее свободное время" in turns[-1]["assistant"],
            "предложена запись, без перевода": s.state.question == "offer_book" and not s.state.escalation}, {})

    # Справка раздела 6 — ответ по регламенту без перевода
    s, turns = talk("faq", ["До скольки вы работаете в субботу?"])
    record("faq", "Справка: режим работы", turns,
           {"ответ по регламенту": "в субботу с девяти" in turns[-1]["assistant"], "без перевода": not s.state.escalation}, {})

    # Медицинский вопрос — граничная фраза и предложение записи; настаивает — перевод R-04 (регламент 6.9, DBI 20.10)
    s, turns = talk("medical", ["Какие таблетки выпить от давления?", "нет, мне нужен совет, что выпить", IDENTITY])
    esc = escalation(s)
    record("medical", "Медицинский вопрос → граничная фраза и запись; настаивает → R-04", turns,
           {"граничная фраза и предложение записи": "не могу давать медицинские советы" in turns[0]["assistant"]
            and turns[0]["assistant"].endswith("Записать вас к терапевту?"),
            "заявка в МИС «ПЕРЕВОД R-04 P3»": header(esc) == "ПЕРЕВОД R-04 P3"}, {"escalation_id": esc and esc["id"]})

    # Экстренный случай — сразу 103/112, без вопросов о ФИО
    s, turns = talk("emergency", ["У мужа давит в груди и отдаёт в руку"])
    esc = escalation(s)
    record("emergency", "С-14 экстренный случай → 103/112 (R-01)", turns,
           {"названы 103 и 112": "сто три" in turns[-1]["assistant"] and "сто двенадцать" in turns[-1]["assistant"],
            "без вопроса о ФИО": "фамилию" not in turns[-1]["assistant"],
            "заявка в МИС «ПЕРЕВОД R-01 P1» (технический пациент)": header(esc) == "ПЕРЕВОД R-01 P1"},
           {"escalation_id": esc and esc["id"]})

    # Просьба соединить с человеком — перевод с карточкой и заявкой в МИС
    s, turns = talk("human", ["Соедините с оператором", IDENTITY])
    esc = escalation(s)
    record("human", "Путь к живому оператору (R-02)", turns,
           {"заявка в МИС «ПЕРЕВОД R-02 P2»": header(esc) == "ПЕРЕВОД R-02 P2", "перевод": s.state.escalation == "human"},
           {"escalation_id": esc and esc["id"]})

    # С-06 Вызов врача на дом — заявку принимает ассистент, задача регистратору в МИС
    s, turns = talk("home_visit", ["Мне нужно вызвать врача на дом", IDENTITY, "улица Тестовая, дом 1, температура и кашель"])
    esc = escalation(s)
    record("home_visit", "С-06 вызов врача на дом", turns,
           {"заявка в МИС «ВЫЗОВ НА ДОМ»": header(esc).startswith("ВЫЗОВ НА ДОМ"),
            "срок по регламенту: до восемнадцати часов": "до восемнадцати часов" in turns[-1]["assistant"]},
           {"escalation_id": esc and esc["id"], "первая строка": header(esc)})

    # С-10 «не смогу» → перенос прямо в звонке-напоминании, итог — в записи МИС
    s, turns = talk("book2", ["Запишите меня к терапевту", "давайте второй", IDENTITY, "да"])
    second = s.state.result
    if second is not None:
        before = visit(second.id)
        state, hello = domain.start_reminder(registry.bookings[second.id])
        s = Session(state=state, history=[{"role": "assistant", "content": hello}])
        _, turns = talk("reminder_move", ["Да, это я.", BORN_SAID, "Нет, завтра не смогу.", "давайте первый", "да"], s)
        after = visit(second.id)
        record("reminder_move", "С-10 напоминание: «не смогу» → перенос", [{"assistant": hello}] + turns,
               {"время изменено в МИС": bool(before and after) and after["scheduled_at"] != before["scheduled_at"]},
               {"visit_id": second.id, "было": before and before["scheduled_at"], "стало": after and after["scheduled_at"]})
        # С-10 «Не звоните мне больше» — отметка «Отказ от обзвона ДД.ММ.ГГГГ» в карточке, запись не меняется
        state, hello = domain.start_reminder(registry.bookings[second.id])
        s = Session(state=state, history=[{"role": "assistant", "content": hello}])
        _, turns = talk("opt_out", ["Да, это я.", BORN_SAID, "Не звоните мне больше"], s)
        comment = client.get_client(int(patient.id)).get("comment") or ""
        line = f"Отказ от обзвона {stand_today():%d.%m.%Y}"
        record("opt_out", "С-10 отказ от обзвона", [{"assistant": hello}] + turns,
               {"в карточке строка «Отказ от обзвона ДД.ММ.ГГГГ»": line in comment.split("\n"),
                "запись не изменилась": bool(visit(second.id)) and visit(second.id)["status"] == STATUS_PLANNED},
               {"comment_line": line})
        registry.cancel(second.id)  # уборка: тестовая запись не занимает время у других

    ok = sum(r["ok"] for r in results)
    out = Path("eval/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "stand_e2e.json").write_text(json.dumps({"date": datetime.now().isoformat(timespec="minutes"), "commit": commit(),
                                                    "ok": ok, "total": len(results), "results": results},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Сквозная проверка на стенде МИС — {datetime.now():%d.%m.%Y %H:%M}, коммит `{commit()}`", "",
             f"Понимание: каскад фразник → ученик → {settings.llm_model if settings.llm_enabled else 'правила'}. "
             f"Пациент вымышленный: {LAST} {FIRST}, {BORN:%d.%m.%Y}. Итог каждого сценария проверен запросом к API стенда. "
             f"Свободное время — по расписанию кейса: сопоставлено {matched} из {len(registry.doctors)} врачей стенда.",
             "", f"**Успешно: {ok} из {len(results)}**", "",
             "| Сценарий | Итог | Проверки в МИС | Доказательство |", "| :--- | :---: | :--- | :--- |"]
    for r in results:
        checks = "; ".join(("✓ " if v else "✗ ") + k for k, v in r["checks"].items())
        lines.append(f"| {r['scenario']} | {'✅' if r['ok'] else '❌'} | {checks} | "
                     f"{', '.join(f'{k}: {v}' for k, v in r['evidence'].items())} |")
    lines += ["", "## Разговоры", ""]
    for r in results:
        lines.append(f"### {r['scenario']}")
        for t in r["turns"]:
            if "patient" in t:
                lines.append(f"- **Пациент:** {t['patient']}")
            wait = f"; ожидание лимита стенда {t['limit_wait_ms'] / 1000:.0f} с" if t.get("limit_wait_ms") else ""
            lines.append(f"- **Ассистент:** {t['assistant']}" + (f" _({t['nlu']}, {t['ms']} мс{wait})_" if "nlu" in t else ""))
        lines.append("")
    Path("eval/stand_e2e.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Успешно: {ok} из {len(results)}. Отчёт: eval/stand_e2e.md")


if __name__ == "__main__":
    main()
