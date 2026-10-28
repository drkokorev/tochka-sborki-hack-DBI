"""Стенд «LLM играет пациента»: персоны звонят ассистенту, итог проверяется по фактам в реестре.

    .venv/bin/python -m eval.simulate                       # NLU через LLM, пациент — LLM
    .venv/bin/python -m eval.simulate --nlu rules           # ассистент без LLM (базовая линия)
    .venv/bin/python -m eval.simulate --only book_basic,emergency --runs 2
    .venv/bin/python -m eval.simulate --stand               # на настоящем стенде МИС (нужен DBI_API_TOKEN)

Пациента играет другая модель (по умолчанию лёгкая GigaChat3-10B — в 15 раз дешевле основной),
чтобы ассистент и «пациент» не подыгрывали друг другу. Дневной лимит шлюза — 100 ₽: если он кончился,
прогон останавливается, а неполный отчёт пишется в eval/report_partial.md и не затирает полный.
Реплики пациента можно пропускать через шум распознавания (--asr-noise): склейки «к лору» → «клору»,
пропуск запятых, «э-э». Отчёт: eval/report.md и eval/report.json.
Метрики: успех по фактам, корректные переводы, отсутствие выдуманных слотов, число ходов, задержка NLU.

С --stand реестр — настоящий стенд МИС с расписанием кейса: записи, переносы, отмены и заявки создаются
на стенде, итог проверяется по нему же. Пациенты вымышленные; перед каждым сценарием у них восстанавливаются
исходные записи (как B0001–B0003 демо-набора), и номера визитов стенда подставляются в сценарий.
Лимит стенда 10 запросов в минуту — прогон идёт десятки минут. Отчёт: eval/report_stand.md.
"""

from __future__ import annotations

import argparse
import os
import json
import random
import re
import statistics
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

from core.agent import Agent, Session
from core.config import get_settings
from core.llm import OpenAICompatLLM, strip_think
from core.version import commit
from domains.clinic.domain import ClinicDomain
from domains.clinic import reminders
from domains.clinic.replies import transfer_header
from domains.clinic.student import Student
from domains.clinic.schedule import seed_demo_bookings

from .scenarios import ANNA, IVANOV, PETROV, SCENARIOS, SMIRNOVA

TODAY = date(2026, 10, 5)  # понедельник; демо-расписание детерминировано
END = "[КОНЕЦ]"

PATIENT_PROMPT = """Ты играешь пациента, который говорит по телефону с голосовым ассистентом регистратуры поликлиники.
Персона: {persona}
Цель звонка: {goal}
Твои данные (называй, только когда спросят): {last} {first} {middle}, дата рождения {birth}.
Правила:
- Отвечай одной короткой репликой, как живой человек по телефону (до 20 слов), без пояснений и кавычек.
- Не подсказывай ассистенту больше, чем знает обычный пациент. Ты не знаешь внутренних кодов и номеров вариантов, пока их не назвали.
- Когда цель достигнута, ассистент перевёл тебя на сотрудника или попрощался — коротко попрощайся и допиши {end}.
- Если разговор зашёл в тупик, попрощайся и допиши {end}."""


def asr_noise(text: str, rng: random.Random) -> str:
    """Типичные ошибки распознавания телефонной речи."""
    text = re.sub(r"\bк ([лт])", lambda m: "к" + m.group(1), text) if rng.random() < 0.5 else text
    text = text.replace(",", "") if rng.random() < 0.5 else text
    if rng.random() < 0.3:
        text = "э-э " + text
    return text.lower() if rng.random() < 0.5 else text


class Run:
    def __init__(self, scenario: dict, agent: Agent, domain: ClinicDomain, patient_llm, *, noise: random.Random | None):
        self.sc, self.agent, self.domain, self.patient, self.noise = scenario, agent, domain, patient_llm, noise
        self.transcript: list[dict] = []
        self.latencies: list[float] = []
        self.offered_ever: set = set()

    def start(self) -> tuple[Session, str]:
        if booking_id := self.sc.get("reminder"):
            state, hello = self.domain.start_reminder(self.domain.schedule.bookings[booking_id])
            return Session(state=state, history=[{"role": "assistant", "content": hello}]), hello
        from domains.clinic.domain import GREETING
        return self.agent.new_session(), GREETING

    def patient_says(self) -> str:
        who = self.sc["who"]
        system = PATIENT_PROMPT.format(persona=self.sc["persona"], goal=self.sc["goal"], end=END, **who)
        # Для модели-пациента роли зеркальны: ассистент регистратуры — «user», пациент — «assistant»
        messages = [{"role": "system", "content": system}] + [
            {"role": "user" if t["role"] == "assistant" else "assistant", "content": t["text"]}
            for t in self.transcript]
        return strip_think(self.patient.complete(messages)).strip().strip('"«»')

    def play(self, max_turns: int = 12) -> dict:
        session, hello = self.start()
        self.transcript.append({"role": "assistant", "text": hello})
        before = {k: v.slot.start for k, v in self.domain.schedule.bookings.items()}
        flags_before = len(self.domain.schedule.escalations)
        asked_identity = False
        for _ in range(max_turns):
            said = self.patient_says()
            done_by_patient = END in said
            said = said.replace(END, "").strip()
            if said:
                heard = asr_noise(said, self.noise) if self.noise else said
                self.transcript.append({"role": "patient", "text": heard})
                t = time.perf_counter()
                step = self.agent.handle(session, heard)
                self.latencies.append(time.perf_counter() - t)
                self.transcript.append({"role": "assistant", "text": step.reply, "nlu": session.last_nlu})
                asked_identity |= "фамилию" in step.reply.lower()
                for slot in getattr(session.state, "offered", []) or []:
                    self.offered_ever.add((slot.doctor_id, slot.start))
            if done_by_patient:
                break
        return self.verdict(before, flags_before, asked_identity)

    def verdict(self, before: dict, flags_before: int, asked_identity: bool) -> dict:
        exp, sched = self.sc["expect"], self.domain.schedule
        new = [b for k, b in sched.bookings.items() if k not in before]
        moved = [k for k, b in sched.bookings.items() if k in before and b.slot.start != before[k]]
        removed = [k for k in before if k not in sched.bookings]
        escalations = sched.escalations[flags_before:]
        replies = " ".join(t["text"] for t in self.transcript if t["role"] == "assistant")
        checks: dict[str, bool] = {}
        if "booked" in exp:
            # На стенде свои названия («лор (отоларинголог)»): ожидание приводим к справочнику реестра
            spec = self.domain._normalize_specialty(exp["booked"]) or exp["booked"]
            checks["запись создана"] = any(exp["booked"] in b.slot.specialty or spec == b.slot.specialty for b in new)
        if "rescheduled" in exp:
            checks["запись перенесена"] = exp["rescheduled"] in moved
        if "cancelled" in exp:
            checks["запись отменена"] = exp["cancelled"] in removed
        if "status" in exp:
            checks["статус назван"] = exp["status"] in replies
        if "faq" in exp:
            checks["справка дана"] = exp["faq"] in replies
            checks["без лишнего перевода"] = not escalations
        if "escalated" in exp:
            # первая строка заявки — правило и приоритет регламента: «ПЕРЕВОД R-01 P1»
            want = exp["escalated"]
            if want == "urgent" and not reminders.registry_open(reminders.clinic_now()):
                want = "emergency"  # острое состояние вне часов регистратуры — по регламенту как угроза жизни: 103, R-01
            route_ok = any(e["description"].split("\n")[0] == transfer_header(want) for e in escalations)
            checks["перевод с верным поводом"] = route_ok
            if exp.get("identity") is False:
                checks["без вопроса о ФИО"] = not asked_identity
        if exp.get("boundary"):
            checks["граничная фраза"] = "не могу давать медицинские советы" in replies
        if exp.get("no_transfer"):
            checks["без перевода"] = not escalations
        if "reminder" in exp:
            codes = [r["code"] for r in sched.reminder_log]
            checks["итог обзвона"] = bool(codes) and codes[-1] == exp["reminder"]
        if "booked" not in exp and "rescheduled" not in exp:
            checks["нет лишних записей"] = not new
        # Защита от выдумывания: каждая созданная или перенесённая запись — из реально предложенных слотов
        touched = new + [sched.bookings[k] for k in moved]
        checks["слоты не выдуманы"] = all((b.slot.doctor_id, b.slot.start) in self.offered_ever for b in touched)
        return {
            "id": self.sc["id"], "ok": all(checks.values()), "checks": checks,
            "turns": sum(1 for t in self.transcript if t["role"] == "patient"),
            "latency_ms": round(1000 * statistics.mean(self.latencies)) if self.latencies else 0,
            "llm_share": round(sum(t.get("nlu") == "llm" for t in self.transcript)
                               / max(1, sum("nlu" in t for t in self.transcript)), 2),
            "transcript": self.transcript,
        }


# Записи демо-набора, на которые ссылаются сценарии: у кого и к какому специалисту
FIXTURES = {"B0001": (IVANOV, "терапевт"), "B0002": (ANNA, "стоматолог"), "B0003": (PETROV, "кардиолог")}


class StandFixtures:
    """Вымышленные пациенты сценариев на стенде МИС: перед каждым сценарием — ровно исходные записи."""

    def __init__(self, registry):
        from datetime import datetime

        self.registry = registry
        self.patients = {}
        for who in (SMIRNOVA, IVANOV, ANNA, PETROV):
            born = datetime.strptime(who["birth"], "%d.%m.%Y").date()
            self.patients[who["last"], who["first"]] = registry.identify(who["last"], who["first"], born, create=True)
        self.ids: dict[str, str] = {}

    def prepare(self, domain: ClinicDomain) -> dict[str, str]:
        reg, keep = self.registry, {}
        for fid, (who, spec) in FIXTURES.items():
            if (b := reg.bookings.get(self.ids.get(fid, ""))) and spec in b.slot.specialty:
                keep[fid] = b.id
        for patient in self.patients.values():  # всё, что сценарии создали сверх исходного, — отменяем
            for b in list(reg.bookings_of(patient)):
                if b.id not in keep.values():
                    reg.cancel(b.id)
        for fid, (who, spec) in FIXTURES.items():
            if fid not in keep:
                patient = self.patients[who["last"], who["first"]]
                slot = reg.find(specialty=domain._normalize_specialty(spec), limit=1)[0]
                keep[fid] = reg.book(slot, patient.full_name, patient.id).id
        self.ids = keep
        return keep


def _bind(sc: dict, ids: dict[str, str], domain: ClinicDomain) -> dict:
    """Сценарий для стенда: номера записей демо-набора → номера визитов стенда, Белова → терапевт стенда."""
    text = json.dumps(sc, ensure_ascii=False)
    for fid, real in ids.items():
        text = text.replace(fid, real)
    sc = json.loads(text)
    if "Беловой" in sc["goal"]:
        doctor = next((d for d in domain.schedule.doctors_for("терапевт") if d.is_active), None)
        if doctor:
            sc["goal"] = sc["goal"].replace("доктору Беловой", f"доктору по фамилии {doctor.surname}")
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlu", choices=["llm", "rules"], default="llm")
    ap.add_argument("--patient-model", default="ai-sage/GigaChat3-10B-A1.8B")
    ap.add_argument("--only", default="")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--asr-noise", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--stand", action="store_true", help="настоящий стенд МИС вместо демо-реестра")
    args = ap.parse_args()
    args.out = args.out or ("eval/report_stand" if args.stand else "eval/report")

    settings = get_settings()
    patient_llm = OpenAICompatLLM(replace(settings, llm_model=args.patient_model, llm_json_mode=False))
    scenarios = [s for s in SCENARIOS if not args.only or s["id"] in args.only.split(",")]
    student = Student.load(os.getenv("STUDENT_DIR", "data/student")) if os.getenv("STUDENT", "1") == "1" else None
    registry = fixtures = None
    if args.stand:
        from domains.clinic.mis import MisClient, MisRegistry, stand_today
        from domains.clinic.timetable import Timetable

        token = os.getenv("DBI_API_TOKEN") or exit("Нужен DBI_API_TOKEN (на ВМ — в /opt/hackathon/ai.env)")
        client = MisClient(os.getenv("DBI_API_URL", ""), token,
                           max_wait=30, attempts=10)  # лимит стенда — ждём по Retry-After сколько нужно
        registry = MisRegistry(client, timetable=Timetable.load(os.getenv("TIMETABLE_DIR", "data")),
                               require_timetable=True)
        fixtures = StandFixtures(registry)
    results = []
    for run in range(args.runs):
        for sc in scenarios:
            if registry is not None:
                domain = ClinicDomain(schedule=registry, student=student)
                sc = _bind(sc, fixtures.prepare(domain), domain)
                today = stand_today
            else:
                domain = ClinicDomain(today=TODAY, student=student)  # тот же каскад, что в звонке: фразник → ученик → LLM
                seed_demo_bookings(domain.schedule)
                today = lambda: TODAY  # noqa: E731
            nlu = OpenAICompatLLM(settings) if args.nlu == "llm" else None
            agent = Agent(domain, nlu, today=today)
            noise = random.Random(run * 1000 + hash(sc["id"]) % 1000) if args.asr_noise else None
            try:
                res = Run(sc, agent, domain, patient_llm, noise=noise).play()
            except Exception as exc:  # сбой шлюза не должен ронять весь прогон
                if "daily" in str(exc) or "429" in str(exc):
                    print(f"Дневной лимит шлюза исчерпан на сценарии {sc['id']} — прогон остановлен", flush=True)
                    write_report(results, args, partial=True)
                    return
                res = {"id": sc["id"], "ok": False, "checks": {"прогон": False}, "error": str(exc)[:200],
                       "turns": 0, "latency_ms": 0, "llm_share": 0, "transcript": []}
            results.append(res)
            print(f"{'✅' if res['ok'] else '❌'} {sc['id']:<18} ходов {res['turns']:>2}  "
                  f"NLU {res['latency_ms']:>4} мс  {', '.join(k for k, v in res['checks'].items() if not v)}",
                  flush=True)
    write_report(results, args)


def write_report(results: list[dict], args, partial: bool = False) -> None:
    if not results:
        return
    ok = sum(r["ok"] for r in results)
    lat = [r["latency_ms"] for r in results if r["latency_ms"]]
    no_fake = sum(r["checks"].get("слоты не выдуманы", True) for r in results)
    lines = [
        f"# Стенд «LLM играет пациента» — {time.strftime('%d.%m.%Y %H:%M')}, коммит `{commit()}`",
        "",
        f"Реестр: **{'стенд МИС «Регистратура» с расписанием кейса' if getattr(args, 'stand', False) else 'демо-данные'}**. "
        f"Понимание ассистента: **{'каскад фразник → ученик → ' + get_settings().llm_model if args.nlu == 'llm' else 'фразник и правила, без LLM'}**, "
        f"пациент: `{args.patient_model}`, шум распознавания: "
        f"{'да' if args.asr_noise else 'нет'}, прогонов: {args.runs}.",
        "",
        "| Метрика | Значение |",
        "| :--- | :--- |",
        f"| Успешных звонков (по фактам в реестре) | **{ok} из {len(results)}** ({100 * ok // max(1, len(results))}%) |",
        f"| Звонков без выдуманных слотов | {no_fake} из {len(results)} |",
        f"| Среднее число реплик пациента | {statistics.mean(r['turns'] for r in results):.1f} |",
        f"| Задержка ассистента на ход, медиана | {statistics.median(lat) if lat else 0:.0f} мс |",
        "",
        "| Сценарий | Итог | Реплик | Проверки |",
        "| :--- | :---: | ---: | :--- |",
    ]
    for r in results:
        checks = ", ".join(f"{'✓' if v else '✗'} {k}" for k, v in r["checks"].items())
        lines.append(f"| {r['id']} | {'✅' if r['ok'] else '❌'} | {r['turns']} | {checks} |")
    lines += ["", "## Неуспешные диалоги", ""]
    for r in results:
        if not r["ok"]:
            lines.append(f"### {r['id']}")
            lines += [f"- **{'Пациент' if t['role'] == 'patient' else 'Ассистент'}:** {t['text']}" for t in r["transcript"]]
            if r.get("error"):
                lines.append(f"- ошибка прогона: {r['error']}")
            lines.append("")
    out = Path(args.out + ("_partial" if partial else ""))
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
    print(f"\nУспешно: {ok}/{len(results)}. Отчёт: {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
