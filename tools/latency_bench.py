"""Замер задержек ответа по звеньям: понимание и шаг диалога, синтез первой фразы, распознавание.

    .venv/bin/python tools/latency_bench.py

Текстовый путь (фразник, ученик, правила, merge, step) меряется всегда — без сети и ключей, на демо-реестре.
Синтез (Silero или Piper) и распознавание (GigaAM) — если модели есть на машине (на ВМ — есть; локально
скачиваются tools/get_voice.sh). GigaChat не меряется: его время зависит от шлюза, медиана по стенду
«LLM-пациент» — в eval/report.md. Итог — таблица и eval/results/latency.json.

Полная задержка «пациент договорил → зазвучал ответ» = ожидание конца фразы + понимание и шаг + синтез первой
фразы (остальные синтезируются, пока звучит первая). Распознавание начинается через 0,25 с тишины, поэтому
ожидание = max(пауза конца фразы, 0,25 с + распознавание): 0,8 с обычно и 0,45 с после вопроса «да/нет».
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.voice import split_speech, speakable  # noqa: E402
from core.agent import Agent  # noqa: E402
from core.version import commit  # noqa: E402
from domains.clinic.domain import ClinicDomain  # noqa: E402
from domains.clinic.schedule import seed_demo_bookings  # noqa: E402

TODAY = date(2026, 10, 19)
ENDPOINT_MS, SHORT_ENDPOINT_MS = 800, 450  # пауза конца фразы; после вопроса «да/нет» — короткая
EARLY_MS = 250  # через столько тишины фраза уже уходит в распознавание (раннее распознавание)
DIALOGS = [
    ["Здравствуйте, мне к врачу", "к терапевту на среду после обеда", "давайте первый",
     "Петров Сергей, второе ноября пятьдесят восьмого года", "да", "нет, спасибо"],
    ["Когда у меня запись?", "Иванов Иван, двенадцатое мая шестидесятого года"],
    ["Хочу перенести запись", "Иванов Иван, двенадцатое мая шестидесятого года", "второй", "да"],
    ["До скольки вы работаете в субботу?", "а сколько стоит УЗИ?", "нет"],
    ["Очень сильно болит зуб, опухла щека", "Смирнова Анна 17.03.1985"],
    ["мне бы талончик к лору на завтра утром", "в девять", "да"],
]


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def stats(values: list[float]) -> dict:
    return {"n": len(values), "median_ms": round(statistics.median(values), 1), "p95_ms": pct(values, 0.95),
            "max_ms": round(max(values), 1)} if values else {"n": 0}


def bench_text() -> tuple[dict, list[str]]:
    from domains.clinic.student import Student

    student = Student.load(os.getenv("STUDENT_DIR", "data/student"))  # None, если модель не обучена
    domain = ClinicDomain(today=TODAY, student=student)
    seed_demo_bookings(domain.schedule)
    agent = Agent(domain, None, today=lambda: TODAY)
    by_stage: dict[str, list[float]] = {}
    replies: list[str] = []
    for _ in range(3):  # первый проход прогревает кэши; считаем все — так честнее
        for phrases in DIALOGS:
            s = agent.new_session()
            for text in phrases:
                t = time.perf_counter()
                step = agent.handle(s, text)
                by_stage.setdefault(s.last_nlu or "none", []).append((time.perf_counter() - t) * 1000)
                replies.append(step.reply)
    total = [v for vs in by_stage.values() for v in vs]
    return {"all": stats(total), **{k: stats(v) for k, v in by_stage.items()}}, replies


def bench_tts(replies: list[str]) -> dict | None:
    from apps.voice import TextToSpeech

    tts = TextToSpeech()
    if not tts.available:
        return None
    firsts = list(dict.fromkeys(split_speech(speakable(r))[0] for r in replies if r))[:20]
    tts.synthesize("Здравствуйте.")  # загрузка модели — не в счёт
    times = []
    for text in firsts:
        t = time.perf_counter()
        tts.synthesize(text)
        times.append((time.perf_counter() - t) * 1000)
    return {"engine": tts.name, **stats(times)}


def bench_stt() -> dict | None:
    from apps.telephony import pcm_to_wav, wav_to_pcm8k
    from apps.voice import LocalSpeechToText, TextToSpeech

    stt, tts = LocalSpeechToText(), TextToSpeech()
    if tts.available:
        stt.load()  # GigaAM грузится явно (на сервере — в фоне при старте)
    if not (tts.available and stt.available):
        return None
    phrases = [p for d in DIALOGS for p in d][:12]
    times = []
    for text in phrases:
        wav = pcm_to_wav(wav_to_pcm8k(tts.synthesize(text)), 8000)  # как в звонке: 8 кГц
        t = time.perf_counter()
        stt.transcribe(wav)
        times.append((time.perf_counter() - t) * 1000)
    return {"engine": "gigaam", **stats(times[1:] or times)}  # первая фраза — прогрев


def main() -> None:
    text, replies = bench_text()
    tts, stt = bench_tts(replies), bench_stt()
    print("Понимание + шаг диалога, мс (без LLM):")
    for k, v in text.items():
        print(f"  {k:10} n={v['n']:3}  медиана {v['median_ms']:6}  p95 {v['p95_ms']:6}  макс {v['max_ms']}")
    for name, r in (("Синтез первой фразы", tts), ("Распознавание фразы (8 кГц)", stt)):
        print(f"{name}: " + (f"{r['engine']}, медиана {r['median_ms']} мс, p95 {r['p95_ms']} мс" if r
                             else "модель не найдена на этой машине — пропущено"))
    totals = None
    if tts and stt:
        work = text["all"]["median_ms"] + tts["median_ms"]
        totals = {
            "без раннего распознавания": round(ENDPOINT_MS + stt["median_ms"] + work),
            "обычная пауза": round(max(ENDPOINT_MS, EARLY_MS + stt["median_ms"]) + work),
            "после вопроса «да/нет»": round(max(SHORT_ENDPOINT_MS, EARLY_MS + stt["median_ms"]) + work),
        }
        print("Итого «договорил → зазвучал ответ» (медианы, без LLM):")
        for k, v in totals.items():
            print(f"  {k:28} ≈ {v} мс")
    out = Path("eval/results/latency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"commit": commit(), "endpoint_ms": ENDPOINT_MS, "short_endpoint_ms": SHORT_ENDPOINT_MS, "early_ms": EARLY_MS, "text": text, "tts_first_phrase": tts, "stt": stt,
                               "total_median_ms": totals}, ensure_ascii=False, indent=1))
    print(f"Сохранено в {out}")


if __name__ == "__main__":
    main()
