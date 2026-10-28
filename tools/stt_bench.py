"""Сравнение распознавания: локальная GigaAM против Whisper через шлюз на «телефонном» звуке.

    .venv/bin/python tools/stt_bench.py

Фразы регистратуры синтезирует Piper, звук приводится к 8 кГц (как в звонке через Asterisk) и обратно.
Метрики: WER (доля ошибок в словах, после нормализации регистра и пунктуации) и задержка на фразу.
Результат печатается и пишется в data/stt_bench.json. Синтетический голос проще живого — это нижняя оценка ошибок.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.telephony import pcm_to_wav, wav_to_pcm8k  # noqa: E402
from apps.voice import LocalSpeechToText, TextToSpeech  # noqa: E402
from core.config import get_settings  # noqa: E402
from openai import OpenAI  # noqa: E402

PHRASES = [
    "Здравствуйте, запишите меня к терапевту на вторник утром",
    "Мне нужно к лор-врачу после обеда",
    "Когда принимает Верещагин Андрей Михайлович",
    "Смирнова Ольга Викторовна, двенадцатое марта семьдесят восьмого года",
    "Давайте второй вариант",
    "Нет, это мне не подходит, есть что-нибудь в пятницу",
    "Хочу отменить запись к окулисту",
    "Перенесите, пожалуйста, мою запись на следующую неделю",
    "Сколько стоит узи щитовидной железы",
    "Готовы ли мои анализы на гормоны",
    "У мужа давит в груди и отдаёт в левую руку",
    "Соедините меня с живым оператором",
    "Скажите, а вы работаете в субботу",
    "Да, всё верно, записывайте",
    "Кузнецов Сергей Андреевич, третье января девяностого",
    "Болит зуб, можно к стоматологу сегодня",
]


def words(text: str) -> list[str]:
    return re.sub(r"[^а-яёa-z0-9 ]", " ", text.lower().replace("ё", "е")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


# Whisper через шлюз — ТОЛЬКО для этого сравнения на синтетических фразах. Ассистент облачное распознавание
# не использует (apps/voice.SpeechToText — только GigaAM на машине)
STT_MODEL = os.getenv("STT_MODEL", "openai/whisper-large-v3")


def stt_prompt(specialties: list[str], surnames: list[str]) -> str:
    """Короткая подсказка Whisper словарём регистратуры. Длинная (больше ~200 токенов) ломает распознавание:
    модель начинает возвращать пустой текст, поэтому берём только частые специальности."""
    common = [s for s in specialties if not s.startswith("детск")][:10]
    return ("Звонок в регистратуру поликлиники. Запишите к врачу: " + ", ".join(common) + ".")[:300]


class WhisperGateway:
    def __init__(self, prompt: str = ""):
        settings = get_settings()
        self.enabled = settings.llm_enabled
        self.prompt = prompt
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "none",
                              timeout=15, max_retries=1) if self.enabled else None

    def transcribe(self, audio: bytes, filename: str = "speech.webm") -> str:
        """Таймаут 15 с и одна повторная попытка: шлюз иногда отвечает долго. Пустой ответ с подсказкой —
        повторяем без неё."""
        if self._client is None:
            raise RuntimeError("Распознавание недоступно: не задан ключ шлюза")
        attempts = [{"prompt": self.prompt}] if self.prompt else []
        attempts.append({})
        last_error: Exception | None = None
        for extra in attempts:
            try:
                result = self._client.audio.transcriptions.create(
                    model=STT_MODEL, file=(filename, audio), language="ru", temperature=0, **extra)
            except Exception as exc:
                last_error = exc
                print("Whisper:", type(exc).__name__)
                continue
            text = (getattr(result, "text", "") or "").strip()
            if text:
                return text
        if last_error is not None:
            raise last_error
        return ""



def main() -> None:
    tts = TextToSpeech()
    audio = [pcm_to_wav(wav_to_pcm8k(tts.synthesize(p))) for p in PHRASES]  # телефонный канал: 8 кГц
    local = LocalSpeechToText()
    t = time.perf_counter()
    local.load()
    load_s = time.perf_counter() - t
    engines = {"gigaam": local, "whisper": WhisperGateway()}
    report: dict = {"phrases": len(PHRASES), "gigaam_load_s": round(load_s, 1)}
    for name, engine in engines.items():
        if name == "gigaam" and not local.available:
            report[name] = {"error": local.error}
            continue
        errs, lat, samples = [], [], []
        for ref, wav in zip(PHRASES, audio):
            t = time.perf_counter()
            try:
                hyp = engine.transcribe(wav, "speech.wav")
            except Exception as exc:
                hyp = f"<{type(exc).__name__}>"
            lat.append(time.perf_counter() - t)
            errs.append(wer(ref, hyp))
            samples.append({"ref": ref, "hyp": hyp})
        report[name] = {"wer": round(statistics.mean(errs), 3), "latency_ms_median": round(1000 * statistics.median(lat)),
                        "latency_ms_p95": round(1000 * sorted(lat)[int(0.95 * (len(lat) - 1))]), "samples": samples}
        print(f"{name:<8} WER {report[name]['wer']:.3f}  медиана {report[name]['latency_ms_median']} мс", flush=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/stt_bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
