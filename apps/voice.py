"""Голос на сервере, всё на машине: распознавание GigaAM и синтез Silero v5 (Piper в резерве) на CPU.

Почему не в браузере: Web Speech API отправляет звук в облако Google, а кейс запрещает бесплатные
облачные STT (152-ФЗ). Звук пациента не покидает машину: внешних сервисов распознавания и синтеза у ассистента нет.
Используется и веб-интерфейсом, и телефонией.
"""

from __future__ import annotations

import io
import re
import logging
import os
import threading
import wave
from functools import lru_cache
from pathlib import Path

from core.resilience import breaker

log = logging.getLogger(__name__)

VOICE_PATH = Path(os.getenv("TTS_VOICE", "voices/ru_RU-irina-medium.onnx"))
SILERO_PATH = Path(os.getenv("SILERO_MODEL", "voices/silero_v5_3_ru.pt"))
SILERO_SPEAKER = os.getenv("TTS_SPEAKER", "xenia")

_LATIN = {"a": "эй", "b": "бэ", "c": "цэ", "d": "дэ", "e": "е", "f": "эф", "k": "ка", "m": "эм", "n": "эн", "p": "пэ",
          "r": "эр", "s": "эс", "t": "тэ", "x": "икс"}


def speakable(text: str) -> str:
    """Текст для синтеза: цифры и латиница словами («номер записи B0004» → «номер записи бэ четыре»).
    Даты и время ассистент и так говорит словами; остаются номера записей, кабинетов и телефонов."""
    from core.ru import cardinal

    text = text.replace("№", "номер").replace("—", ",").replace("«", "").replace("»", "")
    text = re.sub(r"\b([A-Za-z])-?(\d+)\b", lambda m: f"{_LATIN.get(m.group(1).lower(), m.group(1))} {m.group(2)}", text)
    text = re.sub(r"\d+", lambda m: cardinal(int(m.group(0))) if int(m.group(0)) < 1000
                  else " ".join(cardinal(int(d)) for d in m.group(0)), text)
    return re.sub(r"\s{2,}", " ", text).strip()


def split_speech(text: str, min_len: int = 18) -> list[str]:
    """Ответ по фразам для озвучки потоком: первая звучит, пока синтезируются следующие.

    Режем по концу предложения и по «;» между вариантами; короткие куски склеиваем со следующими,
    чтобы интонация не рвалась на «Хорошо.» отдельно."""
    pieces = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", text.strip()) if p.strip()]
    parts: list[str] = []
    for piece in pieces:
        if parts and len(parts[-1]) < min_len:
            parts[-1] = f"{parts[-1]} {piece}"
        else:
            parts.append(piece)
    return parts or [text]


class TextToSpeech:
    """Синтез речи локально, на CPU: Silero v5 (живой голос, ~0,1 с на фразу), если модель скачана; иначе Piper.

    TTS_ENGINE=piper — принудительно Piper; TTS_SPEAKER — голос Silero (xenia, kseniya, baya, aidar, eugene).
    На выходе WAV 16 бит моно."""

    def __init__(self, model_path: Path = VOICE_PATH, silero_path: Path = SILERO_PATH, engine: str | None = None):
        self.model_path = model_path
        self.silero_path = silero_path
        self.engine = engine or os.getenv("TTS_ENGINE", "auto")
        self.kind: str | None = None
        self._voice = None
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def name(self) -> str | None:
        if self._load() is None:
            return None
        return f"Silero v5 ({SILERO_SPEAKER})" if self.kind == "silero" else "Piper"

    def _load(self):
        if self._voice is not None or self.error:
            return self._voice
        with self._lock:
            if self._voice is not None or self.error:
                return self._voice
            if self.engine in ("auto", "silero") and self.silero_path.exists():
                try:
                    import torch

                    model = torch.package.PackageImporter(str(self.silero_path)).load_pickle("tts_models", "model")
                    model.to("cpu")
                    self._voice, self.kind = model, "silero"
                    return self._voice
                except Exception as exc:
                    log.warning("Silero не загрузился, синтез через Piper: %s", exc)
            try:
                from piper import PiperVoice

                if not self.model_path.exists():
                    raise FileNotFoundError(f"нет голоса {self.model_path}")
                self._voice, self.kind = PiperVoice.load(str(self.model_path)), "piper"
            except Exception as exc:  # нет пакета или голоса — браузер озвучит сам
                self.error = str(exc)
                log.warning("Синтез речи недоступен: %s", exc)
        return self._voice

    def synthesize(self, text: str) -> bytes:
        guard = breaker("tts")
        if not guard.allow():
            raise RuntimeError("Синтез временно недоступен (предохранитель tts)")
        try:
            wav = _synthesize_cached(self, text) if _cacheable(text) else self._synthesize(text)
        except Exception as exc:
            guard.fail(exc)
            raise
        guard.ok()
        return wav

    def _synthesize(self, text: str) -> bytes:
        voice = self._load()
        if voice is None:
            raise RuntimeError(self.error or "Синтез речи недоступен")
        text = speakable(text)
        buf = io.BytesIO()
        if self.kind == "silero":
            with self._lock:
                audio = voice.apply_tts(text=text, speaker=SILERO_SPEAKER, sample_rate=48000,
                                        put_accent=True, put_yo=True)
            pcm = (audio.clamp(-1, 1) * 32767).short().numpy().tobytes()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(pcm)
            return buf.getvalue()
        with self._lock, wave.open(buf, "wb") as wav:
            if hasattr(voice, "synthesize_wav"):  # piper-tts ≥ 1.3
                voice.synthesize_wav(text, wav)
            else:
                voice.synthesize(text, wav)
        return buf.getvalue()


def _cacheable(text: str) -> bool:
    """В кэш — только типовые фразы: без имён (слов с заглавной не в начале), цифр и номеров.
    Аудио с ФИО пациента не должно жить в памяти после удаления разговора (152-ФЗ)."""
    return not re.search(r"\d|(?<=[^.!?]\s)[А-ЯЁ]", text.strip())


@lru_cache(maxsize=512)
def _synthesize_cached(tts: TextToSpeech, text: str) -> bytes:
    """Типовые фразы (приветствие, «Верно?», прощание) синтезируем один раз."""
    return tts._synthesize(text)


class LocalSpeechToText:
    """GigaAM-v3 (Сбер, MIT) на CPU: аудио пациента не покидает машину — сильнейший аргумент по 152-ФЗ.

    Модель грузится в фоне при старте (≈1 ГБ, первый раз скачивается). Вход — любой формат, который
    читает ffmpeg (webm из браузера, wav из телефонии); модель сама приводит звук к 16 кГц.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GIGAAM_MODEL", "v3_e2e_ctc")
        self._model = None
        self._lock = threading.Lock()
        self.error: str | None = None
        self.ready = threading.Event()

    def load(self) -> None:
        try:
            import gigaam

            self._model = gigaam.load_model(self.model_name, device="cpu")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("GigaAM недоступна: %s", self.error)
        finally:
            self.ready.set()

    @property
    def available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        if self._model is None:
            raise RuntimeError(self.error or "GigaAM не загружена")
        import tempfile

        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(audio)
            f.flush()
            with self._lock:
                result = self._model.transcribe(f.name)
        return (result if isinstance(result, str) else getattr(result, "text", str(result))).strip()


class SpeechToText:
    """Распознавание речи пациента — только на машине (GigaAM). Внешних сервисов распознавания нет совсем.

    Кейс и 152-ФЗ: звук пациента — персональные данные, бесплатные облачные STT недопустимы. Поэтому облачного
    резерва у ассистента нет: если GigaAM недоступна, веб предлагает написать текстом, а телефон сразу переводит
    на регистратора (apps/telephony.py). Whisper через шлюз остался только в tools/stt_bench.py — для сравнения
    точности на синтетических фразах, без голоса пациентов."""

    def __init__(self, local: LocalSpeechToText | None):
        self.local = local
        self.last_engine = ""

    @property
    def enabled(self) -> bool:
        return bool(self.local and self.local.available)

    @property
    def name(self) -> str | None:
        return "GigaAM (локально)" if self.enabled else None

    def transcribe(self, audio: bytes, filename: str = "speech.webm") -> str:
        guard = breaker("stt")
        if not guard.allow():
            raise RuntimeError("Распознавание временно недоступно (предохранитель stt)")
        if not self.enabled:
            guard.fail(RuntimeError("GigaAM не загружена"))
            raise RuntimeError("Распознавание недоступно: GigaAM не загружена")
        self.last_engine = "gigaam"
        try:
            text = self.local.transcribe(audio, filename)
        except Exception as exc:
            guard.fail(exc)
            raise
        guard.ok()
        return text


# Поддакивания: пациент слушает, а не перебивает. По исследованиям дуплексных диалогов большая часть
# «перебиваний» — именно такие реплики и шум; на них ассистент не замолкает (ADR 045)
BACKCHANNEL = {"угу", "ага", "мгм", "ммм", "хм", "да", "так", "ясно", "понятно", "хорошо", "ок", "окей", "ладно",
               "ну", "слушаю", "конечно", "верно", "понял", "поняла", "ну-ну", "да-да", "так-так", "ага-ага", "угу-угу"}


# Эти слова поверх ответа — всегда «замолчите»
STOP_WORDS = {"стоп", "подождите", "погодите", "подожди", "стойте", "хватит", "нет", "секунду", "минутку", "извините"}


def _words(text: str) -> list[str]:
    return re.findall(r"[а-яё]+(?:-[а-яё]+)*", text.lower().replace("ё", "е"))


def is_backchannel(text: str) -> bool:
    """«Угу», «да-да», «ну да, понятно» — до трёх слов, и все из списка поддакиваний."""
    words = _words(text)
    return 0 < len(words) <= 3 and all(w in BACKCHANNEL for w in words)


def is_filler(text: str, meaningful=None, *, asks: bool = False) -> bool:
    """Реплика поверх ответа, на которую ассистент не замолкает: поддакивание или одно слово без смысла для
    диалога. Короткие звуки распознаются неточно («угу» → «иду»), поэтому одно слово, в котором правила
    не нашли ни цели, ни выбора, ни просьбы, тоже не перебивание. Стоп-слова — всегда перебивание.
    asks — звучащий ответ спрашивает «да/нет»: тогда «угу», «да» — это ответ, а не поддакивание."""
    if is_backchannel(text) and not asks:
        return True
    words = _words(text)
    if len(words) != 1 or words[0] in STOP_WORDS:
        return False
    return meaningful is not None and not meaningful(text)


def meaningful_for(agent, state):
    """Есть ли в реплике смысл для диалога: правила домена нашли цель, специалиста, выбор, «да/нет», просьбу."""
    rule = getattr(getattr(agent, "domain", None), "rule_extract", None)

    def check(text: str) -> bool:
        if rule is None:
            return True
        update = rule(state, text, agent.today())
        return bool(update.model_dump(exclude_none=True, exclude={"smalltalk", "note", "bye"}))
    return check


def load_vad_model():
    """Silero VAD (MIT, ~2 МБ, < 1 мс на окно 32 мс на CPU) — нейросетевой детектор речи для телефонии.
    Отличает голос от шума линии, кашля и стука. None — пакет silero-vad не установлен: остаётся порог громкости."""
    if os.getenv("SILERO_VAD", "1") != "1":
        return None
    try:
        from silero_vad import load_silero_vad

        return load_silero_vad()
    except Exception as exc:  # нет torch или пакета — телефония работает на пороге громкости
        log.info("Silero VAD недоступен: %s", f"{type(exc).__name__}: {exc}"[:120])
        return None
