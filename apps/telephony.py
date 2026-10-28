"""Телефония: звонки через Asterisk ↔ ассистент.

Входящий звонок на номер команды попадает в диалплан Asterisk, который передаёт звук сюда по протоколу
AudioSocket (TCP, кадры 20 мс, 8 кГц, 16 бит). Здесь:
детектор речи → конец фразы → Whisper → агент (тот же, что в вебе) → Piper → 8 кГц обратно в звонок.

- Пациента можно перебивать: голос во время ответа останавливает воспроизведение (регламент, 3.2).
- Тишина: 5 с — «Вы меня слышите?», ещё 5 с — «Вас не слышно. Перезвоните, пожалуйста» и отбой (3.2, R-17).
- Исходящее напоминание (С-10) запускается через AMI Originate; звонок приходит в тот же сервер
  с UUID, по которому мы узнаём запись.

Протокол AudioSocket: https://docs.asterisk.org/Configuration/Channel-Drivers/AudioSocket/
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import struct
import time
import uuid
import warnings
import wave
from dataclasses import dataclass, field
from typing import Any, Callable

from apps.voice import is_filler, load_vad_model, meaningful_for, split_speech
from core.agent import Agent, Session
from core.resilience import breaker

with warnings.catch_warnings():  # audioop есть в Python 3.10–3.12 на ВМ; в 3.13 — пакет audioop-lts
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

log = logging.getLogger(__name__)

KIND_HANGUP, KIND_UUID, KIND_DTMF, KIND_AUDIO, KIND_ERROR = 0x00, 0x01, 0x03, 0x10, 0xFF
RATE = 8000
FRAME_MS = 20
END_MS, SHORT_END_MS = 700, 440  # пауза конца фразы; после вопроса «да/нет» — короче
SPEC_MS = 240  # пауза, после которой фраза уже уходит в распознавание (раннее распознавание)
BARGE_CONFIRM_MS = 900  # пациент говорит поверх ответа дольше — перебивание; короче — может быть «угу» (ADR 045)
DUCK = 0.3  # громкость ответа, пока решаем, перебивание это или поддакивание
FRAME_BYTES = RATE * 2 * FRAME_MS // 1000  # 320 байт: 20 мс, 16 бит, моно

HELLO_AGAIN = "Вы меня слышите?"
NO_SOUND = "Вас не слышно. Перезвоните, пожалуйста."
REMIND_FAILED = ("Здравствуйте! Это поликлиника номер семь. Пожалуйста, перезвоните в регистратуру по номеру "
                 "семьсот, ноль семь, ноль семь. Всего доброго!")
AGENT_FAILED = "Извините, произошёл сбой. Соединяю вас с регистратором, оставайтесь, пожалуйста, на линии."
# Распознавание недоступно (GigaAM не загружена): облачного резерва нет (152-ФЗ) — сразу к регистратору
NO_STT = ("Здравствуйте! Регистратура поликлиники номер семь. Соединяю вас с регистратором, "
          "оставайтесь, пожалуйста, на линии.")
NOT_HEARD = "Извините, не расслышала. Повторите, пожалуйста."
FILLER = "Секунду, проверяю."
FILLER_AFTER = 0.7  # с: если ответ готовится дольше, пациент слышит «Секунду» вместо тишины (регламент 3.7)
# МИС отвечает медленно (лимит запросов стенда, сеть): пока ждём, каждые HOLD_EVERY с — короткая фраза, тишины нет
HOLD = "Ещё мгновение, уточняю в расписании."
HOLD_EVERY = 5.0

# Whisper на тишине и шуме иногда «слышит» титры из обучающих данных — это не реплика пациента
_HALLUCINATIONS = re.compile(r"субтитр|продолжение следует|редактор|корректор|спасибо за просмотр|"
                             r"подписывайтесь|dimatorzok|amara\.org", re.I)


def is_hallucination(text: str) -> bool:
    return bool(_HALLUCINATIONS.search(text)) or not re.search(r"[а-яёa-z0-9]", text, re.I)


def pcm_to_wav(pcm: bytes, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm8k(wav_bytes: bytes) -> bytes:
    """WAV любого формата Piper → 8 кГц, 16 бит, моно (то, что ждёт Asterisk)."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        pcm, rate, width, channels = w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth(), w.getnchannels()
    if channels == 2:
        pcm = audioop.tomono(pcm, width, 0.5, 0.5)
    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
    if rate != RATE:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, RATE, None)
    return pcm


@dataclass
class Vad:
    """Детектор речи: громкость с адаптивным порогом шума и эха, плюс нейросетевой Silero VAD, если он есть
    (голос отличается от шума линии, кашля и стука). Конец фразы — 700 мс тишины после речи."""

    min_speech_ms: int = 240
    end_silence_ms: int = END_MS
    max_utterance_ms: int = 15000
    noise: float = 150.0
    echo: float = 0.0
    play_ms: int = 0
    speech_ms: int = 0
    silence_ms: int = 0
    active: bool = False
    buf: bytearray = field(default_factory=bytearray)
    model: Any = None              # Silero VAD (apps.voice.load_vad_model) или None — только громкость
    prob: float = 0.0              # вероятность речи по последнему окну Silero (32 мс)
    last_silence_ms: int = 0       # сколько тишины было в конце последней фразы (для замера «до первого звука»)
    onset: int = 0                 # кадров с первого громкого кадра, пока речь ещё не подтверждена
    _pcm: bytearray = field(default_factory=bytearray)

    def threshold(self, playing: bool) -> float:
        # Во время ответа ассистента порог выше: эхо собственного голоса не должно считаться перебиванием.
        # С Silero нижняя граница ниже: шум он отсекает сам, а тихий голос теперь слышно
        base = 250.0 if self.model is not None else 400.0
        return max(base, self.noise * 5, self.echo * 1.6) if playing else max(base, self.noise * 3)

    def _speech(self, frame: bytes) -> bool:
        """Silero VAD по окнам 256 отсчётов (32 мс при 8 кГц). Без модели — «да»: решает громкость."""
        if self.model is None:
            return True
        self._pcm += frame
        try:
            while len(self._pcm) >= 512:
                import torch

                chunk = torch.frombuffer(bytearray(self._pcm[:512]), dtype=torch.int16).float() / 32768
                del self._pcm[:512]
                self.prob = float(self.model(chunk, RATE))
        except Exception:
            log.warning("Silero VAD: сбой, дальше — только громкость", exc_info=True)
            self.model = None
            return True
        return self.prob >= 0.5

    def feed(self, frame: bytes, playing: bool = False) -> tuple[bool, bytes | None]:
        """Возвращает (началась ли речь в этом кадре, законченная фраза или None)."""
        rms = audioop.rms(frame, 2)
        self.play_ms = self.play_ms + FRAME_MS if playing else 0
        if self.play_ms == FRAME_MS:
            self.echo = 0.0  # новый ответ — эхо учим заново
        warm = playing and self.play_ms <= 300  # начало ответа: только учим уровень эха
        speech = self._speech(frame)
        loud = not warm and rms > self.threshold(playing) and speech
        if warm or (playing and not loud and not self.speech_ms):
            # Эхо ответа — отдельно от шума линии, по пикам и только пока пациент молчит
            self.echo = 0.7 * self.echo + 0.3 * rms if rms > self.echo else 0.97 * self.echo + 0.03 * rms
        elif not loud and not self.active and not playing:
            # Подстраиваемся под фоновый шум линии. Громкий кадр, который Silero не признал речью (начало слова,
            # пока модель «разгоняется», стук), учитываем медленно: иначе порог за полсекунды взлетает выше голоса
            # и начало фразы теряется
            a = 0.05 if rms <= self.threshold(False) else 0.005
            self.noise = (1 - a) * self.noise + a * rms
        started = False
        if loud:
            self.speech_ms += FRAME_MS
            self.silence_ms = 0
            if not self.active and self.speech_ms >= self.min_speech_ms:
                self.active, started = True, True
        else:
            self.silence_ms += FRAME_MS
            if not self.active:  # провал между слогами не обнуляет счёт — иначе перебить можно только криком
                self.speech_ms = max(0, self.speech_ms - FRAME_MS // 2)
        self.buf += frame  # все кадры подряд: тихие звуки («С» в «Смирнова») — тоже речь, без них слово рвётся
        if not self.active:
            # До подтверждения речи храним её начало целиком и ещё 300 мс до него — первый слог не теряется,
            # даже если Silero «мигает» и подтверждение растягивается на полсекунды и больше
            self.onset = self.onset + 1 if self.speech_ms else 0
            keep = FRAME_BYTES * (15 + self.onset)
            if len(self.buf) > keep:
                self.buf = self.buf[-keep:]
        if self.active and (self.silence_ms >= self.end_silence_ms
                            or len(self.buf) >= self.max_utterance_ms * FRAME_BYTES // FRAME_MS):
            done, self.buf, self.last_silence_ms = bytes(self.buf), bytearray(), self.silence_ms
            self.active, self.speech_ms, self.silence_ms, self.onset = False, 0, 0, 0
            return started, done
        return started, None

    def flush(self) -> bytes | None:
        """Кадры перестали приходить (часть каналов не шлёт тишину) — фраза закончилась."""
        if not self.active:
            self.buf, self.speech_ms, self.onset = bytearray(), 0, 0
            return None
        done, self.buf, self.last_silence_ms = bytes(self.buf), bytearray(), self.end_silence_ms
        self.active, self.speech_ms, self.silence_ms, self.onset = False, 0, 0, 0
        return done


@dataclass
class CallRecord:
    """Звонок для панели в веб-интерфейсе: кто, когда, о чём говорили (без аудио)."""

    id: str
    kind: str  # входящий | напоминание
    started: float = field(default_factory=time.time)
    ended: float | None = None
    turns: list[dict] = field(default_factory=list)
    frames_in: int = 0          # диагностика канала: сколько кадров пришло и пиковый уровень
    peak_rms: int = 0
    utterances: int = 0
    early_stt: int = 0          # фраз, распознанных ещё в паузе (раннее распознавание)
    barge_ins: int = 0          # пациент перебил ассистента
    backchannels: int = 0       # «угу», «да-да» поверх ответа — ассистент не замолчал
    ttfa_ms: list[int] = field(default_factory=list)  # от конца фразы пациента до первого звука ответа
    transferred: bool = False   # звонок переведён на регистратора (тёплый перевод через диалплан)


class Telephony:
    def __init__(self, agent: Agent, stt, tts, *, greeting: str,
                 start_reminder: Callable[[str], tuple] | None = None, journal=None,
                 not_reached: Callable[[str], None] | None = None, answer_timeout: float = 60.0,
                 vad_model: Callable[[], Any] = load_vad_model):
        self.agent, self.stt, self.tts = agent, stt, tts
        self.vad_model = vad_model            # Silero VAD на каждый звонок; без него — порог громкости
        self.not_reached = not_reached        # итог «не дозвонились» — в карточку пациента в МИС
        self.answer_timeout = answer_timeout
        self.journal = journal
        self.greeting = greeting
        self.start_reminder = start_reminder
        self.pending: dict[str, str] = {}  # UUID исходящего звонка → номер записи
        self.calls: list[CallRecord] = []
        self.server: asyncio.base_events.Server | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 9092, warm: tuple[str, ...] = ()) -> None:
        self.server = await asyncio.start_server(self._handle, host, port)
        log.info("AudioSocket слушает %s:%d", host, port)
        # Типовые фразы синтезируем заранее: в звонке они звучат без задержки
        asyncio.get_running_loop().run_in_executor(None, self._warm_up, (self.greeting, FILLER, HOLD, HELLO_AGAIN,
                                                                          NOT_HEARD, NO_SOUND, *warm))

    def _warm_up(self, phrases: tuple[str, ...]) -> None:
        for text in phrases:
            try:
                self.tts.synthesize(text)
            except Exception:
                return

    # ---------- исходящий звонок ----------

    async def call_reminder(self, booking_id: str, endpoint: str | None = None) -> str:
        """Напоминание по телефону: AMI Originate → звонок на софтфон → AudioSocket с нашим UUID."""
        call_id = str(uuid.uuid4())
        self.pending[call_id] = booking_id
        try:
            await ami_originate(endpoint or os.getenv("PHONE_ENDPOINT") or "PJSIP/phone-094@gateway",
                                context="reminder-answered", variables={"REMINDER_UUID": call_id})
        except Exception:
            self.pending.pop(call_id, None)  # звонок не ушёл — не ждём его ответа
            raise
        spawn(self._watch_answer(call_id, booking_id))
        return call_id

    async def _watch_answer(self, call_id: str, booking_id: str) -> None:
        """Пациент не взял трубку за answer_timeout — итог «не дозвонились» фиксируется в МИС."""
        await asyncio.sleep(self.answer_timeout)
        if self.pending.pop(call_id, None) is not None and self.not_reached is not None:
            try:
                await asyncio.to_thread(self.not_reached, booking_id)
            except Exception:
                log.warning("Итог «не дозвонились» не записан для записи %s", booking_id, exc_info=True)

    async def remind_tomorrow(self, booking_ids: list[str], pause: float = 90.0) -> int:
        """Вечерний обзвон: записи на завтра по убыванию риска неявки, по одному звонку за раз."""
        called = 0
        for booking_id in booking_ids:
            try:
                await self.call_reminder(booking_id)
                called += 1
            except Exception:
                log.warning("Напоминание по записи %s не отправлено", booking_id, exc_info=True)
            await asyncio.sleep(pause)
        return called

    # ---------- обработка звонка ----------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        call = None
        try:
            kind, payload = await _read_frame(reader)
            if kind != KIND_UUID:
                return
            call_id = str(uuid.UUID(bytes=payload))
            call = _Call(self, call_id, reader, writer)
            await call.run()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            log.exception("Сбой обработки звонка")
        finally:
            if call is not None:
                call.record.ended = time.time()
            writer.close()


class _Call:
    def __init__(self, tel: Telephony, call_id: str, reader, writer):
        self.tel, self.reader, self.writer = tel, reader, writer
        self.utterances: asyncio.Queue[tuple[bytes, asyncio.Task | None, float]] = asyncio.Queue()
        self.tts_failed = False  # синтез сломался — звонок уходит регистратору (core/resilience.FALLBACKS)
        self.ducked = False      # пациент заговорил поверх ответа: ответ тише, решаем — перебивание или «угу»
        self.reply_asks = False  # звучащий ответ задаёт вопрос «да/нет»: «да» поверх него — ответ, а не поддакивание
        self.turn_end: float | None = None  # когда пациент договорил (monotonic) — для замера до первого звука
        self.ttfa: int | None = None
        self.spec: asyncio.Task | None = None  # распознавание, начатое в паузе до конца фразы
        self.vad = Vad()
        self.playing = False
        self.barge_in = asyncio.Event()
        self.closed = False
        self.call_id = call_id  # UUID канала AudioSocket: по нему диалплан узнаёт, что звонок надо перевести
        self.booking_id = tel.pending.pop(call_id, None) if tel.start_reminder is not None else None
        self.session = tel.agent.new_session()
        self.hello = tel.greeting
        self.record = CallRecord(id=call_id if self.booking_id else str(uuid.uuid4()),
                                 kind="напоминание" if self.booking_id else "входящий")
        self.stt_ms: float | None = None
        tel.calls.append(self.record)
        del tel.calls[:-20]

    async def _prepare(self) -> bool:
        """Исходящее напоминание: запись и пациент — из МИС, в отдельном потоке (сетевой запрос не должен
        останавливать другие звонки). Не вышло — извиняемся и просим перезвонить, а не молчим."""
        if self.booking_id:
            try:
                state, hello = await asyncio.to_thread(self.tel.start_reminder, self.booking_id)
            except Exception as exc:
                log.error("Напоминание не подготовлено (%s)", type(exc).__name__)
                await self.say(REMIND_FAILED)
                return False
            self.session = Session(state=state, history=[{"role": "assistant", "content": hello}])
            self.hello = hello
        if self.tel.journal is not None:
            self.tel.journal.start(self.record.id, "телефон: напоминание" if self.booking_id else "телефон", self.hello)
        return True

    async def run(self) -> None:
        self.vad.model = await asyncio.to_thread(self.tel.vad_model)  # у каждого звонка своё состояние Silero
        if not await self._prepare():
            await self._hangup()
            return
        if not getattr(self.tel.stt, "enabled", True):
            await self.say(NO_STT)
            await self._mark_transfer()
            await self._hangup()
            return
        listener = asyncio.create_task(self._listen())
        try:
            await self.say(self.hello)
            if self.tts_failed:
                await self._mark_transfer()
                return
            silent = 0
            while not self.closed:
                try:
                    pcm, early, self.turn_end = await asyncio.wait_for(self.utterances.get(), timeout=5)
                except asyncio.TimeoutError:
                    st = self.session.state
                    if getattr(st, "intent", None) == "escalate" and not getattr(st, "finished", False):
                        # Пациент просил оператора и не назвал себя — всё равно переводим, карточку дополнит регистратор
                        st.asked_identity = True
                        step = await asyncio.to_thread(self.tel.agent.domain.step, st)
                        await self.say(step.reply)
                        await self._mark_transfer()
                        break
                    silent += 1
                    await self.say(HELLO_AGAIN if silent == 1 else NO_SOUND)
                    if silent >= 2:
                        break
                    continue
                if self.closed:
                    break  # пациент положил трубку
                silent = 0
                if early is not None:
                    self.record.early_stt += 1
                text = await (early if early is not None else self._transcribe(pcm))
                if not text:
                    await self.say(NOT_HEARD)
                    continue
                self.record.turns.append({"role": "patient", "text": text})
                started = time.perf_counter()
                try:
                    step = await self._answer(text)
                except Exception as exc:  # непредвиденный сбой ассистента — не обрываем звонок, а переводим
                    log.error("Сбой ассистента в звонке (%s) — переводим на регистратора", type(exc).__name__)
                    await self.say(AGENT_FAILED)
                    await self._mark_transfer()
                    break
                short = getattr(self.tel.agent.domain, "expects_short_answer", lambda _: False)(self.session.state)
                self.vad.end_silence_ms = SHORT_END_MS if short else END_MS
                if self.tel.journal is not None:
                    self.tel.journal.turn(self.record.id, text, step, self.session.state, self.session.last_update,
                                          self.session.last_nlu, 1000 * (time.perf_counter() - started),
                                          self.stt_ms, getattr(self.tel.stt, "last_engine", None))
                self._report_ttfa()
                await self.say(step.reply)
                self._report_ttfa()
                if self.tts_failed:
                    await self._mark_transfer()
                    break
                st = self.session.state
                if getattr(st, "finished", False) and getattr(st, "question", None) is None:
                    if getattr(st, "escalation", None):
                        # и при экстренном случае: после «звоните 103» пациента соединяют с живым регистратором,
                        # как и обещает фраза регламента
                        await self._mark_transfer()  # после отбоя AudioSocket диалплан соединит с регистратором
                    break  # перевод, прощание, итог напоминания — разговор окончен
        finally:
            listener.cancel()
            await self._hangup()
            if self.tel.journal is not None:
                self.tel.journal.end(self.record.id)

    async def _answer(self, text: str):
        """Ответ агента; если LLM или МИС думают дольше FILLER_AFTER — короткая фраза, чтобы не было тишины,
        и дальше каждые HOLD_EVERY с, пока ответ не готов (МИС может ждать лимит запросов до R-06)."""
        task = asyncio.create_task(asyncio.to_thread(self.tel.agent.handle, self.session, text))
        phrase, timeout = FILLER, FILLER_AFTER
        while not (await asyncio.wait({task}, timeout=timeout))[0]:
            await self.say(phrase, log=False)
            phrase, timeout = HOLD, HOLD_EVERY
        return await task

    async def say(self, text: str, *, log: bool = True) -> None:
        """Озвучка по фразам: первая уходит в линию сразу, следующие синтезируются, пока она звучит."""
        if not text or self.closed:
            return
        if log:
            self.record.turns.append({"role": "assistant", "text": text})
            ask = getattr(self.tel.agent.domain, "expects_short_answer", lambda _: False)
            self.reply_asks = bool(ask(self.session.state))
        parts = split_speech(text)
        synth = lambda part: asyncio.create_task(asyncio.to_thread(self.tel.tts.synthesize, part))  # noqa: E731
        upcoming = synth(parts[0])
        self.playing = True
        self.barge_in.clear()
        try:
            for i in range(len(parts)):
                try:
                    wav = await upcoming
                except Exception as exc:
                    logging.getLogger(__name__).warning(  # имя log здесь занято параметром
                        "Синтез речи недоступен (%s) — переводим на регистратора", type(exc).__name__)
                    self.tts_failed = True  # пациент нас не слышит: запасной путь — живой регистратор
                    return
                if i + 1 < len(parts):
                    upcoming = synth(parts[i + 1])
                if not await self._play(wav_to_pcm8k(wav)):
                    if i + 1 < len(parts):
                        upcoming.cancel()
                    break  # пациент заговорил — замолкаем и слушаем
        finally:
            self.playing = self.ducked = False

    async def _play(self, pcm: bytes) -> bool:
        """Отдаёт звук в линию в реальном времени. False — прервано (пациент заговорил или положил трубку)."""
        started = time.monotonic()
        for i, offset in enumerate(range(0, len(pcm), FRAME_BYTES)):
            if self.barge_in.is_set() or self.closed:
                return False
            chunk = pcm[offset:offset + FRAME_BYTES].ljust(FRAME_BYTES, b"\0")
            if self.ducked:
                chunk = audioop.mul(chunk, 2, DUCK)
            if self.turn_end is not None:  # первый звук после фразы пациента (или «Секунду, проверяю»)
                self.ttfa, self.turn_end = round(1000 * (time.monotonic() - self.turn_end)), None
            self.writer.write(_frame(KIND_AUDIO, chunk))
            await self.writer.drain()
            # кадры отдаём в реальном времени, иначе Asterisk переполнит буфер
            delay = started + (i + 1) * FRAME_MS / 1000 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        return True

    async def _read_frames(self, frames: asyncio.Queue) -> None:
        """Читает кадры из сокета без таймаутов: прерванное чтение сбило бы границы кадров."""
        try:
            while True:
                await frames.put(await _read_frame(self.reader))
        except (asyncio.IncompleteReadError, ConnectionError):
            await frames.put((KIND_HANGUP, b""))

    async def _listen(self) -> None:
        frames: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        reader = asyncio.create_task(self._read_frames(frames))
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(frames.get(), timeout=self.vad.end_silence_ms / 1000)
                except asyncio.TimeoutError:
                    # часть каналов не шлёт кадры в паузах — конец фразы по отсутствию звука
                    if done := self.vad.flush():
                        await self._utterance(done)
                    continue
                if kind in (KIND_HANGUP, KIND_ERROR):
                    self.closed = True
                    await self.utterances.put((b"", None, None))  # будим основной цикл: звонок окончен
                    return
                if kind != KIND_AUDIO:
                    continue
                self.record.frames_in += 1
                self.record.peak_rms = max(self.record.peak_rms, audioop.rms(payload, 2))
                started, done = self.vad.feed(payload, playing=self.playing)
                if started and self.playing:
                    self.ducked = True  # не замолкаем сразу: сначала тише, вдруг это «угу»
                if (self.ducked and self.playing and self.vad.active and not self.barge_in.is_set()
                        and self.vad.speech_ms >= BARGE_CONFIRM_MS):
                    self._barge()  # говорит долго — точно перебивает
                if self.vad.active and self.vad.silence_ms == 0:
                    self.spec = None  # пациент продолжил фразу — раннее распознавание устарело
                elif self.vad.active and self.vad.silence_ms == SPEC_MS < self.vad.end_silence_ms:
                    self.spec = asyncio.create_task(self._transcribe(bytes(self.vad.buf)))
                if done:
                    await self._utterance(done)
        finally:
            reader.cancel()

    async def _utterance(self, pcm: bytes) -> None:
        """Законченная фраза — в очередь вместе с ранним распознаванием, если оно шло по той же речи."""
        self.record.utterances += 1
        early, self.spec = self.spec, None
        ended = time.monotonic() - self.vad.last_silence_ms / 1000
        if self.ducked and self.playing and not self.barge_in.is_set():
            spawn(self._judge(pcm, early, ended))  # короткая фраза поверх ответа: «угу» или ответ?
            return
        self.ducked = False
        await self.utterances.put((pcm, early, ended))

    async def _judge(self, pcm: bytes, early: asyncio.Task | None, ended: float) -> None:
        """Короткая реплика поверх ответа. Поддакивание — ответ продолжается в полную громкость;
        иначе (или если ассистент как раз спросил «да/нет») — перебивание: замолкаем и отвечаем."""
        task = early or asyncio.create_task(self._transcribe(pcm))
        text = await task
        if self.playing and is_filler(text, meaningful_for(self.tel.agent, self.session.state), asks=self.reply_asks):
            self.ducked = False
            self.record.backchannels += 1
            if self.tel.journal is not None:
                self.tel.journal.voice(self.record.id, backchannel=True)
            return
        if self.playing:
            self._barge()
        self.ducked = False
        await self.utterances.put((pcm, task, ended))

    def _barge(self) -> None:
        self.barge_in.set()
        self.record.barge_ins += 1
        if self.tel.journal is not None:
            self.tel.journal.voice(self.record.id, barge=True)

    def _report_ttfa(self) -> None:
        """«До первого звука» — в запись звонка и в журнал (к последнему ответу ассистента)."""
        if self.ttfa is None:
            return
        self.record.ttfa_ms.append(self.ttfa)
        if self.tel.journal is not None:
            self.tel.journal.voice(self.record.id, ttfa_ms=self.ttfa)
        self.ttfa = None

    async def _transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        try:
            started = time.perf_counter()
            text = await asyncio.to_thread(self.tel.stt.transcribe, pcm_to_wav(pcm), "speech.wav")
            self.stt_ms = 1000 * (time.perf_counter() - started)
        except Exception:
            log.exception("Сбой распознавания в звонке")
            return ""
        return "" if is_hallucination(text) else text

    async def _mark_transfer(self) -> None:
        """Тёплый перевод: отметка в AstDB, диалплан после AudioSocket набирает регистратора (OPERATOR).
        Карточка перевода с сутью звонка уже у регистратора на рабочем месте."""
        try:
            await ami_action({"Action": "DBPut", "Family": "transfer", "Key": self.call_id, "Val": "1"})
            self.record.transferred = True
        except Exception:
            log.warning("Перевод на регистратора не отмечен в Asterisk — звонок завершится", exc_info=True)

    async def _hangup(self) -> None:
        self.closed = True
        try:
            self.writer.write(_frame(KIND_HANGUP, b""))
            await self.writer.drain()
        except Exception:
            log.debug("Канал уже закрыт при отбое", exc_info=True)  # пациент положил трубку раньше нас


_background: set[asyncio.Task] = set()


def spawn(coro) -> asyncio.Task:
    """Фоновая задача со ссылкой (иначе её может собрать сборщик мусора) и с логом необработанной ошибки."""
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)

    def done(t: asyncio.Task) -> None:
        _background.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("Фоновая задача звонка упала: %s", type(t.exception()).__name__)
    task.add_done_callback(done)
    return task


def _frame(kind: int, payload: bytes) -> bytes:
    return struct.pack(">BH", kind, len(payload)) + payload


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    header = await reader.readexactly(3)
    kind, length = struct.unpack(">BH", header)
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


async def ami_action(fields: dict[str, str]) -> str:
    """Одно действие AMI (например, DBPut для перевода звонка) с входом и выходом; итог — в предохранитель «ami»."""
    guard = breaker("ami")
    if not guard.allow():
        raise RuntimeError("Asterisk временно недоступен (предохранитель ami)")
    try:
        result = await _ami_action(fields)
    except Exception as exc:
        guard.fail(exc)
        raise
    guard.ok()
    return result


async def _ami_action(fields: dict[str, str]) -> str:
    host, port = os.getenv("AMI_HOST", "127.0.0.1"), int(os.getenv("AMI_PORT", "5038"))
    user, secret = os.getenv("AMI_USER", "assistant"), os.getenv("AMI_SECRET", "")
    if not secret:
        raise RuntimeError("Не задан AMI_SECRET")
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5)
    try:
        await asyncio.wait_for(reader.readline(), timeout=5)  # баннер Asterisk Call Manager

        async def send(lines: dict[str, str]) -> str:
            """Ответ ищем по ActionID: Asterisk вперемешку шлёт события (например, FullyBooted после входа)."""
            action_id = uuid.uuid4().hex
            writer.write(("".join(f"{k}: {v}\r\n" for k, v in (lines | {"ActionID": action_id}).items())
                          + "\r\n").encode())
            await writer.drain()
            while True:
                block = b""
                while not block.endswith(b"\r\n\r\n"):
                    block += await asyncio.wait_for(reader.readline(), timeout=5)
                text = block.decode(errors="replace")
                if f"ActionID: {action_id}" in text:
                    return text

        if "Success" not in await send({"Action": "Login", "Username": user, "Secret": secret, "Events": "off"}):
            raise RuntimeError("AMI: неверная учётка")
        result = await send(fields)
        if "Success" not in result:
            raise RuntimeError(f"AMI: {fields.get('Action')} не выполнено")
        writer.write(b"Action: Logoff\r\n\r\n")
        await writer.drain()
        return result
    finally:
        writer.close()


async def ami_originate(channel: str, *, context: str, variables: dict[str, str], exten: str = "s") -> None:
    """Исходящий звонок через Asterisk Manager Interface (локальный порт, учётка из окружения)."""
    host, port = os.getenv("AMI_HOST", "127.0.0.1"), int(os.getenv("AMI_PORT", "5038"))
    user, secret = os.getenv("AMI_USER", "assistant"), os.getenv("AMI_SECRET", "")
    if not secret:
        raise RuntimeError("Не задан AMI_SECRET — исходящие звонки выключены")
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5)
    try:
        await asyncio.wait_for(reader.readline(), timeout=5)  # баннер Asterisk Call Manager

        async def action(lines: dict[str, str]) -> str:
            writer.write(("".join(f"{k}: {v}\r\n" for k, v in lines.items()) + "\r\n").encode())
            await writer.drain()
            response = b""
            while not response.endswith(b"\r\n\r\n"):
                response += await asyncio.wait_for(reader.readline(), timeout=5)
            return response.decode(errors="replace")

        if "Success" not in await action({"Action": "Login", "Username": user, "Secret": secret}):
            raise RuntimeError("AMI: неверная учётка")
        fields = {"Action": "Originate", "Channel": channel, "Context": context, "Exten": exten, "Priority": "1",
                  "CallerID": "Поликлиника <094>", "Async": "true", "Timeout": "30000"}
        payload = "".join(f"{k}: {v}\r\n" for k, v in fields.items())
        payload += "".join(f"Variable: {k}={v}\r\n" for k, v in variables.items())
        writer.write((payload + "\r\n").encode())
        await writer.drain()
        writer.write(b"Action: Logoff\r\n\r\n")
        await writer.drain()
    finally:
        writer.close()
