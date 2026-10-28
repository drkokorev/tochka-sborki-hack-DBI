"""Телефония без Asterisk: фейковый клиент AudioSocket разговаривает с ассистентом."""

import asyncio
import math
import struct
import uuid

from apps.telephony import (
    FRAME_BYTES, FRAME_MS, KIND_AUDIO, KIND_HANGUP, KIND_UUID, Telephony, Vad, is_hallucination, pcm_to_wav,
    wav_to_pcm8k,
)
from core.agent import Agent
from domains.clinic.domain import GREETING, ClinicDomain
from domains.clinic.schedule import seed_demo_bookings
from tests.conftest import TODAY


def tone(ms, amp=6000):
    n = 8 * ms
    return b"".join(struct.pack("<h", int(amp * math.sin(i / 3))) for i in range(n))


SILENCE = b"\0" * FRAME_BYTES


class FakeStt:
    def __init__(self, *texts):
        self.texts = list(texts)

    calls = 0

    def transcribe(self, audio, filename="speech.wav"):
        assert audio[:4] == b"RIFF"
        self.calls += 1
        return self.texts.pop(0) if self.texts else ""


class FakeTts:
    def synthesize(self, text):
        return pcm_to_wav(b"\1\0" * 160, rate=16000)  # 10 мс тона: проверяем пересэмплирование


def frame(kind, payload):
    return struct.pack(">BH", kind, len(payload)) + payload


async def converse(tel, phrases, call_id=None, linger=0.3, silence_frames=0):
    await tel.start(port=0)
    port = tel.server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(frame(KIND_UUID, uuid.UUID(call_id or str(uuid.uuid4())).bytes))
    await writer.drain()
    await asyncio.sleep(0.3)  # пациент дослушивает приветствие: кадры речи идут не поверх его начала
    for _ in phrases:
        speech = tone(600)
        for i in range(0, len(speech), FRAME_BYTES):
            writer.write(frame(KIND_AUDIO, speech[i:i + FRAME_BYTES]))
        for _ in range(silence_frames):  # канал, который шлёт кадры и в паузах (как Asterisk)
            writer.write(frame(KIND_AUDIO, SILENCE))
        await writer.drain()  # по умолчанию тишину не шлём: конец фразы — по паузе в потоке кадров
        await asyncio.sleep(1.2)
    await asyncio.sleep(linger)  # пациент молчит, но трубку не кладёт
    writer.write(frame(KIND_HANGUP, b""))
    await writer.drain()
    await asyncio.sleep(0.3)
    writer.close()
    tel.server.close()
    return tel.calls[-1]


def make(*texts):
    domain = ClinicDomain(today=TODAY)
    seed_demo_bookings(domain.schedule)
    agent = Agent(domain, None, today=lambda: TODAY)
    # Silero VAD выключен: синтетический тон для него не речь (на ВМ пакет установлен)
    tel = Telephony(agent, FakeStt(*texts), FakeTts(), greeting=GREETING, vad_model=lambda: None,
                    start_reminder=lambda bid: domain.start_reminder(domain.schedule.bookings[bid]))
    return tel, domain


def test_vad_detects_phrase_end():
    vad = Vad()
    for _ in range(20):
        vad.feed(SILENCE)
    speech = tone(500)
    out = None
    for i in range(0, len(speech), FRAME_BYTES):
        vad.feed(speech[i:i + FRAME_BYTES])
    for _ in range(40):
        _, out = vad.feed(SILENCE)
        if out:
            break
    assert out and len(out) >= len(speech) // 2


def test_hallucination_filter_and_resample():
    assert is_hallucination("Субтитры сделал DimaTorzok") and is_hallucination("...")
    assert not is_hallucination("Запишите к терапевту")
    assert len(wav_to_pcm8k(pcm_to_wav(b"\0\0" * 1600, rate=16000))) == 1600  # 100 мс: 16 кГц → 8 кГц


def test_inbound_call_books_through_agent():
    tel, domain = make("Запишите к лору", "первый", "Петров Иван 12.03.1958", "да", "нет, спасибо")
    call = asyncio.run(converse(tel, range(5)))
    said = [t["text"] for t in call.turns if t["role"] == "assistant"]
    assert said[0] == GREETING and call.kind == "входящий"
    assert any(t.startswith("Готово, вы записаны") for t in said)
    assert len(domain.schedule.bookings) == 4


def test_outbound_reminder_call():
    tel, domain = make("да, это я", "двенадцатое мая шестидесятого года", "приду")
    call_id = str(uuid.uuid4())
    tel.pending[call_id] = "B0001"
    call = asyncio.run(converse(tel, range(3), call_id=call_id))
    assert call.kind == "напоминание" and call.turns[0]["text"].endswith("это Иван?")
    assert domain.schedule.reminder_log[-1]["code"] == "CONFIRMED"


def test_escalation_marks_warm_transfer_for_dialplan(monkeypatch):
    """Перевод на регистратора — не заглушка: отметка в AstDB, диалплан соединяет после AudioSocket."""
    import apps.telephony as telephony
    actions = []

    async def fake_ami(fields):
        actions.append(fields)
        return "Response: Success"

    monkeypatch.setattr(telephony, "ami_action", fake_ami)
    tel, domain = make("Соедините с оператором", "Петров Иван 12.03.1958")
    call_id = str(uuid.uuid4())
    call = asyncio.run(converse(tel, range(2), call_id=call_id))
    assert call.transferred and actions[-1] == {"Action": "DBPut", "Family": "transfer", "Key": call_id, "Val": "1"}
    assert domain.schedule.escalations


def test_unanswered_reminder_is_recorded_as_not_reached(monkeypatch):
    """С-10: пациент не взял трубку — итог «не дозвонились» уходит в карточку пациента."""
    import apps.telephony as telephony

    async def fake_originate(*args, **kwargs):
        return None

    monkeypatch.setattr(telephony, "ami_originate", fake_originate)
    tel, domain = make()
    tel.answer_timeout = 0.05
    tel.not_reached = lambda bid: domain.schedule.log_reminder(domain.schedule.bookings[bid], None, "NOT_REACHED")

    async def scenario():
        call_id = await tel.call_reminder("B0001")
        await asyncio.sleep(0.2)
        return call_id

    call_id = asyncio.run(scenario())
    assert call_id not in tel.pending and domain.schedule.reminder_log[-1]["code"] == "NOT_REACHED"


def test_operator_request_then_silence_still_transfers(monkeypatch):
    import apps.telephony as telephony
    actions = []

    async def fake_ami(fields):
        actions.append(fields)
        return "Response: Success"

    monkeypatch.setattr(telephony, "ami_action", fake_ami)
    tel, domain = make("Соедините с оператором")
    call = asyncio.run(converse(tel, range(1), linger=6.5))
    assert call.transferred and actions and domain.schedule.escalations


def test_ami_action_skips_events_and_matches_action_id(monkeypatch):
    """Asterisk шлёт события между ответами (FullyBooted после входа) — ответ ищем по ActionID."""
    import re

    import apps.telephony as telephony

    class FakeAmi:
        def __init__(self):
            self.lines = [b"Asterisk Call Manager/7.0.1\r\n"]
            self.written = b""

        def write(self, data):
            self.written += data
            for block in data.decode().split("\r\n\r\n"):
                if m := re.search(r"ActionID: (\w+)", block):
                    self.lines += [b"Event: FullyBooted\r\n", b"Status: Fully Booted\r\n", b"\r\n",
                                   b"Response: Success\r\n", f"ActionID: {m.group(1)}\r\n".encode(), b"\r\n"]

        async def drain(self):
            pass

        async def readline(self):
            return self.lines.pop(0)

        def close(self):
            pass

    fake = FakeAmi()

    async def fake_open(host, port):
        return fake, fake

    monkeypatch.setenv("AMI_SECRET", "x")
    monkeypatch.setattr(telephony.asyncio, "open_connection", fake_open)
    result = asyncio.run(telephony.ami_action({"Action": "DBPut", "Family": "transfer", "Key": "k", "Val": "1"}))
    assert "Success" in result and b"Events: off" in fake.written


def _speech_frames(seconds: float, voice: float, echo: float, voice_at_ms: int = 1000):
    """Кадры 20 мс: эхо ответа с самого начала, голос пациента — с voice_at_ms. Огибающая — слоги и паузы."""
    import random

    rnd = random.Random(3)

    def env(t: float, peak: float) -> float:
        if t < 0 or peak == 0:
            return 0.0
        return peak * (0.35 + 0.65 * abs(math.sin(math.pi * t / 200))) if t % 900 < 720 else peak * 0.08

    n = FRAME_BYTES // 2
    for i in range(int(seconds * 1000 / FRAME_MS)):
        t = i * FRAME_MS
        level = math.hypot(env(t - voice_at_ms, voice), env(t, echo), 60)
        yield t, struct.pack(f"<{n}h", *(int(max(-32767, min(32767, rnd.gauss(0, level)))) for _ in range(n)))


def _barge_in_after(voice: float, echo: float) -> int | None:
    vad = Vad()
    for _ in range(50):  # пауза перед ответом: шум линии
        vad.feed(b"\0" * FRAME_BYTES)
    for t, frame in _speech_frames(5, voice, echo):
        started, _ = vad.feed(frame, playing=True)
        if started:
            return t - 1000
    return None


def test_barge_in_on_phone_hears_voice_over_echo_but_not_echo_alone():
    # Эхо собственного ответа, даже громкое, ассистента не прерывает
    assert all(_barge_in_after(0, echo) is None for echo in (300, 1000, 2000))
    # Голос пациента вдвое громче эха прерывает за ~0,3 с, несмотря на паузы между слогами
    for voice, echo in ((800, 0), (1500, 300), (3000, 1000)):
        took = _barge_in_after(voice, echo)
        assert took is not None and 0 <= took <= 400, (voice, echo, took)


def test_early_recognition_is_reused_and_yes_no_shortens_end_of_phrase():
    """Канал шлёт кадры и в паузах: фраза уходит в распознавание через 0,24 с тишины, к концу паузы текст готов —
    второй раз не распознаём. После «Верно?» конец фразы — по короткой паузе."""
    from apps.telephony import SHORT_END_MS

    tel, domain = make("Запишите к лору", "первый", "Петров Иван 12.03.1958", "да", "нет, спасибо")
    ends = []
    orig = tel.agent.handle

    def handle(session, text):
        step = orig(session, text)
        ends.append(domain.expects_short_answer(session.state))
        return step

    tel.agent.handle = handle
    call = asyncio.run(converse(tel, range(5), silence_frames=40))
    said = [t["text"] for t in call.turns if t["role"] == "assistant"]
    assert any(t.startswith("Готово, вы записаны") for t in said)
    assert call.early_stt == 5 and tel.stt.calls == 5  # все фразы распознаны в паузе, второй раз — нет
    assert ends[2] is True                          # «…записываю вас: … Верно?» — ждём «да/нет»
    assert ends[0] is False                         # «Какой вариант вам удобнее?» — обычная пауза
    assert SHORT_END_MS < 500



class LongTts(FakeTts):
    def synthesize(self, text):
        return pcm_to_wav(b"\1\0" * 16000 * 2, rate=16000)  # 2 с звука на любую фразу


async def speak_over_greeting(tel, speech_ms):
    """Пациент говорит поверх приветствия (после 0,4 с тишины), потом молчит; звонок кладём через 3 с."""
    await tel.start(port=0)
    port = tel.server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(frame(KIND_UUID, uuid.uuid4().bytes))
    await writer.drain()
    await asyncio.sleep(0.2)
    for _ in range(20):
        writer.write(frame(KIND_AUDIO, SILENCE))
    speech = tone(speech_ms)
    for i in range(0, len(speech), FRAME_BYTES):
        writer.write(frame(KIND_AUDIO, speech[i:i + FRAME_BYTES]))
    for _ in range(40):
        writer.write(frame(KIND_AUDIO, SILENCE))
    await writer.drain()
    await asyncio.sleep(3)
    writer.write(frame(KIND_HANGUP, b""))
    await writer.drain()
    await asyncio.sleep(0.3)
    writer.close()
    tel.server.close()
    return tel.calls[-1]


def make_long(*texts):
    tel, domain = make(*texts)
    tel.tts = LongTts()
    return tel, domain


def test_backchannel_over_reply_does_not_stop_assistant():
    """«Угу» поверх приветствия: ассистент не замолкает (ответ лишь приглушается на время проверки)."""
    tel, _ = make_long("угу")
    call = asyncio.run(speak_over_greeting(tel, 400))
    assert call.backchannels == 1 and call.barge_ins == 0
    assert not any(t["role"] == "patient" for t in call.turns)  # поддакивание — не реплика для агента


def test_short_request_over_reply_interrupts():
    tel, _ = make_long("соедините с оператором")
    call = asyncio.run(speak_over_greeting(tel, 400))
    assert call.barge_ins == 1 and call.backchannels == 0
    assert any(t["role"] == "patient" and "оператором" in t["text"] for t in call.turns)


def test_long_speech_over_reply_interrupts_without_waiting_for_text():
    tel, _ = make_long("я хотел бы записаться к терапевту на завтра")
    call = asyncio.run(speak_over_greeting(tel, 1400))
    assert call.barge_ins == 1 and call.backchannels == 0


def test_hangup_while_listening_ends_call_cleanly(caplog):
    """Пациент положил трубку, пока ассистент слушал: звонок закрыт без «Сбой обработки звонка» в логе."""
    tel, _ = make("До скольки вы работаете в субботу?")
    call = asyncio.run(converse(tel, range(1), linger=0.6))
    assert call.ended is not None
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_time_to_first_audio_is_measured_per_turn():
    from apps.journal import Journal

    tel, _ = make("Запишите к лору", "первый")
    tel.journal = Journal()
    call = asyncio.run(converse(tel, range(2)))
    assert len(call.ttfa_ms) == 2 and all(0 <= x < 3000 for x in call.ttfa_ms)
    stats = tel.journal.stats()
    assert stats["ttfa_count"] == 2 and stats["ttfa_p50"] is not None


def test_silero_gate_rejects_loud_noise_and_hears_quiet_voice():
    """С нейросетевым детектором громкий шум (Silero: не речь) не начинает фразу, а тихий голос — начинает."""
    class FakeSilero:
        def __init__(self, prob):
            self.p = prob

        def __call__(self, chunk, rate):
            assert rate == 8000 and len(chunk) == 256
            return self.p

    loud = struct.pack("<160h", *([3000, -3000] * 80))
    quiet = struct.pack("<160h", *([330, -330] * 80))
    pytest = __import__("pytest")
    try:
        __import__("torch")
    except ImportError:
        pytest.skip("нужен torch (есть на ВМ)")
    def after_silence(vad):  # как в звонке: сначала тишина линии — детектор узнаёт её шум
        for _ in range(60):
            vad.feed(b"\0" * FRAME_BYTES)
        return vad

    noise = after_silence(Vad(model=FakeSilero(0.1)))
    assert not any(noise.feed(loud)[0] for _ in range(40))
    voice = after_silence(Vad(model=FakeSilero(0.9)))
    assert any(voice.feed(quiet)[0] for _ in range(40))
    plain = after_silence(Vad())
    assert not any(plain.feed(quiet)[0] for _ in range(40))  # без Silero порог выше — тихий голос не слышно


def test_without_speech_recognition_call_goes_straight_to_registrar(monkeypatch):
    """GigaAM не загружена: облачного резерва нет (152-ФЗ) — звонок сразу переводится на регистратора."""
    import apps.telephony as telephony
    from apps.telephony import NO_STT

    async def fake_ami(fields):
        return "Response: Success"

    monkeypatch.setattr(telephony, "ami_action", fake_ami)
    async def ring(tel):
        await tel.start(port=0)
        port = tel.server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(frame(KIND_UUID, uuid.uuid4().bytes))
        await writer.drain()
        while True:  # слушаем ответ, пока ассистент сам не положит трубку
            kind, size = struct.unpack(">BH", await reader.readexactly(3))
            await reader.readexactly(size)
            if kind == KIND_HANGUP:
                break
        writer.close()
        tel.server.close()
        return tel.calls[-1]

    tel, _ = make()
    tel.stt.enabled = False
    call = asyncio.run(ring(tel))
    assert call.transferred and call.turns[0]["text"] == NO_STT


def test_slow_answer_is_filled_with_hold_phrases_not_silence(monkeypatch):
    """МИС упёрлась в лимит запросов: пока ответ готовится, пациент слышит «Секунду», затем «Ещё мгновение»."""
    import time

    import apps.telephony as telephony

    monkeypatch.setattr(telephony, "FILLER_AFTER", 0.05)
    monkeypatch.setattr(telephony, "HOLD_EVERY", 0.15)
    said = []

    class RecordingTts(FakeTts):
        def synthesize(self, text):
            said.append(text)
            return super().synthesize(text)

    tel, _ = make("До скольки вы работаете в субботу?")
    tel.tts = RecordingTts()
    handle = tel.agent.handle
    monkeypatch.setattr(tel.agent, "handle", lambda s, t: (time.sleep(0.6), handle(s, t))[1])
    call = asyncio.run(converse(tel, range(1), linger=0.6))
    # по разу фразы синтезируются заранее (прогрев), остальное прозвучало в звонке
    assert said.count(telephony.FILLER) == 2 and said.count(telephony.HOLD) >= 2
    assert any("суббот" in t["text"] for t in call.turns if t["role"] == "assistant")


def test_vad_keeps_quiet_start_of_phrase_when_confirmation_is_slow():
    """Silero «мигает» на начале фразы: подтверждение речи затягивается, но тихое начало («С» в «Смирнова») и
    300 мс до него остаются в записи — распознавание получает слово целиком, без дыр."""
    class FlickerVad(Vad):
        """Silero по очереди «речь» и «не речь», пока фраза не разгонится, — как на синтезированном голосе 8 кГц."""
        n = 0

        def _speech(self, frame):
            self.n += 1
            return self.n > 60 or self.n % 2 == 0

    vad = FlickerVad(model=object())
    for _ in range(20):
        vad.feed(SILENCE)
    speech = tone(1500)
    frames = [speech[i:i + FRAME_BYTES] for i in range(0, len(speech), FRAME_BYTES)]
    for f in frames:
        vad.feed(f)
    out = None
    for _ in range(60):
        _, out = vad.feed(SILENCE)
        if out:
            break
    assert out is not None
    assert out.find(speech[:FRAME_BYTES]) >= 0               # первый кадр речи в записи
    assert len(out) >= len(speech) + 10 * FRAME_BYTES        # речь целиком, без выброшенных тихих кадров
