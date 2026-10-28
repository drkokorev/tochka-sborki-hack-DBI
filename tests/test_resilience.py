"""Отказоустойчивость: предохранители внешних зависимостей, запасные пути, ПДн в логах (core/resilience)."""

import asyncio
import logging
import struct
import uuid

import httpx
import pytest

from core.agent import Agent
from core.llm import extract
from core.resilience import FALLBACKS, Breaker, breaker, health, install_log_masking, mask_pii
from domains.clinic.domain import ClinicDomain, ClinicUpdate
from domains.clinic.mis import MisClient, MisError
from tests.conftest import TODAY, FakeLLM


def test_breaker_opens_after_failures_and_lets_one_trial_through_after_cooldown():
    b = Breaker("x", threshold=3, cooldown=0.05)
    for _ in range(3):
        assert b.allow()
        b.fail(TimeoutError())
    assert b.state == "down" and not b.allow()            # разомкнут — зависимость не зовём
    asyncio.run(asyncio.sleep(0.06))
    assert b.allow() and not b.allow()                    # пауза прошла — одна пробная попытка
    b.ok()
    assert b.state == "ok" and b.allow()
    assert b.snapshot()["last_error"] == "TimeoutError"   # только тип ошибки, без текста


def test_llm_outage_trips_breaker_and_agent_keeps_answering_by_rules():
    llm = FakeLLM(*[TimeoutError("шлюз молчит")] * 10)
    for _ in range(3):
        assert extract(llm, ClinicUpdate, [{"role": "user", "content": "x"}]) == ClinicUpdate()
    calls = len(llm.calls)
    assert breaker("llm").state == "down"
    agent = Agent(ClinicDomain(today=TODAY, use_phrasebook=False), llm, today=lambda: TODAY)
    s = agent.new_session()
    step = agent.handle(s, "Запишите меня к терапевту на завтра")
    assert len(llm.calls) == calls                        # LLM больше не зовём, пока предохранитель не остынет
    assert s.state.intent == "book" and s.last_nlu == "rules" and step.reply


def test_mis_outage_fails_fast_with_mis_error():
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        raise httpx.ConnectError("нет сети")

    client = MisClient("https://stand.example", "token")
    client._http = httpx.Client(base_url="https://stand.example/api/v1", transport=httpx.MockTransport(handler))
    for _ in range(3):
        with pytest.raises(MisError) as err:
            client.doctors()
        assert err.value.status == 503
    with pytest.raises(MisError):
        client.doctors()
    assert len(attempts) == 3                             # четвёртый запрос не ушёл: сразу R-06


def test_mis_business_errors_do_not_trip_breaker():
    def handler(request):
        return httpx.Response(409, json={"code": "КОНФЛИКТ", "message": "Запись уже отменена"})

    client = MisClient("https://stand.example", "token")
    client._http = httpx.Client(base_url="https://stand.example/api/v1", transport=httpx.MockTransport(handler))
    for _ in range(5):
        with pytest.raises(MisError):
            client.update_visit(1, status="Отменена")
    assert breaker("mis").state == "ok"                   # стенд жив, просто отказал по правилам


def test_mis_rate_limit_wait_is_counted_and_batch_runs_pace(monkeypatch):
    """429 от стенда: клиент ждёт по Retry-After и считает ожидание отдельно; pace() освобождает окно лимита."""
    import domains.clinic.mis as mis

    slept, clock = [], [1000.0]
    monkeypatch.setattr(mis._time, "sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(mis._time, "monotonic", lambda: clock[0])
    replies = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json=[])])
    client = MisClient("https://stand.example", "token")
    client._http = httpx.Client(base_url="https://stand.example/api/v1",
                                transport=httpx.MockTransport(lambda request: next(replies)))
    assert client.doctors() == [] and client.waited == 7 and slept == [7]
    assert client.pace(8) == 0                            # 2 запроса в окне + 8 = лимит, ждать не нужно
    client.sent += [clock[0]] * 5                         # окно почти заполнено
    assert client.pace(8) > 50                            # ждём, пока старые запросы выйдут из минутного окна


def test_personal_data_is_masked_in_logs(caplog):
    install_log_masking()
    log = logging.getLogger("tests.pii")
    with caplog.at_level(logging.WARNING):
        log.warning("Пациент %s, %s, тел. %s", "Иванов Иван Петрович", "12.05.1960", "+7 912 345-67-89")
        try:
            raise ValueError("клиент Смирнова Анна 17.03.1985 не найден")
        except ValueError:
            log.exception("Сбой стенда")
    text = caplog.text + "".join(r.exc_text or "" for r in caplog.records)
    for secret in ("Иванов", "12.05.1960", "912", "Смирнова", "17.03.1985"):
        assert secret not in text, secret
    assert "[ФИО]" in text and "[дата]" in text and "[телефон]" in text
    assert mask_pii("терапевт Белова, кабинет 12") == "терапевт Белова, кабинет 12"  # одно слово — не ФИО


def test_every_dependency_has_a_documented_fallback():
    assert {"llm", "mis", "stt", "tts", "ami", "journal"} <= set(FALLBACKS)
    breaker("mis").fail(TimeoutError())
    assert health()["mis"]["fallback"] == FALLBACKS["mis"]


async def _ring(tel):
    """Звонок, который слушает ассистента до отбоя."""
    from tests.test_telephony import frame

    await tel.start(port=0)
    port = tel.server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(frame(0x01, uuid.uuid4().bytes))
    await writer.drain()
    try:
        while True:
            kind, size = struct.unpack(">BH", await asyncio.wait_for(reader.readexactly(3), timeout=10))
            await reader.readexactly(size)
            if kind == 0x00:
                break
    finally:
        writer.close()
        tel.server.close()
    return tel.calls[-1]


def _fake_ami(monkeypatch):
    import apps.telephony as telephony

    async def fake(fields):
        return "Response: Success"

    monkeypatch.setattr(telephony, "ami_action", fake)


def test_phone_tts_failure_transfers_to_registrar(monkeypatch):
    from tests.test_telephony import make

    class BrokenTts:
        def synthesize(self, text):
            raise RuntimeError("Silero упал")

    _fake_ami(monkeypatch)
    tel, _ = make()
    tel.tts = BrokenTts()
    call = asyncio.run(_ring(tel))
    assert call.transferred                               # пациент нас не слышит — к живому регистратору


def test_phone_agent_crash_says_sorry_and_transfers(monkeypatch):
    from apps.telephony import AGENT_FAILED
    from tests.test_telephony import FRAME_BYTES, KIND_AUDIO, frame, make, tone

    _fake_ami(monkeypatch)
    tel, _ = make("Запишите к лору")

    def crash(session, text):
        raise KeyError("неожиданная ошибка")

    tel.agent.handle = crash

    async def call():
        await tel.start(port=0)
        port = tel.server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(frame(0x01, uuid.uuid4().bytes))
        await writer.drain()
        await asyncio.sleep(0.3)
        speech = tone(600)
        for i in range(0, len(speech), FRAME_BYTES):
            writer.write(frame(KIND_AUDIO, speech[i:i + FRAME_BYTES]))
        await writer.drain()
        while True:
            kind, size = struct.unpack(">BH", await asyncio.wait_for(reader.readexactly(3), timeout=10))
            await reader.readexactly(size)
            if kind == 0x00:
                break
        writer.close()
        tel.server.close()
        return tel.calls[-1]

    record = asyncio.run(call())
    assert record.transferred and record.turns[-1]["text"] == AGENT_FAILED
