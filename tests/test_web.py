import os

for _var in ("DBI_API_TOKEN", "LLM_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_var, None)  # тесты веба — на локальных данных и правилах, без сети
# и без тяжёлых локальных моделей, даже если они установлены (на ВМ): ученик и GigaAM проверяются отдельно
os.environ.update({"STUDENT": "0", "TELEPHONY": "0", "JOURNAL_PATH": "", "DEMO_DB": ""})

from fastapi.testclient import TestClient  # noqa: E402

from apps.web import server
from domains.clinic.schedule import DEMO_PATIENTS  # noqa: E402

client = TestClient(server.app)


def test_web_flow_and_clinic_panel():
    client.post("/api/reset")
    start = client.post("/api/session").json()
    assert start["greeting"] and any("Иванов Иван" in p for p in start["demo_patients"])
    before = len(client.get("/api/clinic").json()["bookings"])

    sid = start["session_id"]
    ends = []
    for text in ["Запишите к терапевту", "первый", "Тестов Тест 01.01.1990", "да"]:
        r = client.post("/api/message", json={"session_id": sid, "text": text}).json()
        ends.append(r["end_ms"])
    # «Какой вариант?», «Назовите…» — обычная пауза конца фразы; «…Верно?» — короткая; «Могу ещё помочь?» — обычная
    assert ends == [server.END_MS, server.END_MS, server.SHORT_END_MS, server.END_MS]
    assert r["done"] and r["nlu"] in ("phrasebook", "rules", "llm") and r["latency_ms"] >= 0
    assert len(client.get("/api/clinic").json()["bookings"]) == before + 1


def test_unknown_session_is_404():
    assert client.post("/api/message", json={"session_id": "nope", "text": "hi"}).status_code == 404


def test_index_served():
    page = client.get("/")
    assert "Регистратура" in page.text and page.headers["cache-control"] == "no-cache"


def test_voice_endpoints_degrade_gracefully():
    voice = client.post("/api/session").json()["voice"]
    assert voice["stt"] is None  # GigaAM в тестах не грузится — облачного резерва нет, интерфейс предложит текст
    assert client.post("/api/stt", content=b"", headers={"Content-Type": "audio/webm"}).status_code == 400
    assert client.post("/api/stt", content=b"x" * 10, headers={"Content-Type": "audio/webm"}).status_code == 503
    r = client.post("/api/tts", json={"text": "Здравствуйте"})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.content[:4] == b"RIFF"


def test_reminder_call_via_web():
    client.post("/api/reset")
    r = client.get("/api/reminders").json()
    assert r["items"]
    assert r["window"]["demo"] and "суббота 10:00–15:00" in r["window"]["text"]
    assert r["sms"]["channel"].startswith("не подключён")
    item = r["items"][0]
    call = client.post(f"/api/reminders/{item['id']}/call").json()
    assert call["greeting"].startswith("Здравствуйте! Это виртуальный помощник")
    born = DEMO_PATIENTS[item["patient"]][1]
    for text in ["да", f"{born:%d.%m.%Y}", "приду"]:
        step = client.post("/api/message", json={"session_id": call["session_id"], "text": text}).json()
    assert step["done"] and client.get("/api/reminders").json()["log"][0]["code"] == "CONFIRMED"
    # С-10: по подтверждённой записи больше не звоним — и с рабочего места тоже
    again = client.post(f"/api/reminders/{item['id']}/call")
    assert again.status_code == 409 and "подтверждена" in again.json()["detail"]


def test_transfer_card_and_analytics():
    client.post("/api/reset")
    sid = client.post("/api/session").json()["session_id"]
    for text in ["Соедините с оператором", "Петров Иван 12.03.1958"]:
        r = client.post("/api/message", json={"session_id": sid, "text": text}).json()
    assert r["done"] and r["understood"]
    card = next(c for c in client.get("/api/transfers").json()["items"] if c["id"] == sid)
    assert card["priority"] == "P2" and card["reason"] == "Просит сотрудника"
    assert card["rule"] == "R-02" and card["header"] == "ПЕРЕВОД R-02 P2" and card["result_code"] == "ESCALATED"
    assert any("Петров Иван" in d for d in card["done"])
    assert client.post(f"/api/transfers/{sid}/take").json()["ok"]
    live = client.get("/api/dashboard").json()["live"]
    assert live["finished"] >= 1 and "Перевод" in live["intents"]
    conv = client.get(f"/api/conversations/{sid}").json()
    assert conv["outcome"] == "escalated" and conv["turns"][1]["role"] == "patient"
    assert client.get("/api/status").json()["demo"] is True
    assert "Аналитика" in client.get("/dashboard").text


def test_visit_journal_local():
    client.post("/api/reset")
    sid = client.post("/api/session").json()["session_id"]
    for text in ["Отмените мою запись", "Смирнова Анна 17.03.1985", "нет", "да"]:
        client.post("/api/message", json={"session_id": sid, "text": text})
    j = client.get("/api/visits").json()
    assert j["counts"].get("Отменена") == 1 and j["counts"].get("Запланирована") == 2
    assert any(v["patient"] == "Смирнова Анна" and v["status"] == "Отменена" for v in j["items"])


def test_journal_survives_restart_and_skips_empty(tmp_path):
    from types import SimpleNamespace

    from apps.journal import Journal
    path = tmp_path / "journal.json"
    j = Journal(path=path)
    j.start("a", "веб", "Здравствуйте")
    j.start("b", "веб", "Здравствуйте")  # вкладку открыли и ушли
    step = SimpleNamespace(reply="Слушаю вас.", done=False, payload=None)
    j.turn("a", "алло", step, SimpleNamespace(), None, "rules", 3.0)
    again = Journal(path=path)
    assert [c["id"] for c in again.recent()] == ["a"] and again.stats()["conversations"] == 1


def test_journal_privacy_retention_and_deletion(tmp_path):
    import os
    import stat
    import time
    from types import SimpleNamespace

    from apps.journal import PRIVATE, Journal
    path = tmp_path / "journal.json"
    j = Journal(path=path)
    step = SimpleNamespace(reply="Хорошо, запись разговора отключена.", done=False, payload=None)
    j.start("p", "веб", "Здравствуйте")
    j.turn("p", "Иванов Иван, не записывайте разговор", step, SimpleNamespace(), SimpleNamespace(topic="no_recording"),
           "phrasebook", 1.0)
    assert all(t["text"] == PRIVATE for t in j.get("p")["turns"])
    j.start("a", "веб", "Здравствуйте")
    j.turn("a", "алло", step, SimpleNamespace(), None, "rules", 1.0)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert [c["id"] for c in Journal(path=path).recent()] == ["a"]  # скрытый разговор на диск не попал
    assert j.delete("a") and Journal(path=path).recent() == []
    j.start("old", "веб", "Здравствуйте")
    j.turn("old", "алло", step, SimpleNamespace(), None, "rules", 1.0)
    j.conversations["old"]["started"] = time.time() - 40 * 86400
    j.end("old")
    assert Journal(path=path, days=30).recent() == []  # старше срока хранения — не загружается
    assert client.delete("/api/conversations/nope").status_code == 404


def test_offline_registry_when_stand_is_down_transfers_instead_of_crashing():
    from core.agent import Agent
    from domains.clinic.domain import ClinicDomain
    from apps.container import OfflineRegistry
    domain = ClinicDomain(schedule=OfflineRegistry())
    agent = Agent(domain, None)
    s = agent.new_session()
    assert agent.handle(s, "До скольки вы работаете в субботу?").reply.startswith("Поликлиника работает")
    step = agent.handle(s, "Запишите к терапевту")
    assert s.state.escalation == "mis_error" and step.done



def test_voice_metrics_and_backchannel_flag(monkeypatch):
    class FakeStt:
        last_engine = "gigaam"

        def __init__(self, text):
            self.text = text

        def transcribe(self, audio, filename="speech.wav"):
            return self.text

    sid = client.post("/api/session").json()["session_id"]
    monkeypatch.setattr(server, "stt", FakeStt("угу"))
    r = client.post("/api/stt", content=b"RIFF....", headers={"Content-Type": "audio/wav", "X-Session-Id": sid}).json()
    assert r["backchannel"] is True
    # Ассистент спросил «да/нет» — «да» поверх ответа уже ответ, а не поддакивание
    for text in ["Запишите к терапевту", "первый", "Тестов Тест 01.01.1990"]:
        client.post("/api/message", json={"session_id": sid, "text": text})
    monkeypatch.setattr(server, "stt", FakeStt("да"))
    r = client.post("/api/stt", content=b"RIFF....", headers={"Content-Type": "audio/wav", "X-Session-Id": sid}).json()
    assert r["backchannel"] is False
    assert client.post("/api/voice-metrics", json={"session_id": sid, "ttfa_ms": 640, "barge": True}).json()["ok"]
    assert client.post("/api/voice-metrics", json={"session_id": "nope", "ttfa_ms": 1}).status_code == 404
    live = client.get("/api/dashboard").json()["live"]
    assert live["ttfa_count"] >= 1 and live["barge_ins_per_call"] is not None


def test_health_reports_components_and_fallbacks():
    from core.resilience import breaker

    h = client.get("/api/health").json()
    assert set(h["components"]) >= {"llm", "mis", "stt", "tts", "telephony"}
    assert all(c["fallback"] for name, c in h["components"].items() if name in ("llm", "mis", "stt", "tts"))
    for _ in range(3):
        breaker("mis").fail(TimeoutError())
    h = client.get("/api/health").json()
    assert h["status"] == "degraded" and "mis" in h["degraded"] and h["components"]["mis"]["last_error"] == "TimeoutError"


def test_message_returns_confidence_and_structured_result():
    sid = client.post("/api/session").json()["session_id"]
    r = client.post("/api/message", json={"session_id": sid, "text": "Запишите к терапевту"}).json()
    assert 0 < r["confidence"] <= 1 and r["confidence_level"] in ("high", "medium", "low")
    assert r["result"]["schema"] == "clinic.turn/v1" and r["result"]["awaiting"] == "choice" and r["result"]["offered"]


def test_desk_endpoints_require_token_when_configured(monkeypatch):
    """Рабочее место регистратора (ПДн и действия) — только по токену; пациентская часть открыта."""
    from dataclasses import replace

    monkeypatch.setattr(server.services, "cfg", replace(server.services.cfg, desk_token="desk-secret-1"))
    for method, path in [("get", "/api/conversations"), ("get", "/api/transfers"), ("get", "/api/visits"),
                         ("get", "/api/calls"), ("post", "/api/reset"), ("delete", "/api/conversations/x")]:
        assert getattr(client, method)(path).status_code == 401, path
    assert client.get("/api/transfers", headers={"X-Desk-Token": "desk-secret-1"}).status_code == 200
    assert client.post("/api/session").status_code == 200 and client.get("/api/health").status_code == 200


def test_lab_delay_is_a_callback_with_card():
    client.post("/api/reset")
    sid = client.post("/api/session").json()["session_id"]
    for text in ["сдал анализы, срок прошёл, а результата до сих пор нет", "Петров Иван 12.03.1958"]:
        r = client.post("/api/message", json={"session_id": sid, "text": text}).json()
    assert r["done"]
    card = next(c for c in client.get("/api/transfers").json()["items"] if c["id"] == sid)
    assert card["header"] == "ПЕРЕВОД R-08 P3" and card["result_code"] == "CALLBACK"
    conv = next(c for c in client.get("/api/conversations").json()["items"] if c["id"] == sid)
    assert conv["outcome"] == "callback"
