"""Веб-макет голосового ассистента клиники (кейс DBI).

Запуск: .venv/bin/uvicorn apps.web.server:app --host 0.0.0.0 --port 8000
С переменной DBI_API_TOKEN работает через тестовый стенд МИС, без неё — на локальных тестовых данных.
LLM — OpenAI-совместимый шлюз (LLM_* или OPENAI_* в окружении); без ключа фразы разбираются правилами.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import hmac

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from apps.container import build_services, result_path, stand_registry
from apps.journal import understood
from apps.voice import is_filler, meaningful_for, split_speech
from core.agent import Session
from core.config import get_settings
from core.resilience import FALLBACKS, health as breakers_health, install_log_masking
from domains.clinic import phrasebook
from domains.clinic import reminders as c10
from domains.clinic.domain import GREETING
from domains.clinic.result import level, turn_result
from domains.clinic.schedule import DEMO_PATIENTS

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:  # под uvicorn у логгеров приложения нет обработчика — записи терялись
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
install_log_masking()  # ФИО, даты, телефоны в логах заменяются метками — и при сбоях внешних сервисов

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Голосовой ассистент клиники — макет")


FALLBACK_REPLY = "Не удалось связаться с системой регистратуры. Передаю звонок регистратору, оставайтесь на линии."

# Все зависимости собирает apps/container.py; здесь — только HTTP-маршруты поверх готового приложения
services = build_services()
MIS_TOKEN = services.cfg.mis_token
agent, domain = services.agent, services.domain
local_stt, stt, tts = services.local_stt, services.stt, services.tts
journal, noshow, telephony = services.journal, services.noshow, services.telephony
_result_path = result_path
MAX_AUDIO = 10 * 1024 * 1024

def desk_access(request: Request) -> None:
    """Рабочее место регистратора — ПДн пациентов и действия (обзвон, удаление, сброс): только по токену
    DESK_TOKEN (заголовок X-Desk-Token). Без токена в окружении — открытый демо-режим с вымышленными данными;
    в эксплуатации токен обязателен (README, «Запуск»). Пациентская часть (звонок, голос, health) — без токена."""
    token = services.cfg.desk_token
    if token and not hmac.compare_digest(request.headers.get("x-desk-token", "").encode(), token.encode()):
        raise HTTPException(401, "Нужен токен рабочего места регистратора")


DESK = [Depends(desk_access)]

class SessionStore:
    """Разговоры веба в памяти: под блокировкой (запросы идут из пула потоков), брошенные вкладки — через 2 часа,
    не больше MAX сразу (самые старые вытесняются). Реплики одного разговора обрабатываются по очереди."""

    TTL, MAX = 2 * 3600, 1000

    def __init__(self):
        self._items: dict[str, tuple[Session, float, threading.Lock]] = {}
        self._lock = threading.Lock()

    def put(self, session_id: str, session: Session) -> None:
        now = time.time()
        with self._lock:
            for sid in [sid for sid, (_, t, _) in self._items.items() if now - t > self.TTL]:
                del self._items[sid]
            while len(self._items) >= self.MAX:
                del self._items[min(self._items, key=lambda k: self._items[k][1])]
            self._items[session_id] = (session, now, threading.Lock())

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            item = self._items.get(session_id)
            if item is None:
                return None
            self._items[session_id] = (item[0], time.time(), item[2])
            return item[0]

    def turn_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            return self._items[session_id][2]

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._items

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


sessions = SessionStore()


def _remember(session_id: str, session: Session) -> None:
    sessions.put(session_id, session)


async def _reconnect_mis() -> None:
    """Если стенд был недоступен при старте — подключаемся, как только он ответит."""
    while getattr(domain.schedule, "offline", False):
        await asyncio.sleep(60)
        try:
            domain.set_schedule(await asyncio.to_thread(stand_registry, services.cfg))  # и врачи, и специальности
            log.info("Стенд МИС подключён")
        except Exception:
            log.warning("Стенд МИС всё ещё недоступен")


# Ручной запуск напоминания с рабочего места — объявленный демо-режим: окно С-10 и правило 24 часов не проверяются
# (заказчик разрешил на демо). Отказы, подтверждения и лимит попыток проверяются всегда. REMINDER_DEMO=0 — как в эксплуатации
REMINDER_DEMO = os.getenv("REMINDER_DEMO", "1") == "1"


def _reminder_queue() -> list[str]:
    """Кому звонить сейчас по правилам С-10 (окно, отказы, подтверждения, 24 часа, попытки) — сначала самый высокий
    риск неявки."""
    due = c10.due(domain.schedule, c10.clinic_now())
    return [b.id for b in sorted(due, key=lambda b: -(noshow.risk(b.slot.specialty, b.slot.start) if noshow else 0))]


async def _reminder_scheduler() -> None:
    """С-10 без ручного запуска: в окне обзвона (будни 10–19, суббота 10–15 по часам клиники) каждые REMINDER_EVERY
    минут — обзвон записей на завтра; повторная попытка — не раньше чем через 2 часа, после двух — SMS. Включается
    REMINDER_AUTO=1; номер пациента в эксплуатации берётся из CRM клиники (в карточке стенда его нет), в демо —
    софтфон команды (PHONE_ENDPOINT)."""
    every = 60 * int(os.getenv("REMINDER_EVERY", "30"))
    while True:
        queue = _reminder_queue()
        if queue:
            log.info("Обзвон С-10: %d записей на завтра", len(queue))
        for booking_id in queue:
            booking = domain.schedule.bookings.get(booking_id)
            # правила проверяем перед каждым звонком: окно могло закрыться, пациент — ответить на прошлый звонок
            if booking is None or c10.skip_reason(domain.schedule, booking, c10.clinic_now()):
                continue
            try:
                await telephony.call_reminder(booking_id)
            except Exception:
                log.warning("Напоминание по записи %s не отправлено", booking_id, exc_info=True)
                continue  # звонок не ушёл — попыткой не считаем
            domain.schedule.note_reminder_attempt(booking)
            await asyncio.sleep(90)  # по одному звонку: следующий — когда этот закончится
        await asyncio.sleep(every)


async def _start_telephony() -> None:
    """AudioSocket для Asterisk: включается переменной TELEPHONY=1 (на ВМ — в run.sh).
    Здесь же в фоне грузится локальное распознавание GigaAM."""
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, local_stt.load)
    loop.run_in_executor(None, _warm_tts)  # первая фраза веб-звонка звучит без задержки на загрузку модели
    if getattr(domain.schedule, "offline", False):
        loop.create_task(_reconnect_mis())
    if os.getenv("TELEPHONY") == "1":
        from domains.clinic.domain import ANYTHING_ELSE, CLARIFY, GOODBYE, OFFER_OPERATOR
        await telephony.start(port=int(os.getenv("AUDIOSOCKET_PORT", "9092")),
                              warm=(ANYTHING_ELSE, CLARIFY, GOODBYE, OFFER_OPERATOR))
        if os.getenv("REMINDER_AUTO", "0") == "1":
            asyncio.get_running_loop().create_task(_reminder_scheduler())


def _warm_tts() -> None:
    from domains.clinic.domain import ANYTHING_ELSE, ASK_SPECIALTY, LISTENING
    try:
        for phrase in (GREETING, LISTENING, ASK_SPECIALTY, ANYTHING_ELSE, "Какой вариант вам удобнее?",
                       "Назовите, пожалуйста, фамилию, имя, отчество и дату рождения."):
            for part in split_speech(phrase):  # браузер и телефония озвучивают по фразам — прогреваем фразы
                tts.synthesize(part)
    except Exception:
        log.warning("Синтез речи не прогрет")


app.router.on_startup.append(_start_telephony)


class MessageIn(BaseModel):
    session_id: str = Field(max_length=64)
    text: str = Field(max_length=2000)  # длиннее реплики не бывает; guard дальше режет до лимита агента
    stt_ms: float | None = None      # сколько распознавалась фраза (если сказана голосом)
    stt_engine: str | None = Field(None, max_length=32)


_NO_CACHE = {"Cache-Control": "no-cache"}  # после деплоя браузер сразу берёт новый интерфейс


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers=_NO_CACHE)


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers=_NO_CACHE)  # аналитика — вкладка единого интерфейса


@app.get("/api/health")
def health() -> dict:
    """Состояние зависимостей и запасной путь каждой (core/resilience.FALLBACKS). Без ПДн: только типы ошибок.
    status: ok — всё в строю; degraded — работает запасной путь (ассистент отвечает, часть функций — через
    регистратора); сам сервер при этом жив, поэтому ответ всегда 200."""
    components = {
        "llm": {"state": "ok" if agent.llm else "off", "detail": get_settings().llm_model if agent.llm else "не настроена"},
        "mis": {"state": "down" if getattr(domain.schedule, "offline", False) else ("ok" if MIS_TOKEN else "off"),
                "detail": "стенд МИС" if MIS_TOKEN else "демо-данные (SQLite)" if services.cfg.demo_db else "демо-данные"},
        "stt": {"state": "ok" if stt.enabled else "down", "detail": stt.name or "GigaAM не загружена"},
        "tts": {"state": "ok" if tts.available else "down", "detail": tts.name or "синтез недоступен"},
        "telephony": {"state": "ok" if telephony.server is not None else "off",
                      "detail": "AudioSocket" if telephony.server is not None else "TELEPHONY=0"},
    }
    for name, snap in breakers_health().items():  # живые сбои подряд важнее статической готовности
        comp = components.setdefault(name, {"state": "ok", "detail": ""})
        if snap["state"] != "ok":
            comp["state"] = snap["state"]
        comp |= {"failures_in_row": snap["failures_in_row"], "last_error": snap["last_error"]}
    for name, comp in components.items():
        # не подключено (демо без стенда, без ключа LLM) — это режим, а не сбой: запасной путь не нужен
        comp["fallback"] = FALLBACKS.get(name, "") if comp["state"] != "off" else f"не подключено: {comp['detail']}"
    bad = [n for n, c in components.items() if c["state"] in ("down", "degraded")]
    return {"status": "degraded" if bad else "ok", "degraded": bad, "components": components}


@app.get("/api/status")
def status() -> dict:
    """Из чего сейчас собран ассистент — для индикатора в шапке."""
    settings = get_settings()
    return {
        "speech": stt.name, "voice": f"{tts.name} на сервере" if tts.available else None,
        "understanding": "фразник + " + (("локальная модель + GigaChat" if domain.student else "GigaChat")
                                         if agent.llm else ("локальная модель + правила" if domain.student else "правила")),
        "llm": settings.llm_model if agent.llm else None,
        "mis": "стенд МИС «Регистратура»" if MIS_TOKEN else "демо-данные",
        "phone": "094" if telephony.server is not None else None,
        "demo": not MIS_TOKEN,
        "open_transfers": sum(not c["taken"] for c in journal.transfers()),
    }


@app.post("/api/session")
def new_session() -> dict:
    session_id = uuid.uuid4().hex
    _remember(session_id, agent.new_session())
    journal.start(session_id, "веб", GREETING)
    settings = get_settings()
    return {
        "session_id": session_id,
        "greeting": GREETING,
        "nlu": (settings.llm_model if agent.llm else "правила (LLM не подключена)")
               + (" + ученик rubert-tiny2" if domain.student else ""),
        "mis": "стенд МИС" if MIS_TOKEN else "тестовые данные",
        "voice": {"stt": stt.name,
                  "tts": f"{tts.name} (локально)" if tts.available else None},
        "demo_patients": [] if MIS_TOKEN else [f"{name}, {born:%d.%m.%Y}" for name, (_, born) in DEMO_PATIENTS.items()],
    }


END_MS, SHORT_END_MS = 800, 450


@app.post("/api/message")
def message(body: MessageIn) -> dict:
    session = sessions.get(body.session_id)
    if session is None:
        raise HTTPException(404, "Сессия не найдена — начните новый звонок")
    started = time.perf_counter()
    try:
        with sessions.turn_lock(body.session_id):  # две реплики одного разговора не перемешивают состояние
            step = agent.handle(session, body.text)
        reply, done = step.reply, step.done
    except Exception:
        log.exception("Сбой обработки реплики")
        reply, done = FALLBACK_REPLY, True
        journal.end(body.session_id)
        step = None
    elapsed = (time.perf_counter() - started) * 1000
    if step is not None:
        said = session.history[-2]["content"] if len(session.history) >= 2 else body.text[:500]  # очищенный текст
        journal.turn(body.session_id, said, step, session.state, session.last_update, session.last_nlu,
                     elapsed, body.stt_ms, body.stt_engine)
    return {
        "reply": reply,
        "done": done,
        "nlu": session.last_nlu,
        "latency_ms": round(elapsed),
        "understood": understood(session.state, session.last_update),
        # Пауза конца следующей фразы: после вопроса «да/нет» ответ короткий — ждём меньше
        "end_ms": SHORT_END_MS if getattr(domain, "expects_short_answer", lambda _: False)(session.state) else END_MS,
        "result_id": getattr(getattr(session.state, "result", None), "id", None),
        # Уверенность понимания и машиночитаемый итог реплики (схема clinic.turn/v1, domains/clinic/result.py)
        "confidence": session.last_confidence,
        "confidence_level": level(session.last_confidence),
        "result": turn_result(session.state, session.last_update, reply, getattr(step, "payload", None),
                              source=session.last_nlu, confidence=session.last_confidence) if step is not None else None,
        "state": session.state.model_dump(mode="json", exclude_none=True, exclude_defaults=True,
                                          exclude={"offered", "target", "result", "avoid"}),
    }


class TtsIn(BaseModel):
    text: str = Field(max_length=800)


@app.post("/api/stt")
async def speech_to_text(request: Request) -> dict:
    """Распознаёт фразу пациента: тело запроса — аудио (webm/ogg/wav) из браузера или телефонии."""
    if int(request.headers.get("content-length") or 0) > MAX_AUDIO:
        raise HTTPException(413, "Слишком длинная запись")  # не читаем тело, если оно заведомо больше лимита
    audio = await request.body()
    if not audio:
        raise HTTPException(400, "Пустое аудио")
    if len(audio) > MAX_AUDIO:
        raise HTTPException(413, "Слишком длинная запись")
    kind = request.headers.get("content-type", "audio/webm").split(";")[0].split("/")[-1] or "webm"
    started = time.perf_counter()
    try:
        text = await run_in_threadpool(stt.transcribe, audio, f"speech.{kind}")
    except Exception:
        log.exception("Сбой распознавания речи")
        raise HTTPException(503, "Распознавание речи временно недоступно — напишите текстом")
    # Реплика поверх ответа ассистента: «угу», «да-да» — поддакивание, ответ не прерываем. Если ассистент как раз
    # спросил «да/нет», то «да» — это ответ (состояние разговора — по заголовку X-Session-Id)
    session = sessions.get(request.headers.get("x-session-id", ""))
    asks = session is not None and getattr(domain, "expects_short_answer", lambda _: False)(session.state)
    return {"text": text, "engine": stt.last_engine, "latency_ms": round((time.perf_counter() - started) * 1000),
            "backchannel": is_filler(text, meaningful_for(agent, session.state) if session else None, asks=asks)}


class VoiceMetricsIn(BaseModel):
    session_id: str
    ttfa_ms: float | None = Field(None, ge=0, le=60000)
    barge: bool = False
    backchannel: bool = False


@app.post("/api/voice-metrics")
def voice_metrics(body: VoiceMetricsIn) -> dict:
    """Замеры браузера: от конца фразы пациента до первого звука ответа, перебивания и поддакивания."""
    if body.session_id not in sessions:
        raise HTTPException(404, "Сессия не найдена")
    journal.voice(body.session_id, ttfa_ms=body.ttfa_ms, barge=body.barge, backchannel=body.backchannel)
    return {"ok": True}


@app.post("/api/tts")
def text_to_speech(body: TtsIn) -> Response:
    """Озвучивает фразу ответа локально (Silero или Piper); 503 — браузер озвучит сам."""
    if not tts.available:
        raise HTTPException(503, "Синтез речи недоступен")
    try:
        wav = tts.synthesize(body.text.strip()[:800])
    except Exception:
        log.exception("Сбой синтеза речи")
        raise HTTPException(503, "Синтез речи недоступен")
    return Response(wav, media_type="audio/wav")


@app.get("/api/reminders", dependencies=DESK)
def reminders() -> dict:
    """С-10: кому звонить с напоминанием (записи на завтра) и журнал итогов обзвона."""
    schedule = domain.schedule
    day = schedule.reminder_day()
    upcoming = sorted((b for b in schedule.bookings.values() if b.slot.start.date() > schedule.today),
                      key=lambda b: b.slot.start)[:12]

    def risk(b):
        return noshow.risk(b.slot.specialty, b.slot.start) if noshow else None

    for b in upcoming:  # имена — для тех, кому звонить завтра (карточку со стенда читаем один раз)
        if not b.patient_name and b.slot.start.date() == day:
            try:
                if patient := schedule.patient_of(b):
                    b.patient_name = patient.full_name
            except Exception:
                log.warning("Не удалось прочитать карточку пациента записи %s", b.id)
    # Сначала звоним тем, кому завтра на приём и кто вероятнее не придёт
    upcoming.sort(key=lambda b: (b.slot.start.date() != day, -(risk(b) or 0), b.slot.start))
    now = c10.clinic_now()
    return {
        "day": day.isoformat(),
        "model": noshow.metrics if noshow else None,
        # правила С-10: окно обзвона и демо-режим ручного запуска (объявляется на рабочем месте)
        "window": {"open": c10.in_window(now), "text": c10.window_text(), "demo": REMINDER_DEMO,
                   "auto": os.getenv("REMINDER_AUTO", "0") == "1"},
        "items": [{"id": b.id, "patient": b.patient_name or f"пациент №{b.patient_id}",
                   "specialty": b.slot.specialty, "doctor": b.slot.doctor_name,
                   "start": b.slot.start.isoformat(timespec="minutes"), "due": b.slot.start.date() == day,
                   "risk": round(risk(b), 3) if noshow else None,
                   "skip": c10.skip_reason(schedule, b, now) if b.slot.start.date() == day else None,
                   "attempts": len(c10.attempts(schedule, b.id)),
                   "why": noshow.explain(b.slot.specialty, b.slot.start) if noshow else []}
                  for b in upcoming],
        # SMS-напоминания на завтра (всем, кроме отказавшихся от уведомлений): SMS-шлюза на стенде нет
        "sms": {"channel": "не подключён (на стенде нет SMS-шлюза)",
                "items": [{"id": b.id, "patient": b.patient_name or f"пациент №{b.patient_id}", "reason": why}
                          for b, why in c10.sms_due(schedule)]},
        "log": list(reversed(schedule.reminder_log[-15:])),
    }


def _check_reminder(booking_id: str):
    """Ручной звонок-напоминание: правила С-10, которые действуют и в демо-режиме (отказ, подтверждение, попытки)."""
    booking = domain.schedule.bookings.get(booking_id)
    if booking is None:
        raise HTTPException(404, "Запись не найдена")
    if reason := c10.skip_reason(domain.schedule, booking, c10.clinic_now(), demo=REMINDER_DEMO):
        raise HTTPException(409, f"Звонить не нужно: {reason}")
    return booking


@app.post("/api/reminders/{booking_id}/call", dependencies=DESK)
def reminder_call(booking_id: str) -> dict:
    """Начинает исходящий звонок-напоминание: в вебе — как разговор в окне звонка, в телефонии — через Asterisk."""
    booking = _check_reminder(booking_id)
    try:
        state, hello = domain.start_reminder(booking)
    except Exception:
        log.exception("Не удалось начать напоминание")
        raise HTTPException(503, "МИС не отвечает — попробуйте позже")
    session_id = uuid.uuid4().hex
    _remember(session_id, Session(state=state, history=[{"role": "assistant", "content": hello}]))
    journal.start(session_id, "веб: напоминание", hello)
    return {"session_id": session_id, "greeting": hello}


@app.post("/api/reminders/{booking_id}/phone", dependencies=DESK)
async def reminder_phone(booking_id: str, selftest: bool = False) -> dict:
    """Напоминание настоящим звонком: Asterisk звонит на софтфон команды
    (selftest — в контекст самопроверки диалплана вместо софтфона)."""
    booking = _check_reminder(booking_id)
    if telephony.server is None:
        raise HTTPException(503, "Телефония не запущена")
    try:
        call_id = await telephony.call_reminder(booking_id, "Local/s@selftest" if selftest else None)
        domain.schedule.note_reminder_attempt(booking)
    except Exception:
        log.exception("Не удалось позвонить")
        raise HTTPException(503, "Не удалось позвонить — проверьте Asterisk")
    return {"call_id": call_id}


@app.get("/api/calls", dependencies=DESK)
def calls() -> dict:
    """Последние телефонные звонки с расшифровкой — для панели в интерфейсе."""
    # Без текста разговора: расшифровка — в журнале (/api/conversations), где действуют «не записывайте» и удаление
    return {"enabled": telephony.server is not None,
            "calls": [{"id": c.id, "kind": c.kind, "started": c.started, "ended": c.ended, "turns": len(c.turns),
                       "frames_in": c.frames_in, "peak_rms": c.peak_rms, "utterances": c.utterances,
                       "early_stt": c.early_stt, "barge_ins": c.barge_ins, "backchannels": c.backchannels,
                       "ttfa_ms": c.ttfa_ms}
                      for c in reversed(telephony.calls)]}


@app.get("/api/transfers", dependencies=DESK)
def transfers() -> dict:
    """Очередь переводов для регистратора: карточка тёплого перевода (регламент 7.7)."""
    return {"items": journal.transfers()}


@app.post("/api/transfers/{conv_id}/take", dependencies=DESK)
def take_transfer(conv_id: str) -> dict:
    if not journal.take(conv_id):
        raise HTTPException(404, "Перевод не найден")
    return {"ok": True}


@app.get("/api/conversations", dependencies=DESK)
def conversations() -> dict:
    return {"items": journal.recent()}


@app.delete("/api/conversations/{conv_id}", dependencies=DESK)
def delete_conversation(conv_id: str) -> dict:
    """152-ФЗ: удаление разговора по запросу пациента — из памяти и с диска."""
    if not journal.delete(conv_id):
        raise HTTPException(404, "Разговор не найден")
    telephony.calls[:] = [c for c in telephony.calls if c.id != conv_id]  # и расшифровка звонка в памяти телефонии
    return {"ok": True}


@app.get("/api/conversations/{conv_id}", dependencies=DESK)
def conversation(conv_id: str) -> dict:
    conv = journal.get(conv_id)
    if conv is None:
        raise HTTPException(404, "Разговор не найден")
    return conv


@app.get("/api/dashboard")
def dashboard() -> dict:
    """Эффект на данных кейса, качество модулей и живые метрики пилота."""
    def read(path: str):
        p = Path(path)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    report = read("eval/report.json") or []
    return {
        "analytics": read(_result_path("analytics.json")),
        "noshow_model": noshow.metrics if noshow else None,
        "student": domain.student.metrics if domain.student else None,
        "cascade": read(_result_path("cascade_eval.json")),
        "phrasebook": {"templates": len(phrasebook.TABLE), "phrases": phrasebook.covered_phrases()},
        "stt_bench": {k: {m: v[m] for m in ("wer", "latency_ms_median") if m in v}
                      for k, v in (read(_result_path("stt_bench.json")) or {}).items() if isinstance(v, dict)},
        "eval": {"total": len(report), "ok": sum(r["ok"] for r in report),
                 "scenarios": [{"id": r["id"], "ok": r["ok"], "turns": r["turns"]} for r in report]} if report else None,
        "live": journal.stats() | {"phone_calls": len(telephony.calls)},
    }


@app.get("/api/clinic", dependencies=DESK)
def clinic() -> dict:
    schedule = domain.schedule
    for b in [b for b in schedule.bookings.values() if not b.patient_name][:3]:  # лимит стенда — 10 запросов в минуту
        try:
            if patient := schedule.patient_of(b):
                b.patient_name = patient.full_name
        except Exception:
            break
    return {
        "today": schedule.today.isoformat(),
        "doctors": [{"name": d.name, "specialty": d.specialty, "branch": d.branch, "schedule": d.describe()}
                    for d in schedule.doctors.values()],
        "bookings": [{"id": b.id, "patient": b.patient_name, "doctor": b.slot.doctor_name,
                      "specialty": b.slot.specialty, "branch": b.slot.branch,
                      "start": b.slot.start.isoformat(timespec="minutes")}
                     for b in sorted(schedule.bookings.values(), key=lambda b: b.slot.start)],
    }


@app.get("/api/visits", dependencies=DESK)
def visits(refresh: bool = False) -> dict:
    """Журнал всех записей: со стенда МИС (или из демо-реестра), новые сверху."""
    items = sorted(domain.schedule.visit_log(refresh=refresh), key=lambda v: v["at"], reverse=True)
    return {"items": items, "counts": dict(Counter(v["status"] for v in items)),
            "source": "стенд МИС «Регистратура»" if MIS_TOKEN else "демо-реестр"}


@app.post("/api/reset", dependencies=DESK)
def reset_demo() -> dict:
    """Возвращает расписание и записи к исходному состоянию демо. Со стендом МИС — запрещено: там настоящие записи."""
    global agent, domain
    if MIS_TOKEN:
        raise HTTPException(403, "Сброс доступен только в демо без стенда МИС")
    services.reset_demo()  # и веб, и телефония — на новом реестре
    agent, domain = services.agent, services.domain
    sessions.clear()
    return {"ok": True}
