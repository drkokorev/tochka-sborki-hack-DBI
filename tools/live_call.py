"""Живой звонок без подделок: подключаемся к работающему AudioSocket-серверу ассистента так же, как Asterisk,
и говорим синтезированной речью в реальном времени (8 кГц, кадры по 20 мс).

    make call            # или .venv/bin/python tools/live_call.py  (сервер с TELEPHONY=1 должен работать)
    make call-book       # --book: запись голосом до визита в МИС (вымышленный пациент), визит сверяется по API

Проверяется вся цепочка на настоящих компонентах: детектор речи (Silero VAD), распознавание GigaAM, каскад
понимания, синтез Silero, поддакивание поверх ответа («угу» не должно прерывать ассистента), время до первого
звука ответа. Без флага записи в МИС не создаются: справка о режиме работы и прощание.
С --book — основной сценарий кейса голосом: «запишите к терапевту» → выбор → ФИО и дата рождения → «да»;
пациент слушает, как ассистент повторил его данные, и поправляет («нет, Смирнова Ольга…»), если распознавание
ошиблось, — как живой человек. Созданный визит сверяется запросом к рабочему месту (/api/visits — реестр МИС):
визит есть и записан на названного пациента. Отчёт eval/results/live_call_book.json.
Отчёт помечен коммитом; код выхода 1 — проверка не прошла.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import os
import struct
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.telephony import wav_to_pcm8k  # noqa: E402
from apps.voice import TextToSpeech  # noqa: E402
from core.version import commit  # noqa: E402

SIL = b"\0" * 320
HOST, PORT = "127.0.0.1", int(os.getenv("AUDIOSOCKET_PORT", "9092"))
API = os.getenv("ASSISTANT_URL", "http://localhost:8000")
PATIENT = "Смирнова Ольга, двенадцатое марта тысяча девятьсот семьдесят восьмого года"  # вымышленный пациент стенда
FIX = "Нет, Смирнова Ольга, двенадцатое марта тысяча девятьсот семьдесят восьмого года"


def get(path: str) -> dict:
    """Запрос к рабочему месту регистратора (с токеном, если он задан на сервере)."""
    req = urllib.request.Request(f"{API}{path}", headers={"X-Desk-Token": os.getenv("DESK_TOKEN", "")})
    return json.load(urllib.request.urlopen(req, timeout=15))


async def call() -> dict:
    tts = TextToSpeech()
    say = lambda t: wav_to_pcm8k(tts.synthesize(t))  # noqa: E731
    question, uh, bye = say("Здравствуйте, до скольки вы работаете в субботу?"), say("угу"), say("спасибо, до свидания")
    reader, writer = await asyncio.open_connection(HOST, PORT)
    writer.write(struct.pack(">BH", 1, 16) + uuid.uuid4().bytes)
    got: list[tuple[float, int]] = []  # (время, уровень) кадров ответа ассистента

    async def listen():
        try:
            while True:
                kind, size = struct.unpack(">BH", await reader.readexactly(3))
                payload = await reader.readexactly(size) if size else b""
                if kind == 0x10:
                    got.append((time.monotonic(), audioop.rms(payload, 2)))
                elif kind == 0x00:
                    return
        except (asyncio.IncompleteReadError, ConnectionError):
            return

    listener = asyncio.create_task(listen())

    async def send(pcm: bytes):
        start = time.monotonic()
        for i, off in enumerate(range(0, len(pcm), 320)):
            if listener.done():
                return  # ассистент положил трубку (после прощания) — дальше не говорим
            try:
                writer.write(struct.pack(">BH", 0x10, 320) + pcm[off:off + 320].ljust(320, b"\0"))
                await writer.drain()
            except ConnectionError:
                return
            await asyncio.sleep(max(0.0, start + (i + 1) * 0.02 - time.monotonic()))

    async def until_quiet(limit: float = 25):
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit and not listener.done():
            await send(SIL * 5)
            if got and time.monotonic() - got[-1][0] > 0.6:
                return

    await until_quiet()                                  # приветствие
    await send(question)
    asked = time.monotonic()
    while not any(t > asked for t, _ in got):
        await send(SIL * 5)
    first = next(t for t, _ in got if t > asked)
    await send(SIL * 40)                                 # ответ звучит 0,8 с — пациент поддакивает
    t_uh = time.monotonic()
    await send(uh)
    await send(SIL * 100)
    resumed = [lvl for t, lvl in got if t_uh + 1.2 < t < t_uh + 2.0]
    await until_quiet()
    await send(bye)
    said_bye = time.monotonic()
    while time.monotonic() - said_bye < 10 and not any(t > said_bye for t, _ in got) and not listener.done():
        await send(SIL * 5)                              # ждём ответ на прощание, а не кладём трубку сразу
    await until_quiet(10)
    hung_up_by_assistant = listener.done()
    if not hung_up_by_assistant:
        try:
            writer.write(struct.pack(">BH", 0, 0))
            await writer.drain()
        except ConnectionError:
            pass
    await asyncio.sleep(0.5)
    listener.cancel()

    convs = get("/api/conversations")["items"]
    conv_id = next(c["id"] for c in convs if str(c["channel"]).startswith("телефон"))
    conv = get(f"/api/conversations/{conv_id}")
    turns = [{"role": t["role"], "text": t["text"], "ttfa_ms": t.get("ttfa_ms"), "nlu": t.get("nlu")}
             for t in conv["turns"]]
    answered = any(t["role"] == "assistant" and "суббот" in t["text"] for t in turns)
    report = {
        "at": time.strftime("%Y-%m-%d %H:%M"), "commit": commit(),
        "ttfa_ms_server": next((t["ttfa_ms"] for t in turns if t.get("ttfa_ms")), None),
        "ttfa_ms_client_from_audio_end": round(1000 * (first - asked)),
        "backchannels": conv.get("backchannels", 0), "barge_ins": conv.get("barge_ins", 0),
        "reply_resumed_after_uh": bool(resumed),
        "hung_up_by_assistant": hung_up_by_assistant,
        "checks": {
            "вопрос распознан и получен ответ по регламенту": answered,
            "«угу» поверх ответа не прервало ассистента": conv.get("backchannels", 0) >= 1 and conv.get("barge_ins", 0) == 0,
            "время до первого звука ≤ 1,5 с": bool(turns) and (next((t["ttfa_ms"] for t in turns if t.get("ttfa_ms")), 9e9) <= 1500),
            "прощание завершило звонок": any(t["role"] == "assistant" and "Всего доброго" in t["text"] for t in turns),
        },
        "turns": turns,
    }
    return report


async def book() -> dict:
    """Запись голосом до визита в МИС: каждая фраза — после того, как ассистент договорил."""
    tts = TextToSpeech()
    say = lambda t: wav_to_pcm8k(tts.synthesize(t))  # noqa: E731
    phrases = ["Здравствуйте, запишите меня к терапевту", "давайте первый вариант", PATIENT]
    audio = [say(t) for t in phrases]
    fix, yes = say(FIX), say("да, всё верно")
    reader, writer = await asyncio.open_connection(HOST, PORT)
    writer.write(struct.pack(">BH", 1, 16) + uuid.uuid4().bytes)
    got: list[float] = []

    async def listen():
        try:
            while True:
                kind, size = struct.unpack(">BH", await reader.readexactly(3))
                if size:
                    await reader.readexactly(size)
                if kind == 0x10:
                    got.append(time.monotonic())
                elif kind == 0x00:
                    return
        except (asyncio.IncompleteReadError, ConnectionError):
            return

    listener = asyncio.create_task(listen())

    async def send(pcm: bytes):
        start = time.monotonic()
        for i, off in enumerate(range(0, len(pcm), 320)):
            writer.write(struct.pack(">BH", 0x10, 320) + pcm[off:off + 320].ljust(320, b"\0"))
            await writer.drain()
            await asyncio.sleep(max(0.0, start + (i + 1) * 0.02 - time.monotonic()))

    async def until_quiet(limit: float = 30):
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit:
            await send(SIL * 5)
            if got and time.monotonic() - got[-1] > 0.8:
                return

    async def await_reply(limit: float = 25):
        """Ждём, пока ассистент начнёт отвечать (новый звук после нашей фразы), и пока он договорит."""
        asked = time.monotonic()
        while time.monotonic() - asked < limit and not (got and got[-1] > asked):
            await send(SIL * 5)
        await until_quiet()

    def last_reply() -> str:
        conv = next(c["id"] for c in get("/api/conversations")["items"] if str(c["channel"]).startswith("телефон"))
        return next((t["text"] for t in reversed(get(f"/api/conversations/{conv}")["turns"])
                     if t["role"] == "assistant"), "")

    def heard_right(reply: str) -> bool:
        # пациент найден в МИС («Ольга, записываю вас») или ассистент верно повторил ФИО и год
        return reply.startswith("Ольга, записываю") or ("Смирнова Ольга" in reply and "семьдесят восьмого" in reply)

    await until_quiet()                                  # приветствие
    for pcm in audio:
        await send(pcm)
        await await_reply()
    corrections = 0
    while not heard_right(last_reply()) and corrections < 2:  # распознавание ошиблось — поправляем
        corrections += 1
        await send(fix)
        await await_reply()
    await send(yes)
    await await_reply()
    writer.write(struct.pack(">BH", 0, 0))
    await writer.drain()
    await asyncio.sleep(0.5)
    listener.cancel()

    convs = get("/api/conversations")["items"]
    conv_id = next(c["id"] for c in convs if str(c["channel"]).startswith("телефон"))
    turns = [{"role": t["role"], "text": t["text"], "nlu": t.get("nlu"), "ttfa_ms": t.get("ttfa_ms")}
             for t in get(f"/api/conversations/{conv_id}")["turns"]]
    done = next((t["text"] for t in turns if t["role"] == "assistant" and "Номер записи" in t["text"]), "")
    number = done.split("Номер записи", 1)[1].strip(" .").split(".")[0].strip() if done else ""
    visit = next((v for v in get("/api/visits?refresh=1")["items"] if str(v["id"]) == number), None) if number else None
    return {
        "at": time.strftime("%Y-%m-%d %H:%M"), "commit": commit(), "visit_id": number or None,
        "corrections": corrections,  # сколько раз пациент поправлял ФИО после повтора
        "visit_in_mis": visit and {k: visit.get(k) for k in ("id", "patient", "doctor", "specialty", "at", "status")},
        "checks": {
            "речь распознана, запись подтверждена голосом": bool(done),
            "визит есть в реестре МИС со статусом «Запланирована»": bool(visit) and visit.get("status") == "Запланирована",
            "визит записан на названного пациента": bool(visit) and visit.get("patient") == "Смирнова Ольга",
        },
        "turns": turns,
    }


def main() -> int:
    if "--book" in sys.argv:
        report = asyncio.run(book())
        out = Path("eval/results/live_call_book.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        for name, ok in report["checks"].items():
            print(f"{'✅' if ok else '❌'} {name}")
        print(f"Визит {report['visit_id']}: {report['visit_in_mis']}. Отчёт: {out}")
        return 0 if all(report["checks"].values()) else 1
    report = asyncio.run(call())
    out = Path("eval/results/live_call.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, ok in report["checks"].items():
        print(f"{'✅' if ok else '❌'} {name}")
    print(f"До первого звука: {report['ttfa_ms_server']} мс (сервер), поддакиваний {report['backchannels']}, "
          f"перебиваний {report['barge_ins']}. Отчёт: {out}")
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
