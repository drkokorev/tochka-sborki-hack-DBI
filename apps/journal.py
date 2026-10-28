"""Журнал разговоров: веб и телефон в одном месте.

Из журнала строятся очередь переводов для регистратора (карточка тёплого перевода по регламенту 7.7)
и живая аналитика пилота: исходы, цели обращений, кто понял фразу, задержки по этапам.
Хранится в памяти процесса и, если задан путь, в JSON-файле на машине — чтобы журнал и аналитика переживали
перезапуск сервера. 152-ФЗ: только то, что пациент назвал сам; файл доступен лишь владельцу процесса (0600);
срок хранения ограничен (JOURNAL_DAYS, по умолчанию 30 дней); по просьбе «не записывайте» текст разговора
скрывается и на диск не попадает; разговор можно удалить по запросу пациента. Аудио не сохраняется.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from core.ru import MONTHS_GEN

from core.resilience import breaker
from domains.clinic.replies import ROUTES, transfer_header

log = logging.getLogger(__name__)
PRIVATE = "[скрыто по просьбе пациента]"

# Приоритеты очереди и правила маршрутизации — из той же матрицы регламента, что и заявка в МИС (domains/clinic/replies.py)
PRIORITY = {key: route.priority for key, route in ROUTES.items()}
REASON = {"emergency": "Угроза жизни", "selfharm": "Угроза себе", "urgent": "Острое состояние",
          "dental": "Острая зубная боль", "complaint": "Жалоба", "medical": "Медицинский вопрос",
          "human": "Просит сотрудника", "paid": "Платное отделение", "admin": "Администрация",
          "misunderstood": "Ассистент не понял", "not_found": "Не найден в МИС", "mis_error": "Сбой МИС",
          "waitlist": "Лист ожидания", "pdn": "Персональные данные", "lab_delay": "Задержка анализов"}
INTENT = {"book": "Запись", "schedule": "Расписание", "status": "Статус записи", "reschedule": "Перенос",
          "cancel": "Отмена", "remind": "Напоминание", "escalate": "Перевод", "home_visit": "Вызов врача на дом"}
OUTCOME = {"booked": "Записан", "rescheduled": "Перенесено", "cancelled": "Отменено", "status": "Статус сообщён",
           "info": "Справка дана", "confirmed": "Визит подтверждён", "escalated": "Передан сотруднику",
           "callback": "Обратный звонок", "home_visit": "Вызов на дом принят", "opt_out": "Отказ от обзвона", "not_confirmed": "Визит не подтверждён",
           "goodbye": "Завершён", "open": "Не завершён"}
# Итог исходящего напоминания → итог разговора (подтверждением считаем только CONFIRMED)
REMINDER_OUTCOME = {"CONFIRMED": "confirmed", "OPT_OUT": "opt_out", "OPT_OUT_ALL": "opt_out",
                    "ESCALATED": "escalated", "RESCHEDULED": "rescheduled", "CANCELLED": "cancelled"}


def understood(state: Any, update: Any = None) -> str:
    """«запись · терапевт · 20 октября · после 13:00 · Иванов Иван» — как ассистент понял реплику."""
    parts: list[str] = []
    if update is not None and getattr(update, "red_flag", None):
        parts.append(f"перевод: {REASON.get(update.red_flag, update.red_flag).lower()}")
    elif update is not None and getattr(update, "topic", None):
        parts.append("справка")
    elif getattr(state, "intent", None):
        parts.append(INTENT.get(state.intent, state.intent).lower())
    if getattr(state, "doctor_name", None):
        parts.append(state.doctor_name.split()[0])
    elif getattr(state, "specialty", None):
        parts.append(state.specialty)
    if getattr(state, "day", None):
        parts.append(f"{state.day.day} {MONTHS_GEN[state.day.month - 1]}")
    elif getattr(state, "week_of", None):
        parts.append(f"неделя с {state.week_of.day} {MONTHS_GEN[state.week_of.month - 1]}")
    tf, tt = getattr(state, "time_from", None), getattr(state, "time_to", None)
    if tf and tt:
        parts.append(f"{tf:%H:%M}–{tt:%H:%M}")
    elif tf:
        parts.append(f"после {tf:%H:%M}")
    if getattr(state, "chosen", None):
        parts.append(f"выбран {state.chosen.start:%d.%m %H:%M}")
    if update is not None and getattr(update, "confirm", None) is not None:
        parts.append("да" if update.confirm else "нет")
    patient = getattr(state, "patient", None)
    if patient is not None:
        parts.append(patient.full_name)
    elif getattr(state, "last_name", None):
        parts.append(f"{state.last_name} {state.first_name or ''}".strip())
    return " · ".join(parts)


def _talked(conv: dict) -> bool:
    """Был ли в разговоре хоть один ответ пациента: открытая и брошенная вкладка — не разговор."""
    return any(t["role"] == "patient" for t in conv["turns"])


class Journal:
    def __init__(self, keep: int = 200, path: str | Path | None = None, days: float | None = None):
        self.keep = keep
        self.max_age = (days if days is not None else float(os.getenv("JOURNAL_DAYS", "30"))) * 86400
        self.path = Path(path) if path else None
        self.conversations: OrderedDict[str, dict] = OrderedDict()
        self.taken: set[str] = set()
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            fresh = time.time() - self.max_age
            self.conversations = OrderedDict((c["id"], c) for c in data.get("conversations", []) if c["started"] >= fresh)
            self.taken = set(data.get("taken", []))
        except (OSError, ValueError, KeyError):
            log.exception("Журнал разговоров не прочитан — начинаем с пустого")

    def _save(self) -> None:
        """Под блокировкой: снимок журнала атомарно (через временный файл)."""
        if self.path is None or not breaker("journal").allow():
            return  # диск сбоит — журнал пока в памяти, повторим, когда предохранитель остынет
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fresh = time.time() - self.max_age
            convs = [c for c in self.conversations.values() if _talked(c) and not c.get("private") and c["started"] >= fresh]
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # только владелец процесса
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"conversations": convs, "taken": sorted(self.taken)}, f, ensure_ascii=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
            breaker("journal").ok()
        except OSError as exc:
            breaker("journal").fail(exc)  # разговор продолжается, журнал — в памяти
            log.warning("Журнал разговоров не сохранён: %s", type(exc).__name__)

    def start(self, conv_id: str, channel: str, greeting: str) -> None:
        with self._lock:
            self.conversations[conv_id] = {
                "id": conv_id, "channel": channel, "started": time.time(), "ended": None, "outcome": "open",
                "turns": [{"role": "assistant", "text": greeting, "at": time.time()}], "transfer": None,
                "intents": [],
            }
            # Сначала вытесняем пустые разговоры (открыли вкладку и ушли), потом самые старые
            while len(self.conversations) > self.keep:
                empty = next((k for k, c in self.conversations.items() if not _talked(c) and k != conv_id), None)
                self.conversations.pop(empty) if empty else self.conversations.popitem(last=False)

    def turn(self, conv_id: str, text: str, step: Any, state: Any, update: Any, nlu: str, nlu_ms: float,
             stt_ms: float | None = None, stt_engine: str | None = None) -> None:
        with self._lock:
            conv = self.conversations.get(conv_id)
            if conv is None:
                return
            now = time.time()
            if update is not None and getattr(update, "topic", None) == "no_recording":
                # Пациент против записи: текст разговора скрываем, на диск он не попадёт
                conv["private"] = True
                for t in conv["turns"]:
                    t["text"] = PRIVATE
            if conv.get("private"):
                text = PRIVATE
            conv["turns"].append({"role": "patient", "text": text, "at": now, "stt_ms": stt_ms, "stt": stt_engine})
            private = conv.get("private")
            conv["turns"].append({"role": "assistant", "text": PRIVATE if private else step.reply, "at": now,
                                  "nlu": nlu, "nlu_ms": round(nlu_ms),
                                  "understood": "" if private else understood(state, update)})
            label = self._label(state, update)
            if label and (not conv["intents"] or conv["intents"][-1] != label):
                conv["intents"].append(label)
            if update is not None and getattr(update, "topic", None) and conv["outcome"] == "open":
                conv["outcome"] = "info"
            if getattr(state, "rating", None):
                conv["rating"] = state.rating
            if step.done:
                conv["outcome"] = self._outcome(state, conv["outcome"])
                conv["ended"] = now
                if conv["outcome"] in ("escalated", "callback"):
                    conv["transfer"] = self._card(conv_id, state, step)  # и задача на обратный звонок — с карточкой
                    conv.pop("pending_transfer", None)
            elif getattr(state, "escalation", None) and getattr(state, "intent", None) == "escalate":
                # «Соединяю…» уже прозвучало, ассистент спрашивает ФИО: если пациент положит трубку или замолчит,
                # карточка всё равно должна быть у регистратора — заготовка становится карточкой в end()
                conv["pending_transfer"] = self._card(conv_id, state, step)
            self._save()

    def voice(self, conv_id: str, *, ttfa_ms: float | None = None, barge: bool = False,
              backchannel: bool = False) -> None:
        """Голосовые показатели разговора: время до первого звука ответа (к последнему ответу ассистента),
        перебивания и поддакивания, на которые ассистент не замолчал."""
        with self._lock:
            conv = self.conversations.get(conv_id)
            if conv is None:
                return
            if ttfa_ms is not None:
                last = next((t for t in reversed(conv["turns"]) if t["role"] == "assistant" and "nlu" in t), None)
                if last is not None and "ttfa_ms" not in last:
                    last["ttfa_ms"] = round(ttfa_ms)
            conv["barge_ins"] = conv.get("barge_ins", 0) + int(barge)
            conv["backchannels"] = conv.get("backchannels", 0) + int(backchannel)
            self._save()

    def end(self, conv_id: str) -> None:
        with self._lock:
            if conv := self.conversations.get(conv_id):
                conv["ended"] = conv["ended"] or time.time()
                if conv.get("pending_transfer") and not conv.get("transfer"):
                    card = conv.pop("pending_transfer")
                    conv["transfer"] = card
                    conv["outcome"] = "callback" if card.get("result_code") == "CALLBACK" else "escalated"
                self._save()

    @staticmethod
    def _label(state: Any, update: Any) -> str | None:
        if update is not None and getattr(update, "red_flag", None):
            return "Перевод"
        if update is not None and getattr(update, "topic", None):
            return "Справка"
        return INTENT.get(getattr(state, "intent", None) or "", None)

    @staticmethod
    def _outcome(state: Any, current: str) -> str:
        if getattr(state, "escalation", None):
            # заявка на обратный звонок (задержка анализов, лист ожидания) — не соединение с сотрудником
            route = ROUTES.get(state.escalation)
            return "callback" if route is not None and route.outcome == "CALLBACK" else "escalated"
        if getattr(state, "goodbye", False) or (current == "open" and getattr(state, "question", None) is None
                                                 and not getattr(state, "intent", None)):
            return current if current != "open" else "goodbye"
        intent = getattr(state, "intent", None)
        if intent == "remind":
            code = getattr(state, "reminder_code", None)
            if code is None:
                return current if current != "open" else "open"
            return REMINDER_OUTCOME.get(code, "not_confirmed")
        if getattr(state, "result", None) is not None:
            return "rescheduled" if intent == "reschedule" else "booked"
        if intent == "cancel":
            return "cancelled" if getattr(state, "confirmed", None) else "info"
        if intent == "status":
            return "status"
        if intent == "home_visit" and getattr(state, "finished", False):
            return "home_visit"
        return current if current != "open" else "info"

    @staticmethod
    def _card(conv_id: str, state: Any, step: Any) -> dict:
        reason = state.escalation or "human"
        done: list[str] = []
        if reason in ("emergency", "selfharm"):
            done.append("Рекомендовано звонить 103 / 112")
        if reason == "urgent":
            done.append("Сказано звонить 103 при ухудшении")
        if state.patient is not None and state.patient.id:
            done.append(f"Пациент найден в МИС: {state.patient.full_name}, {state.patient.birth_date:%d.%m.%Y}")
        elif state.patient is not None:
            done.append(f"Пациент назвался (в МИС нет): {state.patient.full_name}, {state.patient.birth_date:%d.%m.%Y}")
        elif state.last_name:
            done.append(f"Пациент назвался: {state.last_name} {state.first_name or ''}".strip())
        if isinstance(step.payload, dict) and step.payload.get("escalation"):
            done.append(f"Заявка в МИС № {step.payload['escalation']}")
        note = state.escalation_note or REASON.get(reason, reason)
        patient = state.patient
        # Карточка для рабочего места регистратора и интеграций (схема clinic.transfer/v1): текстом — для человека,
        # полями — для программ (CRM, очередь колл-центра)
        route = ROUTES.get(reason)
        return {"schema": "clinic.transfer/v1", "id": conv_id, "priority": PRIORITY.get(reason, "P3"),
                "rule": route.rule if route else "", "header": transfer_header(reason),
                "result_code": route.outcome if route else "ESCALATED",
                "reason": REASON.get(reason, reason), "reason_code": reason,
                "summary": note.rsplit(": ", 1)[-1], "done": done, "at": time.time(),
                "patient": ({"mis_id": patient.id or None, "last_name": patient.last_name, "first_name": patient.first_name,
                             "birth_date": patient.birth_date.isoformat(), "found_in_mis": bool(patient.id)}
                            if patient is not None
                            else {"last_name": state.last_name, "first_name": state.first_name,
                                  "birth_date": state.birth_date.isoformat() if state.birth_date else None,
                                  "found_in_mis": False} if state.last_name else None),
                "escalation_id": (step.payload or {}).get("escalation") if isinstance(step.payload, dict) else None,
                "emergency_advised": reason in ("emergency", "selfharm", "urgent")}

    # ---------- для интерфейса ----------

    def transfers(self) -> list[dict]:
        with self._lock:
            cards = []
            for conv in self.conversations.values():
                if card := conv["transfer"]:
                    cards.append(card | {"channel": conv["channel"], "taken": conv["id"] in self.taken,
                                         "dialog": [{"role": t["role"], "text": t["text"]} for t in conv["turns"][-6:]]})
        order = {"P1": 0, "P2": 1, "P3": 2}
        return sorted(cards, key=lambda c: (c["taken"], order[c["priority"]], -c["at"]))

    def delete(self, conv_id: str) -> bool:
        """Удаление по запросу пациента (право на удаление персональных данных)."""
        with self._lock:
            if self.conversations.pop(conv_id, None) is None:
                return False
            self.taken.discard(conv_id)
            self._save()
            return True

    def take(self, conv_id: str) -> bool:
        with self._lock:
            if conv_id in self.conversations and self.conversations[conv_id]["transfer"]:
                self.taken.add(conv_id)
                self._save()
                return True
        return False

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            convs = [c for c in self.conversations.values() if _talked(c)][-limit:]
        return [{"id": c["id"], "channel": c["channel"], "started": c["started"], "ended": c["ended"],
                 "outcome": c["outcome"], "outcome_label": OUTCOME.get(c["outcome"], c["outcome"]),
                 "intents": c["intents"], "turns": sum(t["role"] == "patient" for t in c["turns"])}
                for c in reversed(convs)]

    def get(self, conv_id: str) -> dict | None:
        with self._lock:
            conv = self.conversations.get(conv_id)
            return None if conv is None else {**conv, "turns": list(conv["turns"])}

    def stats(self) -> dict:
        with self._lock:
            convs = [c for c in self.conversations.values() if _talked(c)]
        turns = [t for c in convs for t in c["turns"] if t["role"] == "assistant" and "nlu" in t]
        nlu = Counter(t["nlu"] for t in turns)
        by_nlu = {k: round(statistics.median([t["nlu_ms"] for t in turns if t["nlu"] == k])) for k in nlu}
        lat = sorted(t["nlu_ms"] + (p.get("stt_ms") or 0) for c in convs
                     for p, t in zip(c["turns"][1::2], c["turns"][2::2]) if "nlu_ms" in t)
        ttfa = sorted(t["ttfa_ms"] for t in turns if t.get("ttfa_ms") is not None)
        voiced = [c for c in convs if any(t.get("ttfa_ms") is not None for t in c["turns"])]
        pick = lambda q: ttfa[min(len(ttfa) - 1, int(len(ttfa) * q))] if ttfa else None  # noqa: E731
        finished = [c for c in convs if c["outcome"] != "open"]
        resolved = [c for c in finished if c["outcome"] not in ("escalated", "callback", "goodbye")]
        reasons = Counter(c["transfer"]["reason"] for c in convs if c["transfer"])
        return {
            "conversations": len(convs),
            "finished": len(finished),
            "resolved_share": round(len(resolved) / len(finished), 3) if finished else None,
            "escalated_share": round(sum(c["outcome"] == "escalated" for c in finished) / len(finished), 3)
            if finished else None,
            "channels": dict(Counter(c["channel"] for c in convs)),
            "outcomes": {OUTCOME[k]: v for k, v in Counter(c["outcome"] for c in convs).items()},
            "intents": dict(Counter(i for c in convs for i in c["intents"]).most_common(8)),
            "nlu": dict(nlu),
            "nlu_ms_median": by_nlu,
            "latency_p50": lat[len(lat) // 2] if lat else None,
            "latency_p95": lat[int(len(lat) * 0.95) - 1 if len(lat) > 1 else 0] if lat else None,
            # От конца фразы пациента до первого звука ответа — по живым голосовым разговорам (телефон и веб)
            "ttfa_p50": round(statistics.median(ttfa)) if ttfa else None, "ttfa_p90": pick(0.9), "ttfa_p95": pick(0.95), "ttfa_count": len(ttfa),
            "ttfa_over_2s": round(sum(x > 2000 for x in ttfa) / len(ttfa), 3) if ttfa else None,
            "barge_ins_per_call": round(sum(c.get("barge_ins", 0) for c in voiced) / len(voiced), 2) if voiced else None,
            "backchannels": sum(c.get("backchannels", 0) for c in convs),
            "reasons": dict(reasons),
            "csat": round(statistics.mean(r), 2) if (r := [c["rating"] for c in convs if c.get("rating")]) else None,
            "csat_count": sum(1 for c in convs if c.get("rating")),
            "turns_avg": round(statistics.mean(sum(t["role"] == "patient" for t in c["turns"]) for c in finished), 1)
            if finished else None,
        }
