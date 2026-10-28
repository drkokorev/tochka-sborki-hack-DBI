"""Машиночитаемый итог реплики (схема clinic.turn/v1): что понято и насколько уверенно, что сделано в МИС,
чего ассистент ждёт дальше и как пациенту поправить ошибку. Отдаётся в /api/message рядом с текстом ответа —
для рабочего места регистратора, интеграций и разбора качества.

Пример:
    {"schema": "clinic.turn/v1", "intent": "book", "status": "awaiting",
     "understanding": {"source": "student", "confidence": 0.93, "level": "high",
                       "fields": {"specialty": "терапевт", "day": "2026-10-21"}},
     "awaiting": "choice",
     "offered": [{"n": 1, "doctor": "Белова Анна Сергеевна", "specialty": "терапевт", "at": "2026-10-21T13:00"}],
     "action": null, "hint": "Можно назвать номер варианта, день или время."}
"""

from __future__ import annotations

from typing import Any

from .models import ClinicState, ClinicUpdate
from .schedule import Booking, Slot

SCHEMA = "clinic.turn/v1"
HIGH, MEDIUM = 0.85, 0.6

EXAMPLE_HINT = "Скажите коротко, что нужно, — например: «запишите к терапевту на пятницу» или «перенесите мою запись»."
FIX_HINT = "Если я поняла неверно, скажите «нет» и назовите правильно — например, «нет, к лору» или «нет, в пятницу»."
HINTS = {
    "choice": "Можно назвать номер варианта, день или время.",
    "identity": "Назовите фамилию, имя и дату рождения — например, «Иванов Иван, двенадцатое мая шестидесятого года».",
    "confirm": "Скажите «да», чтобы подтвердить, или «нет», чтобы выбрать другое время.",
    "yes_no": "Ответьте «да» или «нет».",
    "rating": "Назовите оценку от одного до пяти.",
}


def level(confidence: float | None) -> str | None:
    if confidence is None:
        return None
    return "high" if confidence >= HIGH else "medium" if confidence >= MEDIUM else "low"


def _slot(s: Slot) -> dict:
    return {"doctor": s.doctor_name, "specialty": s.specialty, "at": s.start.isoformat(timespec="minutes"),
            "cabinet": s.cabinet or None, "branch": s.branch or None}


def _booking(b: Booking) -> dict:
    return {"id": b.id, **_slot(b.slot)}


def _awaiting(st: ClinicState, reply: str) -> str | None:
    if st.finished and st.question is None:
        return None
    if st.question == "rating":
        return "rating"
    if st.awaiting_confirm:
        return "confirm"
    if st.question:
        return "yes_no"
    if st.offered and not st.chosen or st.candidates and not st.target:
        return "choice"
    if "фамилию" in reply.lower() or "дату рождения" in reply.lower():
        return "identity"
    return None


def _action(st: ClinicState, payload: Any) -> dict | None:
    # Перевод — только на ходу, где он оформлен (в ответе шага номер заявки): «жду» после него — не новый перевод
    if st.escalation and st.finished and isinstance(payload, dict) and "escalation" in payload:
        return {"type": "transfer", "reason": st.escalation, "escalation_id": payload.get("escalation") or None}
    if st.result is not None and st.finished:
        kind = "booking_moved" if st.intent in ("reschedule", "remind") else "booking_created"
        return {"type": kind, "booking": _booking(st.result)}
    if st.finished and st.intent == "cancel" and st.target is not None and st.confirmed:
        return {"type": "booking_cancelled", "booking": _booking(st.target)}
    if st.finished and st.intent == "status" and isinstance(payload, list):
        return {"type": "status_reported", "bookings": [_booking(b) for b in payload if isinstance(b, Booking)]}
    return None


def turn_result(st: ClinicState, update: ClinicUpdate | None, reply: str, payload: Any, *, source: str,
                confidence: float | None) -> dict:
    awaiting = _awaiting(st, reply)
    fields = update.model_dump(mode="json", exclude_none=True, exclude={"note"}) if update is not None else {}
    meaningful = {k for k in fields if k not in ("smalltalk", "bye")}
    if st.misunderstood or (confidence is not None and not meaningful and not st.finished):
        # ассистент переспросил или в реплике нет смысла для диалога («ммм», «...») — уверенность низкая
        confidence = min(confidence or 0.0, 0.3)
    lvl = level(confidence)
    action = _action(st, payload)
    # done — действие в МИС выполнено на этом ходу (дальше лишь «Могу ещё помочь?»); transferred — передан человеку
    status = ("transferred" if st.escalation and st.finished else "done" if action is not None
              else "ended" if st.finished and awaiting is None else "awaiting" if awaiting else "in_progress")
    if status in ("ended", "transferred", "done"):
        hint = None                                   # разговор окончен или передан — подсказывать нечего
    elif awaiting == "identity":
        hint = HINTS["identity"]                      # про ФИО — всегда конкретно, как назвать себя
    elif lvl in ("medium", "low") and meaningful & {"intent", "specialty", "doctor_name", "day", "time_from", "week_of"}:
        hint = FIX_HINT                               # поняли запрос, но не уверены — как поправить
    elif lvl == "low" and not meaningful:
        hint = HINTS.get(awaiting or "") or EXAMPLE_HINT  # ничего не поняли — пример реплики
    else:
        hint = HINTS.get(awaiting or "")
    return {
        "schema": SCHEMA,
        "intent": st.intent,
        "status": status,
        "understanding": {"source": source, "confidence": None if confidence is None else round(confidence, 3),
                          "level": lvl, "fields": fields},
        "awaiting": awaiting,
        "offered": [{"n": i, **_slot(s)} for i, s in enumerate(st.offered, 1)] if awaiting == "choice" else [],
        "action": action,
        "hint": hint,
    }
