"""Правила обзвона С-10 — кому, когда и сколько раз звонить с напоминанием (регламент С-10, уточнение DBI 20.10).

- Звоним накануне приёма (в субботу — про понедельник), только в окне по часам клиники (GMT+4):
  будни 10:00–19:00, суббота 10:00–15:00; в воскресенье и праздники не звоним.
- Не звоним, если запись создана менее 24 часов назад, если пациент отказался от обзвона и если запись уже
  подтверждена (или разговор по ней уже состоялся: перенос, отмена, перевод).
- До 2 попыток с интервалом не меньше 2 часов; не дозвонились — SMS-напоминание.
- SMS накануне получают все записанные, кроме полностью отказавшихся от уведомлений. SMS-шлюза на стенде нет:
  очередь SMS показывается на рабочем месте с пометкой «канал не подключён».

Ручной запуск с рабочего места в объявленном демо-режиме (REMINDER_DEMO=1) не проверяет окно и правило 24 часов —
заказчик разрешил это на демо, если обычный обзвон регламент соблюдает. Отказы, подтверждения и лимит попыток
проверяются всегда.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schedule import Booking, Schedule

CLINIC_TZ = timezone(timedelta(hours=4))
WINDOW = {0: (time(10), time(19)), 1: (time(10), time(19)), 2: (time(10), time(19)), 3: (time(10), time(19)),
          4: (time(10), time(19)), 5: (time(10), time(15))}
# Нерабочие праздничные дни (производственный календарь РФ) на горизонте пилота
HOLIDAYS = {date(2026, 11, 4), date(2026, 12, 31), *(date(2027, 1, d) for d in range(1, 9)), date(2027, 2, 23),
            date(2027, 3, 8), date(2027, 5, 1), date(2027, 5, 10), date(2027, 6, 14)}
MAX_ATTEMPTS = 2
RETRY_AFTER = timedelta(hours=2)
MIN_AGE = timedelta(hours=24)

ATTEMPT = "CALL_PLACED"   # попытка дозвона (звонок ушёл)
FINAL = {"CONFIRMED", "RESCHEDULED", "CANCELLED", "OPT_OUT", "OPT_OUT_ALL", "ESCALATED", "WRONG_PERSON", "UNSURE"}
OPT_OUTS = {"OPT_OUT", "OPT_OUT_ALL"}


def clinic_now() -> datetime:
    """Текущее время клиники без часового пояса — так хранятся время приёма и создания записи."""
    return datetime.now(CLINIC_TZ).replace(tzinfo=None)


def in_window(now: datetime) -> bool:
    if now.date() in HOLIDAYS or now.weekday() not in WINDOW:
        return False
    start, end = WINDOW[now.weekday()]
    return start <= now.time() < end


# Часы работы регистратуры (адресат М2, регламент 7.2): вне них острое состояние (категория Б) — как категория А
REGISTRY_HOURS = {**{d: (time(7), time(20)) for d in range(5)}, 5: (time(8, 30), time(15))}


def registry_open(now: datetime) -> bool:
    if now.date() in HOLIDAYS or now.weekday() not in REGISTRY_HOURS:
        return False
    start, end = REGISTRY_HOURS[now.weekday()]
    return start <= now.time() < end


def window_text() -> str:
    return "будни 10:00–19:00, суббота 10:00–15:00 по часам клиники, без воскресений и праздников"


def opted_out(schedule: "Schedule", patient_id: str | None, *, everything: bool = False) -> bool:
    """Отказался ли пациент от обзвона (everything — полностью от уведомлений, включая SMS)."""
    if not patient_id:
        return False
    codes = {"OPT_OUT_ALL"} if everything else OPT_OUTS
    if schedule.notify_opt_out.get(patient_id) in codes:
        return True

    def owner(e: dict) -> str | None:  # старые строки журнала — без patient_id: пациент по записи
        if e.get("patient_id"):
            return e["patient_id"]
        b = schedule.bookings.get(e.get("booking") or "")
        return b.patient_id if b else None

    return any(e.get("code") in codes and owner(e) == patient_id for e in schedule.reminder_log)


def final_code(schedule: "Schedule", booking_id: str) -> str | None:
    """Итог разговора по записи на её текущее время (подтвердил, отменил, отказался…). Перенесли на другой день —
    итог относился к прежнему времени: накануне нового приёма снова напомним."""
    booking = schedule.bookings.get(booking_id)
    start = booking.slot.start.isoformat() if booking else None
    for e in reversed(schedule.reminder_log):
        if e.get("booking") == booking_id and e.get("code") in FINAL and e.get("start", start) == start:
            return e["code"]
    restored = schedule.reminder_final.get(booking_id)  # со стенда: «Подтверждена запись #id ДД.ММ.ГГГГ ЧЧ:ММ»
    if restored and booking and restored[1] == booking.slot.start:
        return restored[0]
    return None


def attempts(schedule: "Schedule", booking_id: str) -> list[datetime]:
    return [datetime.fromisoformat(e["at"]) for e in schedule.reminder_log
            if e.get("booking") == booking_id and e.get("code") == ATTEMPT]


def skip_reason(schedule: "Schedule", booking: "Booking", now: datetime, *, demo: bool = False) -> str | None:
    """Почему по записи сейчас не звоним; None — звонить можно."""
    if opted_out(schedule, booking.patient_id):
        return "пациент отказался от обзвона"
    if code := final_code(schedule, booking.id):
        return "запись уже подтверждена" if code == "CONFIRMED" else "разговор по записи уже состоялся"
    tries = attempts(schedule, booking.id)
    if len(tries) >= MAX_ATTEMPTS:
        return "две попытки без ответа — отправляется SMS"
    if tries and now - tries[-1] < RETRY_AFTER:
        return "повторная попытка — не раньше чем через 2 часа"
    if demo:
        return None
    if not in_window(now):
        return f"вне окна обзвона ({window_text()})"
    if booking.created and now - booking.created < MIN_AGE:
        return "запись создана менее 24 часов назад"
    return None


def due(schedule: "Schedule", now: datetime, *, demo: bool = False) -> list["Booking"]:
    """Кому звонить сейчас: записи на день напоминания, прошедшие все правила."""
    if not demo and not in_window(now):
        return []
    day = schedule.reminder_day()
    return [b for b in schedule.bookings.values()
            if b.slot.start.date() == day and skip_reason(schedule, b, now, demo=demo) is None]


def sms_due(schedule: "Schedule") -> list[tuple["Booking", str]]:
    """SMS-напоминания на завтра: всем записанным, кроме полностью отказавшихся; и тем, до кого не дозвонились."""
    day = schedule.reminder_day()
    out = []
    for b in schedule.bookings.values():
        if b.slot.start.date() != day or opted_out(schedule, b.patient_id, everything=True):
            continue
        out.append((b, "не дозвонились за 2 попытки" if len(attempts(schedule, b.id)) >= MAX_ATTEMPTS
                    and not final_code(schedule, b.id) else "SMS накануне приёма"))
    return out
