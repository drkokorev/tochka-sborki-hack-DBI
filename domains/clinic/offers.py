"""Как варианты записи звучат в голосе: врач — один раз, дата без повторов, разные дни и часы."""

from __future__ import annotations

from core.ru import WEEKDAYS_ACC, day_ordinal, on_date, time_words

from .schedule import Booking, Slot

_ORDINAL_NAMES = ["Первый", "второй", "третий", "четвёртый", "пятый"]


def _short_when(slot: Slot, prev: Slot | None) -> str:
    """Дата без повторов: тот же день — только время, тот же месяц — без названия месяца."""
    d, t = slot.start.date(), f"в {time_words(slot.start.time())}"
    if prev is not None and prev.start.date() == d:
        return t
    if prev is not None and prev.start.month == d.month:
        return f"{WEEKDAYS_ACC[d.weekday()]}, {day_ordinal(d.day)}, {t}"
    return f"{on_date(d)}, {t}"


def _render_offers(offered: list[Slot], with_doctor: bool = True) -> str:
    """Варианты для голоса: врач — один раз, если он у всех один, дата — без повторов.

    «Терапевт Белова Анна Сергеевна. Первый вариант — в понедельник, девятнадцатого октября, в десять часов,
    второй — в десять тридцать, третий — во вторник, двадцатого, в девять часов.»
    """
    doctors = {s.doctor_id for s in offered}
    branches = {s.branch for s in offered}
    parts, prev = [], None
    for i, s in enumerate(offered):
        label = f"{_ORDINAL_NAMES[i]} вариант" if i == 0 else _ORDINAL_NAMES[i]
        who = ""
        if with_doctor and len(doctors) > 1:
            who = f", врач {s.doctor_name.split()[0]}" + (f", {s.branch}" if len(branches) > 1 and s.branch else "")
        parts.append(f"{label} — {_short_when(s, prev)}{who}")
        prev = s
    first = offered[0]
    if not with_doctor:
        head = "Свободное время"
    elif len(doctors) == 1:
        head = f"{first.specialty.capitalize()} {first.doctor_name}"
    else:
        head = f"{first.specialty.capitalize()}, ближайшее время"
    if with_doctor and len(branches) == 1 and first.branch:
        head += f", {first.branch}"
    return f"{head}. " + "; ".join(parts) + "."


def _spread(slots: list[Slot], n: int = 3) -> list[Slot]:
    """Ближайшие варианты в разные дни — так пациенту проще выбрать, чем из трёх подряд в один день."""
    picked, days = [], set()
    for s in slots:
        if s.start.date() not in days:
            picked.append(s)
            days.add(s.start.date())
        if len(picked) == n:
            break
    picked += [s for s in slots if s not in picked][: n - len(picked)]
    return sorted(picked, key=lambda s: s.start)


def _distinct_times(slots: list[Slot], n: int = 3) -> list[Slot]:
    """Варианты в один день — в разное время, а не «восемь ноль-ноль» у трёх разных врачей."""
    picked, seen = [], set()
    for s in slots:
        hour = (s.start.hour, s.start.minute // 30)
        if hour not in seen:
            picked.append(s)
            seen.add(hour)
        if len(picked) == n:
            break
    picked += [s for s in slots if s not in picked][: n - len(picked)]
    return sorted(picked, key=lambda s: s.start)


def _to_specialist(specialty: str) -> str:
    """«к неврологу», «к детскому кардиологу» — для узких специалистов, у которых имена на согласную."""
    words = specialty.split()
    if not words or words[-1][-1] in "аеиоуыэюя)":
        return f"к специалисту {specialty}"
    adj = [w[:-2] + "ому" if w.endswith(("ий", "ый")) else w for w in words[:-1]]
    return "к " + " ".join([*adj, words[-1] + "у"])


def _short(b: Booking) -> str:
    """«к терапевту Беловой» без склонений: «терапевт Белова, в среду, седьмого октября, в девять тридцать»."""
    return f"{b.slot.specialty} {b.slot.doctor_name.split()[0]}, {b.slot.when()}"

