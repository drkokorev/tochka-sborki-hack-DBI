"""Реальное расписание клиники из материалов кейса (data/*.csv, см. tools/import_schedule.py).

Стенд МИС хранит врачей и записи, но не расписание. Расписание даёт: часы приёма по дням недели,
кабинет, свободные слоты и периоды недоступности (отпуск, больничный, учёба). Врачей стенда
сопоставляем с расписанием по ФИО.
"""

from __future__ import annotations

import csv
import logging
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path

log = logging.getLogger(__name__)
DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def _norm(name: str) -> str:
    return " ".join(name.lower().replace("ё", "е").split())


def parse_hours(value: str) -> tuple[int, int] | None:
    """«08:00–14:00» или «09:00–13:00, 14:00–18:00» → (8, 14) / (9, 18); «—» → None."""
    parts = [p.strip() for p in value.replace("–", "-").split(",") if p.strip() not in ("", "-", "—")]
    if not parts:
        return None
    start = int(parts[0].split("-")[0].split(":")[0])
    end = int(parts[-1].split("-")[-1].split(":")[0])
    return start, end


@dataclass
class TimetableDoctor:
    id: str
    name: str
    specialty: str
    department: str
    cabinet: str
    slot_minutes: int
    hours: dict[int, tuple[int, int]]                      # день недели → (с, до)
    free: list[datetime] = field(default_factory=list)      # свободные слоты по возрастанию
    absent: list[tuple[str, date, date]] = field(default_factory=list)


class Timetable:
    def __init__(self, doctors: list[TimetableDoctor]):
        self.doctors = {d.id: d for d in doctors}
        self._by_name = {_norm(d.name): d for d in doctors}

    @classmethod
    def load(cls, folder: str | Path) -> Timetable | None:
        folder = Path(folder)
        if not (folder / "doctors.csv").exists():
            return None
        doctors: list[TimetableDoctor] = []
        with open(folder / "doctors.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                hours = {i: h for i, day in enumerate(DAYS) if (h := parse_hours(r[day]))}
                doctors.append(TimetableDoctor(r["id"], r["name"], r["specialty"], r["department"], r["cabinet"],
                                               int(r["slot_minutes"]), hours))
        tt = cls(doctors)
        free: dict[str, list[datetime]] = defaultdict(list)
        if (folder / "free_slots.csv").exists():
            with open(folder / "free_slots.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    free[r["doctor_id"]].append(datetime.fromisoformat(f"{r['date']}T{r['start']}"))
        for doc_id, starts in free.items():
            if doc_id in tt.doctors:
                tt.doctors[doc_id].free = sorted(starts)
        if (folder / "unavailable.csv").exists():
            with open(folder / "unavailable.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["doctor_id"] in tt.doctors:
                        tt.doctors[r["doctor_id"]].absent.append(
                            (r["kind"], date.fromisoformat(r["date_from"]), date.fromisoformat(r["date_to"])))
        log.info("Расписание: %d врачей, %d свободных слотов", len(doctors), sum(len(v) for v in free.values()))
        return tt

    def by_name(self, name: str) -> TimetableDoctor | None:
        return self._by_name.get(_norm(name))

    def free_slots(self, doctor_id: str, day_from: date, day_to: date) -> list[datetime]:
        doc = self.doctors[doctor_id]
        lo = bisect_left(doc.free, datetime.combine(day_from, time.min))
        hi = bisect_left(doc.free, datetime.combine(day_to, time.max))
        return doc.free[lo:hi]

    @staticmethod
    def absent_until(doc: TimetableDoctor, today: date) -> tuple[str, date] | None:
        """Если врач сейчас в отпуске или на больничном — до какого числа включительно."""
        for kind, start, end in doc.absent:
            if start <= today <= end:
                return kind, end
        return None
