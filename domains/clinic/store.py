"""Демо-реестр с хранением в SQLite: записи, пациенты, заявки на перевод, итоги обзвона и журнал записей
переживают перезапуск сервера (без стенда МИС). На стенде источник правды — сам стенд (domains/clinic/mis.py).

Запись — сквозная: каждое изменение реестра сразу сохраняется одной транзакцией, а при старте читается обратно.
Файл — с правами 0600, как журнал разговоров: в нём ФИО и даты рождения (пусть и вымышленные в демо).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import date
from pathlib import Path

from .schedule import Booking, Doctor, Patient, Schedule, Slot

_TABLES = {
    "patients": "id TEXT PRIMARY KEY, body TEXT NOT NULL",
    "bookings": "id TEXT PRIMARY KEY, body TEXT NOT NULL",
    "history": "id TEXT PRIMARY KEY, body TEXT NOT NULL",
    "escalations": "n INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL",
    "reminder_log": "n INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL",
    "meta": "key TEXT PRIMARY KEY, value TEXT NOT NULL",
}


class SqliteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._db:
            for name, columns in _TABLES.items():
                self._db.execute(f"CREATE TABLE IF NOT EXISTS {name} ({columns})")
        if fresh:
            os.chmod(self.path, 0o600)

    def rows(self, table: str) -> list[str]:
        order = "n" if table in ("escalations", "reminder_log") else "rowid"
        return [r[0] for r in self._db.execute(f"SELECT body FROM {table} ORDER BY {order}")]

    def meta(self, key: str, default: str = "") -> str:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def write(self, *ops: tuple[str, tuple]) -> None:
        """Несколько изменений одной транзакцией: реестр на диске не бывает «наполовину записан»."""
        with self._lock, self._db:
            for sql, args in ops:
                self._db.execute(sql, args)

    def clear(self) -> None:
        self.write(*((f"DELETE FROM {name}", ()) for name in _TABLES))

    def is_empty(self) -> bool:
        return not self._db.execute("SELECT 1 FROM patients LIMIT 1").fetchone()


def _upsert(table: str, key: str, body: str) -> tuple[str, tuple]:
    return f"INSERT OR REPLACE INTO {table} (id, body) VALUES (?, ?)", (key, body)


class PersistentSchedule(Schedule):
    """Schedule, который сохраняет каждое изменение в SQLite и восстанавливается из него при старте."""

    def __init__(self, doctors: list[Doctor], today: date, store: SqliteStore, **kwargs):
        super().__init__(doctors, today, **kwargs)
        self.store = store
        for body in store.rows("patients"):
            p = Patient.model_validate_json(body)
            self.patients[p.id] = p
        for body in store.rows("bookings"):
            b = Booking.model_validate_json(body)
            if b.slot.doctor_id in self.doctors:
                Schedule._store(self, b)
        self.history = {h["id"]: h for h in map(json.loads, store.rows("history"))}
        self.escalations = [json.loads(x) for x in store.rows("escalations")]
        self.reminder_log = [json.loads(x) for x in store.rows("reminder_log")]
        self._next_id = int(store.meta("next_id", "1"))

    def _save_booking(self, booking: Booking, *extra: tuple[str, tuple]) -> None:
        self.store.write(_upsert("bookings", booking.id, booking.model_dump_json()),
                         _upsert("history", booking.id, json.dumps(self.history.get(booking.id, {}), ensure_ascii=False)),
                         ("INSERT OR REPLACE INTO meta (key, value) VALUES ('next_id', ?)", (str(self._next_id),)),
                         *extra)

    def identify(self, last_name: str, first_name: str, birth_date: date, *, create: bool) -> Patient | None:
        known = len(self.patients)
        patient = super().identify(last_name, first_name, birth_date, create=create)
        if patient is not None and len(self.patients) > known:
            self.store.write(_upsert("patients", patient.id, patient.model_dump_json()))
        return patient

    def book(self, slot: Slot, patient_name: str, patient_id: str | None = None) -> Booking:
        booking = super().book(slot, patient_name, patient_id)
        self._save_booking(booking)
        return booking

    def cancel(self, booking_id: str) -> Booking:
        booking = super().cancel(booking_id)
        self.store.write(("DELETE FROM bookings WHERE id = ?", (booking_id,)),
                         _upsert("history", booking_id, json.dumps(self.history.get(booking_id, {}), ensure_ascii=False)))
        return booking

    def reschedule(self, booking_id: str, slot: Slot) -> Booking:
        moved = super().reschedule(booking_id, slot)  # внутри — cancel (уже сохранён) и новая запись
        self._save_booking(moved)
        return moved

    def escalate(self, patient: Patient | None, description: str) -> str:
        number = super().escalate(patient, description)
        self.store.write(("INSERT INTO escalations (body) VALUES (?)", (json.dumps(self.escalations[-1], ensure_ascii=False),)))
        return number

    def set_opt_out(self, patient: Patient, code: str) -> None:
        super().set_opt_out(patient, code)
        self.store.write(("INSERT INTO reminder_log (body) VALUES (?)", (json.dumps(self.reminder_log[-1], ensure_ascii=False),)))

    def log_reminder(self, booking: Booking, patient: Patient | None, code: str) -> None:
        super().log_reminder(booking, patient, code)
        self.store.write(("INSERT INTO reminder_log (body) VALUES (?)", (json.dumps(self.reminder_log[-1], ensure_ascii=False),)))
