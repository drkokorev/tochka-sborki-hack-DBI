"""Конвертирует «Расписание врачей» кейса (XLSX) в компактные CSV для ассистента.

    .venv/bin/python tools/import_schedule.py путь/к/Расписание_врачей.xlsx [data] [--from 2026-10-15]

Пишет в папку data/ (она в .gitignore — материалы кейса не публикуем):
- doctors.csv     — врач, специальность, отделение, кабинет, длина слота, часы приёма по дням недели;
- free_slots.csv  — свободные слоты начиная с даты --from;
- unavailable.csv — отпуска, больничные, учёба.
Нужен openpyxl (requirements-dev.txt); самому ассистенту он не нужен.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
from pathlib import Path

DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def _rows(ws):
    it = ws.iter_rows(values_only=True)
    header = next(it)
    for row in it:
        if row and row[0]:
            yield dict(zip(header, row))


def _d(value) -> str:
    return (value.date() if isinstance(value, datetime) else value).isoformat()


def main() -> None:
    import openpyxl

    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("out", nargs="?", default="data")
    ap.add_argument("--from", dest="since", default=date.today().isoformat())
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    wb = openpyxl.load_workbook(args.xlsx, read_only=True)

    hours = {r["ID"]: r for r in _rows(wb["График"])}
    with open(out / "doctors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "specialty", "department", "cabinet", "slot_minutes", *DAYS])
        n = 0
        for r in _rows(wb["Врачи"]):
            h = hours.get(r["ID"], {})
            w.writerow([r["ID"], r["ФИО"], r["Специальность (профиль)"], r["Отделение"], r["Кабинет"],
                        r["Слот, мин"], *[(h.get(d) or "—") for d in DAYS]])
            n += 1
    print(f"doctors.csv: {n}")

    since = date.fromisoformat(args.since)
    with open(out / "free_slots.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doctor_id", "date", "start", "minutes"])
        n = 0
        for r in _rows(wb["Слоты"]):
            if r["Статус"] != "Свободен" or r["Дата"].date() < since:
                continue
            w.writerow([r["ID врача"], _d(r["Дата"]), r["Начало"].strftime("%H:%M"), r["Мин"]])
            n += 1
    print(f"free_slots.csv: {n} с {since}")

    with open(out / "unavailable.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doctor_id", "kind", "date_from", "date_to"])
        n = 0
        for r in _rows(wb["Недоступность"]):
            w.writerow([r["ID врача"], r["Тип"], _d(r["С"]), _d(r["По"])])
            n += 1
    print(f"unavailable.csv: {n}")


if __name__ == "__main__":
    main()
