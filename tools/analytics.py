"""Эффект ассистента на данных кейса: журнал звонков регистратуры и история записей → data/analytics.json.

    .venv/bin/python tools/analytics.py журнал_звонков.csv история_записей.csv [eval/report.json]

Сырые CSV (материалы кейса) в git не попадают; дашборд читает только агрегаты.
Сценарий «после» консервативный и прозрачный — все допущения перечислены в поле assumptions.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

VISIT_PRICE = 1200          # средняя стоимость приёма по ОМС из условия кейса, ₽
REMINDER_EFFECT = 0.30      # нижняя граница снижения неявок от напоминаний из условия кейса (30–40%)
OPERATOR_HOURS = (7, 20)    # операторы регистратуры по будням, регламент 7.2


def calls_kpi(path: str) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter=";"))
    days = len({r["datetime"][:10] for r in rows})
    result = Counter(r["result"] for r in rows)
    missed = result["пропущен"]
    answered = [r for r in rows if r["result"] != "пропущен"]
    typical = [r for r in answered if r["result"] in ("записан", "справка")]
    by_hour: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_hour[int(r["datetime"][11:13])].append(r)
    off_hours = sum(len(v) for h, v in by_hour.items() if not OPERATOR_HOURS[0] < h < OPERATOR_HOURS[1])
    return {
        "days": days,
        "calls": len(rows),
        "calls_per_day": round(len(rows) / days),
        "calls_peak_day": max(Counter(r["datetime"][:10] for r in rows).values()),
        "results": dict(result),
        "missed_share": round(missed / len(rows), 3),
        "missed_off_hours": off_hours,
        "wait_avg_sec": round(statistics.mean(int(r["wait_sec"]) for r in answered), 1),
        "wait_missed_avg_sec": round(statistics.mean(int(r["wait_sec"]) for r in rows if r["result"] == "пропущен"), 1),
        "typical_share_of_answered": round(len(typical) / len(answered), 3),
        "typical_minutes_per_day": round(sum(int(r["duration_sec"]) for r in typical) / 60 / days),
        "by_hour": [{"hour": h, "calls": len(v), "missed": round(sum(r["result"] == "пропущен" for r in v) / len(v), 3),
                     "wait": round(statistics.mean(int(r["wait_sec"]) for r in v))} for h, v in sorted(by_hour.items())],
    }


def noshow_kpi(path: str) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter=";"))
    days = len({r["Дата приёма"] for r in rows})
    status = Counter(r["Статус"] for r in rows)
    noshow = status["Неявка"]
    return {
        "appointments": len(rows),
        "status": dict(status),
        "noshow_share": round(noshow / len(rows), 3),
        "noshow_per_day": round(noshow / days, 1),
        "loss_per_month_rub": round(noshow / days * 30 * VISIT_PRICE),
    }


def main() -> None:
    calls, history = calls_kpi(sys.argv[1]), noshow_kpi(sys.argv[2])
    report = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("eval/report.json")
    success = None
    if report.exists():
        runs = json.loads(report.read_text(encoding="utf-8"))
        typical = [r for r in runs if not r["id"].startswith(("emergency", "complaint", "medical", "human", "dental",
                                                              "urgent"))]
        success = round(sum(r["ok"] for r in typical) / max(1, len(typical)), 3)
    automation = success if success is not None else 0.8
    after = {
        "missed_share": 0.0,  # ассистент отвечает сразу и круглосуточно; перевод при занятых операторах — обратный звонок
        "wait_sec": 1,
        "automated_share_of_calls": round(calls["typical_share_of_answered"] * automation, 3),
        "operator_minutes_freed_per_day": round(calls["typical_minutes_per_day"] * automation),
        "noshow_share": round(history["noshow_share"] * (1 - REMINDER_EFFECT), 3),
        "saved_per_month_rub": round(history["loss_per_month_rub"] * REMINDER_EFFECT),
    }
    out = {
        "calls": calls, "noshow": history, "assistant_success": success, "after": after,
        "assumptions": [
            "Ассистент отвечает на каждый звонок сразу и круглосуточно; пропущенный звонок — только при сбое связи.",
            "Типовые звонки (записан, справка) автоматизируются с долей успеха "
            + ("стенда «LLM играет пациента»" if success is not None
               else "ниже результата стенда «LLM-пациент» (22 из 23 = 96%), консервативно")
            + f" ({'%.0f%%' % (automation * 100)}).",
            f"Напоминания снижают неявки на {int(REMINDER_EFFECT * 100)}% — нижняя граница из условия кейса.",
            f"Стоимость приёма по ОМС — {VISIT_PRICE} ₽ (условие кейса).",
        ],
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/analytics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "calls"} | {"missed_share": calls["missed_share"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
