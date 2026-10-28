"""Модель риска неявки по истории записей кейса (CSV «История записей на приём»).

    .venv/bin/python tools/train_noshow.py путь/к/истории.csv [data/noshow_model.json]

Логистическая регрессия на чистом Python (без sklearn): признаки понятны регистратуре и жюри —
прошлые неявки и визиты пациента, специальность, день недели, час приёма.
Проверка честная, по времени: учимся на первых пяти месяцах, качество меряем на последнем.
Модель (веса признаков) пишется в data/ — вместе с данными кейса, в git не попадает. Метрики печатаются.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domains.clinic.noshow import features  # noqa: E402


def load(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter=";"))
    for r in rows:
        r["at"] = datetime.strptime(f"{r['Дата приёма']} {r['Время приёма']}", "%d.%m.%Y %H:%M")
    rows.sort(key=lambda r: r["at"])
    # История пациента — только то, что было ДО этого приёма (без заглядывания в будущее)
    past: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        h = past[r["Пациент (хэш)"]]
        r["prior_visits"], r["prior_noshows"] = len(h), sum(s == "Неявка" for s in h)
        h.append(r["Статус"])
    return [r for r in rows if r["Статус"] != "Отменён"]  # отменённые заранее — не неявка и не визит


def train(rows: list[dict], epochs: int = 12, lr: float = 0.05, l2: float = 1e-4) -> dict[str, float]:
    w: dict[str, float] = defaultdict(float)
    base = sum(r["Статус"] == "Неявка" for r in rows) / len(rows)
    w["bias"] = math.log(base / (1 - base))
    for epoch in range(epochs):
        step = lr / (1 + epoch)
        for r in rows:
            x = feats(r)
            p = _sigmoid(sum(w[k] * v for k, v in x.items()) + w["bias"])
            g = p - (r["Статус"] == "Неявка")
            w["bias"] -= step * g
            for k, v in x.items():
                w[k] -= step * (g * v + l2 * w[k])
    return dict(w)


def feats(r: dict) -> dict[str, float]:
    return features(r["Специальность"], r["at"], r["prior_visits"], r["prior_noshows"])


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def auc(scores: list[float], labels: list[bool]) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    rank_sum = sum(i + 1 for i, (_, y) in enumerate(pairs) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main() -> None:
    src = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/noshow_model.json")
    rows = load(src)
    split = rows[-1]["at"] - timedelta(days=30)
    train_rows, test_rows = [r for r in rows if r["at"] < split], [r for r in rows if r["at"] >= split]
    w = train(train_rows)
    score = lambda r: _sigmoid(w["bias"] + sum(w.get(k, 0.0) * v for k, v in feats(r).items()))  # noqa: E731
    scores = [score(r) for r in test_rows]
    labels = [r["Статус"] == "Неявка" for r in test_rows]
    ranked = sorted(zip(scores, labels), reverse=True)
    top = ranked[: len(ranked) * 3 // 10]
    caught = sum(y for _, y in top) / max(1, sum(labels))
    metrics = {
        "train": len(train_rows), "test": len(test_rows), "test_from": split.date().isoformat(),
        "noshow_rate_test": round(sum(labels) / len(labels), 3),
        "auc": round(auc(scores, labels), 3),
        "top30_share_of_noshows": round(caught, 3),
        "top30_noshow_rate": round(sum(y for _, y in top) / len(top), 3),
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"weights": w, "metrics": metrics}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    strongest = sorted(((k, v) for k, v in w.items() if k != "bias"), key=lambda kv: -abs(kv[1]))[:8]
    print("Самые сильные признаки:", ", ".join(f"{k} {v:+.2f}" for k, v in strongest))


if __name__ == "__main__":
    main()
