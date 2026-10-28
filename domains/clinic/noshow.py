"""Риск неявки на приём: кого обзванивать с напоминанием в первую очередь (С-10).

Модель обучается скриптом tools/train_noshow.py на истории записей кейса; здесь — только признаки и
применение весов. Без файла модели риск не считается, обзвон идёт по времени приёма.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path


def features(specialty: str, at: datetime, prior_visits: int | None, prior_noshows: int = 0) -> dict[str, float]:
    """Признаки, понятные регистратуре: история пациента, специальность, день недели, час.
    prior_visits=None — история пациента неизвестна (стенд не хранит неявок): её признаки не используем."""
    hour = at.hour
    x = {
        f"spec:{specialty.strip().lower()}": 1.0,
        f"weekday:{at.weekday()}": 1.0,
        "hour:early" if hour < 9 else ("hour:evening" if hour >= 18 else "hour:day"): 1.0,
    }
    if prior_visits is None:
        return x
    x["prior_noshows"] = float(min(prior_noshows, 5))
    x["first_visit"] = 1.0 if prior_visits == 0 else 0.0
    if prior_visits:
        x["noshow_rate"] = prior_noshows / prior_visits
    return x


class NoShowModel:
    def __init__(self, weights: dict[str, float], metrics: dict | None = None):
        self.weights, self.metrics = weights, metrics or {}

    @classmethod
    def load(cls, path: str | Path) -> NoShowModel | None:
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["weights"], data.get("metrics"))

    def risk(self, specialty: str, at: datetime, prior_visits: int | None = None, prior_noshows: int = 0) -> float:
        w = self.weights
        # Специальности стенда записаны иначе, чем в истории: «лор (отоларинголог)» → ищем по вхождению
        x = features(specialty, at, prior_visits, prior_noshows)
        spec = next(iter(k for k in x if k.startswith("spec:")))
        if spec not in w:
            name = spec.removeprefix("spec:")
            alt = next((k for k in w if k.startswith("spec:") and (k[5:] in name or name in k[5:])), None)
            x.pop(spec)
            if alt:
                x[alt] = 1.0
        z = w.get("bias", 0.0) + sum(w.get(k, 0.0) * v for k, v in x.items())
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    def explain(self, specialty: str, at: datetime, prior_visits: int | None = None,
                prior_noshows: int = 0) -> list[str]:
        """Что повышает риск — для подсказки регистратору."""
        x = features(specialty, at, prior_visits, prior_noshows)
        parts = sorted(((k, self.weights.get(k, 0.0) * v) for k, v in x.items()), key=lambda kv: -kv[1])
        names = {"prior_noshows": "были неявки", "noshow_rate": "часто не приходит", "first_visit": "первый визит",
                 "hour:early": "ранний приём", "hour:evening": "вечерний приём"}
        days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        names |= {f"weekday:{i}": d for i, d in enumerate(days)}
        return [names.get(k, k.split(":", 1)[-1]) for k, v in parts if v > 0.05][:3]
