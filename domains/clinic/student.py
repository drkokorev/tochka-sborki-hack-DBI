"""Модель-ученик: дистиллированное понимание речевого акта пациента на CPU, без сети.

Энкодер — cointegrated/rubert-tiny2 (29M параметров, русский), сверху — логистическая регрессия.
Обучение — tools/train_student.py на корпусе, который сгенерировал учитель (LLM). Здесь только применение:
вектор CLS → softmax(W·x + b). Порог уверенности подобран на валидации так, чтобы точность принятых
учеником ответов была не ниже заданной; ниже порога — решает LLM (каскад в ClinicDomain.fast_extract).
Без torch/transformers или без файлов модели ученик просто выключен.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)
ENCODER = "cointegrated/rubert-tiny2"


class Encoder:
    def __init__(self, name: str = ENCODER):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(2)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def embed(self, texts: list[str], batch: int = 64):
        import numpy as np

        out = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tokenizer(texts[i:i + batch], padding=True, truncation=True, max_length=64,
                                     return_tensors="pt")
                cls = self.model(**enc).last_hidden_state[:, 0, :]
                out.append(self._torch.nn.functional.normalize(cls, dim=-1).numpy())
        return np.vstack(out)


class Student:
    """Два варианта: «linear» — голова на эмбеддингах энкодера, «finetuned» — дообученный rubert-tiny2."""

    def __init__(self, head: dict, folder: Path):
        self.classes: list[str] = head["classes"]
        self.tau: float = head["tau"]
        self.metrics: dict = head.get("metrics", {})
        self.kind = head.get("type", "linear")
        self._lock = threading.Lock()
        if self.kind == "finetuned":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(2)
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(folder / head["model"])
            self.model = AutoModelForSequenceClassification.from_pretrained(folder / head["model"]).eval()
        else:
            import numpy as np

            self.W = np.array(head["coef"], dtype="float32")
            self.b = np.array(head["intercept"], dtype="float32")
            self.encoder = Encoder(head.get("encoder", ENCODER))

    @classmethod
    def load(cls, folder: str | Path) -> Student | None:
        folder = Path(folder)
        path = folder / "head.json"
        if not path.exists():
            return None
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")), folder)
        except Exception as exc:  # нет torch/transformers или модели — работаем без ученика
            log.warning("Модель-ученик выключена: %s", exc)
            return None

    def predict(self, text: str) -> tuple[str, float]:
        import numpy as np

        with self._lock:
            if self.kind == "finetuned":
                with self._torch.inference_mode():
                    enc = self.tokenizer([text], truncation=True, max_length=48, return_tensors="pt")
                    p = self._torch.softmax(self.model(**enc).logits[0], -1).numpy()
            else:
                z = self.W @ self.encoder.embed([text])[0] + self.b
                p = np.exp(z - z.max())
                p /= p.sum()
        i = int(p.argmax())
        return self.classes[i], float(p[i])
