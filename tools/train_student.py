"""Обучение модели-ученика на корпусе учителя (tools/distill_data.py).

    .venv/bin/python tools/train_student.py [--precision 0.97]

rubert-tiny2 → вектор CLS → логистическая регрессия (scikit-learn). Валидация — 20% корпуса.
Порог уверенности τ — минимальный, при котором точность принятых учеником ответов ≥ --precision
(акт «other» никогда не принимается). Пишет data/student/head.json: веса, классы, τ и метрики.
Нужны torch, transformers, scikit-learn (requirements-ml.txt).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domains.clinic.student import ENCODER, Encoder  # noqa: E402


def pick_tau(probs, preds, gold, target: float) -> tuple[float, float, float]:
    """(τ, покрытие, точность): самый низкий порог, при котором точность принятых ≥ target."""
    items = sorted(((p, pr == g, pr) for p, pr, g in zip(probs, preds, gold)), reverse=True)
    best = (1.01, 0.0, 1.0)
    correct = accepted = 0
    for p, ok, pr in items:
        if pr == "other":
            continue
        accepted += 1
        correct += ok
        if correct / accepted >= target:
            best = (p, accepted / len(items), correct / accepted)
    return best


def finetune(train: list[dict], classes: list[str], out: Path, epochs: int = 6, lr: float = 1e-4):
    """Полное дообучение rubert-tiny2 с головой классификации на CPU (29M параметров — минуты)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER, num_labels=len(classes), id2label=dict(enumerate(classes)),
        label2id={c: i for i, c in enumerate(classes)})
    index = {c: i for i, c in enumerate(classes)}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = random.Random(1)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(train)
        total = 0.0
        for i in range(0, len(train), 32):
            batch = train[i:i + 32]
            enc = tok([r["text"] for r in batch], padding=True, truncation=True, max_length=48, return_tensors="pt")
            labels = torch.tensor([index[r["act"]] for r in batch])
            loss = model(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            total += float(loss) * len(batch)
        print(f"эпоха {epoch + 1}: loss {total / len(train):.3f}", flush=True)
    model.eval()
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    return model, tok


def predict_proba_ft(model, tok, texts: list[str]):
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=48, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, -1).numpy())
    import numpy as np
    return np.vstack(out)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/distill/corpus.jsonl")
    ap.add_argument("--out", default="data/student")
    ap.add_argument("--precision", type=float, default=0.98)
    ap.add_argument("--test", default="eval/nlu_test.jsonl", help="независимые фразы, которых нет в корпусе")
    ap.add_argument("--finetune", action="store_true", help="дообучить энкодер целиком (иначе — голова на эмбеддингах)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.corpus, encoding="utf-8")]
    rng = random.Random(13)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.8)
    train, val = rows[:cut], rows[cut:]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted({r["act"] for r in rows})

    if args.finetune:
        model, tok = finetune(list(train), classes, out, epochs=args.epochs, lr=args.lr)

        def proba(texts):
            return predict_proba_ft(model, tok, texts)
        head = {"type": "finetuned", "model": "model", "classes": classes}
    else:
        enc = Encoder(ENCODER)
        clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
        clf.fit(enc.embed([r["text"] for r in train]), [r["act"] for r in train])
        classes = clf.classes_.tolist()

        def proba(texts):
            return clf.predict_proba(enc.embed(texts))
        head = {"type": "linear", "encoder": ENCODER, "classes": classes, "coef": clf.coef_.round(5).tolist(),
                "intercept": clf.intercept_.round(5).tolist()}

    pv = proba([r["text"] for r in val])
    preds = [classes[i] for i in pv.argmax(1)]
    gold = [r["act"] for r in val]
    tau, coverage, precision = pick_tau(pv.max(1).tolist(), preds, gold, args.precision)

    t = time.perf_counter()  # задержка одной реплики «как в звонке», без батча
    for r in val[:50]:
        proba([r["text"]])
    single_ms = 1000 * (time.perf_counter() - t) / min(50, len(val))

    noisy = [i for i, r in enumerate(val) if r.get("noisy")]
    metrics = {
        "model": head["type"], "corpus": len(rows), "train": len(train), "val": len(val), "acts": len(classes),
        "accuracy": round(accuracy_score(gold, preds), 3),
        "macro_f1": round(f1_score(gold, preds, average="macro"), 3),
        "accuracy_noisy": round(accuracy_score([gold[i] for i in noisy], [preds[i] for i in noisy]), 3) if noisy else None,
        "tau": round(float(tau), 3), "coverage_at_tau": round(coverage, 3), "precision_at_tau": round(precision, 3),
        "latency_ms_single": round(single_ms, 1), "sources": dict(Counter(r["source"] for r in rows)),
    }
    if Path(args.test).exists():
        # Честная проверка: фразы написаны отдельно от корпуса. Базовая линия — правила
        from datetime import date

        from domains.clinic.domain import ClinicDomain, ClinicState

        test = [json.loads(line) for line in open(args.test, encoding="utf-8")]
        tp = proba([r["text"] for r in test])
        t_pred = [classes[i] for i in tp.argmax(1)]
        conf = tp.max(1)
        mask = [c >= tau and p != "other" for c, p in zip(conf, t_pred)]
        accepted = [p == r["act"] for p, r, m in zip(t_pred, test, mask) if m]
        domain = ClinicDomain(today=date(2026, 10, 5))
        rules_ok = 0
        for r in test:
            u = domain.rule_extract(ClinicState(), r["text"], date(2026, 10, 5))
            fam = (f"intent:{u.intent}" if u.intent else f"faq:{u.topic}" if u.topic else
                   f"flag:{u.red_flag}" if u.red_flag else None)
            rules_ok += fam == r["act"] or (fam is None and not r["act"].startswith(("intent", "faq", "flag")))
        metrics |= {"test": len(test),
                    "test_accuracy": round(sum(p == r["act"] for p, r in zip(t_pred, test)) / len(test), 3),
                    "test_coverage_at_tau": round(sum(mask) / len(test), 3),
                    "test_precision_at_tau": round(sum(accepted) / max(1, len(accepted)), 3),
                    "rules_test_accuracy": round(rules_ok / len(test), 3)}
        misses = [(r["text"], r["act"], p, round(float(c), 2)) for r, p, c in zip(test, t_pred, conf) if p != r["act"]]
        print("Ошибки на независимых фразах:", misses[:20])
    (out / "head.json").write_text(json.dumps(head | {"tau": float(tau), "metrics": metrics}, ensure_ascii=False),
                                   encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
