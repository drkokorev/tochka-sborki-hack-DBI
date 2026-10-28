"""Качество каскада «ученик → LLM» на независимых фразах (eval/nlu_test.jsonl).

    .venv/bin/python tools/eval_cascade.py data/student_ft [data/student ...]

Покрытие — доля фраз, на которые каскад ответил сам, без LLM. Точность — доля верных среди них
(верно = цель/тема/флаг/да-нет совпали с разметкой). Остальные фразы ушли бы в LLM.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domains.clinic.acts import act_fields  # noqa: E402
from domains.clinic.domain import ClinicDomain, ClinicState  # noqa: E402
from domains.clinic.student import Student  # noqa: E402

TODAY = date(2026, 10, 5)


def correct(update, act: str) -> bool:
    want = act_fields(act)
    got = {k: v for k in ("intent", "topic", "red_flag", "confirm", "bye", "smalltalk")
           if (v := getattr(update, k)) is not None and not (k in ("bye", "smalltalk") and v is False)}
    if act in ("identity", "choice", "slots"):
        return not any(k in got for k in ("intent", "topic", "red_flag"))
    return all(got.get(k) == v for k, v in want.items())


def main() -> None:
    test = [json.loads(line) for line in open("eval/nlu_test.jsonl", encoding="utf-8")]
    result = {}
    for folder in sys.argv[1:]:
        student = Student.load(folder)
        domain = ClinicDomain(today=TODAY, student=student)
        answered, ok, lat = 0, 0, []
        for r in test:
            t = time.perf_counter()
            upd = domain.fast_extract(ClinicState(), r["text"], TODAY)
            lat.append(time.perf_counter() - t)
            if upd is not None:
                answered += 1
                ok += correct(upd[0] if isinstance(upd, tuple) else upd, r["act"])  # (разбор, кто понял, уверенность)
        result[folder] = {"coverage": round(answered / len(test), 3), "precision": round(ok / max(1, answered), 3),
                          "latency_ms_avg": round(1000 * sum(lat) / len(lat), 1)}
        print(folder, result[folder], flush=True)
    Path("data/cascade_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
