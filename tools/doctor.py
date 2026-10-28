"""Проверка готовности перед демо: что есть, чего нет и как это исправить. Секреты не печатает — только «задан / нет».

    make doctor            # или .venv/bin/python tools/doctor.py [--online]

Режимы складываются: базовый (без ключей и сети) работает всегда, остальные — по мере готовности.
Код выхода 1 — не готов базовый режим (без него не запустится ничего); пробелы в остальных — предупреждения
с готовой командой исправления. --online дополнительно стучится в LLM-шлюз, стенд МИС и Asterisk (AMI).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _env() -> None:
    """Переменные из /opt/hackathon/ai.env и .env — как в run.sh (значения не печатаем)."""
    for path in (Path("/opt/hackathon/ai.env"), ROOT / ".env"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip().removeprefix("export "), value.strip().strip('"'))


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def checks(online: bool) -> list[tuple[str, str, bool, str]]:
    """(режим, что проверяем, готово, как исправить)."""
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    token = os.getenv("DBI_API_TOKEN")
    rows = [
        ("база", f"Python ≥ 3.10 (сейчас {sys.version_info.major}.{sys.version_info.minor})", sys.version_info >= (3, 10),
         "поставьте Python 3.10+"),
        ("база", "зависимости: fastapi, pydantic, httpx, openai", all(map(has, ("fastapi", "pydantic", "httpx", "openai"))),
         "make install"),
        ("база", "папки logs/ и data/ доступны на запись",
         all((ROOT / d).is_dir() and os.access(ROOT / d, os.W_OK) for d in ("logs", "data")), "mkdir -p logs data"),
        ("LLM", "ключ LLM-шлюза (LLM_API_KEY или OPENAI_API_KEY)", bool(key),
         "на ВМ — /opt/hackathon/ai.env; локально — впишите ключ в .env (cp .env.example .env). Без ключа: фразник и правила"),
        ("стенд МИС", "токен стенда (DBI_API_TOKEN)", bool(token),
         "на ВМ — /opt/hackathon/ai.env. Без токена — демо-реестр (SQLite)"),
        ("стенд МИС", "расписание кейса (data/doctors.csv, free_slots.csv)",
         (ROOT / "data/doctors.csv").exists() and (ROOT / "data/free_slots.csv").exists(),
         ".venv/bin/python tools/import_schedule.py <Расписание_врачей.xlsx> data "
         "(без него на стенде время не предлагается — перевод к регистратору)"),
        ("голос", "torch и GigaAM для распознавания", has("torch") and has("gigaam"), "make install-ml"),
        ("голос", "синтез Silero v5 (voices/silero_v5_3_ru.pt)", (ROOT / "voices/silero_v5_3_ru.pt").exists(), "make models"),
        ("голос", "резервный синтез Piper (voices/*.onnx)", any((ROOT / "voices").glob("*.onnx")), "make models"),
        ("голос", "детектор речи Silero VAD (silero-vad)", has("silero_vad"), "make install-ml"),
        ("ученик", "модель-ученик (data/student/head.json)", (ROOT / "data/student/head.json").exists(),
         "make student (нужен ключ LLM: корпус пишет учитель). Без ученика: фразник → GigaChat → правила"),
        ("телефония", "секрет AMI (AMI_SECRET)", bool(os.getenv("AMI_SECRET")),
         "./deploy/setup_asterisk.sh — сам создаёт секрет и дописывает в .env"),
    ]
    if online:
        if key:
            try:
                import httpx

                base = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
                ok = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=10).status_code < 500
            except Exception:
                ok = False
            rows.append(("LLM", "шлюз отвечает", ok, "проверьте LLM_BASE_URL и сеть; ассистент продолжит работать без LLM"))
        if token:
            try:
                from domains.clinic.mis import MisClient

                MisClient(os.getenv("DBI_API_URL", ""), token).doctors()
                ok = True
            except Exception:
                ok = False
            rows.append(("стенд МИС", "стенд отвечает", ok, "проверьте DBI_API_URL и токен; без стенда — перевод к регистратору (R-06)"))
        rows.append(("телефония", "Asterisk AMI на 127.0.0.1:5038", port_open(os.getenv("AMI_HOST", "127.0.0.1"),
                                                                          int(os.getenv("AMI_PORT", "5038"))),
                     "sudo systemctl start asterisk (или ./deploy/setup_asterisk.sh)"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="проверить и связь: LLM-шлюз, стенд МИС, Asterisk")
    args = ap.parse_args()
    _env()
    for folder in ("logs", "data"):  # рабочие папки (в git не входят) — создаём сами, это безопасно
        (ROOT / folder).mkdir(exist_ok=True)
    rows = checks(args.online)
    width = max(len(r[1]) for r in rows)
    for mode, what, ok, fix in rows:
        print(f"{'✓' if ok else '✗'} [{mode:9}] {what:{width}}" + ("" if ok else f"\n    → {fix}"))
    base_ok = all(ok for mode, _, ok, _ in rows if mode == "база")
    ready = sorted({m for m, _, ok, _ in rows} - {m for m, _, ok, _ in rows if not ok})
    print(f"\nГотовы режимы: {', '.join(ready) or '—'}. "
          + ("Базовый режим готов: make demo и make run работают без ключей и сети." if base_ok
             else "Базовый режим НЕ готов — исправьте пункты [база] выше."))
    return 0 if base_ok else 1


if __name__ == "__main__":
    sys.exit(main())
