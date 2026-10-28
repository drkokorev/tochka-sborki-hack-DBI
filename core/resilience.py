"""Единая политика отказоустойчивости: внешние зависимости — через предохранители, у каждой описан запасной путь.

Предохранитель (circuit breaker) считает сбои подряд. После `threshold` сбоев зависимость «размыкается»
на `cooldown` секунд: вызовы не делаются вовсе, ассистент сразу идёт запасным путём, а не ждёт таймаут
на каждой реплике. По истечении паузы пропускается одна пробная попытка: успех — снова «замкнуто».

Защита ПДн при сбоях: в состоянии предохранителя и в логе — только тип ошибки, без текста (в тексте ответа
стенда или шлюза может оказаться ФИО). Логи дополнительно проходят маскирование (`install_log_masking`):
даты, телефоны, ФИО и номера полисов заменяются метками до того, как запись попадёт в файл или консоль.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import traceback
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Что делает ассистент, когда зависимость недоступна. Это же показывает /api/health
FALLBACKS: dict[str, str] = {
    "llm": "понимание без LLM: фразник, локальная модель-ученик и правила; разговор продолжается",
    "mis": "действия с записями — перевод к регистратору с карточкой (R-06); справки работают",
    "stt": "веб — ввод текстом; телефон — сразу перевод на регистратора (облачного распознавания нет)",
    "tts": "веб — голос браузера; телефон — перевод на регистратора",
    "ami": "заявка на перевод уже в МИС — регистратор перезвонит; звонок завершается вежливой фразой",
    "journal": "разговор продолжается; журнал остаётся в памяти до восстановления диска",
}


@dataclass
class Breaker:
    name: str
    threshold: int = 3
    cooldown: float = 30.0
    failures: int = 0                # сбоев подряд
    total_failures: int = 0
    opened_at: float | None = None
    last_error: str = ""             # только тип исключения — без текста (ПДн)
    last_ok: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None and time.monotonic() - self.opened_at < self.cooldown

    def allow(self) -> bool:
        """Можно ли звать зависимость. Разомкнуто — нет; пауза прошла — одна пробная попытка."""
        with self._lock:
            if self.opened_at is None:
                return True
            if time.monotonic() - self.opened_at >= self.cooldown:
                self.opened_at = time.monotonic()  # пробная попытка; следующие ждут её итога
                return True
            return False

    def ok(self) -> None:
        with self._lock:
            self.failures, self.opened_at, self.last_ok = 0, None, time.time()

    def fail(self, exc: BaseException | None = None) -> None:
        with self._lock:
            self.failures += 1
            self.total_failures += 1
            self.last_error = type(exc).__name__ if exc is not None else "failure"
            if self.failures >= self.threshold:
                if self.opened_at is None:
                    log.warning("%s: %d сбоя подряд — запасной путь на %.0f с", self.name, self.failures, self.cooldown)
                self.opened_at = time.monotonic()

    @property
    def state(self) -> str:
        if self.is_open:
            return "down"
        return "degraded" if self.failures else "ok"

    def snapshot(self) -> dict:
        return {"state": self.state, "failures_in_row": self.failures, "failures_total": self.total_failures,
                "last_error": self.last_error or None, "fallback": FALLBACKS.get(self.name, "")}


_breakers: dict[str, Breaker] = {}
_registry_lock = threading.Lock()


def breaker(name: str, **kwargs) -> Breaker:
    """Предохранитель зависимости (один на процесс)."""
    with _registry_lock:
        if name not in _breakers:
            _breakers[name] = Breaker(name, **kwargs)
        return _breakers[name]


def reset_all() -> None:
    """Для тестов: все предохранители снова замкнуты."""
    with _registry_lock:
        _breakers.clear()


def health() -> dict[str, dict]:
    """Состояние всех зависимостей, которые уже вызывались, и их запасные пути."""
    with _registry_lock:
        return {name: b.snapshot() for name, b in sorted(_breakers.items())}


# ---------- маскирование ПДн в логах ----------

_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
_PII = [
    (re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"), "[дата]"),
    (re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{4}})?\b", re.I), "[дата]"),
    (re.compile(r"(?<!\d)(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"), "[телефон]"),
    (re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"), "[полис]"),
    (re.compile(r"(?i)((?:фамили\w*|зовут|пациент\w*|ФИО)[\s:,]+)[а-яё-]+(?:[\s,]+[а-яё-]+){0,2}"), r"\1[ФИО]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    # Фамилия Имя (Отчество) — два-три слова с заглавной кириллицы подряд
    (re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+(?:вич|вна|ич|на))?\b"), "[ФИО]"),
]


def mask_pii(text: str) -> str:
    for pattern, label in _PII:
        text = pattern.sub(label, text)
    return text


_installed = False


def install_log_masking() -> None:
    """Каждая запись лога маскируется до вывода: и текст сообщения, и трассировка исключения."""
    global _installed
    if _installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if record.name.startswith("uvicorn.access"):
            return record  # журнал запросов: метод, путь и код ответа — ПДн там нет (они в теле JSON), а форматтер uvicorn ждёт args
        try:
            record.msg, record.args = mask_pii(record.getMessage()), ()
            if record.exc_info and record.exc_info[0] is not None:
                record.exc_text = mask_pii("".join(traceback.format_exception(*record.exc_info)).rstrip())
        except Exception:  # маскирование не должно ронять логирование
            pass
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
