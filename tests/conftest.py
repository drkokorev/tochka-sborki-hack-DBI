import json
from datetime import date

import pytest

TODAY = date(2026, 10, 5)  # понедельник


class FakeLLM:
    """Возвращает заранее заданные ответы по очереди; dict сериализуется в JSON, Exception — бросается."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages, *, json_mode=False):
        self.calls.append(messages)
        item = self.responses.pop(0) if self.responses else {}
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)


@pytest.fixture
def today():
    return TODAY


@pytest.fixture(autouse=True)
def _fresh_breakers():
    """Предохранители (core/resilience) — общие на процесс: каждый тест начинает с замкнутых."""
    from core.resilience import reset_all

    reset_all()
    yield
    reset_all()


@pytest.fixture(autouse=True)
def _clinic_clock(monkeypatch):
    """Часы клиники в тестах — понедельник тестовой даты, полдень: регистратура работает, окно обзвона открыто.
    Иначе ответы зависели бы от времени запуска (острое состояние ночью — как экстренное, обзвон — вне окна)."""
    from datetime import datetime
    from domains.clinic import reminders
    monkeypatch.setattr(reminders, "clinic_now", lambda: datetime(TODAY.year, TODAY.month, TODAY.day, 12, 0))
