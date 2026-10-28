"""OpenAI-совместимый клиент и структурированное извлечение в Pydantic-модель."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .config import Settings

from .resilience import breaker

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]

_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLM(Protocol):
    def complete(self, messages: list[Message], *, json_mode: bool = False) -> str: ...


class OpenAICompatLLM:
    """Cloud.ru Foundation Models, vLLM, Ollama, OpenRouter — всё через base_url."""

    def __init__(self, settings: Settings):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "none",
            timeout=settings.llm_timeout,
            max_retries=1,
        )
        self._model = settings.llm_model
        self._json_mode = settings.llm_json_mode

    def complete(self, messages: list[Message], *, json_mode: bool = False) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode and self._json_mode else {}
        try:
            resp = self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=0.1, **extra
            )
        except Exception as err:
            # JSON-режим выключаем, только если провайдер ругается именно на него, а не на сами сообщения
            if not extra or getattr(err, "status_code", None) != 400 or not re.search(
                    r"response_format|json", str(err), re.I):
                raise
            log.warning("Провайдер не принял response_format, работаем без него")
            self._json_mode = False
            resp = self._client.chat.completions.create(model=self._model, messages=messages, temperature=0.1)
        return resp.choices[0].message.content or ""


def strip_think(text: str) -> str:
    """Убирает рассуждения reasoning-моделей (<think>…</think>)."""
    return _THINK.sub("", text).strip()


def parse_json_object(raw: str) -> str:
    """Вырезает JSON-объект из ответа: модели любят обрамлять его ```json и пояснениями."""
    text = strip_think(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("в ответе нет JSON-объекта")
    return text[start : end + 1]


def schema_hint(schema: type[BaseModel]) -> str:
    """Шаблон ответа: ключи модели со значением null. Саму JSON-схему модели повторяют дословно."""
    return json.dumps({name: None for name in schema.model_fields}, ensure_ascii=False)


def _unwrap(raw_json: str) -> str:
    """Некоторые модели кладут значения в "properties", копируя форму JSON-схемы."""
    data = json.loads(raw_json)
    if isinstance(data, dict) and isinstance(data.get("properties"), dict):
        return json.dumps(data["properties"], ensure_ascii=False)
    return raw_json


def extract(llm: LLM, schema: type[T], messages: list[Message]) -> T:
    """Извлечение с одной попыткой ремонта. При любом сбое — пустая модель (бот не падает).
    Шлюз сбоит подряд — предохранитель «llm» размыкается, и LLM не зовём, пока он не остынет (core/resilience)."""
    guard = breaker("llm")
    if not guard.allow():
        return schema()
    msgs = list(messages)
    for _ in range(2):
        try:
            raw = llm.complete(msgs, json_mode=True)
        except Exception as exc:
            guard.fail(exc)
            log.warning("LLM недоступна (%s), извлечение пропущено", type(exc).__name__)
            return schema()
        guard.ok()
        try:
            return schema.model_validate_json(_unwrap(parse_json_object(raw)))
        except ValueError as err:  # pydantic.ValidationError — подкласс ValueError
            log.warning("Невалидный JSON от LLM: %s", type(err).__name__)  # без текста — в нём могут быть ПДн
            msgs += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Ответ невалиден: {err}. Верни только исправленный JSON-объект."},
            ]
    return schema()

