"""Настройки из окружения (.env)."""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_base_url: str | None
    llm_api_key: str
    llm_model: str
    llm_json_mode: bool
    llm_timeout: float
    llm_nlg: bool

    @property
    def llm_enabled(self) -> bool:
        """Без ключа LLM не вызывается: работает только разбор по правилам домена."""
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        # Любой OpenAI-совместимый шлюз; пусто — api.openai.com
        llm_base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "GigaChat/GigaChat-3-Pro"),
        llm_json_mode=os.getenv("LLM_JSON_MODE", "1") == "1",
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        llm_nlg=os.getenv("LLM_NLG", "0") == "1",
    )
