"""Очистка пользовательского ввода и простая эвристика против промпт-инъекций."""

import re

MAX_INPUT_CHARS = 1000

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_INJECTION = re.compile(
    "|".join(
        [
            r"ignore (all |the )?(previous|above|prior) (instructions|prompts?)",
            r"игнорир\w* (все |всё )?(предыдущ|инструкц|правил)",
            r"забудь\w* (все |всё )?(инструкц|правил|что тебе)",
            r"(system|системн\w*) (prompt|промпт)",
            r"developer mode|режим разработчика",
            r"\bjailbreak\b",
        ]
    ),
    re.IGNORECASE,
)


def sanitize(text: str) -> str:
    text = _CONTROL.sub("", text)
    text = _SPACES.sub(" ", text).strip()
    return text[:MAX_INPUT_CHARS]


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))
