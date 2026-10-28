"""Цикл обработки реплики: guard → извлечение (LLM, иначе правила) → merge → детерминированный шаг домена."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from .guard import looks_like_injection, sanitize
from .llm import LLM, Message, extract, strip_think
from .resilience import breaker

log = logging.getLogger(__name__)


# Уверенность понимания по источнику, если источник не дал своей (ученик отдаёт свою вероятность):
# фразник — точное совпадение с шаблоном стандартной фразы; LLM — проверена правилами (postprocess);
# правила — разбор без модели
CONFIDENCE = {"phrasebook": 0.95, "llm": 0.8, "student": 0.8, "rules": 0.6}


@dataclass
class Step:
    reply: str            # детерминированный ответ — всегда безопасно показать
    facts: str = ""       # факты для NLG-перефразирования; пусто — не перефразировать
    payload: Any = None   # данные для UI (предложенные слоты, запись)
    done: bool = False    # сценарий завершён (например, запись создана)


class Domain(Protocol):
    # Свойства только для чтения: так протокол принимает конкретные подклассы BaseModel
    @property
    def name(self) -> str: ...
    @property
    def state_model(self) -> type[BaseModel]: ...
    @property
    def update_model(self) -> type[BaseModel]: ...
    @property
    def off_topic_reply(self) -> str: ...

    def extraction_messages(
        self, state: BaseModel, history: list[Message], text: str, today: date
    ) -> list[Message]: ...

    def merge(self, state: BaseModel, update: BaseModel) -> BaseModel: ...

    # Необязательно: def rule_extract(self, state, text, today) -> update_model
    # Детерминированный разбор — fallback, когда LLM нет, она упала или ничего не извлекла.
    # Необязательно: def postprocess(self, state, text, update, today) -> update_model
    # Доводка любого разбора, например обязательные проверки безопасности поверх ответа модели.

    def step(self, state: BaseModel) -> Step:
        """Policy домена. Может обновлять служебные поля состояния (предложенные слоты, запись)."""
        ...


@dataclass
class Session:
    state: BaseModel
    history: list[Message] = field(default_factory=list)
    last: Step | None = None
    last_nlu: str = ""  # кто разобрал последнюю реплику: phrasebook | student | llm | rules | none
    last_confidence: float | None = None  # уверенность понимания последней реплики, 0–1
    last_update: BaseModel | None = None  # как понята последняя реплика — для журнала и отладки


NLG_SYSTEM = (
    "Перефразируй черновик ответа ассистента дружелюбно и кратко, на русском языке. "
    "Используй только факты из черновика и блока фактов: ничего не добавляй, "
    "не меняй цифры, даты, время и названия, сохрани нумерацию вариантов."
)


class Agent:
    def __init__(
        self,
        domain: Domain,
        llm: LLM | None,
        *,
        nlg: bool = False,
        today: Callable[[], date] = date.today,
        history_messages: int = 6,
    ):
        self.domain = domain
        self.llm = llm
        self.nlg = nlg
        self.today = today
        self.history_messages = history_messages

    def new_session(self) -> Session:
        return Session(state=self.domain.state_model())

    def handle(self, session: Session, text: str) -> Step:
        text = sanitize(text)
        injection = bool(text) and looks_like_injection(text)
        safety = getattr(self.domain, "safety_check", None)
        urgent = safety(session.state, text, self.today()) if injection and safety is not None else None
        if urgent is not None:
            # Угроза жизни важнее защиты от «игнорируй инструкции»: правила на машине, без LLM
            session.last_update, session.last_nlu, session.last_confidence = urgent, "rules", 0.6
            session.state = self.domain.merge(session.state, urgent)
            step = self.domain.step(session.state)
        elif not text or injection:
            # пустая реплика или попытка «взлома» — без разбора: прошлый разбор и уверенность не переносим
            session.last_update, session.last_nlu, session.last_confidence = None, "none", None
            step = Step(self.domain.off_topic_reply if text else "Не расслышала. Повторите, пожалуйста.")
        else:
            update, session.last_nlu, session.last_confidence = self._understand(session, text)
            post = getattr(self.domain, "postprocess", None)
            if post is not None:
                update = post(session.state, text, update, self.today())
            session.last_update = update
            session.state = self.domain.merge(session.state, update)
            step = self.domain.step(session.state)
            if self.nlg and step.facts:
                step.reply = self._rephrase(step)
        session.history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": step.reply},
        ]
        session.last = step
        return step

    def _understand(self, session: Session, text: str) -> tuple[BaseModel, str, float]:
        """Разбор реплики: (что понято, кто понял, уверенность 0–1)."""
        today = self.today()
        # Каскад: быстрая локальная модель домена (дистиллированный ученик) отвечает, когда уверена,
        # иначе — LLM. None — «не уверен».
        # Ответ — update или (update, кто понял): например, фразник домена или ученик.
        fast = getattr(self.domain, "fast_extract", None)
        if fast is not None:
            result = fast(session.state, text, today)
            if result is not None:
                # update, (update, кто), (update, кто, уверенность)
                result = result if isinstance(result, tuple) else (result, "student")
                return result if len(result) == 3 else (*result, CONFIDENCE.get(result[1], 0.8))
        # Реплика с персональными данными (домен знает, какая) во внешнюю LLM не уходит — её разбирают правила
        local_only = getattr(self.domain, "local_only", None)
        if self.llm is not None and not (local_only is not None and local_only(session.state, text, today)):
            messages = self.domain.extraction_messages(
                session.state, session.history[-self.history_messages :], text, today
            )
            update = extract(self.llm, self.domain.update_model, messages)
            if update.model_dump(exclude_none=True):
                return update, "llm", CONFIDENCE["llm"]
        rules = getattr(self.domain, "rule_extract", None)
        if rules is not None:
            update = rules(session.state, text, today)
            return update, "rules", CONFIDENCE["rules"] if update.model_dump(exclude_none=True) else 0.2
        return self.domain.update_model(), "none", 0.0

    def _rephrase(self, step: Step) -> str:
        if self.llm is None:
            return step.reply
        hide = getattr(self.domain, "mask_for_llm", lambda text: text)  # ПДн во внешнюю модель не уходят
        messages = [
            {"role": "system", "content": NLG_SYSTEM},
            {"role": "user", "content": f"Факты:\n{hide(step.facts)}\n\nЧерновик ответа:\n{hide(step.reply)}"},
        ]
        guard = breaker("llm")
        if not guard.allow():
            return step.reply  # шлюз сбоит — говорим детерминированный текст, без ожидания таймаута
        try:
            text = strip_think(self.llm.complete(messages))
        except Exception as exc:
            guard.fail(exc)
            log.warning("Перефразирование пропущено (%s)", type(exc).__name__)
            return step.reply
        guard.ok()
        return text or step.reply
