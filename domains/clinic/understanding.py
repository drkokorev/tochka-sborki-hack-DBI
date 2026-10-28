"""Понимание реплики: каскад фразник → ученик → LLM → правила и проверки правилами поверх модели.

Модели только заполняют ClinicUpdate; решения принимают merge и step (dialog.py, scenarios.py).
"""

from __future__ import annotations

import difflib
import re
from datetime import date, timedelta

from core.llm import Message, schema_hint
from core.ru import WEEKDAYS

from . import phrasebook, privacy
from .acts import act_fields
from .faq import FAQ_BY_KEY, topics_hint
from .models import IDENTITY_FIELDS, SEARCH_FIELDS, ClinicState, ClinicUpdate
from .replies import EXTRACTION_PROMPT
from .rules import _APPOINTMENT_DAY, _awaits_identity, _correcting_identity, rule_extract
from .schedule import normalize_specialty
from .speech import strip_fillers

# Основания для выбора варианта в самой фразе: номер, время, день, «давайте», «подходит», фамилия не в счёт
_CHOICE_HINT = re.compile(r"\d|перв|втор|трет|последн|давайте|подходит|устраивает|беру|запиш|"
                          r"понедельн|вторник|\bсред|четверг|пятниц|суббот")
# Согласие на «…Верно?» перед записью, переносом, отменой: «да» модели без такого слова не принимаем —
# обрывок «…ова Ольга» после «нет, Смирнова» модель однажды прочла как «да» и записала не того пациента
_AGREE = re.compile(r"\b(да|ага|угу|конечно|давайте|подтверждаю|верно|правильно|хорошо|согласн\w*|подходит|устраивает|"
                    r"записывайте|отменяйте|переносите|оформляйте|отлично|ок|окей|точно|именно|годится|ладно|"
                    r"можно|пойд[её]т|супер|прекрасно|так и есть|вс[её] так)\b", re.I)
_ASKS_HUMAN = re.compile(r"соедин|позов|переключ|дайте|хочу поговорить|позвать|нужен оператор|нужен человек")

# Общее название → специальность в справочнике стенда, когда вариантов несколько
_PREFERRED_SPECIALTY = {
    "стоматолог": "стоматолог-терапевт",
    "оториноларинголог": "лор (отоларинголог)",
    "лор": "лор (отоларинголог)",
    "дерматолог": "дерматовенеролог",
    "гинеколог": "акушер-гинеколог",
    "окулист": "офтальмолог",
}


class Understanding:
    """Часть ClinicDomain: извлечение данных из реплики. Нужны self.schedule, self.student, self.speech."""

    def extraction_messages(self, state: ClinicState, history: list[Message], text: str, today: date) -> list[Message]:
        context = ""
        options = state.offered if state.offered and not state.chosen else [b.slot for b in state.candidates]
        if options and not state.target:
            context = "Пациенту предложены варианты:\n" + "\n".join(
                f"{i}) {s.spoken()}" for i, s in enumerate(options, 1))
        elif state.awaiting_confirm or state.question:
            context = "Ассистент задал вопрос, на который ждёт ответа да или нет."
        system = EXTRACTION_PROMPT.format(
            today=today.isoformat(),
            weekday=WEEKDAYS[today.weekday()],
            # ФИО и дата рождения в модель не уходят (152-ФЗ): их понимают правила на машине
            state=state.model_dump_json(include={"intent", *SEARCH_FIELDS, "booking_id"}, exclude_none=True),
            context=context,
            calendar="; ".join(f"{WEEKDAYS[d.weekday()]} {d.isoformat()}"
                               for d in (today + timedelta(days=i) for i in range(1, 8))),
            specialties=", ".join(self.specialties),
            topics=topics_hint(),
            schema=schema_hint(ClinicUpdate),
        )
        # Исходящий звонок начинает ассистент, а часть моделей (GigaChat) не принимает диалог,
        # который после системного промпта начинается с реплики ассистента: переносим её в контекст
        known = tuple(v for v in (state.last_name, state.first_name,
                                  state.patient.last_name if state.patient else None,
                                  state.patient.first_name if state.patient else None) if v)
        hide = lambda s: privacy.mask(s, known, self.schedule.find_doctor)  # noqa: E731
        history = [{**m, "content": hide(m["content"])} for m in history]
        while history and history[0]["role"] == "assistant":
            system += f"\nАссистент сказал пациенту: «{history.pop(0)['content']}»"
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": hide(text)}]

    def fast_extract(self, state: ClinicState, text: str, today: date) -> tuple[ClinicUpdate, str, float] | None:
        """Каскад без сети: фразник (хэш-таблица стандартных фраз) → ученик → None (пусть решает LLM).

        Фразник и ученик определяют только речевой акт; слоты достают правила."""
        if state.question == "rating":  # «пять», «отлично», «до свидания» — оценку разбирают правила
            return self.rule_extract(state, text, today), "rules", 0.9
        if not re.search(r"[а-яёa-z0-9]", strip_fillers(text).lower()):
            # «эээ», «ну» — шум, а не непонимание: переспрашиваем без счётчика R-05
            return ClinicUpdate(smalltalk=True), "phrasebook", 0.95
        if self.use_phrasebook and (act := phrasebook.lookup(text, self.speech, self.schedule.find_doctor)):
            if (update := self._act_update(state, text, today, act)) is not None:
                return update, "phrasebook", 0.95
        if self.student is None:
            return None
        act, p = self.student.predict(text)
        if act == "other" or p < self.student.tau:
            return None
        update = self._act_update(state, text, today, act)
        return None if update is None else (update, "student", round(float(p), 3))

    def _act_update(self, state: ClinicState, text: str, today: date, act: str) -> ClinicUpdate | None:
        """Речевой акт + слоты правил. None — данные не сходятся с актом, такой случай решает LLM."""
        rules = self.rule_extract(state, text, today)
        fields = act_fields(act)
        # Ассистент спросил «да/нет», а в ответе явное «да» или «нет» («Да, всё верно. Спасибо!»,
        # «Да, обязательно приду»): контекст вопроса важнее догадки модели о справке или новой цели
        if ((state.awaiting_confirm or state.question) and rules.confirm is not None and act not in ("yes", "no")
                and not rules.red_flag and not rules.topic and (not rules.intent or rules.intent == state.intent)):
            return rules
        # Акт без данных, которые должны были найти правила, — сомнительный случай, отдаём LLM
        if act == "faq:gratitude" and rules.topic != "gratitude":
            act, fields = "smalltalk", {"smalltalk": True}  # «спасибо, этого достаточно» — не благодарность врачу
        if act == "identity" and not (rules.last_name or rules.birth_date):
            return None
        if act == "choice" and rules.choice is None:
            return None
        if act == "slots" and not any(getattr(rules, k) for k in (*SEARCH_FIELDS, "booking_id", "choice")):
            return None
        # Правила и акт расходятся в цели или теме — неоднозначно, решает LLM
        if "intent" in fields and rules.intent and rules.intent != fields["intent"]:
            return None
        if "topic" in fields and rules.topic and rules.topic != fields["topic"]:
            return None
        update = rules.model_copy(update=fields)
        if "topic" in fields or "red_flag" in fields:
            update = update.model_copy(update={"note": text[:300]})
        if "topic" in fields and rules.intent in ("book", "schedule") and not re.search(r"запиш|записат", text.lower()):
            update = update.model_copy(update={"intent": None})  # «что взять на приём» — справка, не запись
        return update

    def local_only(self, state: ClinicState, text: str, today: date) -> bool:
        """Реплику разбирают только правила на машине, без внешней LLM (152-ФЗ): ассистент ждёт ФИО и дату
        рождения или правила нашли их в реплике. Маскирование — второй рубеж, этот — первый."""
        if _awaits_identity(state) or state.question in ("home_address", "birth_check"):
            return True  # ФИО, дата рождения и адрес — персональные данные
        rules = self.rule_extract(state, text, today)
        if any(getattr(rules, k) is not None for k in IDENTITY_FIELDS):
            return True
        return privacy.suspicious_name(text, self.schedule.find_doctor)  # редкое имя строчными — тоже на машине

    def mask_for_llm(self, text: str) -> str:
        return privacy.mask(text, (), self.schedule.find_doctor)

    def rule_extract(self, state: ClinicState, text: str, today: date) -> ClinicUpdate:
        return rule_extract(state, text, today, self.schedule, self.speech)

    def safety_check(self, state: ClinicState, text: str, today: date) -> ClinicUpdate | None:
        """Красный флаг угрозы жизни или острого состояния по локальным правилам — проверяется на каждой реплике,
        в том числе там, где разбор иначе пропускается (защита от инструкций в тексте)."""
        rules = self.rule_extract(state, text, today)
        if rules.red_flag in ("emergency", "selfharm", "urgent"):
            return ClinicUpdate(red_flag=rules.red_flag, note=rules.note)
        return None

    def postprocess(self, state: ClinicState, text: str, update: ClinicUpdate, today: date) -> ClinicUpdate:
        """Правила поверх модели: красные флаги ловим всегда, даты и время правила считают точнее."""
        rules = self.rule_extract(state, text, today)
        fix: dict = {k: getattr(rules, k) for k in ("day", "time_from", "time_to", "choice")
                     if getattr(rules, k) is not None}
        said = re.sub(r"(перв|втор)\w*\s+половин\w*", " ", text.lower())
        if update.choice is not None and rules.choice is None and not _CHOICE_HINT.search(said):
            fix["choice"] = None  # модель «выбрала» вариант, хотя пациент ничего не выбирал — не верим
        if update.topic == "price" and rules.topic is None and not re.search(FAQ_BY_KEY["price"].pattern, said):
            update = update.model_copy(update={"topic": None})  # «как записаться?» после вопроса о цене — не цена
        if update.intent is None and rules.intent and not (update.topic or update.red_flag) and not state.intent:
            fix["intent"] = rules.intent  # «Мне к врачу.» — модель вернула пустоту, правила видят запись
        if update.specialty is None and rules.specialty and not update.doctor_name:
            fix["specialty"] = rules.specialty
        for k in IDENTITY_FIELDS:  # ФИО и дату рождения модель не видит (обезличено) — берём из правил
            if getattr(update, k) is None and getattr(rules, k) is not None:
                fix[k] = getattr(rules, k)
        if rules.reject is not None or rules.avoid_day:  # «первый не подходит»: модель ставит выбор — верим правилам
            fix |= {"choice": None, "reject": rules.reject, "other_day": rules.other_day, "avoid_day": rules.avoid_day}
            if rules.avoid_day and update.day == rules.avoid_day:
                fix["day"] = None
        if (update.day is not None and rules.day is None and not update.birth_date
                and (_awaits_identity(state) or _correcting_identity(state, text.lower()))
                and not re.search(_APPOINTMENT_DAY, text.lower())
                and (rules.last_name or rules.first_name or update.last_name or update.first_name)):
            fix["day"] = None  # ФИО и «двенадцатое марта» без года — неполная дата рождения, а не день приёма
        if (update.intent == "reschedule" and state.intent == "book" and rules.intent != "reschedule"
                and not re.search(r"перенес|перезапис|сдвин|поменя", text.lower())):
            fix["intent"] = None  # «мне нужен другой день» при выборе варианта — не перенос старой записи
        if rules.bye and rules.intent is None and update.intent:
            fix["intent"] = None  # «позвоню позже» — не новая цель, даже если модель её придумала
        if rules.bye and not update.bye:
            fix["bye"] = True  # «нет, спасибо, до свидания» — модель вернула только «нет»
        if rules.red_flag in ("medical", "dental") and update.red_flag == "urgent":
            # «какие таблетки выпить» — медицинский вопрос, «болит зуб, опухла щека» — стоматология, а не общее острое
            fix |= {"red_flag": rules.red_flag, "note": rules.note}
        if rules.red_flag in ("pdn", "lab_delay") and update.red_flag in (None, "admin", "medical", "human"):
            fix |= {"red_flag": rules.red_flag, "note": rules.note}  # ученик знает только «admin» — ПДн различают правила
        if rules.red_flag in ("emergency", "selfharm") and update.red_flag != rules.red_flag:
            fix |= {"red_flag": rules.red_flag, "note": rules.note}  # угроза жизни важнее всего остального
        elif update.red_flag is None and rules.red_flag:
            fix |= {"red_flag": rules.red_flag, "note": rules.note}
        if (state.awaiting_confirm and update.confirm is True and rules.confirm is not True
                and not _AGREE.search(text.lower().replace("ё", "е"))):
            fix["confirm"] = None  # подтверждения в словах пациента нет — повторим вопрос, а не запишем
        if update.confirm is None and rules.confirm is not None and (
                state.awaiting_confirm or state.question or state.intent == "schedule"):
            fix["confirm"] = rules.confirm  # «да» на прямой вопрос модель иногда пропускает
        if state.question == "will_come" and rules.confirm is False and update.intent == "cancel" and rules.intent is None:
            fix |= {"intent": None, "confirm": False}  # «не смогу прийти» на напоминание — перенос, а не отмена
        # Тему справки правила узнают по точным формулировкам регламента — им верим больше, чем модели
        topic = rules.topic or (update.topic if update.topic in FAQ_BY_KEY else None)
        if (topic == "gratitude" and (state.awaiting_confirm or state.question)
                and fix.get("confirm", update.confirm) is not None):
            topic = None  # «Да, спасибо» на вопрос — ответ «да», а не благодарность для передачи
        if topic == "robot" and fix.get("red_flag", update.red_flag) == "human" and not _ASKS_HUMAN.search(text.lower()):
            fix["red_flag"] = None  # «Вы робот или живой человек?» — вопрос, а не просьба соединить
        if topic != update.topic:
            fix["topic"] = topic
        if topic:
            if (fix.get("red_flag", update.red_flag) == "medical") and rules.red_flag != "medical":
                fix["red_flag"] = None  # «нормальные ли анализы» — справка с граничной фразой
            if not fix.get("red_flag", update.red_flag):
                fix["note"] = text[:300]  # по тексту справка уточняет услугу или анализ
        if state.question == "offer_book" and re.search(r"как (можно |мне |бы )?(записаться|попасть на при)", said):
            fix |= {"confirm": None, "intent": "book"}  # «Записать вас?» — «нет, просто скажите, как записаться»
        return update.model_copy(update=fix) if fix else update

    def _normalize_specialty(self, raw: str) -> str | None:
        """Приводит «к лору», «гинеколог» к специальности из справочника (у стенда их 40)."""
        key = raw.strip().lower().replace("ё", "е")
        candidates = list(dict.fromkeys(c for c in (normalize_specialty(key), key) if c))
        for c in candidates:
            if c in self.specialties:
                return c
            preferred = _PREFERRED_SPECIALTY.get(c)
            if preferred in self.specialties:
                return preferred
        adult = [s for s in self.specialties if not s.startswith("детск")]
        for c in candidates:
            pool = self.specialties if "детск" in key else adult
            hits = [s for s in pool if c in s] or [s for s in self.specialties if c in s]
            if hits:
                return min(hits, key=len)
        if key.startswith("детский "):
            return self._normalize_specialty(key.removeprefix("детский "))  # детского врача в клинике нет
        return next(iter(difflib.get_close_matches(key, self.specialties, n=1, cutoff=0.75)), None)

