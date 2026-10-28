"""Обезличивание реплик перед отправкой во внешнюю LLM (152-ФЗ, блок 3 кейса).

ФИО и дата рождения пациента — персональные данные: во внешнюю модель они не уходят. Их достают правила
на машине (rules.py), а LLM получает текст, где вместо них стоят метки «[ФИО]» и «[дата рождения]».
Модели они и не нужны: её задача — понять цель, врача, день и время. Фамилии врачей не маскируются —
это справочник клиники, а не данные пациента.
"""

from __future__ import annotations

import re
from typing import Callable

from .rules import _FIRST_NAMES
from .speech import words_to_digits

_MONTHS = r"(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*"
# Дата рождения: с годом (дата приёма года не содержит) — «12.03.1958», «12 марта 1958 года», «1958 года рождения»
_BIRTH = [
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-](?:19|20)?\d{2}\b"),
    re.compile(r"\b\d{1,2}(?:-?го|-?е)?\s+" + _MONTHS + r"\s+(?:19|20)?\d{2}(?:\s+года?\b|\s*-?го\b)?", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\s*(?:-?го\s*)?(?:года?\s+)?(?:рождения|г\.?\s*р\.?)", re.IGNORECASE),
    re.compile(r"\b(?:родил(?:ся|ась)|дата рождения)[\s:,]+[^,.;!?]{2,30}", re.IGNORECASE),
]
_SURNAME = re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:ов|ова|ев|ева|ёв|ёва|ин|ина|ын|ына|ский|ская|цкий|цкая|енко|ук|юк|их|ых|"
                      r"дзе|швили|ян|янц|ман|берг|штейн)\b")
_COMMON = {"один", "одна", "мужчина", "женщина", "причина", "половина", "машина", "витамин", "магазин", "плохих",
           "других", "каких", "таких", "всех", "них", "своих", "моих", "ваших", "наших", "больших", "малых", "новых",
           "старых", "острых", "детских", "платных", "Готов", "готова", "здоров", "здорова", "основ"}
# «Петрович», «Ильич» — с заглавной; строчными и в падежах — только явные окончания отчеств («петровича», «ивановне»),
# чтобы не задеть «паралич» или «кирпич»
_PATRONYMIC = re.compile(r"\b[А-ЯЁ][а-яё]+(?:ич|овна|евна|ична|инична)\b|"
                         r"\b[А-ЯЁа-яё][а-яё]*(?:ович|евич|овн|евн|ичн|иничн)(?:а|у|ем|ом|е|ы|ой|ну|ны|не)?\b")
# Телефон, полис ОМС (16 цифр), СНИЛС, паспорт — любая цепочка от 6 цифр (время 10:30 и номер записи B0004 короче)
_NUMBER = re.compile(r"(?<![\w.])(?:\+7|8)?[\s(-]*\d(?:[\s()-]*\d){5,}(?![\w])")
_NEAR_SKIP = {"запишите", "запиши", "записать", "меня", "зовут", "это", "мой", "моя", "моего", "мою",
                              "сына", "дочь", "дочку", "маму", "папу", "мужа", "жену", "ребенка", "ребёнка", "пожалуйста",
                              "здравствуйте", "добрый", "день", "привет", "алло", "отменить", "перенести", "к", "на"}


def _first_name(word: str) -> bool:
    """Имя в любом падеже: «ивана», «ивану», «анной» — по основе из списка частых имён."""
    w = word.lower().replace("ё", "е")
    if w in _FIRST_NAMES:
        return True
    return any(w.endswith(end) and w[: len(w) - len(end)] + base in _FIRST_NAMES
               for end in ("а", "у", "ом", "е", "ы", "ой", "ей", "ю", "и") for base in ("", "а", "я", "й", "ь"))
_INTRO = re.compile(r"((?:меня зовут|моя фамилия|фамилия|на имя|пациент(?:ка)?)\s*[:,]?\s+)"
                    r"([А-ЯЁа-яё-]+(?:\s+[А-ЯЁа-яё-]+){0,2})", re.IGNORECASE)
_NOT_PD = {"я", "вы", "мне", "к", "на", "запись", "записаться", "врач", "терапевт", "хочу", "да", "нет", "по", "в"}

NAME, BIRTH = "[ФИО]", "[дата рождения]"


def mask(text: str, known: tuple[str, ...] = (), is_doctor: Callable[[str], object] | None = None) -> str:
    """Текст без ФИО и даты рождения пациента. known — уже известные фамилия и имя (из состояния звонка)."""
    if not text:
        return text
    t = words_to_digits(text)
    for pattern in _BIRTH:
        t = pattern.sub(BIRTH, t)
    for value in known:
        if value and len(value) >= 2:
            t = re.sub(rf"\b{re.escape(value)}\w*", NAME, t, flags=re.IGNORECASE)

    def intro(m: re.Match) -> str:
        words = m.group(2).split()
        if words and words[0].lower() in _NOT_PD:
            return m.group(0)
        return m.group(1) + NAME

    t = _INTRO.sub(intro, t)
    t = _PATRONYMIC.sub(NAME, t)

    def word(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() in _COMMON:
            return w
        if is_doctor is not None and is_doctor(w):
            return w  # «к Беловой» — врач, а не пациент
        return NAME

    t = _SURNAME.sub(word, t)
    t = re.sub(r"\b[А-ЯЁа-яё]+\b", lambda m: NAME if _first_name(m.group(0)) else m.group(0), t)
    # Фамилия рядом с именем или отчеством, даже строчными (так её отдаёт распознавание): «иванова [ФИО]»
    tokens = re.split(r"(\s+|,)", t)
    for i, tok in enumerate(tokens):
        if tok != NAME:
            continue
        for j in (i - 2, i + 2):
            if 0 <= j < len(tokens) and re.fullmatch(r"[А-ЯЁа-яё-]{3,}", tokens[j] or ""):
                w = tokens[j].lower()
                if w not in _NEAR_SKIP and w not in _NOT_PD and not (is_doctor is not None and is_doctor(tokens[j])):
                    tokens[j] = NAME
    t = "".join(tokens)
    t = _mask_rare_names(t, is_doctor)
    t = _NUMBER.sub("[номер]", t)
    t = re.sub(r"(?:\[ФИО\][\s,]*){2,}", NAME + " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


# ---------- редкие имена строчными: консервативное правило (первый рубеж) ----------
# Распознавание речи отдаёт ФИО строчными, а редкое имя («зарина ужахова») не попадает ни в список имён, ни в шаблон
# фамилии с заглавной. Маскировать любое слово на «-ова» нельзя («голова», «снова»), поэтому такие реплики вообще
# не уходят во внешнюю LLM: если рядом стоят два незнакомых слова и одно похоже на фамилию — разбирают правила на машине.
# Только надёжные окончания фамилий: мужские «-ов», «-ин» и прилагательные на «-ский» слишком часто встречаются
# в обычных словах («анализов», «инсулин», «медицинский»); фамилию рядом с известным именем ловит mask()
_SURNAME_TAIL = re.compile(r"(?:ова|ева|ёва|ина|ына|енко|швили|дзе|янц|ян)$")
_COMMON_TAIL = {"причина", "половина", "спина", "машина", "мужчина", "женщина", "глубина", "ангина", "витамина",
                "магазина", "вызова", "голова", "слова", "снова", "основа", "корова", "здорова", "готова", "раствора",
                "перерыва", "часова", "дома", "вина", "лина", "картина", "середина", "вершина", "ширина", "длина",
                "тишина", "резина", "пенициллина", "инсулина", "аспирина", "кофеина", "ибупрофена", "кабина", "клиника",
                "доктора", "диагноза", "приёма", "приема", "журнала", "талона", "полиса", "паспорта", "района",
                "района", "сезона", "телефона", "номера", "адреса", "забора", "раза", "плана", "участкова", "сына",
                "врача", "листа", "вопроса", "анализа", "осмотра", "рецепта", "укола", "кашля", "насморка", "бассейна",
                "медосмотра", "рентгена", "кардиолога", "терапевта", "невролога", "хирурга", "педиатра", "уролога",
                "гинеколога", "окулиста", "стоматолога", "лора", "дерматолога", "эндокринолога"}
_VOCAB: set[str] | None = None


def _vocabulary() -> set[str]:
    """Слова, которые встречаются в обычных репликах пациентов: фразник, справки, проверочный набор и корпус ученика
    (если он есть на машине). Слова, написанные в источнике с заглавной (имена в примерах), не берём."""
    global _VOCAB
    if _VOCAB is not None:
        return _VOCAB
    import json
    from pathlib import Path

    from . import faq, phrasebook
    words: set[str] = set()
    for text in list(phrasebook.TABLE) + [f.answer + " " + f.hint for f in faq.FAQS]:
        words |= set(re.findall(r"\b[а-яё]+\b", text))
    root = Path(__file__).resolve().parents[2]
    for path in (root / "eval/nlu_test.jsonl", root / "data/distill/corpus.jsonl"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    words |= set(re.findall(r"\b[а-яё]+\b", json.loads(line).get("text", "")))
    _VOCAB = {w.replace("ё", "е") for w in words}
    return _VOCAB


def _name_pairs(text: str, is_doctor: Callable[[str], object] | None) -> list[tuple[str, str]]:
    vocab = _vocabulary()
    words = re.findall(r"[А-ЯЁа-яё]+", text)

    def unknown(w: str) -> bool:
        w = w.lower().replace("ё", "е")
        return len(w) >= 3 and w not in vocab and w not in _NOT_PD and w not in _NEAR_SKIP \
            and not (is_doctor is not None and is_doctor(w))

    pairs = []
    for a, b in zip(words, words[1:]):
        for x, y in ((a, b), (b, a)):
            if (len(x) >= 4 and _SURNAME_TAIL.search(x.lower()) and x.lower().replace("ё", "е") not in _COMMON_TAIL
                    and unknown(x) and (unknown(y) or _first_name(y))):
                pairs.append((a, b))
                break
    return pairs


def suspicious_name(text: str, is_doctor: Callable[[str], object] | None = None) -> bool:
    """Похоже ли, что в реплике незнакомые имя и фамилия (тогда — только правила на машине, без LLM)."""
    return bool(_name_pairs(text, is_doctor))


def _mask_rare_names(text: str, is_doctor: Callable[[str], object] | None) -> str:
    """Второй рубеж для истории разговора: такие пары слов заменяем меткой и в уже сказанных репликах."""
    for a, b in _name_pairs(text, is_doctor):
        text = re.sub(rf"\b{re.escape(a)}\s+{re.escape(b)}\b", NAME, text)
    return text
