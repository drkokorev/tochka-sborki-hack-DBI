"""Нормализация транскриптов речи до разбора.

Распознавание речи ошибается предсказуемо: пишет слова-паразиты («э», «ну», «как его»),
склеивает предлог со словом («клору» вместо «к лору») и путает буквы в редких словах
(«тирапевту», «Белавой»). Чиним это словарём специальностей и фамилий врачей, без обучения моделей.
"""

from __future__ import annotations

import difflib
import re

_FILLERS = re.compile(
    r"\b(э+(?:-э+)*|эм+|м+|ну|вот|как его|как е[её]|это самое|короче|типа|блин|в общем|значит)\b[,.]?",
    re.IGNORECASE,
)
_SPECIALTY_WORDS = [
    "терапевт", "стоматолог", "дантист", "лор", "отоларинголог", "оториноларинголог",
    "офтальмолог", "окулист", "невролог", "невропатолог", "кардиолог", "дерматолог", "педиатр",
]
_CASE_ENDINGS = ["", "у", "а", "ом", "ам"]
# Фамилию врача исправляем только там, где её и ждём: «к Белавой», «когда принимает Гусив».
# Иначе «голова» превращается в «Орлова», а пациентка Смирнова — во врача Миронову.
_DOCTOR_CONTEXT = {"к", "ко", "у", "доктор", "доктору", "доктора", "врач", "врачу", "врача", "принимает", "принимают"}


def _surname_forms(surname: str) -> list[str]:
    s = surname.lower().replace("ё", "е")
    if s.endswith("а"):  # Белова → Беловой, Белову
        return [s, s[:-1] + "ой", s[:-1] + "у"]
    return [s, s + "у", s + "а", s + "ым"]  # Орлов → Орлову, Орлова, Орловым


# ---------- числа словами → цифры ----------
# Распознавание речи часто пишет числа словами: «двенадцатое марта пятьдесят восьмого года»,
# «в одиннадцать тридцать», «после пятнадцати». Правила дат и времени понимают цифры.

_CARD = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200, "триста": 300,
    "четыреста": 400, "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
    # родительный падеж: «после пятнадцати», «до двух», «с девяти»
    "одного": 1, "двух": 2, "трех": 3, "четырех": 4, "пяти": 5, "шести": 6, "семи": 7, "восьми": 8, "девяти": 9,
    "десяти": 10, "одиннадцати": 11, "двенадцати": 12, "тринадцати": 13, "четырнадцати": 14, "пятнадцати": 15,
    "шестнадцати": 16, "семнадцати": 17, "восемнадцати": 18, "девятнадцати": 19, "двадцати": 20, "тридцати": 30,
}
_ORD_STEMS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9,
    "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14, "пятнадцат": 15,
    "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19, "двадцат": 20, "тридцат": 30,
    "сороков": 40, "пятидесят": 50, "шестидесят": 60, "семидесят": 70, "восьмидесят": 80, "девяност": 90,
    "сот": 100, "тысячн": 1000, "двухтысячн": 2000,
}
_ORD_END = r"(?:ый|ой|ий|ого|его|ому|ему|ым|ом|ая|ую|ое|ее|ье|ья|ьего|ьему|ьей|ей|ьи)"
_ORD_RE = re.compile(r"^(" + "|".join(sorted(_ORD_STEMS, key=len, reverse=True)) + r")ь?" + _ORD_END + "$")
_MONTH_RE = re.compile(r"^(январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)")
_MINUTES = {"ноль-ноль": "00", "ноль ноль": "00"}


def _number_word(word: str) -> tuple[int, bool] | None:
    """Значение слова-числа и признак порядкового («двенадцатого»)."""
    w = word.lower().replace("ё", "е")
    if w in _CARD:
        return _CARD[w], False
    m = _ORD_RE.match(w)
    if m:
        return _ORD_STEMS[m.group(1)], True
    return None


def _combine(values: list[int]) -> int | None:
    """[1000, 900, 50, 8] → 1958; [20, 2] → 22. None — не одно число («двенадцать тридцать»)."""
    if len(values) == 2 and values[0] == 0:
        return values[1]  # «ноль пять» — минуты
    total, chunk, last = 0, 0, None
    for v in values:
        if v == 1000:
            total, chunk, last = total + (chunk or 1) * 1000, 0, 1000
            continue
        if last is not None and last != 1000 and not (last >= 100 > v or (last >= 20 and last % 10 == 0 and v < 10)):
            return None
        chunk += v
        last = v
    return total + chunk


def words_to_digits(text: str) -> str:
    """«двенадцатое марта тысяча девятьсот пятьдесят восьмого» → «12 марта 1958»; «в одиннадцать тридцать» → «в 11:30».

    Порядковые числа меняем только там, где это дата (рядом с месяцем или «года»): «второй» как выбор
    варианта остаётся словом. Одиночные «один/два/три» без предлога тоже не трогаем — это часто выбор.
    """
    for k, v in _MINUTES.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)

    def half_past(m: re.Match) -> str:  # «в половине десятого» → «в 9:30»
        nv = _number_word(m.group(2))
        return f"{m.group(1)}{nv[0] - 1}:30" if nv and nv[1] and 1 < nv[0] <= 21 else m.group(0)

    text = re.sub(r"(\b(?:в|на)\s+)?половин[аеуы]\s+([а-яё]+)", half_past, text, flags=re.IGNORECASE)
    tokens = re.split(r"(\s+)", text)
    words = [(i, t) for i, t in enumerate(tokens) if t.strip()]
    out = list(tokens)
    j = 0
    while j < len(words):
        run, ordinal = [], False
        k = j
        while k < len(words) and (nv := _number_word(words[k][1].strip(",.?!"))) is not None and not ordinal:
            run.append(nv[0])
            ordinal = nv[1]
            k += 1
        if not run:
            j += 1
            continue
        # «двенадцать тридцать» — часы и минуты: два числа подряд
        value, split = _combine(run), None
        if value is None:
            for cut in range(1, len(run)):
                a, b = _combine(run[:cut]), _combine(run[cut:])
                if a is not None and b is not None:
                    value, split = a, b
                    break
        if value is None:
            j = k
            continue
        prev = words[j - 1][1].lower().strip(",.") if j else ""
        nxt = words[k][1].lower().strip(",.?!") if k < len(words) else ""
        date_context = bool(_MONTH_RE.match(nxt) or nxt.startswith("год") or nxt == "г" or _MONTH_RE.match(prev)
                            or nxt == "числа")
        if ordinal and not date_context:
            if not (prev == "на" and value <= 31 and re.search(r"(о|ь)?е$", words[k - 1][1].lower())):
                j = k
                continue
            digits = f"{value}-е"  # «на двадцать второе» — число месяца
        elif not ordinal and split is None and len(run) == 1 and value <= 3 and prev not in (
                "в", "на", "до", "после", "с", "к", "около") and not nxt.startswith(("час", "минут")):
            j = k
            continue
        else:
            digits = str(value)
            if split is not None:
                digits = f"{value}:{split:02d}" if value <= 23 and split <= 59 else f"{value} {split}"
        first, last = words[j][0], words[k - 1][0]
        tail = re.search(r"[,.?!]+$", words[k - 1][1])
        out[first] = digits + (tail.group(0) if tail else "")
        for idx in range(first + 1, last + 1):
            out[idx] = ""
        j = k
    text = "".join(out)
    # «в 11 30» (минуты сказаны отдельно после цифр) → «в 11:30»
    text = re.sub(r"\b(в|на|после|до|с|к)\s+(\d{1,2})\s+(00|05|10|15|20|25|30|35|40|45|50|55)\b(?!\s*(?:\d|год|г\b))",
                  r"\1 \2:\3", text)
    return re.sub(r"\s{2,}", " ", text)


def strip_fillers(text: str) -> str:
    text = re.sub(r"\b[лЛ][оО]{2,}[рР]", "лор", text)  # GigaAM растягивает гласную: «лоОР-врачу»
    text = words_to_digits(text)
    # Год рождения из распознавания: «178 го» (= 1978) — к виду, который понимают правила
    text = re.sub(r"\b1([0-9]{2})(?:\s*-?го)\b", r"19\1", text)
    text = _FILLERS.sub(" ", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"([,.?!])(?:\s*[,.])+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,.")


class SpeechNormalizer:
    def __init__(self, doctor_surnames: list[str]):
        self._specialties = sorted({w + e for w in _SPECIALTY_WORDS for e in _CASE_ENDINGS})
        self._surnames = sorted({f for s in doctor_surnames for f in _surname_forms(s)})
        self._known = set(self._specialties) | set(self._surnames)

    def fix(self, text: str) -> str:
        """Текст в нижнем регистре → тот же текст с исправленными словами словаря."""
        pieces, last, prev = [], 0, ""
        for m in re.finditer(r"[а-я]+(?:-[а-я]+)*", text):
            pieces += [text[last:m.start()], self._fix_word(m.group(0), surnames_ok=prev in _DOCTOR_CONTEXT)]
            prev, last = m.group(0), m.end()
        pieces.append(text[last:])
        return "".join(pieces)

    def _fix_word(self, word: str, surnames_ok: bool) -> str:
        if len(word) < 4 or word in self._known:
            return word
        # Склейка с предлогом: «клору» → «к лору», «ктерапевту» → «к терапевту»
        for prefix in ("ко", "к"):
            rest = word[len(prefix):]
            if word.startswith(prefix) and rest in self._known:
                return f"{prefix} {rest}"
        if hit := self._fuzzy(word, surnames_ok):
            return hit
        for prefix in ("ко", "к"):
            rest = word[len(prefix):]
            if word.startswith(prefix) and (hit := self._fuzzy(rest, surnames_ok=True)):
                return f"{prefix} {hit}"
        return word

    def _fuzzy(self, word: str, surnames_ok: bool) -> str | None:
        if len(word) < 5:  # короткие слова исправлять опасно
            return None
        if hits := difflib.get_close_matches(word, self._specialties, n=1, cutoff=0.8):
            return hits[0]
        if surnames_ok:
            hits = difflib.get_close_matches(word, self._surnames, n=3, cutoff=0.8)
            return next((h for h in hits if h[0] == word[0]), None)
        return None
