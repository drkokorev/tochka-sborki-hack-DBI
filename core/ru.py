"""Русское форматирование дат, денег и числительных."""

from datetime import date

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
WEEKDAYS_DAT_PL = ["понедельникам", "вторникам", "средам", "четвергам", "пятницам", "субботам", "воскресеньям"]
WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
MONTHS_GEN = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def fmt_date(d: date) -> str:
    return f"{WEEKDAYS_SHORT[d.weekday()]}, {d.day} {MONTHS_GEN[d.month - 1]}"


def fmt_date_long(d: date) -> str:
    """Для голоса: синтезатор плохо читает сокращения «вт», «пн»."""
    return f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS_GEN[d.month - 1]}"


def fmt_money(amount: int) -> str:
    return f"{amount:,}".replace(",", " ") + " ₽"


def plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]


# ---------- числа и время словами (для голоса: регламент требует «в девять тридцать») ----------

_UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
          "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
          "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят"]
_UNITS_GEN = ["", "одного", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти", "десяти",
              "одиннадцати", "двенадцати", "тринадцати", "четырнадцати", "пятнадцати", "шестнадцати",
              "семнадцати", "восемнадцати", "девятнадцати", "двадцати"]
_ORD_GEN = ["", "первого", "второго", "третьего", "четвёртого", "пятого", "шестого", "седьмого", "восьмого",
            "девятого", "десятого", "одиннадцатого", "двенадцатого", "тринадцатого", "четырнадцатого",
            "пятнадцатого", "шестнадцатого", "семнадцатого", "восемнадцатого", "девятнадцатого", "двадцатого"]
WEEKDAYS_ACC = ["в понедельник", "во вторник", "в среду", "в четверг", "в пятницу", "в субботу", "в воскресенье"]


def number_words(n: int) -> str:
    """0–59 словами: для часов и минут."""
    if n < 20:
        return _UNITS[n]
    tens, units = divmod(n, 10)
    return _TENS[tens] + (f" {_UNITS[units]}" if units else "")


def hour_gen(h: int) -> str:
    """«с восьми до двадцати одного»."""
    return _UNITS_GEN[h] if h <= 20 else "двадцати " + _UNITS_GEN[h - 20]


def time_words(t) -> str:
    """9:30 → «девять тридцать», 16:00 → «шестнадцать часов», 10:05 → «десять ноль пять»."""
    if t.minute == 0:
        return f"{number_words(t.hour)} {plural(t.hour, 'час', 'часа', 'часов')}"
    minutes = f"ноль {_UNITS[t.minute]}" if t.minute < 10 else number_words(t.minute)
    return f"{number_words(t.hour)} {minutes}"


def day_ordinal(day: int) -> str:
    if day <= 20:
        return _ORD_GEN[day]
    if day == 30:
        return "тридцатого"
    return ("двадцать " if day < 30 else "тридцать ") + _ORD_GEN[day % 10]


def on_date(d: date) -> str:
    """«в четверг, первого октября» — дата с днём недели, как в регламенте."""
    return f"{WEEKDAYS_ACC[d.weekday()]}, {day_ordinal(d.day)} {MONTHS_GEN[d.month - 1]}"


_TENS_ORD_GEN = ["", "", "двадцатого", "тридцатого", "сорокового", "пятидесятого", "шестидесятого", "семидесятого",
                 "восьмидесятого", "девяностого"]
_TENS_ALL = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]


def year_gen(y: int) -> str:
    """1978 → «тысяча девятьсот семьдесят восьмого», 2017 → «две тысячи семнадцатого» (для «… года»)."""
    head, rest = ("тысяча девятьсот", y - 1900) if y < 2000 else ("две тысячи", y - 2000)
    if rest == 0:
        return "тысяча девятисотого" if y < 2000 else "двухтысячного"
    if rest <= 20:
        tail = _ORD_GEN[rest]
    elif rest % 10 == 0:
        tail = _TENS_ORD_GEN[rest // 10]
    else:
        tail = f"{_TENS_ALL[rest // 10]} {_ORD_GEN[rest % 10]}"
    return f"{head} {tail}"


def birth_words(d: date) -> str:
    """«двенадцатое марта тысяча девятьсот семьдесят восьмого года» — дата рождения для голоса."""
    day = day_ordinal(d.day)  # «двадцать третьего» → «двадцать третье», «двенадцатого» → «двенадцатое»
    day = day[:-2] if day.endswith("третьего") else day[:-3] + "ое"
    return f"{day} {MONTHS_GEN[d.month - 1]} {year_gen(d.year)} года"


_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def cardinal(n: int) -> str:
    """0–999 словами: номера кабинетов («кабинет двести четырнадцать»)."""
    if n < 60:
        return number_words(n)
    hundreds, rest = divmod(n, 100)
    if rest < 20:
        tail = _UNITS[rest] if rest else ""
    else:
        tens, units = divmod(rest, 10)
        tail = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят",
                "девяносто"][tens] + (f" {_UNITS[units]}" if units else "")
    return " ".join(p for p in (_HUNDREDS[hundreds], tail) if p)
