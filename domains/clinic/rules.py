"""Детерминированный разбор реплик пациента. Fallback для LLM; на нём же работает демо без ключа."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

from core.ru import MONTHS_GEN

from .faq import faq_topic
from .speech import SpeechNormalizer, strip_fillers

if TYPE_CHECKING:
    from .models import ClinicState, ClinicUpdate
    from .schedule import Schedule

_INTENTS = [
    ("reschedule", r"перенес|перенест|поменять время|другое время|сдвинуть"),
    ("cancel", r"отмен|не смогу прийти|не приду"),
    ("status", r"статус|\b(мо|сво)(я|ю|ей|и|их|ем)\s+(\w+\s+)?запис|(во сколько|когда)\b.{0,15}\bмо\w* при[её]м|на когда я|когда у меня|провер\w*.{0,12}запис|есть ли у меня запис|узна\w*.{0,15}запис"),
    ("schedule", r"расписан|когда принимает|в какие дни|часы при[её]ма|когда работает|график"),
    ("book", r"запис|запиш|на при[её]м|попасть к|\bко? врачу|к доктору"),
]

# Сначала ищем явно названного специалиста, потом — по жалобам
_EXPLICIT = [
    ("терапевт", r"терапевт"),
    ("стоматолог", r"стоматолог|дантист|зубно\w* врач"),
    ("оториноларинголог", r"\bлор\w{0,2}\b|отоларинголог|оториноларинголог|ухо-горло-нос"),
    ("офтальмолог", r"офтальмолог|окулист|глазно\w* врач"),
    ("невролог", r"невролог|невропатолог"),
    ("кардиолог", r"кардиолог"),
    ("дерматолог", r"дерматолог|кожник"),
    ("педиатр", r"педиатр|детск\w* врач"),
]
_SYMPTOMS = [
    ("стоматолог", r"\bзуб|десн"),
    ("оториноларинголог", r"\bух[оа]\b|\bуш[иа]|насморк|заложен\w* нос"),
    ("офтальмолог", r"глаз|зрени"),
    ("кардиолог", r"сердц|сердеч|давлени"),
    ("дерматолог", r"\bкож|сыпь|родинк|прыщ|зуд"),
    ("невролог", r"голов\w* бол|бол\w* голов|мигрен|\bспин|поясниц|онемен"),
    # «моя дочь», «для сына» — да; «дочка», «сынок» как обращение к ассистенту — нет
    ("педиатр", r"ребен|ребён|\bсын(а|у|ом)?\b|\bдоч(ь|ери|ку|ке|кой)\b"),
    ("терапевт", r"температур|простуд|кашел|кашл|горл|слабост|орви|грипп|больничн|справк|заболел"),
]

_WEEKDAYS = [r"понедельник", r"вторник", r"\bсред[уаы]\b", r"четверг", r"пятниц", r"суббот", r"воскресен"]
_ORDINALS = {
    1: r"\bперв(ый|ого|ому|ым|ая|ую|ое)\b|\bодин\b",
    2: r"\bвтор(ой|ого|ому|ым|ая|ую|ое)\b|\bдва\b",
    3: r"\bтреть?(ий|его|ему|им|ья|ью|е)\b|\bтри\b",
}
_YES = r"\b(да|ага|угу|конечно|давайте|подтверждаю|верно|хорошо|согласен|согласна)\b"
_NO = r"\b(нет|не надо|не нужно|не отменяйте|оставьте)\b"
_REJECT = r"\bне (подход|подойд|устраива|удобн|получится|смогу|могу)|\bнеудобн|\bне то\b"  # «мне подходит» — не отказ
_NAME = r"(?i:меня зовут|моя фамилия|фамилия|на имя|пациент[а-я]*)[\s,:]+([А-ЯЁа-яё-]+(?:[\s,]+[А-ЯЁа-яё-]+){0,2})"
_NOT_NAME = {"и", "я", "а", "хочу", "мне", "на", "к", "запишите", "пожалуйста", "записаться", "нужно", "надо",
             "год", "года", "г", "рождения", "родился", "родилась", "дата", "меня", "зовут", "это", "мой", "моя"}
_NOT_NAME |= set(MONTHS_GEN)
# Ответ не про имя: день, время, номер варианта, «не знаю» — даже когда ждём фамилию
_NOT_NAME_CONTEXT = (r"понедельн|вторник|\bсред[уаы]\b|четверг|пятниц|суббот|воскресен|завтра|сегодня|\bутр|"
                     r"вечер|после обеда|\b(перв|втор|трет|последн)\w*\b|вариант|\bне знаю\b|\bчто\b|\bкак\b|"
                     r"\bзачем\b|\bпочему\b|\bгде\b|\bкогда\b|\bсколько\b")
# Частые имена: по ним понимаем порядок «Иван Петров» / «Петров Иван»
_FIRST_NAMES = set("""александр алексей анатолий андрей антон аркадий артем артемий борис вадим валентин валерий
василий виктор виталий владимир владислав вячеслав геннадий георгий глеб григорий даниил данил денис дмитрий евгений
егор иван игорь илья кирилл константин лев леонид максим марк матвей михаил никита николай олег павел петр роман
руслан семен сергей станислав степан тимофей тимур федор юрий ярослав александра алена алина алла анастасия анна
антонина валентина валерия вера вероника виктория галина дарья диана ева евгения екатерина елена елизавета жанна
зинаида зоя инна ирина карина кира кристина ксения лариса лидия любовь людмила маргарита марина мария надежда
наталья наталия нина оксана олеся ольга полина раиса светлана софия софья таисия тамара татьяна ульяна юлия яна
рустам ринат марат азат айдар ильдар булат дамир самир артур эльдар ильнур фарид рамиль альберт эмиль ислам магомед
ахмед шамиль арсен давид эрик айрат ильшат радик гульнара лейла алия зарина камила эльвира лилия айгуль динара регина
альбина эльмира гузель асель милана аиша амина мадина залина фатима""".split())

# «Красные флаги» по словарю регламента (раздел 7.4) и матрице маршрутизации (7.3). Порядок — приоритет.
_RED_FLAGS = [
    ("selfharm", r"не хочу жить|жить не хоч|не хочется жить|хочу умереть|лучше бы (я )?умер|покончить с собой|"
                 r"убить себя|уйти из жизни|свести счеты|суицид|наглотал\w* таблет|угрожает убить|порезать себя"),
    # Категория А — возможная угроза жизни (R-01)
    ("emergency", r"бол\w* в груди|давит (в груди|за грудин)|плохо с сердцем|сердце (болит|прихват\w*|щемит|колет)|бол\w* сердце|жж\w* в груди|отда\w* в (левую )?руку|сердечн\w* приступ|"
                  r"инфаркт|не (может|могу) дышать|задыха|трудно дышать|синеет|посинел|хрипит|подавил|"
                  r"потерял\w* сознани|без сознания|не приходит в себя|не могу разбудить|обморок|перекосил|"
                  r"отнял\w* (рук|ног)|не может говорить|речь невнятн|онемел\w* половин|инсульт|судорог|припад|"
                  r"кровотеч|кровь не останавлива|рвот\w* кровью|кровь изо рта|\bдтп\b|авари|ударил\w* голов|"
                  r"упал\w* с высоты|ожог|отравил|проглотил|выпил\w* (таблет|хими|уксус)|укусил\w* змея|"
                  r"отека\w* (горло|лицо)|отек\w* (горла|лица)|не (может|могу) глотать|анафилак|младен\w* вял|"
                  r"грудн\w* ребен\w*.{0,20}температур|отошли воды|схватк|скор(ую|ая) помощ|вызовите скорую"),
    # Острая зубная боль, травма зуба, отёк (R-09)
    ("dental", r"(остр|сильн|невыносим)\w* (зубн\w* )?бол\w*.{0,15}зуб|(сильно|очень|жутко) болит зуб|"
               r"зуб\w* (сильно|очень|жутко) болит|зубн\w* бол|флюс|(раздуло|опухл\w*|отек\w*) (щек|десн)|"
               r"(выбил|сломал|откол)\w* зуб|травм\w* зуб"),
    # Категория Б — острое состояние без явной угрозы жизни (R-11)
    ("urgent", r"сильн\w* бол|невыносим|остр\w* бол|бол\w* усилива|температур\w* (сорок|40|не сбива)|не сбивается|"
               r"температур\w*\s+(под\s+)?(39|40|41|тридцать девять|сорок)|"
               r"температур.{0,30}(трет\w*|четверт\w*|пят\w*|несколько|[3-9]) (день|дня|дней|сут)|"
               r"(стало|становится) (намного |гораздо |еще )?хуже|резко (поднял|подскочил)\w* давлени|не могу встать|"
               r"подвернул|распух|порезал\w* глубоко|глубок\w* порез"),
    ("complaint", r"жалоб|жалу(юсь|ется|ешься)|пожаловат|нахам|хамств|хамил|нагруб|груб(о|ост|ил|ый|ая)|невежлив|"
                  r"оскорб|безобраз|недоволен|недовольна|министерств|"
                  r"не (могу|можем) (до вас )?дозвонит"),
    # С-08: срок готовности анализа прошёл, а результата нет — задача в лабораторию с обратным звонком (R-08 P3)
    ("lab_delay", r"(анализ|результат)\w*.{0,40}(до сих пор|всё ещё|все еще|так и) (нет|не готов|не пришл)|"
                  r"срок\w* (уже )?прош(ёл|ел|ли).{0,40}(анализ|результат)|(анализ|результат)\w*.{0,40}срок\w* (уже )?прош|"
                  r"(анализ|результат)\w*.{0,30}задержива"),
    # Персональные данные (R-16) и звонки от организаций (R-14)
    ("pdn", r"персональн\w* данн|удал\w* (мои |все )?(мои )?данн|отзыв\w* согласи|копи\w* запис\w* разговор"),
    ("admin", r"из полиции|прокуратур|журналист|\bсми\b|от организации|юридическ\w* лиц"),
    ("medical", r"можно ли (мне )?(пить|принимать|выпить|принять|колоть|сочетать)|что (мне )?делать\b|"
                r"посоветуйте|нужен (ваш )?совет|подскажите\b.{0,15}\b(что|какие|какое|чем)|что (мне )?(выпить|принять|пить)\b|"
                r"чем (лечить|сбить|снять)|"
                r"\bэто (плохо|опасно|нормально|страшно)|что (означает|значит) (мой |мои |у меня )?(диагноз|анализ|результат)|"
                r"(ибупрофен|парацетамол|нурофен|аспирин|анальгин|антибиотик|цитрамон|но-?шпа)\w*|гемоглобин|"
                r"нормальн\w* ли (у меня )?анализ|что (мне )?(можно |нужно |надо |лучше )?(принять|пить|выпить)|(какие|какую|что за) (таблетк|лекарств)|как(ие|ое|ую) (таблетк|лекарств)|чем (лечить|сбить)|посоветуйте|"
                r"это опасно|нормально ли|какая дозировк|расшифр|выпить от|сбить (температур|давлени)"),
    # Просьба соединить с человеком; пациент явно не справляется с диалогом (R-02, R-15)
    ("human", r"оператор|регистратор|сотрудник|\bживо(й|го|му|м)\b|\bживым\b|позовите человек|соедините с человек|"
              r"нуж(ен|но) (мне )?(живой )?человек|"
              r"с человеком|переключите на|соедините с(о)? специалист|соедините с врач|"
              r"я (вас )?не понимаю|плохо слышу|не слышу вас"),
]


def _first(patterns: list[tuple[str, str]], text: str) -> str | None:
    return next((value for value, pattern in patterns if re.search(pattern, text)), None)


_SPEC_STOP = {"детский", "врач", "диагностики", "акушерство", "терапевт"}


def _directory_specialty(text: str, schedule: "Schedule") -> str | None:
    """Специальности из справочника клиники, которых нет в общих шаблонах: «к урологу», «к хирургу»."""
    parts = {p for d in schedule.doctors.values() for p in re.split(r"[\s()\-]+", d.specialty.lower())
             if len(p) >= 6 and p not in _SPEC_STOP}
    for p in sorted(parts, key=len, reverse=True):
        if re.search(r"\b" + re.escape(p[:-1]), text):
            return p
    return None


def _future(today: date, month: int, day: int) -> date | None:
    """Ближайшая такая дата не раньше сегодня; несуществующая (31 февраля, 29 февраля не в високосный) — None."""
    for year in (today.year, today.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        if d >= today:
            return d
    return None


def _parse_day(text: str, today: date) -> date | None:
    if "послезавтра" in text:
        return today + timedelta(days=2)
    if "завтра" in text:
        return today + timedelta(days=1)
    if "сегодня" in text:
        return today
    m = re.search(r"\b(\d{1,2})\s+(" + "|".join(MONTHS_GEN) + r")", text)
    if m:  # «31 февраля» — такой даты нет: день не берём, а не падаем
        return _future(today, MONTHS_GEN.index(m.group(2)) + 1, int(m.group(1)))
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})\b", text)
    if m and 1 <= int(m.group(2)) <= 12:
        return _future(today, int(m.group(2)), int(m.group(1)))
    m = re.search(r"\bна\s+(\d{1,2})-?е\b|\b(\d{1,2})(?:-?го)?\s+числа\b", text)
    if m:  # «на 22-е», «22 числа» — ближайшее такое число месяца
        n = int(m.group(1) or m.group(2))
        for months_ahead in range(3):
            y, mo = today.year + (today.month - 1 + months_ahead) // 12, (today.month - 1 + months_ahead) % 12 + 1
            try:
                d = date(y, mo, n)
            except ValueError:
                continue
            if d > today:
                return d
    for wd, pattern in enumerate(_WEEKDAYS):
        if re.search(pattern, text):
            d = today + timedelta(days=(wd - today.weekday()) % 7 or 7)
            if re.search(r"следующ\w* недел|через неделю", text):
                # «на следующей неделе во вторник» — день из следующей календарной недели
                next_monday = today + timedelta(days=7 - today.weekday())
                d = next_monday + timedelta(days=wd)
            return d
    return None


def _hm(hour: str, minute: str | None, text_after: str = "") -> time | None:
    h, m = int(hour), int(minute or 0)
    if 1 <= h < 7 and not re.match(r"\s*(час\w*\s*)?(утра|ночи)", text_after):
        h += 12  # поликлиника работает с 7:30 до 20: «в 2», «до 5» — это день
    return time(h, m) if 0 <= h <= 23 and 0 <= m <= 59 else None


def _parse_exact_time(text: str) -> time | None:
    m = re.search(r"\b(?:в|на)\s+(\d{1,2})[:.](\d{2})\b", text) or re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m:
        return _hm(m.group(1), m.group(2))
    m = re.search(r"\b(?:в|на)\s+(\d{1,2})\s*(час\w*)?\s*(утра|дня|вечера)?", text)
    if m and (m.group(2) or m.group(3)):
        return _hm(m.group(1), None, text[m.end(1):])
    return None


def _parse_window(text: str) -> tuple[time | None, time | None]:
    if re.search(r"\bне\s+(?:рано|с утра|утр)", text):
        return time(12), None  # «только не рано утром» — наоборот, после обеда
    if re.search(r"\bне\s+(?:поздно|вечер)", text):
        return None, time(17)
    if re.search(r"\bутр", text):
        return time(8), time(12)
    if re.search(r"дн[её]м|после обеда|в обед", text):
        return time(13), time(17)
    if re.search(r"вечер", text):
        return time(17), time(21)
    no_date = r"(?![:.]?\d*\s+(?:" + "|".join(MONTHS_GEN) + r"))"  # «до 20 октября» — не время
    after = re.search(r"(?:после|\bс)\s+(\d{1,2})(?:[:.](\d{2}))?\b" + no_date, text)
    before = re.search(r"\bдо\s+(\d{1,2})(?:[:.](\d{2}))?\b" + no_date, text)
    return (_hm(after.group(1), after.group(2)) if after else None,
            _hm(before.group(1), before.group(2)) if before else None)


def _correcting_identity(st: "ClinicState", text: str) -> bool:
    """«…Верно?» после повтора ФИО нового пациента — «нет, Сидорова Ольга»: поправка личности, а не времени."""
    return bool(st.awaiting_confirm and st.patient is not None and not st.patient.id and re.search(r"^\s*нет\b", text)
                and any(w in _FIRST_NAMES for w in re.findall(r"[а-я]+", text.replace("ё", "е"))))


# «Иванов Иван, на двенадцатое октября» — день приёма; «Иванов Иван, двенадцатое марта» — дата рождения без года
_APPOINTMENT_DAY = r"\b(на|к)\b|можно|давайте|хочу|лучше|перенес|запиш|удобн|подойд|сможете"


def _awaits_identity(st: "ClinicState") -> bool:
    if st.patient is not None:
        return False
    if st.intent == "book" and st.chosen is not None:
        return True
    if st.intent == "escalate" and st.asked_identity:
        return True
    if st.intent in ("home_visit", "opt_out"):
        return True  # С-06: заявка на вызов врача на дом — по ФИО и дате рождения
    return st.intent in ("status", "cancel", "reschedule") and st.target is None  # и с номером записи — ФИО нужно


_MONTHS = "|".join(MONTHS_GEN)


def _parse_birth_date(text: str, today: date) -> tuple[date | None, str]:
    """Дата рождения и текст без неё (чтобы «12 марта 1958» не стало датой приёма)."""
    for pattern, kind in ((r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b", "num"),
                          (rf"\b(\d{{1,2}})\s+({_MONTHS})\s+(\d{{4}}|\d{{2}})(?:-?го)?(?:\s+года?)?\b", "word"),
                          (rf"\b((?:19|20)\d{{2}})(?:-?го)?(?:\s+года?)?[\s,]+(\d{{1,2}})\s+({_MONTHS})\b", "year_first")):
        m = re.search(pattern, text)
        if not m:
            continue
        if kind == "year_first":  # «1960 год, 12 мая»
            day, year, month = int(m.group(2)), int(m.group(1)), MONTHS_GEN.index(m.group(3)) + 1
        else:
            day, year = int(m.group(1)), int(m.group(3))
            month = int(m.group(2)) if kind == "num" else MONTHS_GEN.index(m.group(2)) + 1
        if year < 100:
            year += 1900 if year > today.year % 100 else 2000
        try:
            born = date(year, month, day)
        except ValueError:
            continue
        if born < today:
            return born, text[:m.start()] + " " + text[m.end():]
    return None, text


def _split_name(words: list[str]) -> dict:
    words = [w.capitalize() for w in words if w.lower() not in _NOT_NAME][:3]
    if len(words) < 2:
        return {"last_name": words[0]} if words else {}
    first, second = words[0], words[1]
    if first.lower().replace("ё", "е") in _FIRST_NAMES and second.lower().replace("ё", "е") not in _FIRST_NAMES:
        return {"first_name": first, "last_name": second}
    return {"last_name": first, "first_name": second}


def rule_extract(
    st: "ClinicState", raw: str, today: date, schedule: "Schedule", speech: SpeechNormalizer | None = None
) -> "ClinicUpdate":
    from .models import ClinicUpdate

    heard = raw.lower().replace("ё", "е")  # до перевода чисел: «второй» как выбор варианта
    raw = strip_fillers(raw)
    text = raw.lower().replace("ё", "е")
    if speech is not None:
        text = speech.fix(text)
    upd: dict = {}

    if re.match(r"\s*(здравствуй|здрасьте|привет|добр(ый|ое|ого)|алло|спасибо|благодарю|до свидания)", text):
        upd["smalltalk"] = True

    for flag, pattern in _RED_FLAGS:
        if re.search(pattern, text):
            upd["red_flag"], upd["note"] = flag, raw[:300]
            break
    if st.question == "home_address":
        # С-06: адрес и что беспокоит — как есть, в заявку. Угроза жизни важнее заявки; «что мне делать» в описании
        # жалобы — не медицинский вопрос, а слова пациента для врача; «передумал» — заявку не оформляем
        if upd.get("red_flag") in ("emergency", "selfharm", "urgent"):
            return ClinicUpdate(red_flag=upd["red_flag"], note=upd["note"])
        if re.search(r"передумал|не надо|не нужно|отмен|не хочу|^\s*нет\b|до свидания|всего доброго", text):
            return ClinicUpdate(confirm=False)
        return ClinicUpdate(note=raw[:300])

    topic = faq_topic(text)
    if topic == "robot" and upd.get("red_flag") == "human" and not re.search(
            r"соедин|позов|переключ|дайте|хочу поговорить|позвать|нужен оператор|нужен человек", text):
        del upd["red_flag"], upd["note"]  # «Вы робот или живой человек?» — вопрос, а не просьба соединить
    if re.search(r"до свидания|всего доброго|позвоню (позже|потом)|перезвоню|перезвоните (позже|потом)|сейчас неудобно|не могу (сейчас )?говорить|не надо,? спасибо|этого достаточно|"
                 r"больше (ничего )?не нужно|вс[её] понятно|"
                 r"ничего не нужно|больше ничего", text):
        upd["bye"] = True
    if topic:
        upd["topic"] = topic
        if upd.get("red_flag") == "medical":
            del upd["red_flag"], upd["note"]  # «нормальные ли анализы» — справка с граничной фразой

    intent = next((name for name, pattern in _INTENTS if re.search(pattern, text)), None)
    if st.question == "cancel_or_move" and re.search(r"отмен", text):
        intent = "cancel"  # «перенести или отменить?» — «отменить»
    if intent == "reschedule" and st.intent == "book" and not re.search(r"перенес|перенест|сдвин|мо\w* запис", text):
        intent = None  # «нет, другое время» при записи — другие варианты, а не перенос старой записи
        if st.awaiting_confirm:
            upd["confirm"] = False
    if intent and not (topic and intent in ("schedule", "book") and not re.search(r"запиш|записат", text)):
        upd["intent"] = intent

    born, text = _parse_birth_date(text, today)
    if born:
        upd["birth_date"] = born
        raw = _parse_birth_date(raw.lower(), today)[1]  # имя ищем в тексте без даты

    m = re.search(_NAME, raw)
    if m:
        words = []
        for w in re.split(r"[\s,]+", m.group(1)):
            if w.lower() in _NOT_NAME:
                break
            words.append(w)
        upd |= _split_name(words)
    elif ((_awaits_identity(st) or (st.awaiting_confirm and born and re.search(r"^\s*нет\b", text))
           or _correcting_identity(st, text))
          and not intent and not re.search(_NOT_NAME_CONTEXT, text)):
        # второе условие — «Записываю вас… Верно?» — «нет, Иванов Иван, 12.05.1960»: поправка личности, а не времени
        # «нет, Иванов Иван» — поправка: «нет» в начале отбрасываем, дальше — имя
        body = re.sub(r"^\s*(?:нет|да|не так)\b[\s,.!-]*", "", raw, flags=re.I)
        words = [w for w in re.findall(r"[А-ЯЁа-яё-]+", body) if w.lower() not in _NOT_NAME]
        if 1 <= len(words) <= 3 and not re.search(_YES + "|" + _NO, body.lower()):
            upd |= _split_name(words)
    name_words = {str(upd[k]).lower() for k in ("last_name", "first_name") if k in upd}

    m = re.search(r"\b[Bb]-?0*(\d{1,4})\b", raw)
    if m:
        upd["booking_id"] = f"B{int(m.group(1)):04d}"
    elif m := re.search(r"номер\w*\s+(?:записи\s+)?0*(\d{1,6})\b", text):
        upd["booking_id"] = str(int(m.group(1)))

    specialty = _first(_EXPLICIT, text) or _directory_specialty(text, schedule)
    if specialty is None and (m := re.search(r"(?:\bк|\bнуж\w*|\bзапиш\w*(?:\s+меня)?(?:\s+к)?)\s+(?:врачу\s+)?"
                                             r"([а-я]{3,}?(?:лог|иатр|хирург))(?:у|а|ам)?\b", text)):
        specialty = m.group(1)  # «к эндокринологу» — врача нет в справочнике: скажем об этом, а не переспросим
    if specialty is None and not topic:  # «нужен больничный» — справка, а не запись по жалобе
        specialty = _first(_SYMPTOMS, text)
    if specialty and specialty != "педиатр" and not specialty.startswith("детский") and re.search(r"детск|ребен", text):
        specialty = "детский " + specialty
    if specialty:
        upd["specialty"] = specialty
    for word in re.findall(r"[а-я-]{4,}", text):
        if word not in name_words and (doctor := schedule.find_doctor(word)):
            upd["doctor_name"] = doctor.name
            break

    day = _parse_day(text, today)
    exact = _parse_exact_time(text)

    # Пациент выбирает из предложенного: «второй», «давайте в 12:30»
    # на «Верно?» тоже: «нет, лучше второй вариант» — другой из названных
    options = (st.offered if st.offered and (st.chosen is None or st.awaiting_confirm)
               else [b.slot for b in st.candidates] if st.candidates and st.target is None else [])
    if options:
        # «во второй половине дня», «двадцать второе октября» — не выбор варианта
        said = re.sub(r"(перв|втор)\w*\s+половин\w*", " ", heard)
        said = re.sub(r"\b(двадцать|тридцать)\s+[а-я]+|[а-я]+\s+(?=(январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|"
                      r"октябр|ноябр|декабр))", " ", said)
        choice = next((n for n, pattern in _ORDINALS.items() if re.search(pattern, said)), None)
        if choice is None and re.search(r"последн", text):
            choice = len(options)
        if exact is None and not re.search(r"\d{1,2}\s+(" + _MONTHS + ")", text):
            # «давайте в 10» при названных вариантах — это время, даже без «часов»
            if bare := re.search(r"\b(?:в|на)\s+(\d{1,2})(?::(\d{2}))?\b(?!-е)", text):
                exact = _hm(bare.group(1), bare.group(2))
        window = _parse_window(text) if exact is None else (None, None)
        if choice is None and (exact or day) and not re.search(_REJECT, heard):  # «во вторник не смогу» — не выбор
            # «давайте в 12:30», «давайте пятницу», «во вторник в девять»: день недели сверяем с вариантами
            timely = [(i, s) for i, s in enumerate(options, 1)
                      if (exact is None or s.start.time() == exact)
                      and (window[0] is None or s.start.time() >= window[0])
                      and (window[1] is None or s.start.time() < window[1])]
            hits = [i for i, s in timely if day is None or s.start.date() == day]
            if day and not hits and not re.search(r"следующ\w* недел", text):
                hits = [i for i, s in timely if s.start.weekday() == day.weekday()]
            if len(hits) == 1:
                choice, day, exact = hits[0], None, None
        if re.search(r"\bни (?:один|одн\w*|как\w*)\b|\bникак(?:ой|ие|ая)\b|\bвсе не подход|\bничего (?:не подход|из)", heard):
            choice = None  # «ни один не подходит» — «один» здесь не выбор первого варианта
        if choice is None and not day and exact is None and re.search(
                r"^\s*нет\b|\bни (?:один|какой)|\bникак(?:ой|ие)\b|\bдруг(?:ой|ие|ое) (?:вариант|врем|день)", heard):
            upd["reject"] = 0  # «нет», «ни один», «другое время» — все названные варианты не подходят
            if re.search(r"друг\w* (день|дат|числ)", heard):
                upd["other_day"] = True
        elif re.search(_REJECT, heard):
            # «Первый вариант не подходит, нужен другой день» — отказ от варианта, а не выбор
            upd["reject"] = choice if choice is not None and choice <= len(options) else 0
            if re.search(r"друг\w* (день|дат|числ)|не в этот день|в другой день", heard):
                upd["other_day"] = True
            if day and not re.search(r"давайте|лучше|можно|хочу|подойд[её]т|а на\b", heard):
                # «В понедельник не получится» — этот день обходим, а не ищем в нём; варианты других дней в силе
                same = [o.start.date() for o in options if o.start.weekday() == day.weekday()]
                upd.pop("reject", None)
                upd["avoid_day"], day = (same[0] if same else day), None
        elif choice is not None and choice <= len(options):
            upd["choice"] = choice

    if (day and not born and ("last_name" in upd or "first_name" in upd) and not re.search(_APPOINTMENT_DAY, text)
            and (_awaits_identity(st) or _correcting_identity(st, text))):
        day = None  # «Смирнова Ольга, двенадцатое марта» — дата рождения без года, а не день приёма
    if day:
        upd["day"] = day
    elif re.search(r"следующ\w* недел|через неделю", text):
        upd["week_of"] = today + timedelta(days=7 - today.weekday())  # понедельник следующей недели
    elif re.search(r"(на )?(этой|текущей) недел", text):
        upd["week_of"] = today - timedelta(days=today.weekday())
    if exact:
        upd["time_from"] = exact
        upd["time_to"] = (datetime.combine(today, exact) + timedelta(minutes=30)).time()
    else:
        time_from, time_to = _parse_window(text)
        if time_from:
            upd["time_from"] = time_from
        if time_to:
            upd["time_to"] = time_to

    if st.question == "rating":  # «Оцените разговор от одного до пяти»
        words = {"один": 1, "единиц": 1, "два": 2, "двойк": 2, "три": 3, "тройк": 3, "четыре": 4, "четв": 4,
                 "пять": 5, "пятер": 5, "отлично": 5, "супер": 5, "хорошо": 4, "нормально": 3, "плохо": 2, "ужасно": 1}
        m = re.search(r"\b([1-5])\b", text)
        rating = int(m.group(1)) if m else next((v for k, v in words.items() if re.search(rf"\b{k}", text)), None)
        if rating:
            upd["rating"] = rating
    if st.question == "will_come":  # исходящее напоминание: «не получится» — это «нет», а не отмена
        if re.search(r"не (смогу|получится|приду|успею|выйдет)|не могу", text):
            upd["confirm"] = False
            if not re.search(r"отмен", text):
                upd.pop("intent", None)
        elif re.search(r"\b(приду|буду|подойду|придем|придём)\b", text):
            upd["confirm"] = True
    if st.question == "operator" and "confirm" not in upd and not re.search(r"^\s*нет\b|не надо|не нужно", text):
        if re.search(r"соедин|переключ|позов|позвать|давайте|хочу поговорить|с человеком|с оператором", text):
            upd["confirm"] = True  # «Если хотите, соединю с сотрудником» — «тогда соедините, пожалуйста»
    if st.question in ("is_patient", "birth_check"):
        if re.search(r"ошиблись|не туда|\bэто не\b|(его|ее|её) нет|не он\b|не она\b", text):
            upd["confirm"] = False
        elif re.search(r"\bэто я\b|\bя слушаю\b|^\s*я\b|слушаю", text):
            upd["confirm"] = True
    if "confirm" in upd:
        return ClinicUpdate(**upd)  # ответ на напоминание уже разобран выше
    if re.search(_NO, text) or (st.question == "referral" and re.search(r"без направлени|не дава", text)):
        upd["confirm"] = False
    elif re.search(_YES, text) or (st.question == "referral" and re.search(r"\bесть\b", text)):
        upd["confirm"] = True
    return ClinicUpdate(**upd)
