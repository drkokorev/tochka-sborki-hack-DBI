"""Корпус для дистилляции понимания речи: учитель (LLM) пишет реплики пациентов по каждому речевому акту,
шаблоны добавляют реальные специальности и фамилии, шум имитирует ошибки распознавания телефонной речи.

    .venv/bin/python tools/distill_data.py [--per-act 40] [--model ai-sage/GigaChat3-10B-A1.8B]

Пишет data/distill/corpus.jsonl: {"text", "act", "source": template|llm, "noisy"}.
Генерирует лёгкая модель шлюза: ~300 запросов, ~4 ₽ на весь корпус.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings  # noqa: E402
from core.llm import OpenAICompatLLM, strip_think  # noqa: E402
from domains.clinic.acts import ACT_HINTS, ACTS  # noqa: E402

PROMPT = """Ты помогаешь собрать обучающие данные для голосового ассистента регистратуры поликлиники.
Ситуация: {hint}.
Напиши {n} разных реплик пациента в этой ситуации — так, как люди говорят по телефону: разный возраст и манера,
вежливо и грубовато, коротко и многословно, с просторечием и оговорками. Не больше 20 слов в реплике.
Только реплики пациента, каждая с новой строки, без нумерации, кавычек и пояснений."""

SPEC = ["к терапевту", "к лору", "к лор-врачу", "к окулисту", "к офтальмологу", "к стоматологу", "к зубному",
        "к неврологу", "к кардиологу", "к гинекологу", "к педиатру", "к хирургу", "к урологу", "к эндокринологу",
        "к дерматологу", "к участковому"]
WHEN = ["", " на завтра", " на понедельник", " на среду", " в пятницу", " после обеда", " утром", " на следующей неделе",
        " на 21 октября", " в 10:30", " вечером", " поскорее"]
LEAD = ["", "здравствуйте, ", "алло, ", "девушка, ", "добрый день, ", "мне бы ", "слушайте, ", "э-э ", "ну "]
SURNAMES = ["Беловой", "Верещагину", "Ковалёвой", "Дорохову", "Нечаевой", "Савельеву", "Грачёвой", "Орлову"]
NAMES = [("Иванов", "Иван", "Петрович"), ("Смирнова", "Ольга", "Викторовна"), ("Кузнецов", "Сергей", "Андреевич"),
         ("Попова", "Анна", "Игоревна"), ("Соколов", "Дмитрий", "Олегович"), ("Морозова", "Нина", "Ивановна")]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]


def templates(rng: random.Random) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for _ in range(90):
        lead, spec, when = rng.choice(LEAD), rng.choice(SPEC), rng.choice(WHEN)
        verb = rng.choice(["запишите меня", "хочу записаться", "нужно записаться", "можно записаться",
                           "нужен талон", "хочу попасть", "запишите пожалуйста"])
        out.append((f"{lead}{verb} {spec}{when}", "intent:book"))
        out.append((f"{lead}{spec}{when}", "slots"))
    for s in SURNAMES:
        out += [(f"когда принимает {s[:-2] + 'а' if s.endswith('ой') else s[:-1]}", "intent:schedule"),
                (f"к {s} можно{rng.choice(WHEN)}", "slots"),
                (f"какие часы приёма у доктора {s[:-2] + 'ой' if s.endswith('ой') else s}", "intent:schedule")]
    for last, first, middle in NAMES:
        for _ in range(6):
            d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1940, 2005)
            date = rng.choice([f"{d:02d}.{m:02d}.{y}", f"{d} {MONTHS[m - 1]} {y}", f"{d} {MONTHS[m - 1]} {y} года"])
            name = rng.choice([f"{last} {first} {middle}", f"{last} {first}", f"{first} {last}",
                               f"меня зовут {first} {last}", f"фамилия {last}, имя {first}"])
            out.append((rng.choice([f"{name}, {date}", f"{name} {date}", name, f"дата рождения {date}"]), "identity"))
    for word in ["первый", "второй", "третий", "давайте первый", "второй вариант", "последний", "давайте в среду",
                 "в пятницу подойдёт", "давайте в 9:30", "первый подходит", "мне третий", "лучше второй",
                 "давайте который в понедельник", "в десять сорок давайте"]:
        out.append((word, "choice"))
    for word in ["да", "да, верно", "ага", "конечно", "да, всё правильно", "угу", "подходит", "да, давайте",
                 "верно", "да, это я", "да, приду", "хорошо"]:
        out.append((word, "yes"))
    for word in ["нет", "нет, не надо", "не подходит", "нет, спасибо", "не нужно", "нет, не верно", "неа",
                 "нет, завтра не получится", "не смогу"]:
        out.append((word, "no"))
    for word in ["до свидания", "спасибо, до свидания", "позвоню позже", "больше ничего не нужно", "всего доброго",
                 "ладно, перезвоню"]:
        out.append((word, "bye"))
    for word in ["здравствуйте", "алло", "добрый день", "спасибо", "алло, слышите", "здрасьте", "благодарю"]:
        out.append((word, "smalltalk"))
    for act, phrases in HANDWRITTEN.items():
        out += [(t, act) for t in phrases]
    for word in ["абырвалг", "кхм кхм", "э-э-э", "да я не знаю даже", "а это кто", "какая погода завтра",
                 "закажите мне пиццу", "ммм"]:
        out.append((word, "other"))
    return out


# Ручные примеры: словарь красных флагов регламента (7.4) и темы, где учитель дал мало вариантов
HANDWRITTEN: dict[str, list[str]] = {
    "flag:emergency": ["у мужа давит в груди", "болит сердце отдаёт в левую руку", "мама потеряла сознание",
                       "ребёнок не может дышать", "у отца перекосило лицо и рука не двигается", "задыхаюсь",
                       "кровь не останавливается", "сын проглотил таблетки", "у неё судороги не прекращаются",
                       "жена беременная, отошли воды", "отравился грибами, рвёт", "упал с лестницы, без сознания",
                       "у ребёнка опухло горло, не может глотать", "не хочу жить", "сильно ударился головой",
                       "речь невнятная, онемела половина тела", "жжёт в груди и холодный пот"],
    "flag:urgent": ["температура сорок не сбивается", "очень сильно болит живот", "резко поднялось давление",
                    "стало намного хуже, кружится голова", "подвернул ногу, распухла", "порезался глубоко",
                    "температура третий день и хуже", "невыносимо болит спина"],
    "flag:dental": ["очень сильно болит зуб", "флюс, щёку раздуло", "выбил зуб", "сломался зуб, острая боль",
                    "опухла десна, не могу терпеть", "зубная боль невыносимая"],
    "flag:complaint": ["хочу пожаловаться на врача", "врач мне нагрубил", "это безобразие, три дня не дозвониться",
                       "я буду жаловаться в министерство", "регистратор хамит", "недоволен обслуживанием",
                       "у вас бардак, запись потеряли"],
    "flag:medical": ["что выпить от давления", "чем сбить температуру", "какие таблетки от головы",
                     "это опасно, если болит бок", "нормально ли что сыпь", "посоветуйте лекарство от кашля",
                     "какую дозировку пить", "что принять от аллергии"],
    "flag:human": ["соедините с оператором", "позовите живого человека", "хочу поговорить с человеком",
                   "переключите на регистратора", "дайте сотрудника", "мне нужен оператор", "с роботом не буду"],
    "flag:admin": ["это из полиции, нужна информация", "звоню из редакции", "хочу отозвать согласие на "
                   "обработку данных", "удалите мои персональные данные", "нужна копия записи разговора"],
    "intent:status": ["на когда я записан", "проверьте мою запись", "во сколько мне к врачу", "у меня есть запись?",
                      "какой у меня кабинет", "напомните время приёма", "я записан к терапевту, на какое число",
                      "хочу узнать свою запись", "подскажите, когда мой приём"],
    "intent:cancel": ["отмените запись", "я не приду", "уже не нужно, отмените", "снимите мою запись",
                      "выздоровел, отменяйте приём", "отмена записи к лору"],
    "faq:waiting": ["сколько ждать приёма у кардиолога", "какие сроки ожидания", "как долго ждать специалиста"],
    "faq:lab_normal": ["нормальные ли у меня анализы", "анализы хорошие?", "что значат мои результаты"],
    "faq:lab_phone": ["продиктуйте мои анализы", "можно узнать результат по телефону",
                      "скажите результаты крови"],
    "faq:holidays": ["как работаете в праздники", "в новогодние открыты", "на майские работаете"],
    "faq:hours": ["до скольки вы работаете", "вы работаете в субботу", "во сколько открываетесь",
                  "режим работы поликлиники", "в воскресенье открыто"],
    "faq:no_slots": ["почему нет талонов", "когда появятся номерки", "почему нет записи к неврологу"],
}


_MERGE = re.compile(r"\bк (л|т|о|с|н|г|п|х|у)")


def noise(text: str, rng: random.Random) -> str:
    """Ошибки распознавания телефонной речи: регистр, пунктуация, склейки, паразиты, опечатки."""
    t = text.lower()
    t = re.sub(r"[,.!?;:«»\"]", "", t) if rng.random() < 0.7 else t
    if rng.random() < 0.4:
        t = _MERGE.sub(lambda m: "к" + m.group(1), t)
    if rng.random() < 0.3:
        t = rng.choice(["э-э ", "ну ", "это самое ", "так "]) + t
    words = t.split()
    if words and rng.random() < 0.4:
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 4:
            j = rng.randrange(1, len(w) - 1)
            words[i] = w[:j] + w[j + 1:] if rng.random() < 0.5 else w[:j] + rng.choice("аоеия") + w[j + 1:]
    return " ".join(words)


def llm_lines(llm, act: str, n: int) -> list[str]:
    raw = strip_think(llm.complete([{"role": "user", "content": PROMPT.format(hint=ACT_HINTS[act], n=n)}]))
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-•*])\s*", "", line).strip().strip('"«»')
        if 2 <= len(line) <= 160 and not line.lower().startswith(("вот", "конечно, вот", "реплики")):
            lines.append(line)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-act", type=int, default=40)
    ap.add_argument("--model", default="ai-sage/GigaChat3-10B-A1.8B")
    ap.add_argument("--no-llm", action="store_true", help="только шаблоны (для проверки без шлюза)")
    ap.add_argument("--reuse-llm", action="store_true",
                    help="взять реплики учителя из уже собранного корпуса, пересобрать шаблоны и шум")
    args = ap.parse_args()
    rng = random.Random(7)
    rows = [{"text": t, "act": a, "source": "template"} for t, a in templates(rng)]
    out = Path("data/distill/corpus.jsonl")
    if args.reuse_llm and out.exists():
        old = [json.loads(line) for line in out.open(encoding="utf-8")]
        rows += [{"text": r["text"], "act": r["act"], "source": "llm"} for r in old
                 if r["source"] == "llm" and not r.get("noisy")]
    elif not args.no_llm:
        llm = OpenAICompatLLM(replace(get_settings(), llm_model=args.model, llm_json_mode=False))
        for act in ACTS:
            got: list[str] = []
            for _ in range(3):  # по 20 за запрос: длинные списки модель повторяет
                try:
                    got += llm_lines(llm, act, 20)
                except Exception as exc:
                    print(f"{act}: {type(exc).__name__}", flush=True)
                    if "429" in str(exc):
                        break
                if len(set(got)) >= args.per_act:
                    break
            uniq = list(dict.fromkeys(got))[: args.per_act]
            rows += [{"text": t, "act": act, "source": "llm"} for t in uniq]
            print(f"{act:<28} {len(uniq)}", flush=True)
    rows += [{**r, "text": noise(r["text"], rng), "noisy": True} for r in rows if rng.random() < 0.6]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"noisy": False, **r}, ensure_ascii=False) + "\n")
    print(f"Корпус: {len(rows)} реплик, {len({r['act'] for r in rows})} актов → {out}")


if __name__ == "__main__":
    main()
