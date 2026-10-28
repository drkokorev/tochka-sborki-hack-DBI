from datetime import date, time

import pytest

from domains.clinic.domain import ClinicDomain, ClinicState
from tests.conftest import TODAY  # понедельник, 5 октября 2026

domain = ClinicDomain(today=TODAY)


def parse(text, state=None):
    return domain.rule_extract(state or ClinicState(), text, TODAY).model_dump(exclude_none=True)


@pytest.mark.parametrize("text, intent", [
    ("запишите меня к врачу", "book"),
    ("хочу перенести запись на другой день", "reschedule"),
    ("отмените мою запись", "cancel"),
    ("на когда я записан?", "status"),
    ("когда принимает кардиолог", "schedule"),
    ("здравствуйте", None),
])
def test_intent(text, intent):
    assert parse(text).get("intent") == intent


@pytest.mark.parametrize("text, specialty", [
    ("к лору", "оториноларинголог"),
    ("к окулисту", "офтальмолог"),
    ("запишите ребёнка к стоматологу", "детский стоматолог"),  # явный специалист важнее жалобы
    ("у ребёнка температура", "педиатр"),
    ("болит голова уже неделю", "невролог"),
    ("кашель и слабость", "терапевт"),
])
def test_specialty(text, specialty):
    assert parse(text)["specialty"] == specialty


@pytest.mark.parametrize("text, day", [
    ("на завтра", date(2026, 10, 6)),
    ("послезавтра", date(2026, 10, 7)),
    ("в пятницу", date(2026, 10, 9)),
    ("в понедельник", date(2026, 10, 12)),  # сегодня понедельник — значит, следующий
    ("на 20 октября", date(2026, 10, 20)),
    ("12.11", date(2026, 11, 12)),
])
def test_day(text, day):
    assert parse(text)["day"] == day


@pytest.mark.parametrize("text, window", [
    ("утром", (time(8), time(12))),
    ("после обеда", (time(13), time(17))),
    ("после 15", (time(15), None)),
    ("в 10:30", (time(10, 30), time(11))),
    ("в 3 часа дня", (time(15), time(15, 30))),
])
def test_time(text, window):
    got = parse(text)
    assert (got.get("time_from"), got.get("time_to")) == window


def test_names_doctors_and_numbers():
    got = parse("меня зовут иван петров и я хочу записаться")
    assert (got["first_name"], got["last_name"]) == ("Иван", "Петров")
    got = parse("моя фамилия Смирнова, Анна, 17 марта 1985 года")
    assert (got["last_name"], got["first_name"], got["birth_date"]) == ("Смирнова", "Анна", date(1985, 3, 17))
    assert "day" not in got  # дата рождения — не дата приёма
    assert parse("к Ковалёву")["doctor_name"] == "Ковалёв Игорь Павлович"
    assert "doctor_name" not in parse("меня зовут Никита Орехов")  # имя пациента — не врач Никитин
    assert parse("номер записи 7")["booking_id"] == "7"
    assert parse("B0012")["booking_id"] == "B0012"


def test_confirm_and_ordinals_only_in_context():
    assert parse("да")["confirm"] is True
    assert parse("нет, не надо")["confirm"] is False
    assert "choice" not in parse("второй")  # без предложенных вариантов выбирать нечего
    assert "choice" not in parse("во вторник")


# ---------- транскрипты речи с ошибками распознавания ----------

@pytest.mark.parametrize("text, field, value", [
    ("э, клору на завтра", "specialty", "оториноларинголог"),
    ("ну это самое ктерапевту в пятницу", "specialty", "терапевт"),
    ("к тирапевту э-э после обеда", "specialty", "терапевт"),
    ("мне к ларингологу", "specialty", "оториноларинголог"),
    ("к акулисту", "specialty", "офтальмолог"),
    ("запишите к белавой", "doctor_name", "Белова Анна Сергеевна"),
    ("ну короче к гусиву на среду", "doctor_name", "Гусев Олег Андреевич"),
    ("когда принимает белава", "doctor_name", "Белова Анна Сергеевна"),
])
def test_noisy_speech(text, field, value):
    assert parse(text)[field] == value


def test_speech_fixes_do_not_touch_ordinary_words():
    got = parse("болит голова и горло")
    assert got["specialty"] == "невролог" and "doctor_name" not in got  # не «Орлова»
    got = parse("меня зовут Анна Смирнова")
    assert (got["first_name"], got["last_name"]) == ("Анна", "Смирнова")  # не врач Миронова
    assert "doctor_name" not in parse("меня зовут Анна Смирнова")


def test_fillers_do_not_break_choice():
    from domains.clinic.domain import ClinicState
    from domains.clinic.schedule import Slot
    from datetime import datetime
    slot = Slot(doctor_id="d1", doctor_name="x", specialty="терапевт", branch="b", start=datetime(2026, 10, 7, 10))
    assert parse("э-э, ну второй", ClinicState(offered=[slot, slot]))["choice"] == 2


def test_gigaam_artifacts():
    from domains.clinic.domain import ClinicState
    got = parse("Смирнова Ольга Викторовна. 12 марта 178 го.", ClinicState(intent="status"))
    assert (got["last_name"], got["first_name"], got["birth_date"]) == ("Смирнова", "Ольга", date(1978, 3, 12))
    assert parse("Мне нужно к лоОР-врачу после обеда.")["specialty"] == "оториноларинголог"


def test_words_to_digits_dates_and_times():
    from domains.clinic.speech import strip_fillers
    assert strip_fillers("петров иван двенадцатое марта пятьдесят восьмого года") == "петров иван 12 марта 58 года"
    assert strip_fillers("Двенадцатое марта, тысяча девятьсот пятьдесят восьмого") == "12 марта, 1958"
    assert strip_fillers("на завтра в одиннадцать тридцать") == "на завтра в 11:30"
    assert strip_fillers("после пятнадцати") == "после 15"
    assert strip_fillers("в половине десятого") == "в 9:30"
    assert strip_fillers("давайте второй") == "давайте второй"  # выбор варианта остаётся словом
    assert strip_fillers("номер записи двести сорок три") == "номер записи 243"
