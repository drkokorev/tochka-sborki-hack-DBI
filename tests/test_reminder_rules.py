"""Правила обзвона С-10: окно по часам клиники, воскресенья и праздники, записи моложе 24 часов, отказы,
подтверждения, до двух попыток с интервалом от двух часов, SMS (регламент С-10, уточнение DBI 20.10)."""

from datetime import date, datetime, timedelta

from domains.clinic import reminders
from domains.clinic.schedule import Schedule, DOCTORS, seed_demo_bookings
from tests.conftest import TODAY


def make():
    schedule = Schedule(DOCTORS, TODAY)
    seed_demo_bookings(schedule)
    booking = next(b for b in schedule.bookings.values() if b.slot.start.date() == schedule.reminder_day())
    return schedule, booking


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute)


def test_window_by_clinic_hours_weekends_and_holidays():
    monday = TODAY
    assert reminders.in_window(at(monday, 10)) and reminders.in_window(at(monday, 18, 59))
    assert not reminders.in_window(at(monday, 9, 59)) and not reminders.in_window(at(monday, 19))
    saturday = monday + timedelta(days=5)
    assert reminders.in_window(at(saturday, 14, 59)) and not reminders.in_window(at(saturday, 15))
    assert not reminders.in_window(at(monday + timedelta(days=6), 12))           # воскресенье
    assert not reminders.in_window(at(date(2026, 11, 4), 12))                     # День народного единства


def test_queue_respects_window_age_attempts_and_final_outcomes():
    schedule, booking = make()
    noon = at(TODAY, 12)
    assert booking in reminders.due(schedule, noon)
    assert reminders.due(schedule, at(TODAY, 20)) == []                           # вне окна — никому
    booking.created = noon - timedelta(hours=3)
    assert reminders.skip_reason(schedule, booking, noon) == "запись создана менее 24 часов назад"
    booking.created = None
    schedule.note_reminder_attempt(booking)                                       # первая попытка
    schedule.reminder_log[-1]["at"] = noon.isoformat()
    assert "через 2 часа" in reminders.skip_reason(schedule, booking, noon + timedelta(hours=1))
    assert reminders.skip_reason(schedule, booking, noon + timedelta(hours=2)) is None
    schedule.note_reminder_attempt(booking)                                       # вторая — и всё
    assert "SMS" in reminders.skip_reason(schedule, booking, noon + timedelta(hours=5))
    assert ("не дозвонились за 2 попытки" in [why for b, why in reminders.sms_due(schedule) if b.id == booking.id])


def test_opted_out_and_confirmed_are_never_called_even_in_demo_mode():
    schedule, booking = make()
    patient = schedule.patient_of(booking)
    late = at(TODAY, 21)
    assert reminders.skip_reason(schedule, booking, late, demo=True) is None     # демо: окно не проверяется
    schedule.log_reminder(booking, patient, "CONFIRMED")
    assert reminders.skip_reason(schedule, booking, late, demo=True) == "запись уже подтверждена"
    schedule2, booking2 = make()
    schedule2.log_reminder(booking2, schedule2.patient_of(booking2), "OPT_OUT")
    assert reminders.skip_reason(schedule2, booking2, at(TODAY, 12), demo=True) == "пациент отказался от обзвона"
    assert booking2 not in reminders.due(schedule2, at(TODAY, 12))
    # отказ только от звонков — SMS по-прежнему приходит; полный отказ — и SMS нет
    assert any(b.id == booking2.id for b, _ in reminders.sms_due(schedule2))
    schedule2.log_reminder(booking2, schedule2.patient_of(booking2), "OPT_OUT_ALL")
    assert not any(b.id == booking2.id for b, _ in reminders.sms_due(schedule2))


def test_registry_hours_for_urgent_cases():
    assert reminders.registry_open(at(TODAY, 7)) and not reminders.registry_open(at(TODAY, 20))
    saturday = TODAY + timedelta(days=5)
    assert reminders.registry_open(at(saturday, 8, 30)) and not reminders.registry_open(at(saturday, 15))
    assert not reminders.registry_open(at(TODAY + timedelta(days=6), 12))


def test_rescheduled_visit_gets_a_reminder_for_its_new_time():
    """Итог «подтвердил» относится к прежнему времени приёма: запись перенесли — накануне нового времени напомним."""
    schedule, booking = make()
    schedule.log_reminder(booking, schedule.patient_of(booking), "CONFIRMED")
    assert reminders.final_code(schedule, booking.id) == "CONFIRMED"
    new_slot = schedule.find(specialty=booking.slot.specialty, day=booking.slot.start.date() + timedelta(days=2), limit=1)[0]
    moved = schedule.reschedule(booking.id, new_slot)
    assert reminders.final_code(schedule, moved.id) is None


def test_old_opt_out_without_patient_id_is_still_respected():
    schedule, booking = make()
    schedule.reminder_log.append({"booking": booking.id, "code": "OPT_OUT", "at": at(TODAY, 11).isoformat()})
    assert reminders.opted_out(schedule, booking.patient_id)
