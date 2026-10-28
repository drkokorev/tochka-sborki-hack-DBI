"""Сборка приложения (apps/container.py) и демо-реестр на SQLite (domains/clinic/store.py)."""

import os
import stat
from datetime import date

from apps.container import Config, build_services
from domains.clinic.schedule import DOCTORS
from domains.clinic.store import PersistentSchedule, SqliteStore


def test_demo_registry_survives_restart(tmp_path):
    """Записи, отмены, заявки и итоги обзвона демо-реестра переживают перезапуск сервера."""
    db = tmp_path / "demo.db"
    reg = PersistentSchedule(DOCTORS, date(2026, 10, 5), SqliteStore(db))
    patient = reg.identify("Тестов", "Тест", date(1990, 1, 1), create=True)
    first = reg.book(reg.find(specialty="терапевт", limit=1)[0], patient.full_name, patient.id)
    second = reg.book(reg.find(specialty="терапевт", limit=2)[1], patient.full_name, patient.id)
    moved = reg.reschedule(second.id, reg.find(specialty="терапевт", limit=5)[-1])
    reg.cancel(first.id)
    reg.escalate(patient, "Жалоба: тест")
    reg.log_reminder(moved, patient, "CONFIRMED")
    assert stat.S_IMODE(os.stat(db).st_mode) == 0o600            # ФИО и даты рождения — только владельцу

    again = PersistentSchedule(DOCTORS, date(2026, 10, 5), SqliteStore(db))
    assert set(again.bookings) == {moved.id} and again.bookings[moved.id].slot.start == moved.slot.start
    assert again.is_busy(moved.slot.doctor_id, moved.slot.start)   # занятый слот не предложат повторно
    assert again.identify("Тестов", "Тест", date(1990, 1, 1), create=False).id == patient.id
    assert again.escalations[-1]["description"] == "Жалоба: тест" and again.reminder_log[-1]["code"] == "CONFIRMED"
    assert {h["status"] for h in again.visit_log()} == {"Отменена", "Запланирована"}
    third = again.book(again.find(specialty="терапевт", limit=1)[0], patient.full_name, patient.id)
    assert third.id not in (first.id, second.id)                    # нумерация продолжается, а не начинается заново


def test_services_are_built_in_one_place_and_reset_rebinds_telephony(tmp_path):
    cfg = Config(demo_db=str(tmp_path / "demo.db"), student=False, journal_path="")
    services = build_services(cfg)
    assert services.telephony.agent is services.agent and services.agent.domain is services.domain
    assert isinstance(services.domain.schedule, PersistentSchedule) and len(services.domain.schedule.bookings) == 3
    services.reset_demo()
    assert services.telephony.agent is services.agent             # после сброса телефония — на новом агенте
    assert len(services.domain.schedule.bookings) == 3              # демо-пациенты снова на месте


def test_doctor_reports_base_mode_ready_without_secrets(capsys, monkeypatch):
    """make doctor: базовый режим готов без ключей; секреты не печатаются, у каждого пробела — команда исправления."""
    import tools.doctor as doctor

    monkeypatch.setattr(doctor, "_env", lambda: None)  # не подмешивать ключи ВМ в окружение тестов
    monkeypatch.setenv("LLM_API_KEY", "секрет-не-печатать")
    monkeypatch.setattr("sys.argv", ["doctor"])
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert "секрет-не-печатать" not in out and "Базовый режим готов" in out
    assert all("→" in block for block in out.split("✗")[1:])
