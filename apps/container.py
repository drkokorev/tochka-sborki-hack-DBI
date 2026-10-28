"""Сборка приложения — единственное место, где создаются зависимости (composition root).

Модули не создают друг друга сами: агент получает домен и LLM, домен — реестр, телефония — агента, голос
и журнал. Здесь же выбираются реализации по окружению: стенд МИС или демо-реестр (SQLite), есть ли LLM,
ученик, модель неявок. Веб-сервер (apps/web/server.py) берёт готовый `Services`, тесты могут собрать свой.
Переменные сборки читаются в `Config.from_env()`; ключи LLM — в core/config.py, голоса и телефонии — в их модулях
(полный список — .env.example).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from apps.journal import Journal
from apps.telephony import Telephony
from apps.voice import LocalSpeechToText, SpeechToText, TextToSpeech
from core.agent import Agent
from core.config import get_settings
from core.llm import OpenAICompatLLM
from domains.clinic.domain import GREETING, ClinicDomain
from domains.clinic.mis import MisClient, MisError, MisRegistry, stand_today
from domains.clinic.noshow import NoShowModel
from domains.clinic.schedule import DOCTORS, Schedule, seed_demo_bookings
from domains.clinic.store import PersistentSchedule, SqliteStore
from domains.clinic.student import Student
from domains.clinic.timetable import Timetable

log = logging.getLogger(__name__)

RESULTS = Path("eval/results")  # результаты замеров в git; data/ — рабочая папка на машине (не в git)


@dataclass(frozen=True)
class Config:
    """Всё, что сборка читает из окружения, — в одном месте."""

    mis_url: str = ""                        # адрес стенда МИС (DBI_API_URL)
    mis_token: str = ""
    timetable_dir: str = "data"
    demo_db: str = "data/demo.db"            # пусто — демо-реестр только в памяти
    student: bool = True
    student_dir: str = "data/student"
    noshow_model: str = ""
    journal_path: str = "data/journal.json"  # пусто — журнал только в памяти
    desk_token: str = ""                     # токен рабочего места регистратора; пусто — открытый демо-режим

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            mis_url=os.getenv("DBI_API_URL", cls.mis_url),
            mis_token=os.getenv("DBI_API_TOKEN", ""),
            timetable_dir=os.getenv("TIMETABLE_DIR", cls.timetable_dir),
            demo_db=os.getenv("DEMO_DB", cls.demo_db),
            student=os.getenv("STUDENT", "1") == "1",
            student_dir=os.getenv("STUDENT_DIR", cls.student_dir),
            noshow_model=os.getenv("NOSHOW_MODEL", ""),
            journal_path=os.getenv("JOURNAL_PATH", cls.journal_path),
            desk_token=os.getenv("DESK_TOKEN", ""),
        )


class OfflineRegistry(Schedule):
    """Реестр на время, пока стенд недоступен: любое действие с записями — сбой МИС (R-06), справки работают."""

    offline = True

    def __init__(self):
        super().__init__([], stand_today())

    def find(self, **kwargs):
        raise MisError(503, "СТЕНД_НЕДОСТУПЕН", "Стенд МИС недоступен")

    def identify(self, *args, **kwargs):
        raise MisError(503, "СТЕНД_НЕДОСТУПЕН", "Стенд МИС недоступен")


def result_path(name: str) -> str:
    """Свежий результат из data/, если он есть на машине, иначе — закоммиченный в eval/results/."""
    return f"data/{name}" if Path(f"data/{name}").exists() else str(RESULTS / name)


def stand_registry(cfg: Config) -> MisRegistry:
    """Реестр поверх стенда МИС; без расписания кейса время не предлагается (перевод к регистратору)."""
    return MisRegistry(MisClient(cfg.mis_url, cfg.mis_token), timetable=Timetable.load(cfg.timetable_dir),
                       require_timetable=True)


def demo_registry(cfg: Config) -> Schedule:
    """Демо без стенда: SQLite (записи переживают перезапуск) или память; демо-пациенты — при первом запуске."""
    if cfg.demo_db:
        schedule: Schedule = PersistentSchedule(DOCTORS, date.today, SqliteStore(cfg.demo_db))  # часы, а не дата старта
        if schedule.store.is_empty():
            seed_demo_bookings(schedule)
        return schedule
    schedule = Schedule(DOCTORS, date.today)
    seed_demo_bookings(schedule)
    return schedule


def build_agent(cfg: Config, student: Student | None) -> tuple[Agent, ClinicDomain]:
    settings = get_settings()
    llm = OpenAICompatLLM(settings) if settings.llm_enabled else None
    today: Callable[[], date] = date.today
    if cfg.mis_token:
        try:
            domain = ClinicDomain(schedule=stand_registry(cfg), student=student)
        except Exception:
            # Стенд недоступен при старте: сервер всё равно поднимается — справки и переводы работают,
            # действия с записями уходят к регистратору (R-06), реестр подключается в фоне (server._reconnect_mis)
            log.exception("Стенд МИС недоступен при старте — работаем без него, переподключимся в фоне")
            domain = ClinicDomain(schedule=OfflineRegistry(), student=student)
        today = stand_today
    else:
        domain = ClinicDomain(schedule=demo_registry(cfg), student=student)
    return Agent(domain, llm, nlg=settings.llm_nlg, today=today), domain


@dataclass
class Services:
    """Собранное приложение: всё, что нужно веб-серверу и телефонии."""

    cfg: Config
    agent: Agent
    domain: ClinicDomain
    student: Student | None
    local_stt: LocalSpeechToText
    stt: SpeechToText
    tts: TextToSpeech
    journal: Journal
    noshow: NoShowModel | None
    telephony: Telephony

    def start_reminder(self, booking_id: str):
        return self.domain.start_reminder(self.domain.schedule.bookings[booking_id])

    def not_reached(self, booking_id: str) -> None:
        """Пациент не взял трубку — «не дозвонились» в карточку (С-10)."""
        booking = self.domain.schedule.bookings.get(booking_id)
        if booking is not None:
            self.domain.schedule.log_reminder(booking, self.domain.schedule.patient_of(booking), "NOT_REACHED")

    def reset_demo(self) -> None:
        """Демо к исходному состоянию: чистый реестр и новый агент — и у веба, и у телефонии."""
        if isinstance(self.domain.schedule, PersistentSchedule):
            self.domain.schedule.store.clear()
        self.agent, self.domain = build_agent(self.cfg, self.student)
        self.telephony.agent = self.agent


def build_services(cfg: Config | None = None) -> Services:
    cfg = cfg or Config.from_env()
    student = Student.load(cfg.student_dir) if cfg.student else None
    agent, domain = build_agent(cfg, student)
    local_stt = LocalSpeechToText()
    stt = SpeechToText(local_stt)  # распознавание — только на машине, облачного резерва нет (152-ФЗ)
    tts = TextToSpeech()
    journal = Journal(path=cfg.journal_path or None)
    noshow = NoShowModel.load(cfg.noshow_model or result_path("noshow_model.json"))
    services = Services(cfg, agent, domain, student, local_stt, stt, tts, journal, noshow, telephony=None)  # type: ignore[arg-type]
    services.telephony = Telephony(agent, stt, tts, greeting=GREETING, journal=journal,
                                   start_reminder=services.start_reminder, not_reached=services.not_reached)
    return services
