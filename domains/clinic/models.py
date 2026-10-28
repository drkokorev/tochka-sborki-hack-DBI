"""Состояние диалога и разбор одной реплики пациента (структура — docs/02_architecture.md)."""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, Field

from .schedule import Booking, Patient, Slot

Intent = Literal["book", "schedule", "status", "reschedule", "cancel", "escalate", "remind", "home_visit", "opt_out"]
RedFlag = Literal["emergency", "selfharm", "urgent", "dental", "complaint", "medical", "human", "paid", "admin", "pdn",
                  "lab_delay"]
Escalation = RedFlag | Literal["misunderstood", "not_found", "mis_error", "waitlist"]
# Вопрос «да/нет», на который ждём ответа (кроме подтверждения записи или отмены)
Question = Literal["more", "operator", "offer_book", "referral", "cancel_or_move", "waitlist",
                   "is_patient", "birth_check", "will_come", "rating", "home_address"]


class ClinicState(BaseModel):
    intent: Intent | None = None
    specialty: str | None = None
    doctor_name: str | None = None
    day: date | None = None
    time_from: time | None = None
    time_to: time | None = None
    week_of: date | None = None         # «на следующей неделе» без дня: понедельник этой недели
    last_name: str | None = None
    first_name: str | None = None
    birth_date: date | None = None
    booking_id: str | None = None
    # Служебные поля: заполняет merge()/step(), не LLM
    patient: Patient | None = None
    offered: list[Slot] = []
    chosen: Slot | None = None
    candidates: list[Booking] = []      # несколько записей пациента: какую переносить или отменять
    target: Booking | None = None       # существующая запись для статуса, переноса или отмены
    awaiting_confirm: bool = False
    confirmed: bool | None = None
    question: Question | None = None
    offer_specialty: str | None = None  # к кому предложили записать («Записать вас?»)
    referral: bool | None = None        # есть ли направление к узкому специалисту (С-02)
    opt_out_code: str | None = None     # входящий отказ от обзвона: OPT_OUT или OPT_OUT_ALL — ждём ФИО для отметки
    home_note: str | None = None        # С-06: адрес и что беспокоит — для заявки на вызов врача на дом
    reminder_code: str | None = None    # итог исходящего напоминания (CONFIRMED, OPT_OUT, WRONG_PERSON…)
    birth_tries: int = 0                # исходящий звонок: сколько раз просили дату рождения для сверки
    referral_asked: int = 0             # сколько раз спросили о направлении (не зацикливаемся)
    move_offered: bool = False          # перед отменой один раз предложили перенос (С-04)
    result: Booking | None = None       # созданная или перенесённая запись
    escalation: Escalation | None = None
    escalation_note: str | None = None
    asked_identity: bool = False
    misunderstood: int = 0              # подряд непонятых реплик (R-05)
    not_found: int = 0                  # неудачных поисков пациента (R-07)
    medical_warned: bool = False        # граничная фраза уже прозвучала (R-04)
    reminder_for: str | None = None     # исходящий звонок-напоминание (С-10) по этой записи
    relaxed: bool = False               # варианты предложены с ослаблением условий пациента
    avoid: list[Slot] = []              # варианты, которые пациент отклонил («первый не подходит»)
    read_back: str = ""                 # ФИО и дата рождения нового пациента, уже повторённые ему вслух
    avoid_days: list[date] = []         # дни, которые не подходят («нужен другой день»)
    say: str | None = None              # справка или реплика перед ответом шага
    rating: int | None = None           # оценка разговора пациентом 1–5 (KPI кейса: удовлетворённость ≥ 4)
    goodbye: bool = False
    finished: bool = False


class ClinicUpdate(BaseModel):
    intent: Intent | None = Field(None, description="Цель пациента, только если он явно её назвал")
    specialty: str | None = Field(None, description="Специальность врача")
    doctor_name: str | None = Field(None, description="Фамилия врача, если названа")
    day: date | None = Field(None, description="Дата приёма YYYY-MM-DD")
    time_from: time | None = Field(None, description="Не раньше HH:MM")
    time_to: time | None = Field(None, description="Не позже HH:MM")
    week_of: date | None = Field(None, description="Понедельник недели YYYY-MM-DD, если пациент назвал неделю "
                                                   "без конкретного дня («на следующей неделе»)")
    choice: int | None = Field(None, description="Номер выбранного варианта из предложенных")
    reject: int | None = Field(None, description="Номер варианта, который пациент отклонил («первый не подходит»); "
                                                 "0 — не подходит ни один")
    other_day: bool | None = Field(None, description="true, если пациент просит другой день")
    avoid_day: date | None = Field(None, description="День YYYY-MM-DD, который пациенту не подходит "
                                                     "(«в понедельник не смогу»)")
    last_name: str | None = Field(None, description="Фамилия пациента в именительном падеже")
    first_name: str | None = Field(None, description="Имя пациента в именительном падеже")
    birth_date: date | None = Field(None, description="Дата рождения пациента YYYY-MM-DD")
    booking_id: str | None = Field(None, description="Номер записи")
    confirm: bool | None = Field(None, description="Ответ да (true) или нет (false) на вопрос ассистента")
    topic: str | None = Field(None, description="Тема справочного вопроса")
    red_flag: RedFlag | None = Field(None, description="Повод перевести на сотрудника")
    note: str | None = Field(None, description="Кратко суть проблемы, если есть red_flag")
    smalltalk: bool | None = Field(None, description="true, если реплика — только приветствие или благодарность")
    bye: bool | None = Field(None, description="true, если пациент прощается или хочет закончить разговор")
    rating: int | None = Field(None, description="Оценка разговора от 1 до 5, если ассистент попросил оценить")


# Поля поиска слота и поля идентификации пациента (как на стенде МИС)
SEARCH_FIELDS = ("specialty", "doctor_name", "day", "time_from", "time_to", "week_of")
IDENTITY_FIELDS = ("last_name", "first_name", "birth_date")
