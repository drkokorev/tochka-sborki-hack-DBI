"""Домен «Точки сборки» (DBI): голосовой ассистент регистратуры. Ответы короткие и пригодны для TTS.

Диалоги следуют регламенту кейса: сценарии С-01…С-05, С-07, С-11, С-14, С-15, справки раздела 6
и матрица маршрутизации 7.3 (красные флаги, перевод на сотрудника, непонимание).
Пациент определяется фамилией, именем и датой рождения — как на стенде МИС.
"""

from __future__ import annotations

from datetime import date

from .dialog import DialogMerge
from .models import (
    IDENTITY_FIELDS, SEARCH_FIELDS, ClinicState, ClinicUpdate, Escalation, Intent, Question, RedFlag,
)
from .replies import (
    ANYTHING_ELSE, ASK_AGAIN, ASK_CARD, ASK_SPECIALTY, CLARIFY, EMERGENCY, EXTRACTION_PROMPT, GOODBYE, GREETING,
    LISTENING, MEDICAL_BOUNDARY, OFFER_OPERATOR, RATE_ASK, RATE_THANKS, REMIND_CONFIRMED, REMIND_HELLO,
    REMIND_UNSURE, REMIND_WRONG, ROUTES, TRANSFER, Route,
)
from .scenarios import REFERRAL_ONLY, Scenarios
from .schedule import DOCTORS, Schedule
from .speech import SpeechNormalizer
from .understanding import Understanding

__all__ = [
    "ClinicDomain", "ClinicState", "ClinicUpdate", "Intent", "RedFlag", "Escalation", "Question",
    "SEARCH_FIELDS", "IDENTITY_FIELDS", "REFERRAL_ONLY", "Route", "ROUTES", "EXTRACTION_PROMPT",
    "GREETING", "CLARIFY", "OFFER_OPERATOR", "ASK_SPECIALTY", "ANYTHING_ELSE", "LISTENING", "ASK_AGAIN", "GOODBYE",
    "RATE_ASK", "RATE_THANKS", "REMIND_HELLO", "REMIND_WRONG", "REMIND_CONFIRMED", "REMIND_UNSURE",
    "MEDICAL_BOUNDARY", "ASK_CARD", "TRANSFER", "EMERGENCY",
]


class ClinicDomain(Understanding, DialogMerge, Scenarios):
    """Домен для core.agent.Agent: понимание (understanding.py), слияние (dialog.py), сценарии (scenarios.py)."""

    name = "clinic"
    state_model = ClinicState
    update_model = ClinicUpdate
    off_topic_reply = "Я помогаю записаться к врачу, перенести или отменить запись и отвечаю на вопросы о поликлинике."

    def __init__(self, today: date | None = None, schedule: Schedule | None = None, student=None,
                 use_phrasebook: bool = True):
        self.schedule = schedule or Schedule(DOCTORS, today or date.today())
        self.student = student  # дистиллированная модель-ученик (domains/clinic/student.py) или None
        self.use_phrasebook = use_phrasebook  # фразник: стандартные фразы из хэш-таблицы, без модели
        self.specialties = sorted({d.specialty for d in self.schedule.doctors.values() if d.is_active})
        self.speech = SpeechNormalizer([d.surname for d in self.schedule.doctors.values()])

    def set_schedule(self, schedule: Schedule) -> None:
        """Новый реестр (стенд подключился после старта): пересчитываем то, что зависит от справочника врачей."""
        self.schedule = schedule
        self.specialties = sorted({d.specialty for d in schedule.doctors.values() if d.is_active})
        self.speech = SpeechNormalizer([d.surname for d in schedule.doctors.values()])

    def expects_short_answer(self, st: ClinicState) -> bool:
        """Ассистент ждёт «да/нет» или оценку: конец фразы можно определять по короткой паузе (0,45 с вместо 0,8).

        «Могу ещё чем-то помочь?» сюда не входит: после «да» часто идёт новый запрос с паузой."""
        return bool(st.awaiting_confirm or st.question) and st.question != "more"

