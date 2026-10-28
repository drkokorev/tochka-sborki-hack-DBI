"""Слияние разобранной реплики с состоянием звонка: ответы на вопросы, справки, перевод, смена сценария.

Детерминированный код без LLM: какая цель у звонка, что пациент выбрал и что подтвердил.
"""

from __future__ import annotations

import re
from typing import Any

from . import reminders

# Настойчивость после граничной фразы (R-04): пациент снова просит совет, а не просто отказывается от записи
_INSISTS = re.compile(r"посоветуйте|нужен .{0,10}совет|подскажите|что (мне )?(выпить|принять|пить)|какие (таблетки|лекарства)|"
                      r"чем (лечить|сбить|снять)|скажите,? что")
from .faq import FAQ_BY_KEY, faq_answer
from .models import IDENTITY_FIELDS, SEARCH_FIELDS, ClinicState, ClinicUpdate
from .replies import (
    ASK_AGAIN, MEDICAL_BOUNDARY, NO_SPECIALIST, RATE_THANKS, REMIND_CONFIRMED, REMIND_UNSURE, REMIND_WRONG, ROUTES,
)


class DialogMerge:
    """Часть ClinicDomain: merge(state, update) → новое состояние."""

    def _keep(self, state: ClinicState) -> dict[str, Any]:
        """Что переживает смену сценария в одном звонке: кто звонит."""
        return {"patient": state.patient, "last_name": state.last_name, "first_name": state.first_name,
                "birth_date": state.birth_date, "medical_warned": state.medical_warned,
                "reminder_for": state.reminder_for}

    def _farewell(self, state: ClinicState, keep: dict) -> ClinicState:
        """Прощание. После выполненной записи, переноса или отмены — один раз просим оценить разговор."""
        if state.finished and state.intent in ("book", "reschedule", "cancel") and not state.escalation:
            return ClinicState(**keep, question="rating")
        return ClinicState(**keep, goodbye=True)

    def merge(self, state: ClinicState, update: ClinicUpdate) -> ClinicState:
        keep = self._keep(state)
        if (state.finished and state.intent == "escalate" and state.escalation
                and update.red_flag in (None, "human", state.escalation) and not update.intent):
            # Пациент уже передан регистратору: «хорошо, жду» — не повод для новой заявки
            st = state.model_copy(deep=True)
            st.say = None
            return st
        if state.question == "rating" and not update.red_flag:
            # Ответ на просьбу оценить: число — сохраняем, что угодно другое — просто прощаемся
            if update.rating and 1 <= update.rating <= 5:
                return ClinicState(**keep, goodbye=True, rating=update.rating, say=RATE_THANKS)
            if not (update.intent or update.specialty or update.topic):
                return ClinicState(**keep, goodbye=True)
        flag, note = update.red_flag, update.note
        faq = FAQ_BY_KEY.get(update.topic or "")
        say = None
        if faq and not flag:
            say, offer = faq_answer(faq.key, update.note or "")
            if faq.key == "price" and update.specialty:
                offer = update.specialty
            if faq.transfer:
                flag, note = faq.transfer, f"{faq.hint}: {update.note or ''}".strip(": ")

        if (flag == "medical" and state.medical_warned and state.question == "offer_book" and update.confirm is False
                and not _INSISTS.search((update.note or "").lower())):
            # «Нет, спасибо, сама решу, что делать» после граничной фразы — отказ от записи, а не настойчивость
            flag = None
            update = update.model_copy(update={"red_flag": None})
        if flag == "medical" and not state.medical_warned:
            # Регламент 6.9 и уточнение DBI 20.10: без красного флага на сотрудника не переводим — граничная фраза
            # и предложение записи. Перевод по R-04 — только если пациент настаивает (повторный вопрос, отказ от записи)
            if state.intent and not state.finished:
                # вопрос посреди записи или напоминания — граничная фраза, и разговор продолжается с того же места
                st = state.model_copy(deep=True)
                st.medical_warned, st.say = True, MEDICAL_BOUNDARY
                return st
            # «К какому врачу идти с …? — Начните с участкового терапевта» (регламент 6.9)
            return ClinicState(**(keep | {"medical_warned": True}), say=MEDICAL_BOUNDARY, question="offer_book",
                               offer_specialty=self._normalize_specialty("терапевт"))
        if flag == "urgent" and not reminders.registry_open(reminders.clinic_now()):
            flag = "emergency"  # вне часов регистратуры категория Б обрабатывается как А: рекомендация 103 (регламент 7.4)
        if flag:
            route = ROUTES[flag]
            st = ClinicState(**keep, intent="escalate", escalation=flag,
                             escalation_note=f"{route.label}: {note}" if note else route.label, say=say)
            return self._merge_identity(st, update)

        if state.awaiting_confirm and update.confirm is False and any(getattr(update, k) for k in IDENTITY_FIELDS):
            # «Верно?» — «нет, я Иванов Иван, 12.05.1960»: исправляют личность, выбранное время остаётся
            update = update.model_copy(update={"confirm": None})
        if state.question == "home_address" and not update.red_flag:
            if update.confirm is False:
                return ClinicState(**keep, say="Хорошо, заявку на вызов врача не оформляю.")
            if update.note:
                st = state.model_copy(deep=True)
                st.home_note, st.question = update.note, None
                return st
        if state.question == "birth_check" and not update.red_flag and not update.topic:
            st = state.model_copy(deep=True)
            if update.birth_date or update.day:
                return self._check_birth(st, update, keep)
            if update.confirm is False:  # «это не я», «вы ошиблись»
                return self._remind_end(st, "WRONG_PERSON", REMIND_WRONG)
            if update.bye:  # «перезвоните позже», «до свидания» — деталей не раскрыли, итог «не удалось поговорить»
                return self._remind_end(st, "NO_ANSWER", "Хорошо, извините за беспокойство. До свидания!")
            return self._check_birth(st, update, keep)
        if (answered := self._answer_question(state, update, keep)) is not None:
            if faq and say and state.question == "offer_book" and update.confirm is False:
                answered.say = say  # «Нет, записываться не нужно. До скольки в субботу?» — справку повторяем
            return answered
        if update.bye and not (update.intent or update.choice or update.confirm):
            return self._farewell(state, keep)  # «Позвоню позже, спасибо» — прощаемся, не держим пациента

        if faq:
            st = ClinicState(**keep) if state.finished or state.intent is None else state.model_copy(deep=True)
            st.question = None
            has_request = update.intent or (update.specialty and faq.key != "price") or update.doctor_name
            if has_request:
                st = self._merge_request(st, update)
            if faq.key == "opt_out":
                # С-10: «не звоните больше» — только звонки, SMS приходят; «и SMS не надо» — отказ от всех уведомлений
                full = bool(re.search(r"смс|sms|сообщени|уведомлени|ничего не (присылайте|отправляйте)",
                                      (update.note or "").lower()))
                code = "OPT_OUT_ALL" if full else "OPT_OUT"
                if full:
                    say = "Хорошо, напоминаний больше не будет — ни звонком, ни SMS."
                if state.reminder_for and state.target:
                    return self._remind_end(st, code, say)
                if state.patient is not None and state.patient.id:
                    self.schedule.set_opt_out(state.patient, code)  # входящий звонок опознанного пациента
                else:
                    # отметку ставим в карточку — сначала узнаём, чья она
                    return ClinicState(**keep, intent="opt_out", opt_out_code=code, say=say)
            if faq.key == "home_visit":
                # С-06: заявку принимает ассистент — дальше ФИО и дата рождения, адрес и что беспокоит
                st = ClinicState(**keep, intent="home_visit", say=say)
                return self._merge_identity(st, update)
            if faq.key == "robot":
                st.question = "operator"  # «Если хотите, соединю с сотрудником»
            elif st.intent is None and offer is not None:
                st.question = "offer_book"
                st.offer_specialty = self._normalize_specialty(offer) if offer else None
            st.say, st.misunderstood = say, 0
            return self._merge_identity(st, update)

        if update.smalltalk and state.finished and not update.model_dump(exclude_none=True, exclude={"smalltalk", "bye"}):
            return self._farewell(state, keep)  # «спасибо, до свидания» после результата
        if not update.model_dump(exclude_none=True, exclude={"smalltalk", "note", "bye"}):
            st = state.model_copy(deep=True)
            if st.finished:
                st = ClinicState(**keep)
            if update.smalltalk:
                return st  # «здравствуйте», «спасибо» — не повод переводить на регистратора
            if state.question == "will_come":  # «не знаю, может быть» — статус не меняем
                return self._remind_end(st, "UNSURE", REMIND_UNSURE)
            if state.question == "is_patient" and st.misunderstood >= 1:
                return self._remind_end(st, "NO_ANSWER", REMIND_WRONG)
            if state.question == "is_patient":
                st.misunderstood += 1
                return st
            st.misunderstood += 1  # R-05: переспросить, потом предложить оператора, на третий раз — перевод
            if st.misunderstood >= 3 and st.intent != "escalate":
                route = ROUTES["misunderstood"]
                st = ClinicState(**keep, intent="escalate", escalation="misunderstood", escalation_note=route.label)
            return st

        return self._merge_identity(self._merge_request(state, update), update)

    def _answer_question(self, state: ClinicState, update: ClinicUpdate, keep: dict) -> ClinicState | None:
        """Ответ «да/нет» на вопрос ассистента. None — реплика не про вопрос."""
        q = state.question
        if q == "offer_book" and update.confirm is None and (update.day or update.time_from or update.week_of) \
                and not (update.intent or update.specialty or update.doctor_name):
            update = update.model_copy(update={"confirm": True})  # «Записать вас?» — «в понедельник»: это «да»
        if q is None or update.confirm is None:
            return None
        if self._is_new_request(state, update):
            return None  # «да, и ещё к лору» — новый запрос важнее
        yes = update.confirm
        if q == "is_patient":
            st = state.model_copy(deep=True)
            if yes:
                if st.patient is None:
                    st.question = "will_come"  # карточка не прочиталась — сверять не с чем
                    return st
                st.question = "birth_check"  # «да» — ещё не проверка личности: дату рождения называет сам пациент
                return self._check_birth(st, update, keep) if update.birth_date else st
            return self._remind_end(st, "WRONG_PERSON", REMIND_WRONG)
        if q == "will_come":
            st = state.model_copy(deep=True)
            if yes:
                return self._remind_end(st, "CONFIRMED", REMIND_CONFIRMED)
            # «Не получится» — регламент: сразу предложить перенос
            return ClinicState(**keep, intent="reschedule", target=state.target, say="Давайте перенесу.")
        if q == "more":
            return ClinicState(**keep, say="Слушаю вас.") if yes else self._farewell(state, keep)
        if q == "operator":
            if yes or update.red_flag == "human":  # «нет, спасибо, мне нужен человек» — «нет» относится к роботу
                reason = "misunderstood" if state.misunderstood else "human"
                return self._merge_identity(ClinicState(**keep, intent="escalate", escalation=reason,
                                                        escalation_note=ROUTES[reason].label), update)
            if update.bye:
                return self._farewell(state, keep)  # «Нет, спасибо. Всего доброго!» — прощаемся, а не переспрашиваем
            st = state.model_copy(deep=True)
            # после непонятой реплики (R-05) — переспросить; после «Вы робот?» — просто предложить помощь
            st.question, st.say = None, ASK_AGAIN if state.misunderstood else "Хорошо. Чем могу помочь?"
            return st
        if q == "offer_book":
            if not yes:  # «Нет, спасибо. До свидания» — прощаемся, а не спрашиваем «Могу ещё чем-то помочь?»
                return self._farewell(state, keep) if update.bye else ClinicState(**keep, say="Хорошо.")
            if state.intent == "schedule":  # «Когда принимает Белова?» — «Записать вас?» — «Да»: к ней же
                st = state.model_copy(update={"intent": "book", "question": None})
            else:
                st = ClinicState(**keep, intent="book", specialty=state.offer_specialty)
            # «Да, на субботу» — уточнения модели применяем к новой записи
            st = self._merge_request(st, update.model_copy(update={"intent": None, "confirm": None, "choice": None}))
            return self._merge_identity(st, update)
        st = state.model_copy(deep=True)
        st.question = None
        if q == "referral":
            st.referral = yes
        elif q == "cancel_or_move":
            if yes:
                return ClinicState(**keep, intent="reschedule", target=state.target)
        elif q == "waitlist":
            if not yes:
                return ClinicState(**keep, say="Хорошо.")
            st = ClinicState(**keep, intent="escalate", escalation="waitlist",
                             escalation_note=f"{ROUTES['waitlist'].label}: {state.specialty or state.doctor_name}")
        return self._merge_identity(st, update)

    def _check_birth(self, st: ClinicState, update: ClinicUpdate, keep: dict) -> ClinicState:
        """Исходящий звонок: собеседник сам называет дату рождения, сверяем с карточкой (регламент 3.5).
        Совпала — напоминание; не совпала дважды или так и не названа — извиняемся, деталей не раскрываем."""
        if update.day and not update.birth_date:
            st.question, st.say = "birth_check", "Нужна дата рождения полностью — с годом."
            return st  # «двенадцатого мая» — без года сверить нельзя, попыткой не считаем
        st.birth_tries += 1
        if update.birth_date and st.patient is not None and update.birth_date == st.patient.birth_date:
            st.question = "will_come"
            return st
        if st.birth_tries >= 3 or (update.birth_date and st.birth_tries >= 2):
            return self._remind_end(st, "WRONG_PERSON" if update.birth_date else "NO_ANSWER", REMIND_WRONG)
        st.question = "birth_check"
        if update.birth_date:
            st.say = "Дата не совпала с карточкой."
        return st

    def _is_new_request(self, state: ClinicState, update: ClinicUpdate) -> bool:
        """Есть ли в реплике новый запрос, а не повтор известного (модель любит повторять контекст)."""
        if update.intent and update.intent != state.intent and not (
                state.question == "offer_book" and update.intent == "book"):
            return True
        # врач и специальность записи, о которой говорим (напоминание, перенос), — повтор контекста, а не новый запрос:
        # на «Вы придёте?» модель отвечает «да» и заодно повторяет «терапевт Ковалёв»
        target = state.target.slot if state.target is not None else None
        if update.specialty and self._normalize_specialty(update.specialty) not in (
                state.specialty, state.offer_specialty, target and target.specialty):
            return True
        if update.doctor_name:
            doctor = self.schedule.find_doctor(update.doctor_name)
            if doctor and doctor.name not in (state.doctor_name, target and target.doctor_name):
                return True
        return bool(update.choice and state.offered and not state.chosen)

    @staticmethod
    def _drop_echo(st: ClinicState, changes: dict) -> dict:
        """Убирает поля, повторяющие уже выбранный слот или запись, о которой идёт речь: модель повторяет врача,
        день и время, а «новый запрос» сбросил бы подтверждение («Отменяю запись… Верно?» — «Да» — снова вопрос)."""
        same: dict[str, bool] = {}
        slots = [(st.chosen, True)]
        if st.target is not None:  # у переноса новый день — законный запрос, поэтому для него только врач
            slots.append((st.target.slot, st.intent in ("cancel", "status", "remind")))
        for slot, with_time in slots:
            if slot is None:
                continue
            checks = {"specialty": changes.get("specialty") == slot.specialty,
                      "doctor_name": changes.get("doctor_name") == slot.doctor_name}
            if with_time:
                checks |= {
                    "day": changes.get("day") == slot.start.date(),
                    "time_from": changes.get("time_from") is not None and changes["time_from"] <= slot.start.time(),
                    "time_to": changes.get("time_to") is not None and slot.start.time() < changes["time_to"],
                }
            same = {k: same.get(k, False) or v for k, v in {**same, **checks}.items()}
        return {k: v for k, v in changes.items() if not same.get(k)}

    def _merge_request(self, state: ClinicState, update: ClinicUpdate) -> ClinicState:
        keep = self._keep(state)
        intent = update.intent
        if state.intent == "reschedule" and intent == "book":
            intent = None  # «запишите на второй» внутри переноса — это выбор, а не новая запись
        if state.reminder_for and intent in ("reschedule", "cancel") and state.target and state.intent == "remind":
            # «Завтра не смогу, перенесите» в звонке-напоминании: «завтра» — это старая запись, не новое время
            st = ClinicState(**keep, target=state.target)
            update = update.model_copy(update={k: None for k in SEARCH_FIELDS})
        elif state.finished or (intent and state.intent and intent != state.intent):
            st = ClinicState(**keep)
        else:
            st = state.model_copy(deep=True)
        st.misunderstood, st.question = 0, None
        if intent:
            st.intent = intent

        if (st.awaiting_confirm and update.confirm is False and update.choice and st.offered
                and 1 <= update.choice <= len(st.offered) and st.offered[update.choice - 1] != st.chosen):
            # «Верно?» — «нет, лучше второй вариант»: берём другой из названных и снова переспрашиваем
            st.chosen, st.confirmed, st.awaiting_confirm = st.offered[update.choice - 1], None, False
            update = update.model_copy(update={"confirm": None, "choice": None})
        if update.confirm is not None and st.awaiting_confirm:
            st.confirmed = update.confirm
        elif update.confirm and st.intent == "schedule":
            st.intent = "book"  # «Записать вас?» — «Да»

        if update.reject is not None and st.offered and not st.chosen:
            # Отклонённые варианты больше не предлагаем; «нужен другой день» — и их дни тоже
            bad = st.offered if not 1 <= update.reject <= len(st.offered) else [st.offered[update.reject - 1]]
            st.avoid = [*st.avoid, *bad]
            if update.other_day:
                st.avoid_days = sorted({*st.avoid_days, *(s.start.date() for s in bad)})
            st.offered, st.relaxed, st.say = [], False, "Хорошо."
        if update.avoid_day:
            st.avoid_days = sorted({*st.avoid_days, update.avoid_day})
            if any(o.start.date() == update.avoid_day for o in st.offered) and not st.chosen:
                st.offered, st.relaxed, st.say = [], False, "Хорошо."
        if st.offered and not st.chosen and (update.day or update.time_from):
            # День и время из реплики, указывающие ровно на один предложенный слот, надёжнее номера от модели
            hits = [o for o in st.offered if (update.day is None or o.start.date() == update.day)
                    and (update.time_from is None or o.start.time() == update.time_from)]
            if len(hits) == 1:
                st.chosen = hits[0]
        if update.confirm and st.offered and len(st.offered) == 1 and not st.chosen and not st.awaiting_confirm:
            st.chosen = st.offered[0]  # «Свободно одно время. Подходит?» — «Да»
        if update.choice and st.offered and not st.chosen and 1 <= update.choice <= len(st.offered):
            # Выбор применяем до сравнения полей: «первый» + повтор даты от модели не должен сбрасывать варианты
            st.chosen = st.offered[update.choice - 1]
        changes = update.model_dump(include=set(SEARCH_FIELDS), exclude_none=True)
        if "specialty" in changes:
            changes["specialty"] = self._normalize_specialty(changes["specialty"])
            if (changes["specialty"] is None and not changes.get("doctor_name") and not st.specialty
                    and st.intent in (None, "book", "schedule") and self.specialties):  # реестр недоступен — R-06
                # «к эндокринологу», а такого врача в клинике нет — говорим прямо и предлагаем терапевта,
                # а не переспрашиваем «к какому врачу» по кругу
                return st.model_copy(update={"intent": None, "question": "offer_book", "say": NO_SPECIALIST,
                                             "offer_specialty": self._normalize_specialty("терапевт")})
        if "doctor_name" in changes:
            doctor = self.schedule.find_doctor(changes["doctor_name"])
            changes["doctor_name"] = doctor.name if doctor else None
            if doctor:
                changes["specialty"] = doctor.specialty
        changes = {k: v for k, v in self._drop_echo(st, changes).items() if v is not None and v != getattr(st, k)}
        if "day" in changes and st.week_of:
            changes["week_of"] = None  # назван конкретный день — неделя больше не нужна
        if changes:
            if "specialty" in changes:
                st.referral, st.referral_asked = None, 0
            if "specialty" in changes or "doctor_name" in changes:
                st.avoid, st.avoid_days = [], []  # другой врач — прежние отказы не в счёт
            st = st.model_copy(update=changes)
            st.offered, st.chosen, st.confirmed, st.awaiting_confirm = [], None, None, False
            if st.candidates:
                st.candidates = [b for b in st.candidates if self._same_specialty(b, st.specialty)] or st.candidates

        if update.choice:
            if st.offered and not st.chosen and 1 <= update.choice <= len(st.offered):
                st.chosen = st.offered[update.choice - 1]
            elif st.candidates and not st.target and 1 <= update.choice <= len(st.candidates):
                st.target, st.candidates = st.candidates[update.choice - 1], []
        if update.booking_id and update.booking_id.strip().upper() != st.booking_id:
            st.booking_id = update.booking_id.strip().upper()
            if not (st.target and st.target.id == st.booking_id):
                st.target = None
        return st

    def _merge_identity(self, st: ClinicState, update: ClinicUpdate) -> ClinicState:
        ident = update.model_dump(include=set(IDENTITY_FIELDS), exclude_none=True)
        ident = {k: (v.strip().capitalize() if isinstance(v, str) else v) for k, v in ident.items()}
        if ident and any(getattr(st, k) != v for k, v in ident.items()):
            st = st.model_copy(update=ident)
            st.patient = None  # данные поменялись — пациента ищем заново
            if not st.booking_id:
                st.target = None
        return st

