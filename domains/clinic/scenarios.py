"""Сценарии регламента (С-01…С-05, С-07, С-10, С-11, 7.3): что ответить и что сделать в МИС.

step() выбирает сценарий по цели звонка; ответы собираются из данных МИС и расписания по шаблонам replies.py.
"""

from __future__ import annotations

from datetime import timedelta

import httpx

from core.agent import Step
from core.ru import birth_words, on_date

from . import reminders
from .mis import MisError
from .models import ClinicState
from .offers import _distinct_times, _render_offers, _short, _spread, _to_specialist
from .replies import (
    ANYTHING_ELSE, ASK_AGAIN, ASK_CARD, ASK_SPECIALTY, CLARIFY, GOODBYE, LISTENING, OFFER_OPERATOR, RATE_ASK, REMIND_HELLO,
    ROUTES, TRANSFER, transfer_header,
)
from .schedule import Booking, Patient, Slot, SlotTakenError, _cabinet, on_day

_QUIET_SAY = {"Слушаю вас.", ASK_AGAIN}

# С-02: к этим специалистам — только по направлению терапевта (педиатра)
REFERRAL_ONLY = ("кардиолог", "невролог", "эндокринолог", "гастроэнтеролог", "уролог", "пульмонолог")


class Scenarios:
    """Часть ClinicDomain: step(state) → реплика и действие в МИС."""

    def step(self, st: ClinicState) -> Step:
        try:
            step = self._dispatch(st)
        except (MisError, httpx.HTTPError):  # R-06: сбой МИС — к регистратору, а не «перезвоните позже»
            st.intent, st.escalation, st.finished = "escalate", "mis_error", True
            step = Step(ROUTES["mis_error"].phrase, done=True)
        if st.say:
            step.reply = f"{st.say} {step.reply}".strip()
            st.say = None
        return step

    def _dispatch(self, st: ClinicState) -> Step:
        if st.goodbye:
            st.goodbye, st.finished = False, True
            return Step(GOODBYE, done=True)
        if st.question == "rating":
            return Step(RATE_ASK)
        if st.intent is None and (st.specialty or st.doctor_name or st.day or st.time_from or st.time_to):
            st.intent = "book"  # «к лору», «а на завтра есть?» — это запись
        if getattr(self.schedule, "offline", False) and st.intent in ("book", "schedule", "status", "reschedule", "cancel"):
            raise MisError(503, "СТЕНД_НЕДОСТУПЕН", "Реестр недоступен")  # R-06: к регистратору, справки работают
        if st.intent is None:
            if st.question == "offer_book":
                to = " к терапевту" if (st.offer_specialty or "").startswith("терапевт") else ""
                return Step(f"Записать вас{to}?")
            if st.say:
                quiet = st.question == "operator" or st.say.endswith("?") or st.say in _QUIET_SAY
                if not quiet:
                    st.question = "more"  # «Могу ещё чем-то помочь?» — «да» значит «слушаю», «нет» — прощаемся
                return Step("" if quiet else ANYTHING_ELSE)
            if st.misunderstood == 1:
                return Step(CLARIFY)
            if st.misunderstood >= 2:
                st.question = "operator"
                return Step(OFFER_OPERATOR)
            return Step(ANYTHING_ELSE if st.patient else LISTENING)
        return getattr(self, f"_step_{st.intent}")(st)

    # ---------- исходящее напоминание (С-10) ----------

    def start_reminder(self, booking: Booking) -> tuple[ClinicState, str]:
        """Состояние и первая реплика исходящего звонка. Называем только имя и отчество (уточнение DBI 20.10);
        врача и время — только после того, как собеседник сам назовёт дату рождения и она совпадёт с карточкой."""
        patient = self.schedule.patient_of(booking)
        name = (" ".join(x for x in (patient.first_name, patient.middle_name) if x) if patient
                else (booking.patient_name.split() + ["", ""])[1] or "пациент")
        st = ClinicState(intent="remind", patient=patient, target=booking, reminder_for=booking.id,
                         question="is_patient")
        return st, REMIND_HELLO.format(name=name)

    def _remind_end(self, st: ClinicState, code: str, reply: str) -> ClinicState:
        if st.target is not None:
            self.schedule.log_reminder(st.target, st.patient, code)
        return ClinicState(**self._keep(st), intent="remind", target=st.target, finished=True, say=reply,
                           reminder_code=code)

    def _step_remind(self, st: ClinicState) -> Step:
        if st.finished:
            return Step("", done=True)
        assert st.target is not None
        if st.question == "is_patient":
            p = st.patient
            name = " ".join(x for x in (p.first_name, p.middle_name) if x) if p else "пациент"
            return Step(f"Это поликлиника номер семь, напоминание о записи. Скажите, пожалуйста, это {name}?")
        if st.question == "birth_check":
            return Step("Чтобы я убедилась, что говорю с вами, назовите, пожалуйста, вашу дату рождения.")
        slot = st.target.slot
        tomorrow = slot.start.date() == self.schedule.today + timedelta(days=1)
        when = f"завтра, {slot.when()}" if tomorrow else slot.when()
        where = f", кабинет {_cabinet(slot.cabinet)}" if slot.cabinet else ""
        st.question = "will_come"
        reply = (f"Напоминаю: у вас запись {when}, {slot.specialty} {slot.doctor_name.split()[0]}{where}. "
                 "Вы придёте?")
        return Step(reply, facts=reply)

    def _reminder_outcome(self, st: ClinicState, code: str) -> None:
        if st.reminder_for and st.target is not None and st.target.id == st.reminder_for:
            self.schedule.log_reminder(st.target, st.patient, code)

    def _done(self, st: ClinicState, reply: str, payload=None) -> Step:
        st.finished, st.question = True, "more"
        return Step(f"{reply} {ANYTHING_ELSE}", facts=reply, payload=payload, done=True)

    def _step_book(self, st: ClinicState) -> Step:
        if not st.specialty and not st.doctor_name:
            return Step(ASK_SPECIALTY)
        if st.specialty and any(r in st.specialty for r in REFERRAL_ONLY) and not st.referral:
            return self._ask_referral(st)
        if st.chosen is None:
            return self._offer(st, doctor_id=None)
        if ask := self._ensure_patient(st, create=True):
            return ask
        assert st.patient is not None
        if st.confirmed is None:
            st.awaiting_confirm = True
            p = st.patient
            heard = f"{p.full_name}, {birth_words(p.birth_date)}"
            if p.id or st.read_back == heard:
                reply = f"{p.first_name}, записываю вас: {st.chosen.spoken()}. Верно?"
            else:
                # в базе пациента нет — один раз повторяем ФИО и дату рождения: ошибку распознавания слышно до записи
                st.read_back = heard
                reply = f"{heard}. В базе вас пока нет — заведу карточку. Записываю вас: {st.chosen.spoken()}. Верно?"
            return Step(reply, facts=reply)
        if not st.confirmed:
            st.avoid = [*st.avoid, st.chosen]  # от этого времени пациент отказался — снова его не предлагаем
            st.offered, st.chosen, st.confirmed, st.awaiting_confirm = [], None, None, False
            step = self._offer(st, doctor_id=None)
            step.reply = "Хорошо, выберем другое время. " + step.reply
            return step
        st.awaiting_confirm = False
        self._register(st)  # карточка нового пациента — только после «да»
        try:
            st.result = self.schedule.book(st.chosen, st.patient.full_name, st.patient.id)
        except SlotTakenError:
            return self._slot_taken(st)
        return self._done(st, f"Готово, вы записаны: {st.chosen.spoken()}. Номер записи {st.result.id}. "
                              "Возьмите паспорт и полис ОМС. Накануне я напомню о приёме.", st.result)

    def _ask_referral(self, st: ClinicState) -> Step:
        if st.referral is None:
            st.referral_asked += 1
            if st.referral_asked == 2:  # ответили не «да» и не «нет» — переспрашиваем короче
                st.question = "referral"
                return Step("Скажите, пожалуйста, есть ли у вас направление от врача: да или нет?")
            if st.referral_asked < 3:
                st.question = "referral"
                return Step(f"{_to_specialist(st.specialty).capitalize()} запись идёт по направлению лечащего врача. "
                            "У вас есть направление?")
            st.referral = False  # третий раз без ответа — считаем, что направления нет, и предлагаем терапевта
        # Направления нет — сначала участковый терапевт (BOOKED_GP), либо платно
        therapist = self._normalize_specialty("терапевт")
        spec = st.specialty
        st.intent, st.specialty, st.question, st.offer_specialty = None, None, "offer_book", therapist
        return Step(f"Тогда сначала нужен приём участкового терапевта — он выдаст направление {_to_specialist(spec)}. "
                    "Без направления можно платно. Записать вас к терапевту?")

    def _step_schedule(self, st: ClinicState) -> Step:
        if not st.specialty and not st.doctor_name:
            return Step("Расписание какого врача или специалиста вам подсказать?")
        doctor = self.schedule.find_doctor(st.doctor_name) if st.doctor_name else None
        doctors = [doctor] if doctor else self.schedule.doctors_for(st.specialty)
        if not doctors:
            return Step("Такого врача в расписании нет. Могу записать к другому врачу этой специальности.")
        parts = []
        for d in doctors[:3]:
            if absent := self.schedule.absence(d):
                # С-07: причину отсутствия не называем
                parts.append(f"{d.specialty.capitalize()} {d.name} не принимает до {on_day(absent[1])} включительно.")
                if doctor:
                    others = [o for o in self.schedule.doctors_for(d.specialty) if o.id != d.id and o.is_active]
                    nearest = self.schedule.find(specialty=d.specialty, limit=1)
                    if others and nearest:
                        parts.append(f"Ближайшее время к другому врачу — {nearest[0].spoken()}.")
                continue
            nearest = self.schedule.find(doctor_id=d.id, limit=1)
            free = (f"Ближайшее свободное время — {nearest[0].when()}." if nearest
                    else "Свободного времени в ближайшие две недели пока нет.")
            parts.append(f"{d.describe()}. {free}")
        st.question, st.offer_specialty = "offer_book", st.specialty
        reply = " ".join(parts) + " Записать вас?"
        return Step(reply, facts=reply)

    def _step_status(self, st: ClinicState) -> Step:
        if ask := self._find_target(st, pick=False):
            return ask
        assert st.target is not None
        items = (self.schedule.bookings_of(st.patient) if st.patient else [st.target])[:3]
        listed = "; ".join(f"{b.slot.spoken()}, номер записи {b.id}" for b in items)
        who = f"{st.patient.first_name}, в" if st.patient else "В"
        return self._done(st, f"{who}ы записаны: {listed}. Если нужно, я помогу перенести или отменить запись.",
                          items)

    def _step_cancel(self, st: ClinicState) -> Step:
        if ask := self._find_target(st, pick=True):
            return ask
        assert st.target is not None
        if not st.move_offered:
            st.move_offered, st.question = True, "cancel_or_move"
            who = f"{st.patient.first_name}, у" if st.patient else "У"
            return Step(f"{who} вас запись: {_short(st.target)}. Могу перенести её на другой день вместо отмены. "
                        "Перенести?")
        if st.confirmed is None:
            st.awaiting_confirm = True
            return Step(f"Отменяю запись: {_short(st.target)}. Верно?")
        st.awaiting_confirm = False
        if st.confirmed:
            self.schedule.cancel(st.target.id)
            self._reminder_outcome(st, "CANCELLED")
            return self._done(st, "Готово, запись отменена. Спасибо, что предупредили — это время достанется "
                                  "другому пациенту.")
        return self._done(st, "Хорошо, запись сохранена.")

    def _step_reschedule(self, st: ClinicState) -> Step:
        if ask := self._find_target(st, pick=True):
            return ask
        assert st.target is not None
        if st.chosen is None:
            doctor_id = self._reschedule_doctor(st)
            step = self._offer(st, doctor_id=doctor_id)
            if st.offered and doctor_id != st.target.slot.doctor_id and st.offered[0].doctor_name not in step.reply:
                o = st.offered[0]
                step.reply = f"{o.specialty.capitalize()} {o.doctor_name}. {step.reply}"  # врач другой — называем его
            if st.offered and not step.reply.startswith("Пожалуйста") and not st.reminder_for:
                step.reply = f"Сейчас вы записаны: {_short(st.target)}. {step.reply}"
            return step
        if st.confirmed is None:
            st.awaiting_confirm = True
            c = st.chosen
            to = f"{c.specialty} {c.doctor_name}, " if c.doctor_id != st.target.slot.doctor_id else ""
            return Step(f"Переношу запись: {to}{c.when()}. Прежнюю отменяю. Верно?")
        st.awaiting_confirm = False
        if not st.confirmed:
            st.offered, st.chosen, st.confirmed = [], None, None
            step = self._offer(st, doctor_id=self._reschedule_doctor(st))
            step.reply = "Хорошо, выберем другое время. " + step.reply
            return step
        try:
            st.result = self.schedule.reschedule(st.target.id, st.chosen)
        except SlotTakenError:
            return self._slot_taken(st)
        self._reminder_outcome(st, "RESCHEDULED")
        return self._done(st, f"Готово, запись перенесена: {st.result.slot.spoken()}. "
                              f"Номер записи прежний — {st.result.id}.", st.result)

    def _step_opt_out(self, st: ClinicState) -> Step:
        """Входящий отказ от напоминаний: отметка «Отказ от обзвона / уведомлений ДД.ММ.ГГГГ» в карточке пациента."""
        if ask := self._ensure_patient(st, create=False):
            return ask
        assert st.patient is not None
        self.schedule.set_opt_out(st.patient, st.opt_out_code or "OPT_OUT")
        return self._done(st, "Готово, отметила в вашей карточке.")

    def _step_home_visit(self, st: ClinicState) -> Step:
        """С-06: заявка на вызов врача на дом (регламент 6.3). Повод не оцениваем, время визита не обещаем."""
        if ask := self._ensure_patient(st, create=True):
            return ask
        if not st.home_note:
            st.question = "home_address"
            return Step("Назовите, пожалуйста, адрес, куда прийти врачу, и коротко — что беспокоит. "
                        "Оценить, нужен ли визит, может только врач: я передам ваши слова.")
        now = reminders.clinic_now()
        cutoff = {5: 12}.get(now.weekday(), 14)
        today = now.weekday() != 6 and now.date() not in reminders.HOLIDAYS and now.hour < cutoff
        day = now.date()
        if not today:  # следующий рабочий день: без воскресений и праздников
            day += timedelta(days=1)
            while day.weekday() == 6 or day in reminders.HOLIDAYS:
                day += timedelta(days=1)
        self._register(st)
        number = self.schedule.escalate(st.patient, f"ВЫЗОВ НА ДОМ {day:%d.%m.%Y} P3\n{st.home_note}")
        when = "сегодня" if today else ("завтра" if day == now.date() + timedelta(days=1) else on_date(day))
        late = "" if today else "Заявки на сегодня уже не принимаются. "
        return self._done(st, f"{late}Заявка на вызов врача на дом принята. Врач придёт {when}, в течение рабочего "
                              "дня, до восемнадцати часов. Точное время назвать не могу — врач обходит пациентов "
                              "по маршруту. Регистратор перезвонит, только если понадобится уточнение. При ухудшении "
                              "звоните сто три.", {"escalation": number})

    def _step_escalate(self, st: ClinicState) -> Step:
        route = ROUTES[st.escalation or "human"]
        if st.finished:  # перевод уже оформлен — пациент ждёт регистратора
            return Step("Регистратор уже подключается, оставайтесь, пожалуйста, на линии.", done=True)
        has_identity = bool(st.last_name and st.first_name and st.birth_date)
        if st.patient is None and has_identity:
            st.patient = self.schedule.identify(st.last_name, st.first_name, st.birth_date, create=True)
        self._register(st)
        if route.ask_card and st.patient is None and not st.asked_identity:
            st.asked_identity = True
            ask = (ASK_CARD.replace("Чтобы сотрудник сразу увидел вашу карточку", "Для листа ожидания")
                   if st.escalation == "waitlist" else ASK_CARD)
            return Step(f"{route.phrase} {ask}".strip())
        # первая строка заявки — правило и приоритет регламента («ПЕРЕВОД R-02 P2»), дальше суть для регистратора
        key = st.escalation or "human"
        number = self.schedule.escalate(st.patient, f"{transfer_header(key)}\n{st.escalation_note or route.label}")
        if st.reminder_for and (target := self.schedule.bookings.get(st.reminder_for)):
            self.schedule.log_reminder(target, st.patient, "ESCALATED")
        st.finished = True
        if st.escalation == "waitlist":
            reply = "Готово, вы в листе ожидания. Мы позвоним, как только освободится время."
        elif st.escalation in ("emergency", "selfharm", "not_found", "mis_error"):
            reply = route.phrase
        elif route.outcome == "CALLBACK":
            reply = "Спасибо, вопрос передан — вам перезвонят." if st.asked_identity else route.phrase
        elif st.asked_identity:
            reply = "Спасибо, соединяю. " + TRANSFER
        else:
            reply = f"{route.phrase} {TRANSFER}"
        return Step(reply, done=True, payload={"escalation": number})

    # ---------- помощники ----------

    def _ensure_patient(self, st: ClinicState, *, create: bool) -> Step | None:
        """Находит (или заводит) пациента по фамилии, имени и дате рождения; иначе — вопрос."""
        if st.patient is not None:
            return None
        if not (st.last_name and st.first_name):
            tail = "" if st.birth_date else " и дату рождения"
            return Step(f"Назовите, пожалуйста, фамилию, имя, отчество{tail}.")
        if not st.birth_date:
            return Step("Назовите, пожалуйста, дату рождения полностью — число, месяц и год.")
        today = self.schedule.today
        if st.birth_date > today or today.year - st.birth_date.year > 110:
            st.birth_date = None  # «12.03.1890», год из будущего с цифрами: распознавание исказило год
            return Step("Не расслышала год рождения. Назовите, пожалуйста, дату рождения ещё раз, с годом полностью.")
        st.patient = self.schedule.identify(st.last_name, st.first_name, st.birth_date, create=False)
        if st.patient is None and create:
            # нового пациента заводим в МИС только после «да» на повтор записи (_register)
            st.patient = Patient(id="", last_name=st.last_name, first_name=st.first_name, birth_date=st.birth_date)
        if st.patient is None:
            st.not_found += 1
            if st.not_found >= 2:  # R-07
                st.intent, st.escalation = "escalate", "not_found"
                st.escalation_note = (f"{ROUTES['not_found'].label}: {st.last_name} {st.first_name}, "
                                      f"{st.birth_date:%d.%m.%Y}")
                return self._step_escalate(st)
            st.last_name = st.first_name = st.birth_date = None
            return Step("Не нашла вас в базе. Назовите, пожалуйста, ещё раз фамилию, имя и дату рождения "
                        "или номер записи.")
        return None

    def _reschedule_doctor(self, st: ClinicState) -> str:
        """Перенос — к тому же врачу; «а можно к Беловой?» — к названному врачу той же специальности."""
        assert st.target is not None
        if st.doctor_name:
            doctor = self.schedule.find_doctor(st.doctor_name)
            if doctor is not None and doctor.specialty == st.target.slot.specialty:
                if doctor.id != st.target.slot.doctor_id and any(o.doctor_id != doctor.id for o in st.offered):
                    st.offered = []  # прозвучали варианты прежнего врача — ищем заново у названного
                return doctor.id
        return st.target.slot.doctor_id

    def _register(self, st: ClinicState) -> None:
        """Новый пациент, которого ещё нет в МИС, — заводим карточку (запись подтверждена или нужен перевод)."""
        if st.patient is not None and not st.patient.id:
            st.patient = self.schedule.identify(st.patient.last_name, st.patient.first_name, st.patient.birth_date,
                                                create=True)

    @staticmethod
    def _same_specialty(b: Booking, specialty: str | None) -> bool:
        return bool(specialty) and (specialty in b.slot.specialty or b.slot.specialty in specialty)

    def _find_target(self, st: ClinicState, *, pick: bool) -> Step | None:
        """Находит существующую запись: по номеру или по пациенту; из нескольких — спрашивает, какую."""
        if st.target is not None:
            return None
        if st.booking_id:
            booking = self.schedule.bookings.get(st.booking_id)
            if booking is None and st.booking_id.isdigit():
                booking = self.schedule.bookings.get(f"B{int(st.booking_id):04d}")
            if booking:
                # Номер записи только сужает поиск: назвать, перенести или отменить запись можно, когда пациент назвал
                # себя (номера визитов на стенде идут подряд — их легко угадать, а запись — чужие данные)
                if ask := self._ensure_patient(st, create=False):
                    return ask
                if booking.patient_id and st.patient is not None and booking.patient_id != st.patient.id:
                    st.booking_id = None
                    return Step("Запись с таким номером на ваше имя не найдена. Могу найти ваши записи по фамилии, "
                                "имени и дате рождения — назвать их?")
                st.target = booking
                return None
            number, st.booking_id = st.booking_id, None
            return Step(f"Запись с номером {number} не найдена. Назовите, пожалуйста, фамилию, имя и дату рождения.")
        if st.candidates:
            return self._ask_which(st)
        if ask := self._ensure_patient(st, create=False):
            return ask
        assert st.patient is not None
        items = self.schedule.bookings_of(st.patient)
        if not items:
            st.intent, st.question = None, "offer_book"
            st.offer_specialty = st.specialty
            return Step(f"{st.patient.first_name}, активных записей не найдено. Хотите записаться?")
        if pick and len(items) > 1:
            st.candidates = [b for b in items if self._same_specialty(b, st.specialty)] or items
            if len(st.candidates) > 1:
                return self._ask_which(st)
            items = st.candidates
            st.candidates = []
        st.target = items[0]
        return None

    def _ask_which(self, st: ClinicState) -> Step:
        if len(st.candidates) > 3:
            return Step("У вас несколько записей. К какому врачу запись, о которой идёт речь?")
        listed = "; ".join(f"вариант {i} — {_short(b)}" for i, b in enumerate(st.candidates, 1))
        return Step(f"У вас несколько записей: {listed}. О какой идёт речь?")

    def _offer(self, st: ClinicState, doctor_id: str | None) -> Step:
        if st.offered and st.relaxed:
            # Пациент снова просит своё время, а его нет: честно говорим и предлагаем лист ожидания (С-11)
            st.question = "waitlist"
            return Step("В удобное вам время свободных окон нет. Могу записать на один из названных вариантов "
                        "или поставить вас в лист ожидания на удобное время — мы позвоним, когда оно освободится. "
                        "Поставить в лист ожидания?", payload=st.offered)
        if st.offered and len(st.offered) > 1:
            # Варианты уже прозвучали: в голосе не зачитываем их снова целиком
            names = ["первый", "второй", "третий"][: len(st.offered)]
            reply = f"Какой вариант вам удобнее: {', '.join(names[:-1])} или {names[-1]}? Можно назвать время."
            return Step(reply, facts=reply, payload=st.offered)
        if st.offered:
            note = "Это единственное свободное время."
        else:
            st.offered, note = self._find_offers(st, doctor_id)
            st.relaxed = bool(note)
        if not st.offered:
            # С-11: не отпускать «ни с чем» — лист ожидания вместо «талонов нет»
            who = f"к специалисту {st.specialty}" if st.specialty else "к этому врачу"
            st.question = "waitlist"
            return Step(f"Свободного времени {who} на ближайшие две недели пока нет. Могу поставить вас в лист "
                        "ожидания: как только время освободится, мы позвоним. Поставить?")
        if len(st.offered) == 1:
            reply = f"{note} Свободно одно время: {st.offered[0].spoken()}. Подходит? Если нет — назовите другой день.".strip()
            return Step(reply, facts=reply, payload=st.offered)
        body = _render_offers(st.offered, with_doctor=doctor_id is None)
        if note.endswith(":"):
            body = body[0].lower() + body[1:]  # «В тот же день свободно: терапевт Белова…»
        reply = f"{note} {body} Какой вариант вам удобнее?".strip()
        return Step(reply, facts=reply, payload=st.offered)

    def _slot_taken(self, st: ClinicState) -> Step:
        st.offered, st.chosen, st.confirmed, st.awaiting_confirm = [], None, None, False
        retry = self._dispatch(st)
        retry.reply = "Извините, это время только что заняли. " + retry.reply
        return retry

    def _find_offers(self, st: ClinicState, doctor_id: str | None) -> tuple[list[Slot], str]:
        if doctor_id is None and st.doctor_name:
            doctor = self.schedule.find_doctor(st.doctor_name)
            doctor_id = doctor.id if doctor else None
        base = {"specialty": None if doctor_id else st.specialty, "doctor_id": doctor_id}
        times = {"time_from": st.time_from, "time_to": st.time_to}
        attempts: list[tuple[str, dict]] = []
        week = None if st.day or not st.week_of else (st.week_of, st.week_of + timedelta(days=7))
        if week:  # «на следующей неделе»: сначала только эта неделя, потом ближайшее
            attempts.append(("", {**times, "week": week}))
            if st.time_from or st.time_to:
                attempts.append(("В это время на ту неделю мест нет. На той неделе свободно:", {"week": week}))
            attempts.append(("На ту неделю мест нет. Ближайшие свободные окна:", times))
        attempts.append(("", {"day": st.day, **times}))
        if st.day and (st.time_from or st.time_to):
            attempts.append(("На это время свободных окон нет. В тот же день свободно:", {"day": st.day}))
        if st.day:
            attempts.append(("В этот день мест нет. Ближайшие свободные окна:",
                             {"time_from": st.time_from, "time_to": st.time_to}))
        if st.time_from or st.time_to:
            attempts.append(("В удобное вам время мест нет. Ближайшие свободные окна:", {}))
        # «Завтра не получится» в напоминании — день исходной записи не предлагаем
        skip_days = {st.target.slot.start.date()} if st.reminder_for and st.target and not st.day else set()
        if not st.day:  # названный день важнее прежнего «нужен другой день»
            skip_days |= set(st.avoid_days)
        avoid = {(s.doctor_id, s.start) for s in st.avoid}
        for note, filters in attempts:
            span = filters.pop("week", None)
            slots = [s for s in self.schedule.find(**base, **filters, limit=1000)
                     if s.start.date() not in skip_days and (s.doctor_id, s.start) not in avoid
                     and (span is None or span[0] <= s.start.date() < span[1])]
            if slots:
                return (_distinct_times(slots) if filters.get("day") else _spread(slots)), note
        return [], ""
