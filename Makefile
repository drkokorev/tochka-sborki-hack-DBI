# Основные команды. Без ключей и сети работают install, test, demo и run (демо-данные, фразник и правила).
PY := .venv/bin/python

.PHONY: install install-ml models student doctor test demo run e2e eval call call-book docker

install:          ## окружение и зависимости
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt && mkdir -p logs data

install-ml:       ## библиотеки локальных моделей: torch (CPU), transformers, GigaAM, Silero VAD
	.venv/bin/pip install -q --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt

models:           ## скачать модели: голоса Silero v5 и Piper (voices/), GigaAM-v3 в кэш (~1 ГБ, один раз)
	./tools/get_voice.sh && $(PY) -c "import gigaam; gigaam.load_model('v3_e2e_ctc', device='cpu'); print('GigaAM готова')"

student:          ## модель-ученик: корпус от учителя (нужен ключ LLM, ~4 ₽) и дообучение rubert-tiny2 на CPU (~3 мин)
	$(PY) tools/distill_data.py && $(PY) tools/train_student.py --finetune

doctor:           ## проверка готовности: что есть, чего нет и как исправить (секреты не печатает)
	$(PY) tools/doctor.py

test:             ## все тесты, сеть не нужна
	$(PY) -m pytest -q

demo:             ## основной сценарий кейса в консоли: 7 разговоров с проверкой итога в реестре
	$(PY) tools/demo_dialog.py

PORT ?= 8000

run:              ## веб-интерфейс на http://localhost:8000 (другой порт: make run PORT=8080)
	.venv/bin/uvicorn apps.web.server:app --host 0.0.0.0 --port $(PORT)

e2e:              ## сквозная проверка на стенде МИС (нужен DBI_API_TOKEN) → eval/stand_e2e.md
	$(PY) tools/stand_e2e.py

call:             ## живой звонок в работающий AudioSocket: голос, распознавание, ответ, «угу» → eval/results/live_call.json
	$(PY) tools/live_call.py

call-book:        ## живой звонок: запись голосом до визита в МИС (вымышленный пациент) → eval/results/live_call_book.json
	$(PY) tools/live_call.py --book

eval:             ## стенд «LLM играет пациента» (нужен ключ LLM-шлюза) → eval/report.md
	$(PY) -m eval.simulate

docker:           ## облегчённый образ: фразник, правила, LLM-шлюз; данные — в томе ./data
	docker compose up --build
