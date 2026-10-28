#!/usr/bin/env bash
# Запуск демо на ВМ хакатона: публичный адрес машины проксирует 0.0.0.0:8000.
# Ключи LLM-шлюза и токен стенда МИС берутся из /opt/hackathon/ai.env.
set -euo pipefail
cd "$(dirname "$0")"
set -a
[ -f /opt/hackathon/ai.env ] && . /opt/hackathon/ai.env
[ -f .env ] && . ./.env          # AMI_SECRET для исходящих звонков (создаёт deploy/setup_asterisk.sh)
TELEPHONY="${TELEPHONY:-1}"       # AudioSocket для Asterisk на 127.0.0.1:9092
set +a
exec .venv/bin/uvicorn apps.web.server:app --host 0.0.0.0 --port 8000
