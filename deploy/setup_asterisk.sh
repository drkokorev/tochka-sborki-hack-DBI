#!/usr/bin/env bash
# Настраивает Asterisk на ВМ хакатона: регистрация на SIP-шлюзе (готовые конфиги организаторов),
# диалплан ассистента (AudioSocket) и AMI для исходящих напоминаний. Секрет AMI — случайный, в .env (не в git).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:/usr/sbin:/sbin"   # asterisk ставится в /usr/sbin
command -v asterisk >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq asterisk < /dev/null
SECRET="$(grep -s '^AMI_SECRET=' .env | cut -d= -f2- || true)"
if [ -z "$SECRET" ]; then
  SECRET="$(openssl rand -hex 16)"
  [ -f .env ] && sed -i '/^AMI_SECRET=$/d;/^AMI_USER=assistant$/d' .env   # пустые строки из .env.example — без дублей
  printf 'AMI_USER=assistant\nAMI_SECRET=%s\n' "$SECRET" >> .env
  chmod 600 .env
fi
sudo cp /opt/hackathon/sip/pjsip.conf /opt/hackathon/sip/modules.conf /etc/asterisk/
sudo cp deploy/asterisk/extensions.conf /etc/asterisk/extensions.conf
sed "s/__SECRET__/$SECRET/" deploy/asterisk/manager.conf | sudo tee /etc/asterisk/manager.conf >/dev/null
# Звуки (самопроверка, IVR, «регистраторы заняты») — нашим синтезом речи, без ручной записи
SOUNDS=/usr/share/asterisk/sounds/custom
.venv/bin/python tools/make_prompts.py logs/prompts && sudo mkdir -p "$SOUNDS" && sudo cp logs/prompts/*.wav "$SOUNDS"/
sudo systemctl restart asterisk
sleep 5
sudo asterisk -rx "module show like audiosocket"
sudo asterisk -rx "pjsip show registrations"
