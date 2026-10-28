#!/usr/bin/env bash
# Скачивает голоса в voices/ (в git не храним): Piper (MIT, rhasspy/piper-voices, ~60 МБ) — резерв, Silero v5 — основной.
set -euo pipefail
VOICE="${1:-ru_RU-irina-medium}"
IFS=- read -r LANG_CODE NAME QUALITY <<< "$VOICE"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG_CODE%%_*}/${LANG_CODE}/${NAME}/${QUALITY}"
mkdir -p voices
for f in "$VOICE.onnx" "$VOICE.onnx.json"; do
  curl -fsSL -o "voices/$f" "$BASE/$f"   # -f: ошибка 404 — это ошибка, а не HTML вместо модели
done
# Основной голос — Silero v5 (snakers4/silero-models, CC BY-NC 4.0), ~140 МБ
[ -f voices/silero_v5_3_ru.pt ] || curl -fsSL -o voices/silero_v5_3_ru.pt https://models.silero.ai/models/tts/ru/v5_3_ru.pt
ls -la voices
