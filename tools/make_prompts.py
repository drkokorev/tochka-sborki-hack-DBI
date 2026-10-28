"""Звуки для Asterisk из нашего синтеза речи (Silero, резерв — Piper): 8 кГц, 16 бит, моно.

    .venv/bin/python tools/make_prompts.py /usr/share/asterisk/sounds/custom   # вызывает deploy/setup_asterisk.sh

Самопроверка телефонии (selftest-*), приветствие IVR клиники и фраза «регистраторы заняты» с обещанием
перезвонить — всё воспроизводимо, без ручной записи.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.telephony import pcm_to_wav, wav_to_pcm8k  # noqa: E402
from apps.voice import TextToSpeech  # noqa: E402

PROMPTS = {
    "selftest-1": "Здравствуйте. Запишите меня к терапевту на вторник.",
    "selftest-2": "Давайте первый вариант.",
    "ivr-welcome": "Поликлиника номер семь. Чтобы записаться, перенести запись или задать вопрос, нажмите один — "
                   "вам ответит виртуальный помощник. Чтобы соединиться с регистратором, нажмите ноль.",
    "registry-busy": "Все регистраторы сейчас заняты. Ваше обращение с карточкой уже передано, "
                     "регистратор перезвонит вам в ближайшее время. Всего доброго!",
}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/prompts")
    out.mkdir(parents=True, exist_ok=True)
    tts = TextToSpeech()
    if not tts.available:
        sys.exit("Синтез речи недоступен — запустите tools/get_voice.sh")
    for name, text in PROMPTS.items():
        pcm = wav_to_pcm8k(tts.synthesize(text))
        (out / f"{name}.wav").write_bytes(pcm_to_wav(pcm, rate=8000))
        print(out / f"{name}.wav")


if __name__ == "__main__":
    main()
