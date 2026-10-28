"""Перебивание голосом в веб-звонке: функция bargeStep из index.html на модели уровней микрофона (tools/bargein_sim.mjs)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="нужен Node.js")
def test_barge_in_reacts_to_voice_and_ignores_echo():
    out = subprocess.run(["node", "tools/bargein_sim.mjs", "--no-save"], cwd=ROOT, capture_output=True,
                         text=True, timeout=60, check=True).stdout
    summary = json.loads(out.strip().splitlines()[-1])
    ref, level = summary["эталон"], summary["уровень"]
    # Эхо ответа само себя не прерывает — ни с эталоном (Silero/Piper), ни без него (голос браузера)
    assert ref["false_triggers"] == [] and level["false_triggers"] == []
    # С эталоном голос пациента ловится в ≥ 85% случаев (не ловится тихий голос поверх громкого эха
    # и речь в первые 0,3 с ответа), реакция — до 0,8 с (медиана ~0,33 с)
    assert len(ref["missed"]) <= 0.15 * ref["voice_cases"]
    assert ref["reaction_ms_max"] <= 800
