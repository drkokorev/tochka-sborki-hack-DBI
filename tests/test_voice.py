from types import SimpleNamespace

from tools.stt_bench import WhisperGateway, stt_prompt  # облачный Whisper — только в инструменте сравнения


class FakeAudio:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.texts.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


def make(*texts, prompt="подсказка"):
    stt = WhisperGateway.__new__(WhisperGateway)
    stt.prompt, stt.enabled = prompt, True
    fake = FakeAudio(*texts)
    stt._client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake))
    return stt, fake


def test_prompt_is_short():
    prompt = stt_prompt([f"специальность{i}" for i in range(40)], [f"Фамилия{i}" for i in range(85)])
    assert len(prompt) <= 300 and "Фамилия" not in prompt


def test_empty_result_with_prompt_is_retried_without_it():
    stt, fake = make("", " Запишите к терапевту. ")
    assert stt.transcribe(b"x") == "Запишите к терапевту."
    assert "prompt" in fake.calls[0] and "prompt" not in fake.calls[1]


def test_error_then_success_and_all_empty():
    stt, _ = make(TimeoutError("медленно"), "да")
    assert stt.transcribe(b"x") == "да"
    stt, _ = make("", "")
    assert stt.transcribe(b"x") == ""


def test_assistant_speech_recognition_is_local_only():
    """Звук пациента распознаётся только на машине: облачного резерва у ассистента нет (152-ФЗ)."""
    import inspect

    import apps.voice as voice
    from apps.voice import SpeechToText

    class Local:
        def __init__(self, text, available=True):
            self.text, self.available = text, available

        def transcribe(self, audio, filename):
            return self.text

    stt = SpeechToText(Local("локально"))
    assert stt.transcribe(b"x") == "локально" and stt.last_engine == "gigaam" and stt.name.startswith("GigaAM")
    off = SpeechToText(Local("", available=False))
    assert not off.enabled and off.name is None
    try:
        off.transcribe(b"x")
        raise AssertionError("без GigaAM распознавание должно отказать, а не уйти в облако")
    except RuntimeError:
        pass
    source = inspect.getsource(voice)
    assert "import openai" not in source and "from openai" not in source and "transcriptions" not in source
