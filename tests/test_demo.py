"""`make demo` — основной сценарий кейса одной командой: все разговоры доходят до действия в реестре."""

from tools.demo_dialog import run


def test_demo_dialog_reaches_registry_actions(capsys):
    assert run() == 0
    assert "7 из 7" in capsys.readouterr().out
