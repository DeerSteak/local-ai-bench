import pytest

from scripts.app import benchmark_gui, benchmark_launcher
from scripts.app.benchmark_launcher import parse_launcher_request


@pytest.mark.parametrize("flag", ["--ui", "--interface"])
def test_launcher_accepts_ui_modes_and_compatibility_alias(flag):
    assert parse_launcher_request([flag, "gui"]) == ("gui", [])
    assert parse_launcher_request([flag, "terminal"]) == ("terminal", [])


def test_launcher_defaults_to_auto():
    assert parse_launcher_request([]) == ("auto", [])


def test_none_preserves_noninteractive_benchmark_arguments():
    assert parse_launcher_request(
        ["--ui", "none", "--tests", "llm", "--out", "my results.json"],
    ) == ("none", ["--tests", "llm", "--out", "my results.json"])


def test_none_requires_a_benchmark_command_and_interactive_modes_reject_one():
    with pytest.raises(SystemExit):
        parse_launcher_request(["--ui", "none"])
    with pytest.raises(SystemExit):
        parse_launcher_request(["--ui", "gui", "--tests", "llm"])


def test_gui_dispatch_uses_packaged_frontend(monkeypatch):
    monkeypatch.setattr(benchmark_launcher.sys, "argv", ["benchmark_launcher", "--ui", "gui"])
    monkeypatch.setattr(benchmark_launcher, "tkinter_available", lambda: True)
    monkeypatch.setattr(benchmark_launcher, "select_interface_mode", lambda *args, **kwargs: "gui")
    monkeypatch.setattr(benchmark_gui, "run_benchmark_gui", lambda: 17)
    with pytest.raises(SystemExit) as exc:
        benchmark_launcher.main()
    assert exc.value.code == 17
