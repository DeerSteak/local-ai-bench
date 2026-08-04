import pytest

from benchmark_launcher import parse_launcher_request


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
