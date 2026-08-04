from pathlib import Path

from benchmark_frontend import (
    GUI_OPTION_DEFAULTS,
    MenuEntry,
    build_benchmark_command,
    build_frontend_state,
    load_frontend_state,
    validate_gui_options,
)
from benchmark_gui import effective_gui_options, open_path_command


def test_effective_gui_options_uses_defaults_without_saved_gui_settings():
    assert effective_gui_options(None) == GUI_OPTION_DEFAULTS
    assert effective_gui_options({"tests": ["llm"]}) == GUI_OPTION_DEFAULTS
    assert effective_gui_options(None) is not GUI_OPTION_DEFAULTS


def test_gui_options_round_trip_in_frontend_state(tmp_path):
    path = tmp_path / "state.json"
    options = dict(GUI_OPTION_DEFAULTS, runs=5, cpu_only=True, out="results/custom.json")
    state = build_frontend_state(
        "llamacpp", ["llm"], [MenuEntry("model", "Model", "llm", "LLM", True)],
        gui_options=options,
    )
    from benchmark_frontend import save_frontend_state
    assert save_frontend_state(state, path)
    assert load_frontend_state(path)["gui_options"] == options


def test_validate_gui_options_rejects_bounds_types_and_missing_keys():
    assert validate_gui_options(dict(GUI_OPTION_DEFAULTS, runs=0))
    assert validate_gui_options(dict(GUI_OPTION_DEFAULTS, timeout="300"))
    assert validate_gui_options({})
    assert validate_gui_options(GUI_OPTION_DEFAULTS) == []


def test_build_command_includes_every_gui_execution_setting():
    options = dict(
        GUI_OPTION_DEFAULTS, warmup=2, runs=4, timeout=600, acc_timeout=90,
        acc_token_budget=2048, cpu_only=True, force_all=True,
        out="chosen.json", comfyui="/chosen/ComfyUI",
    )
    command = build_benchmark_command(
        "llamacpp", Path("/detected/ComfyUI"), ["llm"],
        [MenuEntry("model", "Model", "llm", "LLM", True)],
        python_executable="python", benchmark_path=Path("benchmark.py"), gui_options=options,
    )
    assert command[command.index("--comfyui") + 1] == "/chosen/ComfyUI"
    for flag, value in (
        ("--warmup", "2"), ("--runs", "4"), ("--timeout", "600"),
        ("--acc-timeout", "90"), ("--acc-token-budget", "2048"),
        ("--out", "chosen.json"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--cpu-only" in command
    assert "--force-all" in command


def test_open_path_command_uses_each_desktop_platform_launcher():
    path = Path("/tmp/results")
    assert open_path_command(path, "Darwin") == ["open", "/tmp/results"]
    assert open_path_command(path, "Linux") == ["xdg-open", "/tmp/results"]
    assert open_path_command(path, "Windows") == ["explorer", "/tmp/results"]
