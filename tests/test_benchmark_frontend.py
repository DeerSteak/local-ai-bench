from datetime import datetime
from pathlib import Path

import pytest

import config
import shared
from benchmark_frontend import (
    FrontendCancelled,
    MenuEntry,
    build_benchmark_command,
    build_model_entries,
    build_test_entries,
    choose_engine,
    choose_models,
    choose_tests,
    missing_catalog_hint,
    model_selection_error,
    parse_toggle_numbers,
    render_model_menu,
    run_frontend,
    toggle_group,
)
from models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


class InputSequence:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeEngine:
    def __init__(self, name="fake"):
        self.name = name


def sample_inventory():
    return {
        "llm": [LLM_MODELS[0], LLM_MODELS[-1]],
        "custom": [{
            "tag": "custom-folder",
            "label": "custom-folder (custom)",
            "short": "custom-folder",
            "size": 1,
        }],
        "embedding": list(EMBED_MODELS),
        "image": [IMAGE_MODELS[0], IMAGE_MODELS[-1]],
    }


def empty_inventory():
    return {"llm": [], "custom": [], "embedding": [], "image": []}


def full_inventory():
    inventory = sample_inventory()
    inventory["llm"] = list(LLM_MODELS)
    inventory["image"] = list(IMAGE_MODELS)
    return inventory


def output_collector():
    messages = []
    return messages, messages.append


def test_default_test_state_matches_documented_matrix():
    entries = {entry.value: entry for entry in build_test_entries(sample_inventory())}
    assert {name for name, entry in entries.items() if entry.checked} == {
        "llm", "conv", "emb", "img",
    }
    assert all(entries[name].available for name in entries)
    assert all(not entries[name].checked for name in (
        "mcq", "math", "code", "tool", "conc_tool", "conc_chat",
    ))


def test_empty_inventory_makes_every_test_unavailable_and_unchecked():
    entries = build_test_entries(empty_inventory())
    assert all(not entry.available and not entry.checked for entry in entries)


def test_family_specific_inventory_only_enables_its_tests():
    inventory = empty_inventory()
    inventory["embedding"] = [EMBED_MODELS[0]]
    entries = {entry.value: entry for entry in build_test_entries(inventory)}
    assert entries["emb"].available and entries["emb"].checked
    assert all(not entry.available for name, entry in entries.items() if name != "emb")


def test_model_defaults_select_nonlarge_catalog_and_embeddings_only():
    entries = build_model_entries(full_inventory(), ["llm", "emb", "img"])
    selected = {entry.value for entry in entries if entry.checked}
    assert {model["tag"] for model in LLM_MODELS if model["tier"] != "large"} <= selected
    assert not {model["tag"] for model in LLM_MODELS if model["tier"] == "large"} & selected
    assert "custom-folder" not in selected
    assert {model["tag"] for model in EMBED_MODELS} <= selected
    assert {model["short"] for model in IMAGE_MODELS if model["tier"] != "large"} <= selected
    assert not {model["short"] for model in IMAGE_MODELS if model["tier"] == "large"} & selected


def test_model_entries_only_include_families_used_by_selected_tests():
    llm_entries = build_model_entries(sample_inventory(), ["mcq", "conc_chat"])
    assert {entry.kind for entry in llm_entries} == {"llm", "custom"}
    image_entries = build_model_entries(sample_inventory(), ["img"])
    assert {entry.kind for entry in image_entries} == {"image"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2", {2}),
        ("2,4 6", {2, 4, 6}),
        ("3-5", {3, 4, 5}),
        ("2 4-5", {2, 4, 5}),
    ],
)
def test_parse_toggle_numbers(raw, expected):
    assert parse_toggle_numbers(raw, 6) == expected


@pytest.mark.parametrize("raw", ["", "0", "7", "4-2", "a", "2-x"])
def test_parse_toggle_numbers_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        parse_toggle_numbers(raw, 6)


def test_tier_toggle_spans_catalog_llm_and_images_but_not_custom_or_embedding():
    entries = [
        MenuEntry("llm", "LLM", "llm", "LLM", False, tier="small"),
        MenuEntry("image", "Image", "image", "Images", True, tier="small"),
        MenuEntry("custom", "Custom", "custom", "Custom", False),
        MenuEntry("embed", "Embed", "embedding", "Embeddings", False),
    ]
    assert toggle_group(
        entries, lambda entry: entry.kind in ("llm", "image") and entry.tier == "small",
    )
    assert entries[0].checked and entries[1].checked
    assert not entries[2].checked and not entries[3].checked
    toggle_group(entries, lambda entry: entry.kind in ("llm", "image") and entry.tier == "small")
    assert not entries[0].checked and not entries[1].checked


def test_group_toggle_reports_when_no_entries_match():
    assert not toggle_group([], lambda entry: True)


def test_custom_and_embedding_bulk_toggles_are_independent():
    entries = [
        MenuEntry("c1", "C1", "custom", "Custom", False),
        MenuEntry("c2", "C2", "custom", "Custom", True),
        MenuEntry("e1", "E1", "embedding", "Embeddings", True),
    ]
    toggle_group(entries, lambda entry: entry.kind == "custom")
    assert entries[0].checked and entries[1].checked and entries[2].checked
    toggle_group(entries, lambda entry: entry.kind == "embedding")
    assert entries[0].checked and entries[1].checked and not entries[2].checked


def test_choose_engine_auto_selects_only_registered_engine():
    messages, output = output_collector()
    assert choose_engine(["llamacpp"], InputSequence([]), output) == "llamacpp"
    assert messages == ["Engine: llamacpp"]


def test_choose_engine_accepts_number_when_multiple_registered():
    messages, output = output_collector()
    selected = choose_engine(["llamacpp", "mlx"], InputSequence(["2"]), output)
    assert selected == "mlx"
    assert any("CLI-only" in message for message in messages)


def test_choose_engine_can_cancel():
    with pytest.raises(FrontendCancelled):
        choose_engine(["llamacpp", "mlx"], InputSequence(["q"]), lambda _: None)


def test_choose_engine_accepts_default_and_reprompts_after_invalid_input():
    messages, output = output_collector()
    assert choose_engine(
        ["llamacpp", "mlx"], InputSequence(["invalid", ""]), output,
    ) == "llamacpp"
    assert any("Couldn't parse" in message for message in messages)


def test_choose_tests_toggles_individual_entries_and_rejects_unavailable():
    entries = build_test_entries(sample_inventory())
    entries[2].available = False
    entries[2].checked = False
    messages, output = output_collector()
    selected = choose_tests(entries, InputSequence(["5", "3", ""]), output)
    assert "mcq" in selected
    assert "emb" not in selected
    assert any("cannot be selected" in message for message in messages)


def test_choose_tests_reprompts_when_everything_is_deselected():
    entries = build_test_entries(sample_inventory())
    messages, output = output_collector()
    selected = choose_tests(entries, InputSequence(["1-4", "", "1", ""]), output)
    assert selected == ["llm"]
    assert any("Select at least one" in message for message in messages)


def test_choose_tests_eof_cancels():
    with pytest.raises(FrontendCancelled):
        choose_tests(build_test_entries(sample_inventory()), InputSequence([EOFError()]), lambda _: None)


def test_choose_tests_q_cancels_and_invalid_input_reprompts():
    entries = build_test_entries(sample_inventory())
    messages, output = output_collector()
    with pytest.raises(FrontendCancelled):
        choose_tests(entries, InputSequence(["invalid", "q"]), output)
    assert any("Couldn't parse" in message for message in messages)


def test_choose_models_tier_partial_to_all_then_all_to_none():
    entries = build_model_entries(sample_inventory(), ["llm", "img"])
    small_tier = [entry for entry in entries if entry.tier == LLM_MODELS[0]["tier"]]
    assert small_tier and all(entry.checked for entry in small_tier)
    messages, output = output_collector()
    choose_models(entries, ["llm", "img"], None, InputSequence(["xs", "xs", ""]), output)
    assert all(entry.checked for entry in small_tier)
    assert any("Tier keys" in message for message in messages)


def test_choose_models_custom_and_embedding_commands():
    entries = build_model_entries(sample_inventory(), ["llm", "emb"])
    choose_models(entries, ["llm", "emb"], None, InputSequence(["custom", "emb", "emb", ""]), lambda _: None)
    assert all(entry.checked for entry in entries if entry.kind == "custom")
    assert all(entry.checked for entry in entries if entry.kind == "embedding")


def test_choose_models_reprompts_when_required_family_is_empty():
    entries = build_model_entries(sample_inventory(), ["img"])
    messages, output = output_collector()
    choose_models(entries, ["img"], None, InputSequence(["1", "", "1", ""]), output)
    assert any("Select at least one image" in message for message in messages)


def test_choose_models_handles_unavailable_groups_invalid_input_and_cancel():
    entries = [MenuEntry("llm", "LLM", "llm", "LLM", True, tier="small")]
    messages, output = output_collector()
    with pytest.raises(FrontendCancelled):
        choose_models(
            entries, ["llm"], None,
            InputSequence(["l", "custom", "invalid", "q"]), output,
        )
    assert any("tier large" in message for message in messages)
    assert any("custom models" in message for message in messages)
    assert any("Couldn't parse" in message for message in messages)


def test_model_selection_error_covers_each_required_family():
    assert "LLM" in model_selection_error([], ["conv"])
    assert "embedding" in model_selection_error([], ["emb"])
    assert "image" in model_selection_error([], ["img"])
    assert model_selection_error(
        [MenuEntry("x", "X", "custom", "Custom", True)], ["tool", "conc_chat"],
    ) is None


def test_render_model_menu_has_one_shared_llm_list_and_cross_family_help():
    entries = build_model_entries(sample_inventory(), ["llm", "conv", "mcq", "conc_tool", "img"])
    messages, output = output_collector()
    render_model_menu(entries, "missing hint", output)
    assert sum("LLM —" in message for message in messages) == 2
    assert any("catalog LLM and image models together" in message for message in messages)
    assert messages[-1] == "missing hint"


def test_missing_catalog_hint_counts_families_and_ignores_custom_models():
    inventory = sample_inventory()
    hint = missing_catalog_hint(inventory, "Linux")
    assert f"{len(LLM_MODELS) - 2} LLM" in hint
    assert f"{len(IMAGE_MODELS) - 2} image" in hint
    assert "custom" not in hint
    assert "`bash setup.sh`" in hint


def test_missing_catalog_hint_uses_windows_command():
    assert "`setup.bat`" in missing_catalog_hint(empty_inventory(), "Windows")


def test_missing_catalog_hint_omitted_when_every_catalog_model_installed():
    inventory = {
        "llm": list(LLM_MODELS), "custom": [],
        "embedding": list(EMBED_MODELS), "image": list(IMAGE_MODELS),
    }
    assert missing_catalog_hint(inventory, "Linux") is None


def test_build_command_emits_every_applicable_explicit_selector(tmp_path):
    entries = build_model_entries(sample_inventory(), ["llm", "emb", "img"])
    command = build_benchmark_command(
        "fake", tmp_path / "Comfy UI", ["llm", "emb", "img"], entries,
        python_executable="python-test", benchmark_path=Path("/repo/scripts/benchmark.py"),
    )
    assert command[:8] == [
        "python-test", "/repo/scripts/benchmark.py", "--engine", "fake",
        "--comfyui", str(tmp_path / "Comfy UI"), "--tests", "llm",
    ]
    assert "--llm-models" in command
    assert "--embedding-models" in command
    assert "--image-models" in command
    assert "--maxtier" not in command


@pytest.mark.parametrize(
    ("tests", "entry", "selector"),
    [
        (["llm"], MenuEntry("phi4-mini", "Phi", "llm", "LLM", True), "--llm-models"),
        (["emb"], MenuEntry("nomic-embed-text", "Nomic", "embedding", "Embeddings", True),
         "--embedding-models"),
        (["img"], MenuEntry("sdxl", "SDXL", "image", "Images", True), "--image-models"),
    ],
)
def test_build_command_is_exact_for_each_isolated_workload_family(tests, entry, selector):
    assert build_benchmark_command(
        "fake", Path("/comfy"), tests, [entry],
        python_executable="python", benchmark_path=Path("/benchmark.py"),
    ) == [
        "python", "/benchmark.py", "--engine", "fake", "--comfyui", "/comfy",
        "--tests", *tests, selector, entry.value,
    ]


def test_build_command_uses_one_llm_selection_for_accuracy_and_concurrency():
    entries = [
        MenuEntry("phi4-mini", "Phi", "llm", "LLM", True),
        MenuEntry("custom-one", "Custom", "custom", "Custom", True),
    ]
    command = build_benchmark_command(
        "fake", Path("/comfy"), ["mcq", "conc_chat"], entries,
        python_executable="python", benchmark_path=Path("/benchmark.py"),
    )
    index = command.index("--llm-models")
    assert command[index + 1:] == ["phi4-mini", "custom-one"]
    assert command.count("--llm-models") == 1


def test_build_command_standalone_conversation_has_no_other_model_flags():
    entries = [MenuEntry("phi4-mini", "Phi", "llm", "LLM", True)]
    command = build_benchmark_command(
        "fake", Path("/comfy"), ["conv"], entries,
        python_executable="python", benchmark_path=Path("/benchmark.py"),
    )
    assert "--llm-models" in command
    assert "--embedding-models" not in command
    assert "--image-models" not in command


def test_build_command_uses_default_benchmark_path():
    command = build_benchmark_command(
        "fake", Path("/comfy"), ["llm"],
        [MenuEntry("phi4-mini", "Phi", "llm", "LLM", True)],
        python_executable="python",
    )
    assert command[1] == str(config.SCRIPT_DIR / "scripts" / "benchmark.py")


def test_run_frontend_launches_argument_list_and_propagates_exit_code(tmp_path):
    commands = []
    messages, output = output_collector()

    class Result:
        returncode = 7

    result = run_frontend(
        input_fn=InputSequence(["", "", ""]),
        output_fn=output,
        process_runner=lambda command: commands.append(command) or Result(),
        engine_names_fn=lambda: ["fake"],
        engine_factory=FakeEngine,
        inventory_builder=lambda engine, path: sample_inventory(),
        system="Linux",
        python_executable="python-test",
        benchmark_path=tmp_path / "benchmark.py",
    )
    assert result == 7
    assert len(commands) == 1
    assert commands[0][0] == "python-test"
    assert str(config.COMFYUI_DIR) in commands[0]
    assert any("Launching benchmark.py" in message for message in messages)
    assert "Start this benchmark? [Y/n]" in messages


def test_run_frontend_default_output_function_is_timestamped(monkeypatch, capsys):
    monkeypatch.setattr(shared, "_console_now", lambda: datetime(2026, 7, 22, 9, 8, 7))
    result = run_frontend(
        input_fn=InputSequence(["", "", "y"]),
        process_runner=lambda command: 0,
        engine_names_fn=lambda: ["fake"],
        engine_factory=FakeEngine,
        inventory_builder=lambda engine, path: sample_inventory(),
    )
    lines = capsys.readouterr().out.splitlines()
    assert result == 0
    assert lines
    assert all(line.startswith("[09:08:07] ") for line in lines)


def test_run_frontend_uses_selected_engine_and_setup_comfyui_path():
    commands = []
    seen = []
    result = run_frontend(
        input_fn=InputSequence(["2", "", "", "y"]),
        output_fn=lambda _: None,
        process_runner=lambda command: commands.append(command) or 0,
        engine_names_fn=lambda: ["llamacpp", "mlx"],
        engine_factory=lambda name: FakeEngine(name),
        inventory_builder=lambda engine, path: seen.append((engine.name, path)) or sample_inventory(),
        system="Windows",
        python_executable="python",
        benchmark_path=Path("/benchmark.py"),
    )
    assert result == 0
    assert seen == [("mlx", config.COMFYUI_DIR)]
    assert commands[0][commands[0].index("--engine") + 1] == "mlx"
    assert commands[0][commands[0].index("--comfyui") + 1] == str(config.COMFYUI_DIR)


def test_run_frontend_shows_tests_without_a_path_prompt():
    messages, output = output_collector()
    result = run_frontend(
        input_fn=InputSequence(["q"]),
        output_fn=output,
        process_runner=lambda command: 0,
        engine_names_fn=lambda: ["fake"],
        engine_factory=FakeEngine,
        inventory_builder=lambda engine, path: sample_inventory(),
    )
    assert result == 0
    assert any(message == "Choose benchmark tests:" for message in messages)
    assert not any(message.startswith("ComfyUI directory [") for message in messages)


def test_run_frontend_explicit_no_cancels_final_confirmation():
    called = []
    messages, output = output_collector()
    result = run_frontend(
        input_fn=InputSequence(["", "", "n"]),
        output_fn=output,
        process_runner=lambda command: called.append(command),
        engine_names_fn=lambda: ["fake"],
        engine_factory=FakeEngine,
        inventory_builder=lambda engine, path: sample_inventory(),
    )
    assert result == 0
    assert called == []
    assert messages[-1] == "Benchmark selection cancelled."


def test_run_frontend_q_eof_or_interrupt_cancels_without_process():
    for first_input in ("q", EOFError(), KeyboardInterrupt()):
        called = []
        result = run_frontend(
            input_fn=InputSequence([first_input]),
            output_fn=lambda _: None,
            process_runner=lambda command: called.append(command),
            engine_names_fn=lambda: ["fake"],
            engine_factory=FakeEngine,
            inventory_builder=lambda engine, path: sample_inventory(),
        )
        assert result == 0
        assert called == []


def test_run_frontend_returns_error_without_any_installed_models():
    called = []
    result = run_frontend(
        input_fn=InputSequence([]),
        output_fn=lambda _: None,
        process_runner=lambda command: called.append(command),
        engine_names_fn=lambda: ["fake"],
        engine_factory=FakeEngine,
        inventory_builder=lambda engine, path: empty_inventory(),
    )
    assert result == 1
    assert called == []
