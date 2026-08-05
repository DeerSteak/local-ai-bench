import json

import pytest

from scripts.app.benchmark_frontend import GUI_OPTION_DEFAULTS, MenuEntry, build_frontend_state
from scripts.app.benchmark_presets import (
    PORTABLE_GUI_KEYS, build_portable_preset, compare_portable_presets,
    load_portable_preset, save_portable_preset, validate_portable_preset,
)


def sample_state():
    options = {**GUI_OPTION_DEFAULTS, "out": "/private/results.json", "comfyui": "/private/ComfyUI"}
    return build_frontend_state(
        "llamacpp", ["llm"], [MenuEntry("model", "Model", "llm", "LLM", True)],
        max_prompt_tokens=32768, tg_tokens=[128], gui_options=options,
    )


def test_portable_preset_excludes_private_paths_and_round_trips(tmp_path):
    preset = build_portable_preset("Vendor validation", sample_state())
    serialized = json.dumps(preset)
    assert "/private" not in serialized
    assert set(preset["configuration"]["options"]) == set(PORTABLE_GUI_KEYS)
    path = tmp_path / "preset.json"
    save_portable_preset(path, preset)
    assert load_portable_preset(path) == preset


def test_legacy_preset_without_offline_setting_remains_valid():
    preset = build_portable_preset("Legacy", sample_state())
    del preset["configuration"]["options"]["offline"]
    assert validate_portable_preset(preset) == []


def test_compare_reports_changed_sections():
    original = build_portable_preset("Original", sample_state())
    changed = build_portable_preset("Changed", sample_state())
    changed["configuration"]["options"]["runs"] = 7
    assert compare_portable_presets(original, changed) == ["options"]


@pytest.mark.parametrize("mutation", [
    lambda preset: preset.update(schema_version=99),
    lambda preset: preset.update(name=""),
    lambda preset: preset["configuration"].pop("tests"),
    lambda preset: preset["configuration"]["options"].update(out="private"),
])
def test_portable_preset_validation_rejects_malformed_or_private_fields(mutation):
    preset = build_portable_preset("Valid", sample_state())
    mutation(preset)
    assert validate_portable_preset(preset)
