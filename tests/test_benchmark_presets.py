import json

import pytest

from scripts.app.benchmark_frontend import GUI_OPTION_DEFAULTS, MenuEntry, build_frontend_state
from scripts.app.benchmark_presets import (
    PORTABLE_GUI_KEYS, PRESET_SCHEMA_VERSION, build_portable_preset, compare_portable_presets,
    load_portable_preset, migrate_portable_preset, save_portable_preset, validate_portable_preset,
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


def test_presets_carry_no_engine_so_importing_one_keeps_the_engine_selection():
    preset = build_portable_preset("No engine", sample_state())
    assert "engine" not in preset["configuration"]
    assert preset["schema_version"] == PRESET_SCHEMA_VERSION


def test_v1_preset_still_imports_with_its_engine_dropped(tmp_path):
    legacy = build_portable_preset("Legacy v1", sample_state())
    legacy["schema_version"] = 1
    legacy["configuration"]["engine"] = "llamacpp"
    assert validate_portable_preset(legacy) == []
    migrated = migrate_portable_preset(legacy)
    assert migrated["schema_version"] == PRESET_SCHEMA_VERSION
    assert "engine" not in migrated["configuration"]
    assert migrated["configuration"]["tests"] == legacy["configuration"]["tests"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_portable_preset(path) == migrated
    # Migrating an already-current preset is a no-op.
    assert migrate_portable_preset(migrated) == migrated


def test_current_schema_rejects_a_preset_that_still_carries_an_engine():
    preset = build_portable_preset("Stale engine", sample_state())
    preset["configuration"]["engine"] = "llamacpp"
    assert validate_portable_preset(preset)


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
