"""Portable benchmark preset serialization without secrets or private paths."""

import json
from pathlib import Path

from scripts.app.benchmark_options import PUBLIC_OPTION_SCHEMA
from scripts.results.result_store import atomic_write_json


PRESET_SCHEMA_VERSION = 2
# v1 carried an "engine"; v2 drops it so importing a preset never overrides the
# engine selection on screen. v1 files still import, with that field ignored.
SUPPORTED_PRESET_SCHEMA_VERSIONS = (1, PRESET_SCHEMA_VERSION)
PORTABLE_GUI_KEYS = (
    "warmup", "runs", "timeout", "acc_timeout", "acc_token_budget", "gpu_split_mode",
    "cpu_only", "force_all", "offline",
)


def build_portable_preset(name: str, state: dict) -> dict:
    options = state.get("gui_options", {})
    return {
        "schema_version": PRESET_SCHEMA_VERSION,
        "name": name.strip(),
        "configuration": {
            "tests": list(state["tests"]),
            "models": {key: list(values) for key, values in state["models"].items()},
            "max_prompt_tokens": state.get("max_prompt_tokens"),
            "tg_tokens": list(state["tg_tokens"]) if state.get("tg_tokens") else None,
            "options": {key: options[key] for key in PORTABLE_GUI_KEYS},
        },
    }


def validate_portable_preset(preset: object) -> list[str]:
    if not isinstance(preset, dict) or set(preset) != {"schema_version", "name", "configuration"}:
        return ["Preset must contain schema_version, name, and configuration."]
    if preset["schema_version"] not in SUPPORTED_PRESET_SCHEMA_VERSIONS:
        return [f"Unsupported preset schema version: {preset['schema_version']}."]
    if not isinstance(preset["name"], str) or not preset["name"].strip():
        return ["Preset name cannot be blank."]
    config = preset["configuration"]
    required = {"tests", "models", "max_prompt_tokens", "tg_tokens", "options"}
    allowed = required | ({"engine"} if preset["schema_version"] == 1 else set())
    if not isinstance(config, dict) or not required <= set(config) <= allowed:
        return ["Preset configuration is incomplete."]
    if not isinstance(config["tests"], list):
        return ["Preset tests are invalid."]
    if not isinstance(config["models"], dict) or set(config["models"]) != {"llm", "embedding", "image"}:
        return ["Preset model selections are invalid."]
    option_keys = set(config["options"]) if isinstance(config["options"], dict) else set()
    expected_keys = set(PORTABLE_GUI_KEYS)
    if (option_keys - expected_keys
            or not (expected_keys - option_keys).issubset({"offline", "gpu_split_mode"})):
        return ["Preset execution options are invalid."]
    split_mode = config["options"].get("gpu_split_mode", "layer")
    if split_mode not in PUBLIC_OPTION_SCHEMA["--gpu-split-mode"].choices:
        return ["Preset GPU split mode is invalid."]
    return []


def save_portable_preset(path: Path, preset: dict) -> None:
    errors = validate_portable_preset(preset)
    if errors:
        raise ValueError(errors[0])
    atomic_write_json(Path(path), preset)


def migrate_portable_preset(preset: dict) -> dict:
    """Bring a validated preset up to the current schema by dropping v1's engine."""
    if preset["schema_version"] == PRESET_SCHEMA_VERSION:
        return preset
    configuration = {key: value for key, value in preset["configuration"].items()
                     if key != "engine"}
    return {**preset, "schema_version": PRESET_SCHEMA_VERSION,
            "configuration": configuration}


def load_portable_preset(path: Path) -> dict:
    preset = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_portable_preset(preset)
    if errors:
        raise ValueError(errors[0])
    return migrate_portable_preset(preset)


def compare_portable_presets(left: dict, right: dict) -> list[str]:
    left_config = left["configuration"]
    right_config = right["configuration"]
    return [
        key for key in sorted(left_config)
        if left_config.get(key) != right_config.get(key)
    ]
