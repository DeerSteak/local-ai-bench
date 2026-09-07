"""Durable, non-secret selections shared by terminal and graphical setup."""

from pathlib import Path

from scripts.runtime.comfyui_installation import normalize_comfyui_dir
from scripts.setup.setup_config import load_setup_config, write_setup_data, SCHEMA_VERSION
from scripts.workloads.models import (
    EMBED_MODELS, IMAGE_MODELS, LLM_MODELS_XSMALL, LLM_MODELS_SMALL,
    LLM_MODELS_MEDIUM, LLM_MODELS_LARGE,
)
from scripts.workloads.model_variants import expanded_variant_catalog


def model_keys() -> dict[str, list[str]]:
    return {
        "llm_tags": [model["tag"] for tier in (
            LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE,
        ) for model in expanded_variant_catalog(tier)],
        "embedding_tags": [model["tag"] for model in EMBED_MODELS],
        "image_shorts": [model["short"] for model in IMAGE_MODELS],
    }


def setup_preferences(data: dict, *, qualification: bool = False) -> dict:
    saved = data.get("setup_preferences")
    if qualification or not isinstance(saved, dict) or type(saved.get("version")) is not int or saved.get("version") != 1:
        return {}
    result: dict = {"version": 1}
    for key in ("models", "engines"):
        values = saved.get(key)
        if isinstance(values, dict):
            result[key] = {name: checked for name, checked in values.items()
                           if isinstance(name, str) and type(checked) is bool}
    if type(saved.get("save_token_preference")) is bool:
        result["save_token_preference"] = saved["save_token_preference"]
    if saved.get("comfyui_mode") in ("existing", "detected", "download"):
        result["comfyui_mode"] = saved["comfyui_mode"]
    if isinstance(saved.get("comfyui_path"), str):
        result["comfyui_path"] = saved["comfyui_path"]
    return result


def restore_model_selection(defaults: dict[str, bool], preferences: dict) -> dict[str, bool]:
    saved = preferences.get("models", {})
    return {name: saved.get(name, checked) for name, checked in defaults.items()}


def restore_engine_selection(entries: list[dict], preferences: dict) -> None:
    saved = preferences.get("engines", {})
    checked = [entry["enabled"] and saved.get(entry["name"], entry["checked"]) for entry in entries]
    if any(checked):
        for entry, value in zip(entries, checked):
            entry["checked"] = value


def restored_comfyui_options(preferences: dict, detected: Path | None,
                              explicit: str | None = None) -> tuple[str, str]:
    if explicit:
        return "existing", str(normalize_comfyui_dir(Path(explicit)) or "")
    mode = preferences.get("comfyui_mode")
    if mode == "download":
        return "download", ""
    if mode == "existing" and preferences.get("comfyui_path"):
        path = normalize_comfyui_dir(Path(preferences["comfyui_path"]))
        if path:
            return "existing", str(path)
    return ("detected", str(detected)) if detected else ("download", "")


def write_setup_preferences(path: Path, plan: dict, *, qualification: bool = False) -> None:
    if qualification:
        return
    models = {}
    for key, names in model_keys().items():
        selected = set(plan.get(key, []))
        models.update({name: name in selected for name in names})
    selected_engines = set(plan.get("engines", []))
    saved = {
        "version": 1, "models": models,
        "engines": {name: name in selected_engines for name in ("llamacpp", "llamacpp-vulkan", "vllm")},
        "comfyui_mode": plan.get("comfyui_mode"),
        "comfyui_path": plan.get("comfyui_path", ""),
        "save_token_preference": plan.get("save_token_preference", True),
    }
    data = load_setup_config(path)
    data.update(schema_version=SCHEMA_VERSION, setup_preferences=saved)
    write_setup_data(path, data)
