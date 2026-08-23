"""Catalog validation and expansion for optional GGUF quantization variants."""

from copy import deepcopy
import re


VARIANT_KEYS = ("tag", "short", "quantization", "hf_repo", "hf_file", "download_size")
SIZE_PATTERN = re.compile(r"~?\s*([0-9]+(?:\.[0-9]+)?)\s*GB", re.IGNORECASE)


def _nonempty_string(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_hf_file(value) -> bool:
    return _nonempty_string(value) or (
        isinstance(value, list) and bool(value)
        and all(_nonempty_string(item) for item in value)
    )


def validate_model_variants(model: dict) -> None:
    variants = model.get("variants")
    if variants is None:
        return
    if not _nonempty_string(model.get("base_model")):
        raise ValueError("multi-variant model requires base_model")
    if not isinstance(variants, list) or len(variants) < 2:
        raise ValueError("model variants must contain at least two entries")
    if any(not isinstance(variant, dict) for variant in variants):
        raise ValueError("model variant must be an object")
    for variant in variants:
        missing = [key for key in VARIANT_KEYS if key not in variant]
        if missing:
            raise ValueError(f"model variant is missing fields: {', '.join(missing)}")
        if any(not _nonempty_string(variant[key]) for key in VARIANT_KEYS if key != "hf_file") \
                or not _valid_hf_file(variant["hf_file"]):
            raise ValueError("model variant fields must be non-empty strings")
        if "default" in variant and not isinstance(variant["default"], bool):
            raise ValueError("model variant default must be boolean")
    for key in ("tag", "short", "quantization"):
        values = [variant[key] for variant in variants]
        if len(values) != len(set(values)):
            raise ValueError(f"model variants contain duplicate {key}")
    defaults = [variant for variant in variants if variant.get("default")]
    if len(defaults) != 1:
        raise ValueError("model variants require exactly one default")
    default = defaults[0]
    for key in ("tag", "short", "hf_repo", "hf_file", "download_size"):
        if default[key] != model.get(key):
            raise ValueError(f"default model variant must match top-level {key}")


def expanded_model_variants(model: dict) -> list[dict]:
    """Return executable model records; legacy single-variant records stay unchanged."""
    validate_model_variants(model)
    if "variants" not in model:
        return [deepcopy(model)]
    common = {key: deepcopy(value) for key, value in model.items() if key != "variants"}
    expanded = []
    for variant in model["variants"]:
        record = {
            **common,
            **deepcopy(variant),
            "base_model": model["base_model"],
            "base_label": model["label"],
            "variant": variant["quantization"],
            "label": (
                f"{model['label']} — {variant['quantization']} "
                f"({variant['download_size']})"
            ),
        }
        if not variant.get("default"):
            for key in [name for name in record if name.startswith("vllm_")]:
                record.pop(key)
            record.pop("native_mtp", None)
        expanded.append(record)
    return expanded


def variant_selection_state(tags, selected_tags) -> str:
    """Return the aggregate checkbox state for one quantization family."""
    members = set(tags)
    selected = members & set(selected_tags)
    if not selected:
        return "none"
    return "all" if selected == members else "some"


def variant_selection_target(tags, selected_tags) -> set[str]:
    """A family click clears a full selection; otherwise it selects every child."""
    members = set(tags)
    selected = set(selected_tags)
    return selected - members if members <= selected else selected | members


def collapse_variant_selection(models, selected_tags) -> set[str]:
    """Replace each selected quantization family with only its default variant."""
    selected = set(selected_tags)
    grouped: dict[str, list[dict]] = {}
    for model in models:
        if isinstance(model.get("base_model"), str) and isinstance(model.get("variant"), str):
            grouped.setdefault(model["base_model"], []).append(model)
    for variants in grouped.values():
        tags = {model["tag"] for model in variants}
        if selected & tags:
            selected -= tags
            default = next((model["tag"] for model in variants if model.get("default")), None)
            if default is not None:
                selected.add(default)
    return selected


def default_model_variant(model: dict) -> dict:
    variants = expanded_model_variants(model)
    return next((variant for variant in variants if variant.get("default")), variants[0])


def expanded_variant_catalog(catalog: list[dict]) -> list[dict]:
    return [variant for model in catalog for variant in expanded_model_variants(model)]


def normalize_variant_selectors(selectors: list[str] | None, catalog: list[dict]) -> dict[str, tuple[str, ...]]:
    if selectors is None:
        return {}
    if not selectors:
        raise ValueError("variant selection must not be empty")
    available = {
        model["base_model"]: model for model in catalog if model.get("variants") is not None
    }
    selected: dict[str, list[str]] = {}
    seen = set()
    for selector in selectors:
        if not isinstance(selector, str) or selector.count("=") != 1:
            raise ValueError("variant selectors must use BASE=VARIANT")
        base_model, variant = (part.strip() for part in selector.split("=", 1))
        if not base_model or not variant:
            raise ValueError("variant selectors must use non-empty BASE=VARIANT")
        model = available.get(base_model)
        known = {
            item["quantization"] for item in model.get("variants", [])
        } if model else set()
        if variant not in known:
            raise ValueError(f"unknown model/variant pair: {base_model}={variant}")
        pair = (base_model, variant)
        if pair in seen:
            raise ValueError(f"duplicate model/variant pair: {base_model}={variant}")
        seen.add(pair)
        selected.setdefault(base_model, []).append(variant)
    return {base_model: tuple(variants) for base_model, variants in selected.items()}


def select_model_variants(models: list[dict], selections: dict[str, tuple[str, ...]]) -> list[dict]:
    if not selections:
        return deepcopy(models)
    selected_bases = {
        base_model for model in models
        if isinstance(base_model := model.get("base_model"), str)
    }
    missing = sorted(set(selections) - selected_bases)
    if missing:
        raise ValueError(f"variant base model is not selected: {', '.join(missing)}")
    resolved = []
    for model in models:
        base_model = model.get("base_model")
        requested = selections.get(base_model) if isinstance(base_model, str) else None
        if requested is None:
            resolved.append(deepcopy(model))
            continue
        variants = {
            item["variant"]: item for item in expanded_model_variants(model)
        }
        resolved.extend(variants[name] for name in requested)
    return resolved


def variant_sweep_cost(models: list[dict], catalog: list[dict], installed_tags=()) -> dict | None:
    selected_variants = [model for model in models if model.get("base_model") and model.get("variant")]
    if not selected_variants:
        return None
    installed = set(installed_tags)
    catalog_by_base = {model.get("base_model"): model for model in catalog if model.get("variants")}

    def size_gb(model: dict) -> float | None:
        match = SIZE_PATTERN.fullmatch(str(model.get("download_size", "")).strip())
        return float(match.group(1)) if match else None

    added_disk = 0.0
    download = 0.0
    grouped: dict[str, list[dict]] = {}
    for model in selected_variants:
        grouped.setdefault(model["base_model"], []).append(model)
        if model["tag"] not in installed and (size := size_gb(model)) is not None:
            download += size
    for base_model, selected in grouped.items():
        catalog_model = catalog_by_base[base_model]
        default = default_model_variant(catalog_model)
        selected_size = sum(size_gb(model) or 0 for model in selected)
        added_disk += max(0.0, selected_size - (size_gb(default) or 0))
    baseline_units = len(models) - sum(len(selected) - 1 for selected in grouped.values())
    return {
        "added_disk_gb": round(added_disk, 1),
        "download_gb": round(download, 1),
        "runtime_multiplier": round(len(models) / baseline_units, 2),
    }
