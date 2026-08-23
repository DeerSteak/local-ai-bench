"""Catalog validation and expansion for optional GGUF quantization variants."""

from copy import deepcopy


VARIANT_KEYS = ("tag", "short", "quantization", "hf_repo", "hf_file", "download_size")


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
    return [
        {
            **common,
            **deepcopy(variant),
            "base_model": model["base_model"],
            "variant": variant["quantization"],
        }
        for variant in model["variants"]
    ]


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
