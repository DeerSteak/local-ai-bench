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
