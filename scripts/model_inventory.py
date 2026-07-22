"""Read-only installed-model discovery shared by the CLI and frontend."""

import re
from pathlib import Path

from models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


def sanitize_tag_to_short(tag: str) -> str:
    """Turn a raw tag into a filesystem/JSON-key-safe short identifier."""
    return re.sub(r"[:/]", "-", tag)


def classify_engine_models(installed: list[dict], llm_catalog: list[dict] | None = None,
                           embed_catalog: list[dict] | None = None) -> dict[str, list[dict]]:
    """Split an engine inventory into catalog LLM, embedding, and custom entries."""
    llm_catalog = LLM_MODELS if llm_catalog is None else llm_catalog
    embed_catalog = EMBED_MODELS if embed_catalog is None else embed_catalog
    installed_by_tag = {entry["tag"]: entry for entry in installed}
    llm_tags = {model["tag"] for model in llm_catalog}
    embed_tags = {model["tag"] for model in embed_catalog}

    def installed_catalog(catalog):
        return [
            {**model, "size": installed_by_tag[model["tag"]].get("size")}
            for model in catalog if model["tag"] in installed_by_tag
        ]

    custom = []
    for tag in sorted(set(installed_by_tag) - llm_tags - embed_tags):
        custom.append({
            "tag": tag,
            "label": f"{tag} (custom)",
            "short": sanitize_tag_to_short(tag),
            "size": installed_by_tag[tag].get("size"),
        })

    return {
        "llm": installed_catalog(llm_catalog),
        "embedding": installed_catalog(embed_catalog),
        "custom": custom,
    }


def installed_image_models(comfyui_dir: Path, image_catalog: list[dict] | None = None) -> list[dict]:
    """Return catalog image entries whose primary checkpoint exists locally."""
    image_catalog = IMAGE_MODELS if image_catalog is None else image_catalog
    checkpoints_dir = Path(comfyui_dir) / "models" / "checkpoints"
    installed = []
    for model in image_catalog:
        path = checkpoints_dir / model["checkpoint"]
        if path.exists():
            installed.append({**model, "size": path.stat().st_size, "path": path})
    return installed


def build_model_inventory(engine, comfyui_dir: Path) -> dict[str, list[dict]]:
    """Build the complete read-only inventory for one engine and ComfyUI path."""
    inventory = classify_engine_models(engine.list_installed_models())
    inventory["image"] = installed_image_models(comfyui_dir)
    return inventory


def format_model_inventory(inventory: dict[str, list[dict]], engine_name: str) -> list[str]:
    """Format an installed inventory for `benchmark.py --list-models`."""
    lines = [f"Downloaded models ({engine_name})"]
    groups = (
        ("LLM", "llm", lambda model: model["tag"]),
        ("Embeddings", "embedding", lambda model: model["tag"]),
        ("Custom LLM", "custom", lambda model: model["tag"]),
        ("Image generation", "image", lambda model: model["short"]),
    )
    for label, key, identifier in groups:
        models = inventory.get(key, [])
        if not models:
            continue
        lines.append(f"  {label}:")
        for model in models:
            size = model.get("size")
            size_gb = f"{size / 1e9:.1f} GB" if size is not None else "? GB"
            lines.append(f"    {identifier(model):<40} {size_gb:>10}   {model['label']}")

    counts = {key: len(inventory.get(key, [])) for key in ("llm", "embedding", "custom", "image")}
    lines.append(
        "  "
        f"{counts['llm']} LLM, {counts['embedding']} embedding, "
        f"{counts['custom']} custom, {counts['image']} image installed"
    )
    return lines
