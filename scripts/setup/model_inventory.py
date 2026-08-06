"""Installed-model discovery and cleanup helpers."""

import re
import shutil
from pathlib import Path

from scripts.runtime import config, hardware
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


def sanitize_tag_to_short(tag: str) -> str:
    """Turn a raw tag into a filesystem/JSON-key-safe short identifier."""
    return re.sub(r"[:/]", "-", tag)


def model_tag_slug(tag: str) -> str:
    """Return the llama.cpp model-directory name for a catalog tag."""
    return tag.replace(":", "_").replace("/", "_")


def engine_model_dir(models_root: Path, engine: str, tag: str) -> Path:
    """Per-engine model directory — mirrors each engine's own `_models_dir()`.
    Not used by vLLM, which resolves weights from an HF cache by repo id."""
    return Path(models_root) / engine / model_tag_slug(tag)


def engine_download_size(model: dict, engine: str) -> str | None:
    """Download size under `engine`, or None for entries with no engine weights
    (image checkpoints). Engines carry different weights for the same model."""
    if engine == "vllm":
        return model.get("vllm_download_size") or model.get("download_size")
    return model.get("download_size")


def engine_model_complete(model_dir: Path, engine: str, filenames=()) -> bool:
    """True once `model_dir` holds every file `engine` needs. vLLM keeps its weights in
    an HF cache instead — see vllm_install.hf_cache_model_complete."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return False
    return all((model_dir / Path(name).name).exists() for name in filenames)


ENGINE_LABELS = {"llamacpp": "llama.cpp", "vllm": "vLLM"}


def engine_fit_report(model: dict, engines, ceiling_gb: float | None) -> dict[str, dict]:
    """Per-engine size, memory need, and fit — engines carry different weights."""
    report = {}
    for engine in engines:
        if engine == "vllm" and not model.get("vllm_repo"):
            continue
        size = engine_download_size(model, engine)
        if size is None:
            continue
        report[engine] = {
            "size": size,
            "needed_gb": hardware.model_memory_requirement_gb(size),
            "fits": hardware.model_fits(size, ceiling_gb),
        }
    return report


def fits_any_engine(report: dict[str, dict]) -> bool | None:
    """True if at least one engine can hold it, None when the ceiling is unknown."""
    verdicts = [entry["fits"] for entry in report.values()]
    if not verdicts or all(verdict is None for verdict in verdicts):
        return None
    return any(verdict for verdict in verdicts)


def format_engine_sizes(report: dict[str, dict]) -> str:
    """'~6.2 GB' for one engine, 'llama.cpp ~6.2 GB · vLLM ~12.4 GB' for several."""
    if len(report) == 1:
        return next(iter(report.values()))["size"]
    return " · ".join(f"{ENGINE_LABELS.get(engine, engine)} {entry['size']}"
                      for engine, entry in report.items())


def engine_fit_warnings(report: dict[str, dict], ceiling_gb: float | None) -> list[str]:
    """One warning per engine that can't hold this model."""
    if ceiling_gb is None:
        return []
    prefix_needed = len(report) > 1
    return [
        (f"{ENGINE_LABELS.get(engine, engine)} needs " if prefix_needed else "needs ")
        + f"~{entry['needed_gb']:.1f} GB, ~{ceiling_gb:.1f} GB available"
        for engine, entry in report.items() if entry["fits"] is False
    ]


def models_missing_engine_support(models: list[dict], engine: str) -> list[str]:
    """Catalog tags with no weights defined for `engine`."""
    if engine != "vllm":
        return []
    return [model["tag"] for model in models if not model.get("vllm_repo")]


def find_non_catalog_model_dirs(models_dir: Path, llm_catalog: list[dict] | None = None,
                                embed_catalog: list[dict] | None = None) -> list[Path]:
    """Return installed model directories not owned by the current catalog."""
    llm_catalog = LLM_MODELS if llm_catalog is None else llm_catalog
    embed_catalog = EMBED_MODELS if embed_catalog is None else embed_catalog
    catalog_slugs = {model_tag_slug(model["tag"])
                     for model in llm_catalog + embed_catalog}
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(
        (path for path in models_dir.iterdir()
         if (path.is_dir() or path.is_symlink())
         and path.name not in catalog_slugs
         and any(path.glob("*.gguf"))),
        key=lambda path: path.name,
    )


def delete_non_catalog_model_dirs(models_dir: Path, directory_names: list[str],
                                  llm_catalog: list[dict] | None = None,
                                  embed_catalog: list[dict] | None = None,
                                  ) -> tuple[list[str], dict[str, str]]:
    """Delete explicitly named non-catalog directories without following symlinks."""
    llm_catalog = LLM_MODELS if llm_catalog is None else llm_catalog
    embed_catalog = EMBED_MODELS if embed_catalog is None else embed_catalog
    catalog_slugs = {model_tag_slug(model["tag"])
                     for model in llm_catalog + embed_catalog}
    models_dir = Path(models_dir)
    removed = []
    failures = {}
    for name in directory_names:
        if name != Path(name).name or name in catalog_slugs:
            failures[name] = "not an eligible non-catalog directory"
            continue
        target = models_dir / name
        if not target.is_dir() and not target.is_symlink():
            failures[name] = "directory no longer exists"
            continue
        if not any(target.glob("*.gguf")):
            failures[name] = "directory does not contain a GGUF model"
            continue
        try:
            if target.is_symlink():
                target.unlink()
            else:
                shutil.rmtree(target)
            removed.append(name)
        except OSError as exc:
            failures[name] = str(exc)
    return removed, failures


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


def installed_image_models(models_dir: Path, image_catalog: list[dict] | None = None) -> list[dict]:
    """Return catalog image entries in benchmark-managed model storage."""
    image_catalog = IMAGE_MODELS if image_catalog is None else image_catalog
    checkpoints_dir = Path(models_dir) / "checkpoints"
    installed = []
    for model in image_catalog:
        path = checkpoints_dir / model["checkpoint"]
        if path.exists():
            installed.append({**model, "size": path.stat().st_size, "path": path})
    return installed


def build_model_inventory(engine, image_models_dir: Path) -> dict[str, list[dict]]:
    """Build the complete read-only inventory with benchmark-managed images."""
    inventory = classify_engine_models(engine.list_installed_models())
    inventory["image"] = installed_image_models(image_models_dir)
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
