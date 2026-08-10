"""Engine-aware execution for an inspected custom-model import."""

import os
import shutil
from pathlib import Path

from scripts.runtime import config
from scripts.setup.custom_models import custom_model, save_custom_model
from scripts.setup.model_import import ImportVariant, RepositoryInspection, valid_custom_tag
from scripts.workloads.models import EMBED_MODELS, LLM_MODELS


def load_hf_token(env=None, token_path: Path | None = None) -> str | None:
    env = os.environ if env is None else env
    token = str(env.get("HF_TOKEN", "")).strip()
    if token:
        return token
    path = token_path or config.SCRIPT_DIR / "hf.txt"
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def import_model(*, inspection: RepositoryInspection, engine: str, variant: ImportVariant,
                 tag: str, label: str, vllm_cache: Path | None = None,
                 token: str | None = None, models_dir: Path = config.MODELS_DIR,
                 registry_path: Path = config.CUSTOM_MODELS_PATH) -> dict:
    if engine not in {"llamacpp", "vllm"}:
        raise ValueError("engine must be llamacpp or vllm")
    if not valid_custom_tag(tag):
        raise ValueError("model tag may contain only letters, numbers, dots, underscores, and hyphens")
    if not label.strip():
        raise ValueError("display name is required")
    if tag in {model["tag"] for model in LLM_MODELS + EMBED_MODELS}:
        raise ValueError("model tag conflicts with a catalog model")
    if custom_model(engine, tag, registry_path) is not None:
        raise ValueError("model tag is already registered for this engine")
    if engine == "llamacpp":
        if variant not in inspection.llama_variants:
            raise ValueError("selected GGUF variant is not available")
        destination = Path(models_dir) / "llamacpp" / tag
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"model destination already exists: {destination}")
        created_destination = not destination.exists()
        destination.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download
        try:
            for filename in variant.files:
                downloaded = Path(hf_hub_download(
                    repo_id=inspection.repo, filename=filename, revision=inspection.revision,
                    local_dir=destination, token=token,
                ))
                target = destination / Path(filename).name
                if downloaded != target:
                    shutil.move(str(downloaded), target)
        except BaseException:
            if created_destination and destination.exists():
                shutil.rmtree(destination)
            else:
                for path in destination.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink(missing_ok=True)
            raise
        record = {
            "tag": tag, "label": label.strip(), "engine": engine,
            "repo": inspection.repo, "revision": inspection.revision,
            "format": "gguf", "files": [Path(name).name for name in variant.files],
        }
    else:
        if inspection.vllm_variant is None or variant != inspection.vllm_variant:
            raise ValueError("repository has no importable vLLM snapshot")
        if vllm_cache is None:
            raise ValueError("vLLM cache location is unavailable")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=inspection.repo, revision=inspection.revision, token=token,
            cache_dir=str(Path(vllm_cache) / "hub"),
            allow_patterns=[*variant.files, *variant.support_files],
            ignore_patterns=["*.pth", "*.bin", "original/*"],
        )
        record = {
            "tag": tag, "label": label.strip(), "engine": engine,
            "repo": inspection.repo, "revision": inspection.revision,
            "format": "safetensors", "files": [],
        }
    save_custom_model(record, registry_path)
    return record


def enough_disk_space(variant: ImportVariant, destination: Path) -> bool | None:
    if variant.size is None:
        return None
    destination = Path(destination)
    probe = destination if destination.exists() else destination.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free >= variant.size
