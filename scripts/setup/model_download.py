"""Engine-aware execution for an inspected custom-model import."""

import os
import shutil
from pathlib import Path

from scripts.runtime import config
from scripts.setup.custom_models import custom_model, save_custom_model
from scripts.setup.model_import import ImportVariant, RepositoryInspection, valid_custom_tag
from scripts.setup.vllm_install import hf_cache_model_complete
from scripts.workloads.models import EMBED_MODELS, LLM_MODELS


def cancellable_tqdm(cancel_check):
    from tqdm.auto import tqdm

    class CancellableTqdm(tqdm):
        def update(self, n=1):
            if cancel_check():
                raise InterruptedError("model import cancelled")
            return super().update(n)

        def __iter__(self):
            for item in super().__iter__():
                if cancel_check():
                    raise InterruptedError("model import cancelled")
                yield item

    return CancellableTqdm


def _cache_tree_state(*roots: Path) -> dict[Path, set[Path] | None]:
    return {
        root: ({path.relative_to(root) for path in root.rglob("*")} if root.exists() else None)
        for root in roots
    }


def _remove_cache_changes(state: dict[Path, set[Path] | None]) -> None:
    for root, previous in state.items():
        if not root.exists():
            continue
        if previous is None:
            shutil.rmtree(root)
            continue
        for path in root.rglob("*.incomplete"):
            path.unlink(missing_ok=True)
        current = sorted(
            (path for path in root.rglob("*") if path.relative_to(root) not in previous),
            key=lambda path: len(path.parts), reverse=True,
        )
        for path in current:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass


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


def custom_model_artifacts_present(entry: dict, *, models_dir: Path = config.MODELS_DIR,
                                   vllm_cache: Path | None = None) -> bool:
    engine, tag = entry.get("engine"), entry.get("tag")
    if engine == "llamacpp" and isinstance(tag, str):
        files = entry.get("files")
        destination = Path(models_dir) / "llamacpp" / tag
        return isinstance(files, list) and bool(files) and all(
            isinstance(name, str) and (destination / Path(name).name).is_file()
            for name in files
        )
    repo = entry.get("repo")
    return bool(
        engine == "vllm" and isinstance(repo, str) and vllm_cache is not None
        and hf_cache_model_complete(Path(vllm_cache), repo)
    )


def import_model(*, inspection: RepositoryInspection, engine: str, variant: ImportVariant,
                 tag: str, label: str, vllm_cache: Path | None = None,
                 token: str | None = None, models_dir: Path = config.MODELS_DIR,
                 registry_path: Path = config.CUSTOM_MODELS_PATH,
                 cancel_check=lambda: False) -> dict:
    if engine not in {"llamacpp", "vllm"}:
        raise ValueError("engine must be llamacpp or vllm")
    if not valid_custom_tag(tag):
        raise ValueError("model tag may contain only letters, numbers, dots, underscores, and hyphens")
    if not label.strip():
        raise ValueError("display name is required")
    if tag in {model["tag"] for model in LLM_MODELS + EMBED_MODELS}:
        raise ValueError("model tag conflicts with a catalog model")
    registered = custom_model(engine, tag, registry_path)
    if registered is not None:
        if custom_model_artifacts_present(
            registered, models_dir=models_dir, vllm_cache=vllm_cache,
        ):
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
                if cancel_check():
                    raise InterruptedError("model import cancelled")
                downloaded = Path(hf_hub_download(
                    repo_id=inspection.repo, filename=filename, revision=inspection.revision,
                    local_dir=destination, token=token,
                    tqdm_class=cancellable_tqdm(cancel_check),
                ))
                target = destination / Path(filename).name
                if downloaded != target:
                    shutil.move(str(downloaded), target)
                if cancel_check():
                    raise InterruptedError("model import cancelled")
            if cancel_check():
                raise InterruptedError("model import cancelled")
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
        hub = Path(vllm_cache) / "hub"
        cache_name = f"models--{inspection.repo.replace('/', '--')}"
        cache_state = _cache_tree_state(hub / cache_name, hub / ".locks" / cache_name)
        if cancel_check():
            raise InterruptedError("model import cancelled")
        try:
            snapshot_download(
                repo_id=inspection.repo, revision=inspection.revision, token=token,
                cache_dir=str(hub),
                allow_patterns=[*variant.files, *variant.support_files],
                ignore_patterns=["*.pth", "*.bin", "original/*"],
                tqdm_class=cancellable_tqdm(cancel_check),
            )
            if cancel_check():
                raise InterruptedError("model import cancelled")
        except InterruptedError:
            _remove_cache_changes(cache_state)
            raise
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
