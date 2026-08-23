"""Engine-aware execution for an inspected custom-model import."""

import os
import shutil
import subprocess
from pathlib import Path

from scripts.runtime import config
from scripts.setup.custom_models import custom_model, save_custom_model
from scripts.setup.model_import import ImportVariant, RepositoryInspection, valid_custom_tag
from scripts.setup.vllm_install import hf_cache_model_complete
from scripts.setup.model_inventory import (
    engine_download_size, engine_model_complete, engine_model_dir,
    models_missing_engine_support,
)
from scripts.runtime.mtp import native_mtp_config
from scripts.workloads.models import EMBED_MODELS, LLM_MODELS


def download_hf_files(repo: str, filenames: str | list[str], destination: Path, *,
                      token: str | None = None, save_as: str | None = None,
                      warn=lambda _message: None) -> bool:
    destination.mkdir(parents=True, exist_ok=True)
    requested = filenames if isinstance(filenames, list) else [filenames]
    env = {**os.environ, **({"HF_TOKEN": token} if token else {})}
    success = True
    for filename in requested:
        downloaded = False
        for cli in ("hf", "huggingface-cli"):
            if not shutil.which(cli):
                continue
            result = subprocess.run(
                [cli, "download", repo, filename, "--local-dir", str(destination)],
                env=env, capture_output=True, text=True,
            )
            downloaded = result.returncode == 0
            if not downloaded:
                detail = (result.stderr or result.stdout or "").strip()
                if detail:
                    warn(f"{cli} error: {detail}")
            break
        if not downloaded:
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=repo, filename=filename, local_dir=str(destination), token=token,
                )
                downloaded = True
            except Exception as exc:
                warn(f"Python API download failed: {exc}")
        success = success and downloaded
        if downloaded:
            source = destination / filename
            target_name = save_as if save_as and len(requested) == 1 else Path(filename).name
            target = destination / target_name
            if source.exists() and source != target:
                shutil.move(str(source), str(target))
                try:
                    source.parent.rmdir()
                except OSError:
                    pass
    return success


def download_hf_snapshot(repo: str, cache_home: Path, *, token: str | None = None,
                         warn=lambda _message: None) -> bool:
    cache_home.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HOME": str(cache_home), **({"HF_TOKEN": token} if token else {})}
    ignored = ["*.pth", "*.bin", "original/*"]
    for cli in ("hf", "huggingface-cli"):
        if not shutil.which(cli):
            continue
        command = [cli, "download", repo]
        for pattern in ignored:
            command += ["--exclude", pattern]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            warn(f"{cli} error: {detail}")
        break
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo, token=token, ignore_patterns=ignored,
            cache_dir=str(cache_home / "hub"),
        )
        return True
    except Exception as exc:
        warn(f"Python API download failed: {exc}")
        return False


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


def catalog_model_downloaded(model: dict, engine: str, *, models_dir: Path,
                             vllm_cache: Path) -> bool:
    if engine == "vllm":
        return hf_cache_model_complete(vllm_cache, model["vllm_repo"])
    filenames = model["hf_file"] if isinstance(model["hf_file"], list) else [model["hf_file"]]
    directory = engine_model_dir(models_dir, engine, model["tag"])
    return engine_model_complete(directory, engine, filenames)


def catalog_mtp_artifact_downloaded(model: dict, engine: str, *, models_dir: Path) -> bool:
    config = native_mtp_config(model, engine)
    if config is None or "draft_file" not in config:
        return True
    directory = engine_model_dir(models_dir, engine, model["tag"])
    return (directory / Path(config["draft_file"]).name).is_file()


def catalog_mtp_artifact_download_size(model: dict, engine: str) -> str | None:
    config = native_mtp_config(model, engine)
    if config is None or "draft_file" not in config:
        return None
    value = model["native_mtp"][engine].get("draft_download_size")
    return value if isinstance(value, str) and value.strip() else None


def provision_catalog_models(models: list[dict], engines: list[str], *,
                             models_dir: Path, vllm_cache: Path, load_token,
                             issues: list[str], info, warn, fail, ok) -> None:
    for engine in engines:
        if len(engines) > 1:
            info(f"Models for {engine} ...")
        engine_models = [
            model for model in models
            if engine != "vllm" or not model.get("variant") or model.get("default", False)
        ]
        unsupported = models_missing_engine_support(engine_models, engine)
        for tag in unsupported:
            warn(f"{tag} — no {engine} weights defined; skipping for this engine")
            issues.append(f"No {engine} weights for {tag}")
        for model in engine_models:
            tag, label = model["tag"], model["label"]
            if tag in unsupported:
                continue
            size = engine_download_size(model, engine)
            primary_downloaded = catalog_model_downloaded(
                model, engine, models_dir=models_dir, vllm_cache=vllm_cache,
            )
            success = True
            if not primary_downloaded:
                warn(f"{label} [{engine}] ({size}) — downloading ...")
                if engine == "vllm":
                    repository, destination = model["vllm_repo"], vllm_cache
                    success = download_hf_snapshot(
                        repository, vllm_cache, token=load_token(), warn=warn,
                    )
                else:
                    repository = model["hf_repo"]
                    destination = engine_model_dir(models_dir, engine, tag)
                    success = download_hf_files(
                        repository, model["hf_file"], destination,
                        token=load_token(), warn=warn,
                    )
                if not success:
                    fail(f"{label} [{engine}] — download failed")
                    issues.append(f"Download {repository} manually into {destination}")
                    continue

            mtp_config = native_mtp_config(model, engine)
            if (mtp_config is not None and "draft_file" in mtp_config
                    and not catalog_mtp_artifact_downloaded(
                        model, engine, models_dir=models_dir,
                    )):
                destination = engine_model_dir(models_dir, engine, tag)
                draft_size = catalog_mtp_artifact_download_size(model, engine) or "size unknown"
                warn(f"{label} [{engine}] MTP predictor ({draft_size}) — downloading ...")
                success = download_hf_files(
                    mtp_config["draft_repo"], mtp_config["draft_file"], destination,
                    token=load_token(), warn=warn,
                )
                if not success:
                    fail(f"{label} [{engine}] MTP predictor — download failed")
                    issues.append(
                        f"Download {mtp_config['draft_repo']}/{mtp_config['draft_file']} "
                        f"manually into {destination}"
                    )
                    continue
            state = "already downloaded" if primary_downloaded else "downloaded successfully"
            ok(f"{label} [{engine}] — {state}")
