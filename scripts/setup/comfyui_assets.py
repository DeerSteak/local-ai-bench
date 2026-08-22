"""Provision selected ComfyUI checkpoints and supporting model assets."""

from pathlib import Path

from scripts.runtime.hardware import CHECKPOINT_SIZES_GB, ENCODER_SIZES_GB
from scripts.workloads.models import image_checkpoint_folder


CHECKPOINT_REPOS = {
    "sd15": "Comfy-Org/stable-diffusion-v1-5-archive",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux2-dev": "black-forest-labs/FLUX.2-dev",
}
GATED_MODELS = {"flux-dev", "flux2-dev"}


def provision(selected_images: list[dict], models_dir: Path, *, find_asset, download,
              load_token, info, warn, fail, ok) -> list[str]:
    found = []
    for model in selected_images:
        existing = find_asset(model["checkpoint"], image_checkpoint_folder(model))
        if existing:
            size_gb = existing.stat().st_size / (1024 ** 3)
            ok(f"Checkpoint found: {model['checkpoint']} ({size_gb:.1f} GB)")
            found.append(model["checkpoint"])
    missing = [model for model in selected_images if model["checkpoint"] not in found]
    if missing:
        info(f"Downloading {len(missing)} missing checkpoint(s): "
             f"{', '.join(model['checkpoint'] for model in missing)}")
    for model in missing:
        short, checkpoint = model["short"], model["checkpoint"]
        token = load_token()
        if short in GATED_MODELS and not token:
            info(f"Skipping {model['label']} — no token provided")
            continue
        destination = models_dir / image_checkpoint_folder(model)
        destination.mkdir(parents=True, exist_ok=True)
        repo = model.get("checkpoint_repo", CHECKPOINT_REPOS.get(short))
        remote = model.get("checkpoint_remote", checkpoint)
        save_as = checkpoint if remote != checkpoint else None
        if download(repo, remote, token=token, dest_dir=destination, save_as=save_as):
            ok(f"{checkpoint} downloaded")
            found.append(checkpoint)
        elif short in GATED_MODELS:
            fail(f"{model['label']} download failed — check token and license acceptance")
        else:
            warn(f"{model['label']} download failed — image benchmarks will run without it")
    for model in selected_images:
        if model["checkpoint"] not in found:
            continue
        for asset in model.get("support_assets", ()):
            _download_support(
                asset["name"], asset["folder"], asset["repo"], asset["remote"],
                models_dir / asset["folder"], find_asset, download, load_token, warn, ok,
                save_as=asset["name"], gated=asset.get("gated", False),
            )

    return found


def missing_download_size_gb(selected_images: list[dict], find_asset) -> float:
    total = 0.0
    counted = set()
    for model in selected_images:
        checkpoint = model["checkpoint"]
        checkpoint_key = (image_checkpoint_folder(model), checkpoint)
        if checkpoint_key not in counted:
            counted.add(checkpoint_key)
            if not find_asset(checkpoint, checkpoint_key[0]):
                total += CHECKPOINT_SIZES_GB.get(checkpoint, 0.0)
        for asset in model.get("support_assets", ()):
            key = (asset["folder"], asset["name"])
            if key not in counted:
                counted.add(key)
                if not find_asset(asset["name"], asset["folder"]):
                    total += ENCODER_SIZES_GB.get(asset["name"], 0.0)
    return total


def _download_support(filename: str, subdir: str, repo: str, remote: str,
                      destination: Path, find_asset, download, load_token,
                      warn, ok, *, save_as: str | None = None,
                      gated: bool = False) -> None:
    if find_asset(filename, subdir):
        ok(f"{filename} already present")
        return
    token = load_token()
    if gated and not token:
        warn(f"Skipping {filename} — no token provided")
        return
    if download(repo, remote, token=token, dest_dir=destination, save_as=save_as):
        ok(f"{filename} downloaded")
    else:
        warn(f"{filename} download failed — image generation will error")
