"""Provision selected ComfyUI checkpoints and supporting model assets."""

from pathlib import Path


CHECKPOINT_REPOS = {
    "sd15": "Comfy-Org/stable-diffusion-v1-5-archive",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd35-large": "stabilityai/stable-diffusion-3.5-large",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux2-dev": "black-forest-labs/FLUX.2-dev",
}
GATED_MODELS = {"sd35-large", "flux-dev", "flux2-dev"}


def provision(selected_images: list[dict], models_dir: Path, *, find_asset, download,
              load_token, info, warn, fail, ok) -> list[str]:
    checkpoints = models_dir / "checkpoints"
    clip_dir = models_dir / "clip"
    vae_dir = models_dir / "vae"
    found = []
    for model in selected_images:
        existing = find_asset(model["checkpoint"], "checkpoints")
        if existing:
            size_gb = existing.stat().st_size / (1024 ** 3)
            ok(f"Checkpoint found: {model['checkpoint']} ({size_gb:.1f} GB)")
            found.append(model["checkpoint"])
    missing = [model for model in selected_images if model["checkpoint"] not in found]
    if missing:
        info(f"Downloading {len(missing)} missing checkpoint(s): "
             f"{', '.join(model['checkpoint'] for model in missing)}")
        checkpoints.mkdir(parents=True, exist_ok=True)
    for model in missing:
        short, checkpoint = model["short"], model["checkpoint"]
        token = load_token()
        if short in GATED_MODELS and not token:
            info(f"Skipping {model['label']} — no token provided")
            continue
        if download(CHECKPOINT_REPOS[short], checkpoint, token=token, dest_dir=checkpoints):
            ok(f"{checkpoint} downloaded")
            found.append(checkpoint)
        elif short in GATED_MODELS:
            fail(f"{model['label']} download failed — check token and license acceptance")
        else:
            warn(f"{model['label']} download failed — image benchmarks will run without it")
    sd35 = any("sd3.5" in name for name in found)
    flux1 = "flux1-dev.safetensors" in found
    flux2 = "flux2-dev.safetensors" in found
    if flux1 or sd35:
        for filename in ("t5xxl_fp16.safetensors", "clip_l.safetensors"):
            _download_support(
                filename, "clip", "comfyanonymous/flux_text_encoders", filename,
                clip_dir, find_asset, download, load_token, warn, ok,
            )
    if sd35:
        _download_support(
            "clip_g.safetensors", "clip", "stabilityai/stable-diffusion-3.5-large",
            "text_encoders/clip_g.safetensors", clip_dir, find_asset, download,
            load_token, warn, ok, save_as="clip_g.safetensors", gated=True,
        )
    if flux1:
        _download_support(
            "ae.safetensors", "vae", "black-forest-labs/FLUX.1-schnell",
            "ae.safetensors", vae_dir, find_asset, download, load_token, warn, ok,
            gated=True,
        )
    if flux2:
        text_dir = models_dir / "text_encoders"
        filename = "mistral_3_small_flux2_fp8.safetensors"
        _download_support(
            filename, "text_encoders", "Comfy-Org/flux2-dev",
            f"split_files/text_encoders/{filename}", text_dir, find_asset, download,
            load_token, warn, ok, save_as=filename,
        )
        _download_support(
            "flux2-vae.safetensors", "vae", "Comfy-Org/flux2-dev",
            "split_files/vae/flux2-vae.safetensors", vae_dir, find_asset, download,
            load_token, warn, ok, save_as="flux2-vae.safetensors",
        )
    return found


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
