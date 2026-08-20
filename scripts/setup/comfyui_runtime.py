"""Prepare an installed ComfyUI runtime and its accelerator dependencies."""

import subprocess
from pathlib import Path

from scripts.runtime.comfyui_installation import (
    add_managed_models_to_comfyui, find_comfyui_python,
    legacy_models_dir_with_assets, write_extra_model_paths,
)


def prepare(comfyui_dir: Path, models_dir: Path, extra_paths: Path, *,
            portable_python: Path, intel_xpu: bool, rocm: bool,
            issues: list[str], info, warn, fail, ok) -> bool:
    if not comfyui_dir.exists():
        return False
    python = find_comfyui_python(comfyui_dir)
    requirements = comfyui_dir / "requirements.txt"
    if portable_python.exists():
        ok("Windows portable build detected — using bundled python_embeded")
    elif requirements.exists():
        installed = subprocess.run(
            [python, "-m", "pip", "show", "aiohttp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if installed:
            ok("ComfyUI requirements already installed")
        else:
            info("Installing ComfyUI requirements ...")
            if subprocess.run([python, "-m", "pip", "install", "-r", str(requirements)]).returncode:
                fail("ComfyUI requirements install failed")
                issues.append(f"pip install -r {requirements}")
            else:
                ok("ComfyUI requirements installed")
    else:
        warn("ComfyUI requirements.txt not found — clone may be incomplete")
    write_extra_model_paths(extra_paths, models_dir, legacy_models_dir_with_assets(comfyui_dir))
    try:
        model_config = add_managed_models_to_comfyui(comfyui_dir, models_dir)
        ok(f"ComfyUI model path configured in {model_config}")
    except OSError as exc:
        warn(f"Could not update ComfyUI's extra model paths: {exc}")
        issues.append(f"Add {models_dir} to ComfyUI's extra model paths")
    if not portable_python.exists() and intel_xpu:
        _ensure_torch_backend(python, "xpu", "XPU", issues, info, fail, ok)
    if not portable_python.exists() and rocm:
        _ensure_torch_backend(python, "rocm6.4", "ROCm", issues, info, fail, ok)
    return True


def torch_backend_available(python: str, marker: str, *, run=subprocess.run) -> bool:
    expression = (
        "import torch; assert torch.version.hip; torch.zeros(1, device='cuda'); "
        "print(torch.version.hip)" if marker == "ROCm"
        else "import torch; assert torch.xpu.is_available(); torch.zeros(1, device='xpu'); "
        "print('xpu')"
    )
    result = run([python, "-c", expression], capture_output=True, text=True)
    output = result.stdout.strip().lower() if result.returncode == 0 else ""
    return bool(output) and output not in {"false", "none"}


def _ensure_torch_backend(python: str, index: str, marker: str,
                          issues: list[str], info, fail, ok) -> None:
    if torch_backend_available(python, marker):
        ok(f"{marker}-enabled PyTorch already installed")
        return
    url = f"https://download.pytorch.org/whl/{index}"
    info(f"Installing {marker}-enabled PyTorch from {url} ...")
    command = [
        python, "-m", "pip", "install", "--upgrade", "--force-reinstall", "--index-url", url,
        "torch", "torchvision", "torchaudio",
    ]
    if subprocess.run(command).returncode == 0 and torch_backend_available(python, marker):
        ok(f"{marker}-enabled PyTorch installed")
    else:
        fail(f"{marker}-enabled PyTorch install failed")
        issues.append(" ".join(command))
