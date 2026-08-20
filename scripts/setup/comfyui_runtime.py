"""Prepare an installed ComfyUI runtime and its accelerator dependencies."""

import subprocess
from pathlib import Path

from scripts.runtime.comfyui_installation import (
    add_managed_models_to_comfyui, find_comfyui_python,
    legacy_models_dir_with_assets, write_extra_model_paths,
)


AMD_ROCM_72_WHEELS = {
    False: (
        "torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl",
        "torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl",
        "torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl",
        "triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl",
    ),
    True: (
        "torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl",
        "torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl",
        "torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl",
        "triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl",
    ),
}


def prepare(comfyui_dir: Path, models_dir: Path, extra_paths: Path, *,
            portable_python: Path, intel_xpu: bool, rocm: bool,
            rocm_version: tuple[int, int] | None = None, wsl: bool = False,
            issues: list[str], info, warn, fail, ok) -> bool:
    if not comfyui_dir.exists():
        return False
    python = find_comfyui_python(comfyui_dir)
    requirements = comfyui_dir / "requirements.txt"
    if not portable_python.exists() and intel_xpu:
        _ensure_torch_backend(python, "xpu", "XPU", issues, info, fail, ok)
    if not portable_python.exists() and rocm:
        _ensure_rocm_torch_backend(
            python, rocm_version, wsl=wsl, issues=issues,
            info=info, fail=fail, ok=ok,
        )
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


def rocm_torch_install_command(python: str, version: tuple[int, int] | None, *,
                               wsl: bool) -> list[str]:
    if version is None or version < (7, 2):
        return [
            python, "-m", "pip", "install", "--upgrade", "--force-reinstall",
            "--index-url", "https://download.pytorch.org/whl/rocm6.4",
            "torch", "torchvision", "torchaudio",
        ]
    release = "7.2" if wsl else "7.2.1"
    base = f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{release}"
    return [
        python, "-m", "pip", "install", "--upgrade", "--force-reinstall",
        "numpy==1.26.4", *(f"{base}/{wheel}" for wheel in AMD_ROCM_72_WHEELS[wsl]),
    ]


def remove_wsl_bundled_hsa_runtime(python: str, *, run=subprocess.run) -> bool:
    script = (
        "import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()['purelib'])/"
        "'torch'/'lib'; [f.unlink() for f in p.glob('libhsa-runtime64.so*')]"
    )
    return run([python, "-c", script]).returncode == 0


def _ensure_rocm_torch_backend(python: str, version: tuple[int, int] | None, *,
                               wsl: bool, issues: list[str], info, fail, ok) -> None:
    if torch_backend_available(python, "ROCm"):
        ok("ROCm-enabled PyTorch already installed")
        return
    command = rocm_torch_install_command(python, version, wsl=wsl)
    source = "AMD's ROCm wheel repository" if version and version >= (7, 2) \
        else "PyTorch's ROCm 6.4 index"
    info(f"Installing ROCm-enabled PyTorch from {source} ...")
    installed = subprocess.run(command).returncode == 0
    if installed and wsl:
        installed = remove_wsl_bundled_hsa_runtime(python)
    if installed and torch_backend_available(python, "ROCm"):
        ok("ROCm-enabled PyTorch installed")
    else:
        fail("ROCm-enabled PyTorch install failed")
        issues.append(" ".join(command))
