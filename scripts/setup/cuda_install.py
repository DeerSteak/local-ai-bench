"""CUDA toolkit prerequisites for the llama.cpp source build — see docs/setup.md."""

import shutil
import subprocess

CUDA_WSL_KEYRING_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/"
    "cuda-keyring_1.1-1_all.deb"
)
CUDA_KEYRING_DEB = "/tmp/cuda-keyring.deb"

# Driver-free meta-package. `cuda`, `cuda-12-x`, and `cuda-drivers` each pull a Linux
# driver into WSL2 and break the /dev/dxg passthrough the Windows driver provides.
CUDA_TOOLKIT_PACKAGE = "cuda-toolkit"


def cuda_toolkit_plan(*, is_wsl: bool, nvidia_ok: bool, nvcc_found: bool,
                      which_fn=shutil.which) -> list[list[str]]:
    """Commands installing the WSL-Ubuntu CUDA toolkit, empty when they do not apply.
    Deliberately WSL2-only: native Linux needs its own distro repository, not this one."""
    if nvcc_found or not is_wsl or not nvidia_ok:
        return []
    if which_fn("apt-get") is None:
        return []
    if which_fn("wget"):
        fetch = ["wget", "-qO", CUDA_KEYRING_DEB, CUDA_WSL_KEYRING_URL]
    elif which_fn("curl"):
        fetch = ["curl", "-fsSLo", CUDA_KEYRING_DEB, CUDA_WSL_KEYRING_URL]
    else:
        return []
    return [
        fetch,
        ["sudo", "dpkg", "-i", CUDA_KEYRING_DEB],
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", CUDA_TOOLKIT_PACKAGE],
    ]


def run_cuda_toolkit_install(plan: list[list[str]], *, log=print,
                             run=subprocess.run) -> bool:
    """Execute a `cuda_toolkit_plan`, returning whether every command succeeded."""
    for command in plan:
        log(f"  Running: {' '.join(command)}")
        try:
            failed = run(command).returncode != 0
        except OSError as exc:
            log(f"  Could not run {command[0]}: {exc}")
            return False
        if failed:
            log("  CUDA toolkit install failed — llama.cpp will build CPU-only")
            return False
    return True
