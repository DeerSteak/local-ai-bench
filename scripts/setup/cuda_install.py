"""CUDA toolkit prerequisites for the llama.cpp source build — see docs/setup.md."""

import shutil
import subprocess
from pathlib import Path

from scripts.runtime.hardware import detect_wsl

CUDA_WSL_KEYRING_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/"
    "cuda-keyring_1.1-1_all.deb"
)
CUDA_UBUNTU_REPO_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{release}/x86_64/"
    "cuda-keyring_1.1-1_all.deb"
)
CUDA_KEYRING_DEB = "/tmp/cuda-keyring.deb"

# Driver-free meta-package. `cuda`, `cuda-12-x`, and `cuda-drivers` each pull a Linux
# driver into WSL2 and break the /dev/dxg passthrough the Windows driver provides.
CUDA_TOOLKIT_PACKAGE = "cuda-toolkit"
NATIVE_NVIDIA_UBUNTU_VERSIONS = {"24.04", "26.04"}
NATIVE_NVIDIA_REBOOT_EXIT_CODE = 75
NOUVEAU_BLOCKLIST_COMMAND = (
    "printf '%s\\n' 'blacklist nouveau' 'options nouveau modeset=0' "
    "> /etc/modprobe.d/disable-nouveau.conf"
)


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


def native_cuda_toolkit_plan(os_release: dict[str, str], architecture: str, *,
                             nvidia_ok: bool, nvcc_found: bool,
                             which_fn=shutil.which) -> list[list[str]]:
    if nvcc_found or not nvidia_ok:
        return []
    distribution = os_release.get("ID", "").lower()
    version = os_release.get("VERSION_ID", "")
    if (distribution != "ubuntu" or version not in NATIVE_NVIDIA_UBUNTU_VERSIONS
            or architecture.lower() not in {"x86_64", "amd64"}):
        return []
    if which_fn("apt-get") is None:
        return []
    url = CUDA_UBUNTU_REPO_URL.format(release=version.replace(".", ""))
    if which_fn("wget"):
        fetch = ["wget", "-qO", CUDA_KEYRING_DEB, url]
    elif which_fn("curl"):
        fetch = ["curl", "-fsSLo", CUDA_KEYRING_DEB, url]
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


def qualification_needs_native_nvidia_driver(target: dict, *, os_name: str,
                                              release: str,
                                              nvidia_available: bool) -> bool:
    if (target["platform"] != "linux" or target["backend"] != "cuda"
            or target["architecture"] != "x86_64"):
        return False
    if os_name != "Linux" or detect_wsl(os_name, release):
        raise ValueError(f"target {target['id']} requires native Linux; detected {os_name} {release}")
    return not nvidia_available


def nouveau_loaded(module_path: Path = Path("/sys/module/nouveau")) -> bool:
    return module_path.is_dir()


def native_nvidia_driver_plan(os_release: dict[str, str], kernel_release: str, *,
                              disable_nouveau: bool = False) \
        -> tuple[tuple[str, ...], ...]:
    distribution = os_release.get("ID", "").lower()
    version = os_release.get("VERSION_ID", "")
    if distribution != "ubuntu" or version not in NATIVE_NVIDIA_UBUNTU_VERSIONS:
        detected = f"{distribution or 'unknown'} {version or 'unknown'}"
        raise ValueError(
            "automatic NVIDIA driver installation requires Ubuntu 24.04 or 26.04; "
            f"detected {detected}"
        )
    if not kernel_release:
        raise ValueError("automatic NVIDIA driver installation could not identify the kernel")
    commands = [
        ("apt-get", "update"),
        (
            "apt-get", "install", "-y", "ubuntu-drivers-common",
            f"linux-headers-{kernel_release}",
        ),
        ("ubuntu-drivers", "install"),
    ]
    if disable_nouveau:
        commands.extend((
            ("sh", "-c", NOUVEAU_BLOCKLIST_COMMAND),
            ("update-initramfs", "-u"),
        ))
    return tuple(commands)


def run_native_nvidia_driver_install(plan: tuple[tuple[str, ...], ...], *,
                                     run=subprocess.run) -> None:
    for command in plan:
        completed = run(["sudo", *command])
        if completed.returncode:
            raise RuntimeError(
                f"NVIDIA driver install command exited with {completed.returncode}: "
                f"{' '.join(command)}"
            )
