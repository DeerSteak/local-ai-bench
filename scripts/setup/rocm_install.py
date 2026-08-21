"""Install AMD's pinned ROCm stack for supported qualification targets."""

from dataclasses import dataclass
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from scripts.runtime.hardware import detect_wsl


ROCM_VERSION = "7.2"
ROCM_INSTALLER_BUILD = "7.2.70200-1"
NATIVE_ROCM_VERSION = "7.2.1"
NATIVE_ROCM_INSTALLER_BUILD = "7.2.1.70201-1"
WINDOWS_DRIVER = "AMD Software: Adrenalin Edition 26.1.1 for WSL2"
UBUNTU_CODENAMES = {"22.04": "jammy", "24.04": "noble"}
NATIVE_UBUNTU_VERSIONS = {"24.04"}
NATIVE_KERNELS = {(6, 8), (6, 17)}
GET_EFFECTIVE_UID = getattr(os, "geteuid", lambda: 1)


@dataclass(frozen=True)
class RocmInstallPlan:
    package_url: str
    package_path: Path
    commands: tuple[tuple[str, ...], ...]


def parse_os_release(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip("\"'")
    return values


def wsl_rocm_install_plan(os_release: str, *, temp_dir: Path | None = None) -> RocmInstallPlan:
    release = parse_os_release(os_release)
    distribution = release.get("ID", "").lower()
    version = release.get("VERSION_ID", "")
    if distribution != "ubuntu" or version not in UBUNTU_CODENAMES:
        detected = f"{distribution or 'unknown'} {version or 'unknown'}"
        raise ValueError(
            f"ROCm {ROCM_VERSION} on WSL supports Ubuntu 22.04 or 24.04; detected {detected}"
        )
    codename = UBUNTU_CODENAMES[version]
    package = f"amdgpu-install_{ROCM_INSTALLER_BUILD}_all.deb"
    package_url = (
        f"https://repo.radeon.com/amdgpu-install/{ROCM_VERSION}/ubuntu/{codename}/{package}"
    )
    package_path = (temp_dir or Path(tempfile.gettempdir())) / package
    commands = (
        ("apt-get", "update"),
        ("apt-get", "install", "-y", str(package_path)),
        ("amdgpu-install", "-y", "--usecase=wsl,rocm", "--no-dkms"),
    )
    return RocmInstallPlan(package_url, package_path, commands)


def native_rocm_install_plan(os_release: str, kernel_release: str, *, user: str | None,
                             temp_dir: Path | None = None) -> RocmInstallPlan:
    release = parse_os_release(os_release)
    distribution = release.get("ID", "").lower()
    version = release.get("VERSION_ID", "")
    if distribution != "ubuntu" or version not in NATIVE_UBUNTU_VERSIONS:
        detected = f"{distribution or 'unknown'} {version or 'unknown'}"
        raise ValueError(
            f"native ROCm {NATIVE_ROCM_VERSION} qualification requires Ubuntu 24.04; "
            f"detected {detected}"
        )
    match = re.match(r"(\d+)\.(\d+)", kernel_release)
    kernel = tuple(map(int, match.groups())) if match else None
    if kernel not in NATIVE_KERNELS:
        supported = " or ".join(".".join(map(str, item)) for item in sorted(NATIVE_KERNELS))
        raise ValueError(
            f"native ROCm {NATIVE_ROCM_VERSION} qualification requires kernel {supported}; "
            f"detected {kernel_release or 'unknown'}"
        )
    if user and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user):
        raise ValueError("could not safely identify the user for AMD GPU permissions")
    package = f"amdgpu-install_{NATIVE_ROCM_INSTALLER_BUILD}_all.deb"
    package_url = (
        f"https://repo.radeon.com/amdgpu-install/{NATIVE_ROCM_VERSION}/ubuntu/noble/"
        f"{package}"
    )
    package_path = (temp_dir or Path(tempfile.gettempdir())) / package
    commands = [
        ("apt-get", "install", "-y", str(package_path)),
        ("apt-get", "update"),
        ("amdgpu-install", "-y", "--usecase=graphics,rocm"),
    ]
    if user:
        commands.append(("usermod", "-aG", "render,video", user))
    return RocmInstallPlan(package_url, package_path, tuple(commands))


def qualification_needs_wsl_rocm(target: dict, *, os_name: str, release: str,
                                 rocm_available: bool) -> bool:
    if target["platform"] != "wsl2" or target["backend"] != "rocm":
        return False
    if not detect_wsl(os_name, release):
        raise ValueError(f"target {target['id']} requires WSL2; detected {os_name} {release}")
    return not rocm_available


def qualification_needs_native_rocm(target: dict, *, os_name: str, release: str,
                                    rocm_available: bool) -> bool:
    if target["platform"] != "linux" or target["backend"] != "rocm":
        return False
    if os_name != "Linux" or detect_wsl(os_name, release):
        raise ValueError(f"target {target['id']} requires native Linux; detected {os_name} {release}")
    return not rocm_available


def run_rocm_install(plan: RocmInstallPlan, *,
                     download=urllib.request.urlretrieve,
                     run=subprocess.run, geteuid=GET_EFFECTIVE_UID) -> None:
    download(plan.package_url, plan.package_path)
    prefix = () if geteuid() == 0 else ("sudo",)
    for command in plan.commands:
        completed = run([*prefix, *command])
        if completed.returncode:
            raise RuntimeError(
                f"ROCm install command exited with {completed.returncode}: {' '.join(command)}"
            )
