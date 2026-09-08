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
RYZEN_AI_HALO_TARGET_PREFIX = "ryzen-ai-halo-"
RYZEN_AI_HALO_OEM_PACKAGE = "linux-oem-24.04"
RYZEN_AI_HALO_MIN_OEM_KERNEL = (6, 14, 0, 1018)
RYZEN_AI_HALO_FORBIDDEN_DKMS_PACKAGES = ("amdgpu-dkms", "amdgpu-dkms-firmware")
GET_EFFECTIVE_UID = getattr(os, "geteuid", lambda: 1)


@dataclass(frozen=True)
class RocmInstallPlan:
    package_url: str
    package_path: Path
    commands: tuple[tuple[str, ...], ...]
    reboot_required: bool = False


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


def ryzen_ai_halo_oem_kernel_ready(target_id: str, kernel_release: str) -> bool:
    if not target_id.startswith(RYZEN_AI_HALO_TARGET_PREFIX):
        return True
    match = re.match(r"(\d+)\.(\d+)\.(\d+)-(\d+)-oem(?:$|[-+])", kernel_release)
    return bool(match and tuple(map(int, match.groups())) >= RYZEN_AI_HALO_MIN_OEM_KERNEL)


def ryzen_ai_halo_dkms_packages(target_id: str, *,
                                run=subprocess.run) -> tuple[str, ...]:
    if not target_id.startswith(RYZEN_AI_HALO_TARGET_PREFIX):
        return ()
    installed = []
    for package in RYZEN_AI_HALO_FORBIDDEN_DKMS_PACKAGES:
        result = run(
            ["dpkg-query", "-W", "-f=${db:Status-Abbrev}", package],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip().startswith("i"):
            installed.append(package)
    return tuple(installed)


def native_rocm_install_plan(os_release: str, kernel_release: str, *, target_id: str,
                             user: str | None, temp_dir: Path | None = None) -> RocmInstallPlan:
    release = parse_os_release(os_release)
    distribution = release.get("ID", "").lower()
    version = release.get("VERSION_ID", "")
    if distribution != "ubuntu" or version not in NATIVE_UBUNTU_VERSIONS:
        detected = f"{distribution or 'unknown'} {version or 'unknown'}"
        raise ValueError(
            f"native ROCm {NATIVE_ROCM_VERSION} qualification requires Ubuntu 24.04; "
            f"detected {detected}"
        )
    halo = target_id.startswith(RYZEN_AI_HALO_TARGET_PREFIX)
    match = re.match(r"(\d+)\.(\d+)", kernel_release)
    kernel = tuple(map(int, match.groups())) if match else None
    if not halo and kernel not in NATIVE_KERNELS:
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
    packages = ["python3-setuptools", "python3-wheel", str(package_path)]
    oem_kernel_ready = ryzen_ai_halo_oem_kernel_ready(target_id, kernel_release)
    reboot_required = halo
    if halo and not oem_kernel_ready:
        packages.insert(0, RYZEN_AI_HALO_OEM_PACKAGE)
    commands = []
    if halo:
        commands.extend((
            ("dpkg", "--purge", *RYZEN_AI_HALO_FORBIDDEN_DKMS_PACKAGES),
            ("update-initramfs", "-u"),
        ))
    commands.extend((
        ("apt-get", "update"),
        ("apt-get", "install", "-y", *packages),
        (
            "amdgpu-install", "-y", "--usecase=rocm", "--no-dkms",
        ) if halo else (
            "amdgpu-install", "-y", "--usecase=graphics,rocm",
        ),
    ))
    if user:
        commands.append(("usermod", "-aG", "render,video", user))
    return RocmInstallPlan(
        package_url, package_path, tuple(commands), reboot_required=reboot_required,
    )


def qualification_needs_wsl_rocm(target: dict, *, os_name: str, release: str,
                                 rocm_available: bool) -> bool:
    if target["platform"] != "wsl2" or target["backend"] != "rocm":
        return False
    if not detect_wsl(os_name, release):
        raise ValueError(f"target {target['id']} requires WSL2; detected {os_name} {release}")
    return not rocm_available


def setup_needs_wsl_rocm(*, os_name: str, release: str, amd_gpus: list[dict],
                         rocm_available: bool) -> bool:
    return bool(
        detect_wsl(os_name, release) and amd_gpus and not rocm_available
    )


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
