"""Read-only host and accelerator discovery for setup."""

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.runtime import hardware


@dataclass(frozen=True)
class SystemDiscovery:
    os_name: str
    release: str
    machine: str
    node: str
    total_ram_gb: float | None
    chip: str | None = None


@dataclass(frozen=True)
class NvidiaDiscovery:
    gpus: list[dict]
    compute_capability: str | None
    max_cuda_version: str | None

    @property
    def available(self) -> bool:
        return bool(self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return sum(device["vram_gb"] or 0.0 for device in self.gpus)


def discover_system(meminfo_path: Path = Path("/proc/meminfo")) -> SystemDiscovery:
    os_name = platform.system()
    total_ram_gb = None
    chip = None
    if os_name == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            ).strip()
        except Exception:
            chip = "unknown"
        try:
            memory = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True,
            ).strip())
            total_ram_gb = memory / (1024 ** 3)
        except Exception:
            pass
    elif os_name == "Linux":
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal"):
                    total_ram_gb = int(line.split()[1]) / (1024 ** 2)
                    break
        except (OSError, ValueError, IndexError):
            pass
    elif os_name == "Windows":
        try:
            output = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            total_ram_gb = int(output.splitlines()[-1].strip()) / (1024 ** 3)
        except Exception:
            pass
    return SystemDiscovery(
        os_name=os_name, release=platform.release(), machine=platform.machine(),
        node=platform.node(), total_ram_gb=total_ram_gb, chip=chip,
    )


def discover_nvidia() -> NvidiaDiscovery:
    try:
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL,
        )
        gpus = hardware.parse_nvidia_gpus(inventory)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return NvidiaDiscovery([], None, None)
    try:
        capability = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        capability = None
    try:
        summary = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.DEVNULL,
        )
        max_cuda = hardware.parse_nvidia_max_cuda_version(summary)
    except (FileNotFoundError, subprocess.CalledProcessError):
        max_cuda = None
    return NvidiaDiscovery(gpus, capability, max_cuda)
