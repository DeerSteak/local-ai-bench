"""Read-only host and accelerator discovery for setup."""

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemDiscovery:
    os_name: str
    release: str
    machine: str
    node: str
    total_ram_gb: float | None
    chip: str | None = None


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
