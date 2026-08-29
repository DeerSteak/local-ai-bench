"""Read-only host and accelerator discovery for setup."""

import platform
import json
import shutil
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


@dataclass(frozen=True)
class RocmDiscovery:
    names: list[str]
    gfx_targets: list[str]
    kind: str | None
    gpus: list[dict]

    @property
    def available(self) -> bool:
        return bool(self.names)

    @property
    def total_vram_gb(self) -> float | None:
        return sum(device["vram_gb"] for device in self.gpus) if self.gpus else None


@dataclass(frozen=True)
class DisplayDiscovery:
    vendor: str | None
    kind: str | None
    name: str | None


_WINDOWS_AMD_INVENTORY_SCRIPT = r"""
$devices = Get-CimInstance Win32_VideoController | Where-Object {
    $_.PNPDeviceID -like 'PCI\VEN_1002*' -and ($_.Name -match 'AMD|Radeon')
}
$inventory = @($devices | ForEach-Object {
    $memory = $null
    try {
        $driver = (Get-PnpDeviceProperty -InstanceId $_.PNPDeviceID `
            -KeyName DEVPKEY_Device_Driver -ErrorAction Stop).Data
        $properties = Get-ItemProperty -LiteralPath `
            "HKLM:\SYSTEM\CurrentControlSet\Control\Class\$driver" -ErrorAction Stop
        $memory = $properties.'HardwareInformation.qwMemorySize'
    } catch {}
    [pscustomobject]@{
        name = $_.Name
        pnp_device_id = $_.PNPDeviceID
        driver = $_.DriverVersion
        vram_bytes = $memory
    }
})
ConvertTo-Json -InputObject $inventory -Compress
"""


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
            [hardware.nvidia_smi_executable(), "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL,
        )
        gpus = hardware.parse_nvidia_gpus(inventory)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return NvidiaDiscovery([], None, None)
    try:
        capability = subprocess.check_output(
            [hardware.nvidia_smi_executable(), "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        capability = None
    try:
        summary = subprocess.check_output(
            [hardware.nvidia_smi_executable()], text=True, stderr=subprocess.DEVNULL,
        )
        max_cuda = hardware.parse_nvidia_max_cuda_version(summary)
    except (FileNotFoundError, subprocess.CalledProcessError):
        max_cuda = None
    return NvidiaDiscovery(gpus, capability, max_cuda)


def discover_rocm() -> RocmDiscovery:
    try:
        output = subprocess.check_output(
            [hardware.rocm_executable("rocminfo") or "rocminfo"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return RocmDiscovery([], [], None, [])
    names = hardware.rocminfo_gpu_names(output)
    targets = hardware.rocminfo_gfx_targets(output)
    kind = None
    gpus = []
    if names:
        kind = (
            "discrete" if any(hardware.classify_gpu(name) == "discrete" for name in names)
            else "integrated"
        )
    if kind == "discrete":
        try:
            memory = subprocess.check_output(
                [hardware.rocm_executable("rocm-smi") or "rocm-smi",
                 "--showmeminfo", "vram", "--json"],
                text=True, stderr=subprocess.DEVNULL,
            )
            gpus = hardware.parse_rocm_smi_gpus(memory, names)
        except (FileNotFoundError, subprocess.CalledProcessError,
                json.JSONDecodeError, ValueError):
            pass
    return RocmDiscovery(names, targets, kind, gpus)


def rocm_version(version_path: Path = Path("/opt/rocm/.info/version")) -> tuple[int, int] | None:
    try:
        output = subprocess.check_output(
            [hardware.rocm_executable("hipconfig") or "hipconfig", "--version"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return hardware.parse_rocm_version(output)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        return hardware.parse_rocm_version(version_path.read_text(encoding="utf-8"))
    except OSError:
        return None


def discover_metal() -> tuple[bool, list[str]]:
    if platform.system() != "Darwin":
        return False, []
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True,
        )
    except Exception:
        return False, []
    available = "Metal" in output or "Apple" in output
    details = [
        line.strip() for line in output.splitlines()
        if "Chipset Model" in line or "Metal" in line
    ]
    return available, details if available else []


def discover_windows_gpu() -> DisplayDiscovery:
    if platform.system() != "Windows":
        return DisplayDiscovery(None, None, None)
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return DisplayDiscovery(None, None, None)
    for name in (line.strip() for line in output.splitlines() if line.strip()):
        if "AMD" in name or "Radeon" in name:
            return DisplayDiscovery("amd", hardware.classify_gpu(name), name)
        if hardware.is_intel_xpu_display(name):
            return DisplayDiscovery("intel", hardware.classify_gpu(name), name)
    return DisplayDiscovery(None, None, None)


def parse_windows_amd_gpus(output: str) -> list[dict]:
    """Parse active Windows AMD adapters and their driver-reported 64-bit VRAM."""
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    devices = []
    seen = set()
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        identity = item.get("pnp_device_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            continue
        seen.add(identity)
        try:
            vram_bytes = int(item.get("vram_bytes") or 0)
        except (TypeError, ValueError):
            vram_bytes = 0
        devices.append({
            "name": item["name"].strip(),
            "vram_gb": vram_bytes / (1024 ** 3) if vram_bytes > 0 else None,
            "driver": str(item.get("driver") or ""),
            "vendor": "amd",
            "backend": "vulkan",
            "kind": hardware.classify_gpu(item["name"]),
            "pnp_device_id": identity,
        })
    return devices


def discover_windows_amd_gpus() -> list[dict]:
    if platform.system() != "Windows":
        return []
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", _WINDOWS_AMD_INVENTORY_SCRIPT],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return parse_windows_amd_gpus(output)


def discover_wsl_windows_amd_gpus() -> list[dict]:
    if not hardware.detect_wsl(platform.system(), platform.release()):
        return []
    try:
        output = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", _WINDOWS_AMD_INVENTORY_SCRIPT],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return parse_windows_amd_gpus(output)


def discover_linux_intel_gpu() -> DisplayDiscovery:
    if platform.system() != "Linux":
        return DisplayDiscovery(None, None, None)
    try:
        output = subprocess.check_output(["lspci", "-nn"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return DisplayDiscovery(None, None, None)
    for line in output.splitlines():
        if (any(key in line for key in ("VGA", "3D controller", "Display"))
                and hardware.is_intel_xpu_display(line)):
            name = line.split(":", 2)[-1].strip()
            return DisplayDiscovery("intel", hardware.classify_gpu(name), name)
    return DisplayDiscovery(None, None, None)


def discover_intel_vram_gb(*, sysfs_root: Path = Path("/sys/class/drm")) -> float | None:
    """Read total Intel device-local memory from XPU-SMI or the Linux DRM driver."""
    if executable := shutil.which("xpu-smi"):
        try:
            output = subprocess.check_output(
                [executable], text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
            snapshot = hardware.parse_xpu_smi_memory_gb(output)
            if snapshot is not None:
                return snapshot[1]
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    if platform.system() != "Linux":
        return None
    totals = []
    for device in sorted(sysfs_root.glob("card[0-9]*/device")):
        try:
            if (device / "vendor").read_text(encoding="utf-8").strip().casefold() != "0x8086":
                continue
            total = int((device / "mem_info_vram_total").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if total > 0:
            totals.append(total / (1024 ** 3))
    return sum(totals) if totals else None


def discover_linux_amd_gpu() -> DisplayDiscovery:
    if platform.system() != "Linux":
        return DisplayDiscovery(None, None, None)
    try:
        output = subprocess.check_output(["lspci", "-nn"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return DisplayDiscovery(None, None, None)
    for line in output.splitlines():
        if (any(key in line for key in ("VGA", "3D controller", "Display"))
                and ("AMD/ATI" in line or "[1002:" in line)):
            name = line.split(":", 2)[-1].strip()
            return DisplayDiscovery("amd", hardware.classify_gpu(name), name)
    return DisplayDiscovery(None, None, None)


def discover_linux_nvidia_gpu() -> DisplayDiscovery:
    if platform.system() != "Linux":
        return DisplayDiscovery(None, None, None)
    try:
        output = subprocess.check_output(["lspci", "-nn"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return DisplayDiscovery(None, None, None)
    for line in output.splitlines():
        if (any(key in line for key in ("VGA", "3D controller", "Display"))
                and ("NVIDIA" in line or "[10de:" in line.casefold())):
            name = line.split(":", 2)[-1].strip()
            return DisplayDiscovery("nvidia", "discrete", name)
    return DisplayDiscovery(None, None, None)
