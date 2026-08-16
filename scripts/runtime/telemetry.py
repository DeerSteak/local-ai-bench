"""Shared resource queries, sampling, and memory aggregation."""

from dataclasses import asdict, dataclass
import ctypes
import platform
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import psutil

from scripts.runtime import config
from scripts.runtime import hardware
from scripts.runtime.shared import Shared


MEMORY_CHANNELS = (
    "host_ram_used_gb", "process_rss_gb",
    "accelerator_memory_used_gb", "accelerator_memory_total_gb",
)
POWER_SCOPES = ("processor_package", "accelerator", "cpu_package", "whole_system", "unknown")
ADL_PMLOG_MAX_SENSORS = 256
ADL_PMLOG_ASIC_POWER = 23
ADL_PMLOG_BOARD_POWER = 73


class _PsutilProcess(Protocol):
    @property
    def pid(self) -> int: ...
    def children(self, recursive: bool = False) -> Sequence["_PsutilProcess"]: ...
    def cpu_percent(self, interval: float | None = None) -> float: ...
    def memory_info(self) -> Any: ...


class _PsutilMemory(Protocol):
    @property
    def used(self) -> int: ...
    @property
    def total(self) -> int: ...


class PsutilLike(Protocol):
    Error: Any
    def Process(self, pid: int, /) -> _PsutilProcess: ...
    def virtual_memory(self) -> _PsutilMemory: ...


@dataclass(frozen=True)
class TelemetrySample:
    timestamp_sec: float
    window: str
    host_ram_used_gb: float | None = None
    process_rss_gb: float | None = None
    accelerator_memory_used_gb: float | None = None
    accelerator_memory_total_gb: float | None = None
    power_watts: float | None = None


@dataclass(frozen=True)
class ChannelSummary:
    peak_gb: float | None
    mean_gb: float | None
    final_gb: float | None
    valid_samples: int


@dataclass(frozen=True)
class WindowSummary:
    name: str
    sample_count: int
    duration_sec: float
    channels: Mapping[str, ChannelSummary]


@dataclass(frozen=True)
class Headroom:
    absolute_gb: float | None
    fraction: float | None
    state: str


@dataclass(frozen=True)
class PowerReading:
    watts: float
    source: str
    scope: str


@dataclass(frozen=True)
class PowerAvailability:
    available: bool
    source: str
    scope: str
    reason: str | None = None
    location: str | None = None


def power_availability_dict(value: PowerAvailability) -> dict[str, Any]:
    return {
        "available": value.available,
        "source": value.source,
        "scope": value.scope,
        "reason": value.reason,
    }


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def parse_powermetrics_power(output: str) -> PowerReading | None:
    patterns = (
        r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([0-9.]+)\s*(m?W)\b",
        r"Package Power:\s*([0-9.]+)\s*(m?W)\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        values = []
        for raw, unit in matches:
            value = _finite_nonnegative(raw)
            if value is not None:
                values.append(value / 1000 if unit.lower() == "mw" else value)
        if values:
            return PowerReading(values[-1], "powermetrics", "processor_package")
    return None


def parse_nvidia_power(output: str) -> PowerReading | None:
    values = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s*([0-9.]+)\s*(?:W)?\s*", line, flags=re.IGNORECASE)
        value = _finite_nonnegative(match.group(1)) if match else None
        if value is not None:
            values.append(value)
    return PowerReading(sum(values), "nvidia-smi", "accelerator") if values else None


def parse_rocm_power(output: str) -> PowerReading | None:
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    keys = ("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)")
    values = []
    for card in payload.values():
        if not isinstance(card, dict):
            continue
        value = next((_finite_nonnegative(card.get(key)) for key in keys if key in card), None)
        if value is not None:
            values.append(value)
    return PowerReading(sum(values), "rocm-smi", "accelerator") if values else None


class _AdlSensorData(ctypes.Structure):
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _AdlPmLogData(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("sensors", _AdlSensorData * ADL_PMLOG_MAX_SENSORS),
    ]


def parse_amd_adl_power(sensors: Sequence[tuple[int, int]]) -> float | None:
    for sensor in (ADL_PMLOG_ASIC_POWER, ADL_PMLOG_BOARD_POWER):
        if sensor >= len(sensors):
            continue
        supported, raw_value = sensors[sensor]
        value = _finite_nonnegative(raw_value) if supported else None
        if value is not None:
            return value
    return None


class AmdAdlPowerSource:
    def __init__(self, availability: PowerAvailability, *, library_factory=None):
        self.availability = availability
        self.library_factory = library_factory or _load_amd_adl_library
        self._library = None
        self._context = ctypes.c_void_p()
        self._allocations: list[Any] = []
        self._allocator = None
        self._adapter_count = 0

    def start(self) -> None:
        try:
            library = self.library_factory()
            if library is None:
                return
            self._configure(library)
            self._library = library
            allocator_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)

            def allocate(size):
                buffer = ctypes.create_string_buffer(size)
                self._allocations.append(buffer)
                return ctypes.addressof(buffer)

            self._allocator = allocator_type(allocate)
            if library.ADL2_Main_Control_Create(
                    self._allocator, 1, ctypes.byref(self._context)) != 0:
                self.stop()
                return
            count = ctypes.c_int()
            if library.ADL2_Adapter_NumberOfAdapters_Get(
                    self._context, ctypes.byref(count)) != 0:
                self.stop()
                return
            self._adapter_count = max(0, min(count.value, 250))
        except (AttributeError, OSError, TypeError, ValueError):
            self.stop()

    def read_watts(self) -> float | None:
        if self._library is None or not self._context.value:
            return None
        values = []
        for adapter_index in range(self._adapter_count):
            output = _AdlPmLogData()
            output.size = ctypes.sizeof(output)
            try:
                result = self._library.ADL2_New_QueryPMLogData_Get(
                    self._context, adapter_index, ctypes.byref(output),
                )
            except (OSError, ValueError):
                continue
            if result != 0:
                continue
            value = parse_amd_adl_power([
                (sensor.supported, sensor.value) for sensor in output.sensors
            ])
            if value is not None:
                values.append(value)
        return sum(values) if values else None

    def stop(self) -> None:
        if self._library is not None and self._context.value:
            try:
                self._library.ADL2_Main_Control_Destroy(self._context)
            except (AttributeError, OSError, ValueError):
                pass
        self._library = None
        self._context = ctypes.c_void_p()
        self._adapter_count = 0
        self._allocator = None
        self._allocations.clear()

    @staticmethod
    def _configure(library) -> None:
        library.ADL2_Main_Control_Create.restype = ctypes.c_int
        library.ADL2_Main_Control_Destroy.restype = ctypes.c_int
        library.ADL2_Adapter_NumberOfAdapters_Get.restype = ctypes.c_int
        library.ADL2_New_QueryPMLogData_Get.restype = ctypes.c_int


def _load_amd_adl_library():
    if platform.system() != "Windows":
        return None
    name = "atiadlxx.dll" if ctypes.sizeof(ctypes.c_void_p) == 8 else "atiadlxy.dll"
    try:
        return ctypes.CDLL(name, winmode=0x00000800)
    except OSError:
        return None


def discover_amd_adl_power(*, library_factory=None) -> PowerAvailability | None:
    factory = library_factory or _load_amd_adl_library
    try:
        library = factory()
    except OSError:
        return None
    if library is None:
        return None
    source = AmdAdlPowerSource(
        PowerAvailability(True, "amd-adl", "accelerator", location="atiadlxx.dll"),
        library_factory=lambda: library,
    )
    source.start()
    try:
        reading = source.read_watts()
    finally:
        source.stop()
    if reading is None:
        return PowerAvailability(False, "amd-adl", "accelerator",
                                 "AMD driver power counters are unreadable")
    return PowerAvailability(True, "amd-adl", "accelerator", location="atiadlxx.dll")


def parse_rapl_energy_uj(output: str) -> float | None:
    value = _finite_nonnegative(output.strip())
    return value / 1_000_000 if value is not None else None


def integrate_power_joules(samples: Sequence[tuple[float, float | None]]) -> float | None:
    energy = 0.0
    intervals = 0
    for (before_time, before_watts), (after_time, after_watts) in zip(samples, samples[1:]):
        if before_watts is None or after_watts is None or after_time <= before_time:
            continue
        if _finite_nonnegative(before_watts) is None or _finite_nonnegative(after_watts) is None:
            continue
        energy += (before_watts + after_watts) / 2 * (after_time - before_time)
        intervals += 1
    return energy if intervals else None


def efficiency_per_joule(work_count: float | int | None,
                          energy_joules: float | None) -> float | None:
    work = _finite_nonnegative(work_count)
    energy = _finite_nonnegative(energy_joules)
    if work is None or energy is None or work == 0 or energy == 0:
        return None
    return work / energy


def add_power_efficiency(power: dict[str, Any] | None, unit: str,
                         work_count: float | int | None) -> dict[str, Any] | None:
    if power is None:
        return None
    result = dict(power)
    result["efficiency"] = {
        "unit": unit,
        "work_count": work_count,
        "per_joule": efficiency_per_joule(work_count, power.get("energy_joules")),
    }
    return result


def discover_power_source(platform_name: str | None = None, *, which_fn=shutil.which,
                          run_fn=subprocess.run,
                          rapl_paths: Sequence[Path] | None = None,
                          is_file_fn: Callable[[str], bool] | None = None,
                          adl_discovery_fn=discover_amd_adl_power) -> PowerAvailability:
    platform_name = platform_name or platform.system()
    if platform_name == "Darwin":
        executable = which_fn("powermetrics") or "/usr/bin/powermetrics"
        exists = is_file_fn or (lambda path: Path(path).is_file())
        if not exists(executable):
            return PowerAvailability(False, "powermetrics", "processor_package",
                                     "powermetrics is not installed")
        sudo = which_fn("sudo") or "/usr/bin/sudo"
        try:
            result = run_fn([sudo, "-n", "-v"], capture_output=True, text=True,
                            timeout=2, check=False)
        except (OSError, subprocess.SubprocessError):
            return PowerAvailability(False, "powermetrics", "processor_package",
                                     "power permission check failed")
        if result.returncode:
            return PowerAvailability(
                False, "powermetrics", "processor_package",
                "administrator permission is not active; run sudo -v before the benchmark",
            )
        return PowerAvailability(True, "powermetrics", "processor_package",
                                 location=executable)
    failed_sources = []
    if executable := which_fn("nvidia-smi"):
        try:
            result = run_fn(
                [executable, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and parse_nvidia_power(result.stdout):
            return PowerAvailability(True, "nvidia-smi", "accelerator", location=executable)
        failed_sources.append(PowerAvailability(
            False, "nvidia-smi", "accelerator", "GPU power counters are unreadable",
        ))
    if platform_name == "Windows" and (adl_status := adl_discovery_fn()) is not None:
        if adl_status.available:
            return adl_status
        failed_sources.append(adl_status)
    if executable := which_fn("rocm-smi"):
        try:
            result = run_fn(
                [executable, "--showpower", "--json"], capture_output=True, text=True,
                timeout=2, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and parse_rocm_power(result.stdout):
            return PowerAvailability(True, "rocm-smi", "accelerator", location=executable)
        failed_sources.append(PowerAvailability(
            False, "rocm-smi", "accelerator", "GPU power counters are unreadable",
        ))
    paths = tuple(
        rapl_paths if rapl_paths is not None
        else Path("/sys/class/powercap").glob("intel-rapl*/energy_uj")
    )
    readable = next((path for path in paths if os.access(path, os.R_OK)), None)
    if readable:
        return PowerAvailability(True, "rapl", "cpu_package", location=str(readable))
    if paths:
        return PowerAvailability(False, "rapl", "cpu_package",
                                 "RAPL energy counter permission is denied")
    if failed_sources:
        return failed_sources[-1]
    return PowerAvailability(False, "unsupported", "unknown",
                             "no supported power source was detected")


def query_power_reading(availability: PowerAvailability,
                        *, run_fn=subprocess.run) -> PowerReading | None:
    if not availability.available or not availability.location:
        return None
    try:
        if availability.source == "nvidia-smi":
            result = run_fn(
                [availability.location, "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            return parse_nvidia_power(result.stdout) if result.returncode == 0 else None
        if availability.source == "rocm-smi":
            result = run_fn(
                [availability.location, "--showpower", "--json"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            return parse_rocm_power(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


class PowerSourceLike(Protocol):
    def start(self) -> None: ...
    def read_watts(self) -> float | None: ...
    def stop(self) -> None: ...


class PollingPowerSource:
    def __init__(self, availability: PowerAvailability, *, run_fn=subprocess.run,
                 monotonic=time.monotonic,
                 read_text_fn: Callable[[str], str] | None = None):
        self.availability = availability
        self.run_fn = run_fn
        self.monotonic = monotonic
        self.read_text_fn = read_text_fn or (
            lambda path: Path(path).read_text(encoding="utf-8")
        )
        self._last_rapl: tuple[float, float] | None = None

    def start(self) -> None:
        self._last_rapl = None

    def read_watts(self) -> float | None:
        if self.availability.source != "rapl":
            reading = query_power_reading(self.availability, run_fn=self.run_fn)
            return reading.watts if reading else None
        if not self.availability.available or not self.availability.location:
            return None
        try:
            joules = parse_rapl_energy_uj(self.read_text_fn(self.availability.location))
        except OSError:
            return None
        now = self.monotonic()
        previous = self._last_rapl
        if joules is None:
            return None
        self._last_rapl = (now, joules)
        if previous is None or now <= previous[0] or joules < previous[1]:
            return None
        return (joules - previous[1]) / (now - previous[0])

    def stop(self) -> None:
        pass


class PowermetricsPowerSource:
    def __init__(self, availability: PowerAvailability, interval_sec: float, *,
                 popen_fn: Callable[..., Any] = subprocess.Popen,
                 monotonic=time.monotonic):
        self.availability = availability
        self.interval_sec = interval_sec
        self.popen_fn = popen_fn
        self.monotonic = monotonic
        self._process: Any = None
        self._reader: threading.Thread | None = None
        self._latest: float | None = None
        self._latest_at: float | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.availability.available or not self.availability.location:
            return
        command = [
            shutil.which("sudo") or "/usr/bin/sudo", "-n", self.availability.location,
            "--samplers", "cpu_power,gpu_power,ane_power", "--sample-rate",
            str(max(100, round(self.interval_sec * 1000))), "--sample-count", "-1",
            "--buffer-size", "1",
        ]
        try:
            self._process = self.popen_fn(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except OSError:
            self._process = None
            return
        self._reader = threading.Thread(
            target=self._read_output, name="powermetrics-reader", daemon=True,
        )
        self._reader.start()

    def read_watts(self) -> float | None:
        with self._lock:
            if (self._latest_at is None
                    or self.monotonic() - self._latest_at > max(2.0, self.interval_sec * 3)):
                return None
            poll = getattr(self._process, "poll", None)
            if callable(poll) and poll() is not None:
                return None
            return self._latest

    def stop(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            try:
                self._process.kill()
            except OSError:
                pass
        if self._reader:
            self._reader.join(timeout=2)

    def _read_output(self) -> None:
        stdout = getattr(self._process, "stdout", None)
        if stdout is None:
            return
        for line in stdout:
            reading = parse_powermetrics_power(line)
            if reading:
                with self._lock:
                    self._latest = reading.watts
                    self._latest_at = self.monotonic()


def create_power_source(availability: PowerAvailability, interval_sec: float,
                        **kwargs) -> PowerSourceLike:
    if availability.source == "powermetrics":
        return PowermetricsPowerSource(availability, interval_sec, **kwargs)
    if availability.source == "amd-adl":
        return AmdAdlPowerSource(availability, **kwargs)
    return PollingPowerSource(availability, **kwargs)


def power_block(samples: Sequence[TelemetrySample], interval_sec: float,
                availability: PowerAvailability, failed_samples: int) -> dict[str, Any]:
    windows = []
    idle_values = []
    for name in dict.fromkeys(sample.window for sample in samples):
        selected = [sample for sample in samples if sample.window == name]
        values = [value for sample in selected
                  if (value := _finite_nonnegative(sample.power_watts)) is not None]
        energy = integrate_power_joules([
            (sample.timestamp_sec, sample.power_watts) for sample in selected
        ])
        if name == "idle":
            idle_values.extend(values)
        windows.append({
            "name": name,
            "sample_count": len(selected),
            "duration_sec": (
                max(0.0, selected[-1].timestamp_sec - selected[0].timestamp_sec)
                if selected else 0.0
            ),
            "mean_watts": sum(values) / len(values) if values else None,
            "peak_watts": max(values) if values else None,
            "energy_joules": energy,
            "samples": [
                {"timestamp_sec": sample.timestamp_sec, "watts": sample.power_watts}
                for sample in selected
            ],
        })
    all_values = [value for sample in samples
                  if (value := _finite_nonnegative(sample.power_watts)) is not None]
    measured_samples = [sample for sample in samples if sample.window.startswith("measured")]
    measured_energy = integrate_power_joules([
        (sample.timestamp_sec, sample.power_watts) for sample in measured_samples
    ])
    complete_timeline = all(
        _finite_nonnegative(sample.power_watts) is not None for sample in measured_samples
    )
    recorded = measured_energy is not None and complete_timeline
    return {
        "status": "recorded" if recorded else "unavailable",
        "reason": None if recorded else (
            availability.reason or "insufficient valid samples for energy integration"
        ),
        "source": availability.source,
        "scope": availability.scope,
        "energy_joules": measured_energy if recorded else None,
        "mean_watts": sum(all_values) / len(all_values) if all_values else None,
        "peak_watts": max(all_values) if all_values else None,
        "idle_baseline_watts": (
            sum(idle_values) / len(idle_values) if idle_values else None
        ),
        "windows": windows,
        "provenance": {
            "interval_sec": interval_sec,
            "failed_samples": failed_samples,
        },
    }


def process_resource_usage(pid: int, psutil_module: PsutilLike = psutil) -> tuple[float, float] | None:
    try:
        parent = psutil_module.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
        cpu = sum(item.cpu_percent(interval=None) for item in processes)
        memory_gb = sum(item.memory_info().rss for item in processes) / (1024 ** 3)
        return cpu, memory_gb
    except (psutil_module.Error, OSError):
        return None


def system_memory_usage(psutil_module: PsutilLike = psutil) -> tuple[float, float]:
    memory = psutil_module.virtual_memory()
    return memory.used / (1024 ** 3), memory.total / (1024 ** 3)


def parse_gpu_usage(platform_name: str, output: str) -> float | None:
    if platform_name == "Darwin":
        values = re.findall(r'"Device Utilization %"\s*=\s*([0-9.]+)', output)
    elif "GPU use (%)" in output:
        values = re.findall(r'"?GPU use \(%\)"?\s*[:=]\s*"?([0-9.]+)', output)
    else:
        values = re.findall(r"(?m)^\s*([0-9.]+)\s*%?\s*$", output)
    percentages = [float(value) for value in values if 0 <= float(value) <= 100]
    return max(percentages) if percentages else None


def query_gpu_usage(platform_name: str | None = None, run_fn=subprocess.run,
                    which_fn=shutil.which) -> float | None:
    platform_name = platform_name or platform.system()
    if platform_name == "Darwin":
        executable = which_fn("ioreg") or "/usr/sbin/ioreg"
        command = [executable, "-r", "-d", "1", "-c", "AGXAccelerator"]
    elif executable := which_fn("nvidia-smi"):
        command = [executable, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    elif executable := which_fn("rocm-smi"):
        command = [executable, "--showuse", "--json"]
    else:
        return None
    try:
        result = run_fn(command, capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_gpu_usage(platform_name, result.stdout) if result.returncode == 0 else None


def parse_gpu_process_memory(output: str, process_ids: set[int]) -> float | None:
    used_mib = 0.0
    found = False
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            memory = float(re.sub(r"\s*MiB$", "", parts[1], flags=re.IGNORECASE))
        except ValueError:
            continue
        if pid in process_ids:
            used_mib += memory
            found = True
    return used_mib / 1024 if found else None


def query_gpu_process_memory(pid: int, run_fn=subprocess.run, which_fn=shutil.which,
                             psutil_module: PsutilLike = psutil) -> float | None:
    executable = which_fn("nvidia-smi")
    if not executable:
        return None
    try:
        parent = psutil_module.Process(pid)
        process_ids = {parent.pid, *(child.pid for child in parent.children(recursive=True))}
        result = run_fn(
            [executable, "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (psutil_module.Error, OSError, subprocess.SubprocessError):
        return None
    return parse_gpu_process_memory(result.stdout, process_ids) if result.returncode == 0 else None


def query_vram_usage() -> tuple[float, float] | None:
    snapshot = Shared.sample_memory_gb()
    used = snapshot["gpu_vram_used_gb"]
    total = snapshot["gpu_vram_total_gb"]
    return (used, total) if used is not None and total is not None else None


def query_sampler_vram_usage(run_fn=subprocess.run, which_fn=shutil.which) -> tuple[float, float] | None:
    if executable := which_fn("nvidia-smi"):
        command = [executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
        source = "nvidia"
    elif executable := which_fn("rocm-smi"):
        command = [executable, "--showmeminfo", "vram", "--json"]
        source = "rocm"
    else:
        return None
    try:
        result = run_fn(command, capture_output=True, text=True, timeout=2, check=False)
        if result.returncode:
            return None
        if source == "nvidia":
            pairs = [line.split(",") for line in result.stdout.splitlines() if "," in line]
            used = sum(float(pair[0].strip()) for pair in pairs) / 1024
            total = sum(float(pair[1].strip()) for pair in pairs) / 1024
        else:
            payload = json.loads(result.stdout)
            used = sum(float(card.get("VRAM Total Used Memory (B)", 0)) for card in payload.values()) / (1024 ** 3)
            total = sum(float(card.get("VRAM Total Memory (B)", 0)) for card in payload.values()) / (1024 ** 3)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError):
        return None
    return (used, total) if total > 0 else None


def summarize_samples(name: str, samples: Sequence[TelemetrySample]) -> WindowSummary:
    channels = {}
    for channel in MEMORY_CHANNELS:
        values = [value for sample in samples if (value := getattr(sample, channel)) is not None]
        channels[channel] = ChannelSummary(
            peak_gb=max(values) if values else None,
            mean_gb=sum(values) / len(values) if values else None,
            final_gb=values[-1] if values else None,
            valid_samples=len(values),
        )
    duration = max(0.0, samples[-1].timestamp_sec - samples[0].timestamp_sec) if samples else 0.0
    return WindowSummary(name, len(samples), duration, channels)


def summarize_windows(samples: Sequence[TelemetrySample]) -> list[WindowSummary]:
    names = list(dict.fromkeys(sample.window for sample in samples))
    return [summarize_samples(name, [sample for sample in samples if sample.window == name])
            for name in names]


def summarize_case(windows: Sequence[WindowSummary]) -> Mapping[str, ChannelSummary]:
    result = {}
    for channel in MEMORY_CHANNELS:
        summaries = [window.channels[channel] for window in windows]
        valid = sum(summary.valid_samples for summary in summaries)
        weighted_total = sum(
            (summary.mean_gb or 0.0) * summary.valid_samples for summary in summaries
        )
        peaks = [summary.peak_gb for summary in summaries if summary.peak_gb is not None]
        finals = [summary.final_gb for summary in summaries if summary.final_gb is not None]
        result[channel] = ChannelSummary(
            peak_gb=max(peaks) if peaks else None,
            mean_gb=weighted_total / valid if valid else None,
            final_gb=finals[-1] if finals else None,
            valid_samples=valid,
        )
    return result


def calculate_headroom(peak_used_gb: float | None, ceiling_gb: float | None) -> Headroom:
    if peak_used_gb is None or ceiling_gb is None or ceiling_gb <= 0:
        return Headroom(None, None, "unknown")
    absolute = ceiling_gb - peak_used_gb
    fraction = absolute / ceiling_gb
    if absolute < 0:
        state = "exceeded"
    elif fraction < config.MEMORY_HEADROOM_COMFORTABLE_FRACTION:
        state = "tight"
    else:
        state = "comfortable"
    return Headroom(absolute, fraction, state)


def sample_dict(sample: TelemetrySample) -> dict[str, Any]:
    return {
        "timestamp_sec": sample.timestamp_sec,
        **{channel: getattr(sample, channel) for channel in MEMORY_CHANNELS},
    }


def window_summary_dict(summary: WindowSummary,
                        samples: Sequence[TelemetrySample]) -> dict[str, Any]:
    return {
        "name": summary.name,
        "sample_count": summary.sample_count,
        "duration_sec": summary.duration_sec,
        "channels": {name: asdict(channel) for name, channel in summary.channels.items()},
        "samples": [sample_dict(sample) for sample in samples],
    }


def memory_block(samples: Sequence[TelemetrySample], interval_sec: float,
                 failed_samples: int, channel_failures: Mapping[str, int],
                 sources: Mapping[str, str],
                 ceiling_gb: float | None = None) -> dict[str, Any]:
    windows = summarize_windows(samples)
    case = summarize_case(windows)
    basis_channel = "accelerator_memory_used_gb"
    if sources.get(basis_channel, "unsupported") == "unsupported":
        basis_channel = "process_rss_gb"
    peak = case[basis_channel].peak_gb
    headroom = asdict(calculate_headroom(peak, ceiling_gb))
    headroom["basis_channel"] = basis_channel
    return {
        "windows": [
            window_summary_dict(
                window, [sample for sample in samples if sample.window == window.name],
            )
            for window in windows
        ],
        "summary": {name: asdict(summary) for name, summary in case.items()},
        "headroom": headroom,
        "provenance": {
            "interval_sec": interval_sec,
            "failed_samples": failed_samples,
            "channels": {name: {
                "source": sources.get(name, "unsupported"),
                "failed_samples": channel_failures.get(name, 0),
            }
                         for name in MEMORY_CHANNELS},
        },
    }


def default_memory_sources(which_fn=shutil.which) -> dict[str, str]:
    accelerator = "unsupported"
    if which_fn("nvidia-smi"):
        accelerator = "nvidia-smi"
    elif which_fn("rocm-smi"):
        accelerator = "rocm-smi"
    return {
        "host_ram_used_gb": "psutil",
        "process_rss_gb": "psutil",
        "accelerator_memory_used_gb": accelerator,
        "accelerator_memory_total_gb": accelerator,
    }


def memory_ceiling_gb(sources: Mapping[str, str], run_fn=subprocess.run,
                      psutil_module: PsutilLike = psutil,
                      which_fn=shutil.which) -> float | None:
    accelerator = sources.get("accelerator_memory_total_gb")
    if accelerator == "nvidia-smi":
        executable = which_fn("nvidia-smi")
        if not executable:
            return None
        try:
            result = run_fn(
                [executable, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            totals = [float(line.strip()) / 1024 for line in result.stdout.splitlines()
                      if line.strip()]
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        if result.returncode or not totals:
            return None
        return sum(max(0.0, total - hardware.VRAM_RESERVE_GB) for total in totals)
    if accelerator == "rocm-smi":
        snapshot = query_vram_usage()
        return (snapshot[1] - hardware.VRAM_RESERVE_GB) if snapshot else None
    try:
        total = psutil_module.virtual_memory().total / (1024 ** 3)
    except (psutil_module.Error, OSError):
        return None
    return max(0.0, total - hardware.RAM_RESERVE_GB)


def derive_run_memory_summary(sections: Mapping[str, object]) -> dict[str, Any] | None:
    channels: dict[str, dict[str, Any]] = {}
    tightest = None

    def visit(value: object, path: tuple[str, ...]) -> None:
        nonlocal tightest
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
            return
        if not isinstance(value, dict):
            return
        memory = value.get("memory")
        if isinstance(memory, dict):
            summary = memory.get("summary", {})
            if isinstance(summary, dict):
                for channel, values in summary.items():
                    peak = values.get("peak_gb") if isinstance(values, dict) else None
                    if not isinstance(peak, (int, float)):
                        continue
                    current = channels.setdefault(channel, {"peak_gb": peak})
                    current["peak_gb"] = max(current["peak_gb"], peak)
            headroom = memory.get("headroom", {})
            absolute = headroom.get("absolute_gb") if isinstance(headroom, dict) else None
            if isinstance(absolute, (int, float)) and (
                    tightest is None or absolute < tightest["absolute_gb"]):
                tightest = {
                    "absolute_gb": absolute,
                    "fraction": headroom.get("fraction"),
                    "state": headroom.get("state"),
                    "case_id": memory.get("case_id"),
                    "case_path": "/".join(path),
                }
                if isinstance(headroom.get("basis_channel"), str):
                    tightest["basis_channel"] = headroom["basis_channel"]
        for key, child in value.items():
            if key != "memory":
                visit(child, (*path, str(key)))

    visit(dict(sections), ())
    if not channels and tightest is None:
        return None
    return {"channels": channels, "tightest_headroom": tightest}


def derive_run_power_summary(sections: Mapping[str, object]) -> dict[str, Any] | None:
    cases = []

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
            return
        if not isinstance(value, dict):
            return
        power = value.get("power")
        if isinstance(power, dict):
            cases.append((power, "/".join(path)))
        for key, child in value.items():
            if key != "power":
                visit(child, (*path, str(key)))

    visit(dict(sections), ())
    if not cases:
        return None
    scopes = sorted({power.get("scope") for power, _ in cases
                     if isinstance(power.get("scope"), str)})
    sources = sorted({power.get("source") for power, _ in cases
                      if isinstance(power.get("source"), str)})
    energies = [value for power, _ in cases
                if (value := _finite_nonnegative(power.get("energy_joules"))) is not None]
    idle = [value for power, _ in cases
            if (value := _finite_nonnegative(power.get("idle_baseline_watts"))) is not None]
    return {
        "status": "recorded" if energies else "unavailable",
        "reason": None if energies else next(
            (power.get("reason") for power, _ in cases if isinstance(power.get("reason"), str)),
            "no case recorded usable power samples",
        ),
        "scope": scopes[0] if len(scopes) == 1 else "mixed",
        "source": sources[0] if len(sources) == 1 else "mixed",
        "energy_joules": sum(energies) if energies and len(scopes) == 1 else None,
        "idle_baseline_watts": sum(idle) / len(idle) if idle else None,
        "recorded_cases": len(energies),
        "total_cases": len(cases),
    }


class TelemetrySampler:
    def __init__(self, pid: int, interval_sec: float = config.TELEMETRY_INTERVAL_SEC,
                 sample_fn: Callable[[float, str], TelemetrySample] | None = None,
                 power_source: PowerSourceLike | None = None):
        if interval_sec <= 0:
            raise ValueError("telemetry interval must be positive")
        self.pid = pid
        self.interval_sec = interval_sec
        self._sample_fn = sample_fn or self._sample
        self.power_source = power_source
        self._samples: list[TelemetrySample] = []
        self._failed_samples = 0
        self._channel_failures = {channel: 0 for channel in MEMORY_CHANNELS}
        self._power_failures = 0
        self._window = "idle"
        self._started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()

    @property
    def samples(self) -> tuple[TelemetrySample, ...]:
        with self._lock:
            return tuple(self._samples)

    @property
    def failed_samples(self) -> int:
        with self._lock:
            return self._failed_samples

    @property
    def channel_failures(self) -> Mapping[str, int]:
        with self._lock:
            return dict(self._channel_failures)

    @property
    def power_failures(self) -> int:
        with self._lock:
            return self._power_failures

    def set_window(self, name: str) -> None:
        if not name:
            raise ValueError("telemetry window must not be empty")
        with self._lock:
            self._window = name

    def mark_window(self, name: str) -> None:
        self.set_window(name)
        self.capture()

    def capture(self) -> TelemetrySample:
        with self._capture_lock:
            timestamp = time.monotonic() - self._started_at
            with self._lock:
                window = self._window
            sample = self._capture(timestamp, window)
            with self._lock:
                self._samples.append(sample)
            return sample

    def start(self) -> "TelemetrySampler":
        if self._thread and self._thread.is_alive():
            raise RuntimeError("telemetry sampler is already running")
        self._stop_event.clear()
        self._started_at = time.monotonic()
        if self.power_source:
            self.power_source.start()
        self._thread = threading.Thread(target=self._run, name="resource-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> tuple[TelemetrySample, ...]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(3.0, self.interval_sec * 2))
        if self.power_source:
            self.power_source.stop()
        return self.samples

    def __enter__(self) -> "TelemetrySampler":
        return self.start()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()

    def _sample(self, timestamp_sec: float, window: str) -> TelemetrySample:
        host_used, _ = system_memory_usage()
        process = process_resource_usage(self.pid)
        vram = query_sampler_vram_usage()
        return TelemetrySample(
            timestamp_sec=timestamp_sec,
            window=window,
            host_ram_used_gb=host_used,
            process_rss_gb=process[1] if process else None,
            accelerator_memory_used_gb=vram[0] if vram else None,
            accelerator_memory_total_gb=vram[1] if vram else None,
            power_watts=self.power_source.read_watts() if self.power_source else None,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.capture()
            self._stop_event.wait(self.interval_sec)

    def _capture(self, timestamp: float, window: str) -> TelemetrySample:
        try:
            sample = self._sample_fn(timestamp, window)
        except Exception:
            with self._lock:
                self._failed_samples += 1
                for channel in MEMORY_CHANNELS:
                    self._channel_failures[channel] += 1
                if self.power_source:
                    self._power_failures += 1
            return TelemetrySample(timestamp, window)
        with self._lock:
            for channel in MEMORY_CHANNELS:
                if getattr(sample, channel) is None:
                    self._channel_failures[channel] += 1
            if self.power_source and sample.power_watts is None:
                self._power_failures += 1
        return sample


class CaseTelemetry:
    def __init__(self, pid: int | None = None, interval_sec: float = config.TELEMETRY_INTERVAL_SEC,
                 sampler: TelemetrySampler | None = None, sources: Mapping[str, str] | None = None,
                 power_availability: PowerAvailability | None = None):
        self.power_availability = power_availability
        power_source = (
            create_power_source(power_availability, interval_sec) if power_availability else None
        )
        self.sampler = sampler or TelemetrySampler(
            pid or os.getpid(), interval_sec, power_source=power_source,
        )
        self.sources = dict(sources or default_memory_sources())
        self.ceiling_gb = memory_ceiling_gb(self.sources)
        self._cursor = 0
        self._failed_cursor = 0
        self._channel_failure_cursor = {channel: 0 for channel in MEMORY_CHANNELS}
        self._power_failure_cursor = 0
        self.last_power: dict[str, Any] | None = None

    def start(self) -> "CaseTelemetry":
        self.sampler.start()
        self.sampler.mark_window("idle")
        self._cursor = 0
        self._failed_cursor = 0
        self._channel_failure_cursor = {channel: 0 for channel in MEMORY_CHANNELS}
        self._power_failure_cursor = 0
        self.last_power = None
        return self

    def stop(self) -> None:
        self.sampler.stop()

    def begin_model_load(self) -> None:
        self.sampler.mark_window("model_load")

    def begin_measured(self, subwindow: str = "measured") -> None:
        self.sampler.mark_window(subwindow)

    def finish_case(self, ceiling_gb: float | None = None) -> dict[str, Any]:
        self.sampler.capture()
        all_samples = self.sampler.samples
        samples = all_samples[self._cursor:]
        self._cursor = len(all_samples)
        failed_samples = self.sampler.failed_samples - self._failed_cursor
        self._failed_cursor = self.sampler.failed_samples
        current_failures = self.sampler.channel_failures
        case_failures = {
            channel: current_failures[channel] - self._channel_failure_cursor[channel]
            for channel in MEMORY_CHANNELS
        }
        self._channel_failure_cursor = dict(current_failures)
        power_failures = self.sampler.power_failures - self._power_failure_cursor
        self._power_failure_cursor = self.sampler.power_failures
        block = memory_block(
            samples, self.sampler.interval_sec, failed_samples,
            {
                channel: (case_failures[channel]
                          if self.sources.get(channel) != "unsupported" else 0)
                for channel in MEMORY_CHANNELS
            },
            self.sources,
            self.ceiling_gb if ceiling_gb is None else ceiling_gb,
        )
        self.last_power = (
            power_block(
                samples, self.sampler.interval_sec, self.power_availability, power_failures,
            )
            if self.power_availability else None
        )
        self.sampler.mark_window("idle")
        return block
