"""Shared resource queries, sampling, and memory aggregation."""

from dataclasses import asdict, dataclass
import platform
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import psutil

from scripts.runtime import config
from scripts.runtime.shared import Shared


MEMORY_CHANNELS = (
    "host_ram_used_gb", "process_rss_gb",
    "accelerator_memory_used_gb", "accelerator_memory_total_gb",
)


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


def window_summary_dict(summary: WindowSummary) -> dict[str, Any]:
    return {
        "name": summary.name,
        "sample_count": summary.sample_count,
        "duration_sec": summary.duration_sec,
        "channels": {name: asdict(channel) for name, channel in summary.channels.items()},
    }


class TelemetrySampler:
    def __init__(self, pid: int, interval_sec: float = config.TELEMETRY_INTERVAL_SEC,
                 sample_fn: Callable[[float, str], TelemetrySample] | None = None):
        if interval_sec <= 0:
            raise ValueError("telemetry interval must be positive")
        self.pid = pid
        self.interval_sec = interval_sec
        self._sample_fn = sample_fn or self._sample
        self._samples: list[TelemetrySample] = []
        self._failed_samples = 0
        self._window = "idle"
        self._started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def samples(self) -> tuple[TelemetrySample, ...]:
        with self._lock:
            return tuple(self._samples)

    @property
    def failed_samples(self) -> int:
        with self._lock:
            return self._failed_samples

    def set_window(self, name: str) -> None:
        if not name:
            raise ValueError("telemetry window must not be empty")
        with self._lock:
            self._window = name

    def start(self) -> "TelemetrySampler":
        if self._thread and self._thread.is_alive():
            raise RuntimeError("telemetry sampler is already running")
        self._stop_event.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="memory-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> tuple[TelemetrySample, ...]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2))
        return self.samples

    def __enter__(self) -> "TelemetrySampler":
        return self.start()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()

    def _sample(self, timestamp_sec: float, window: str) -> TelemetrySample:
        host_used, _ = system_memory_usage()
        process = process_resource_usage(self.pid)
        vram = query_vram_usage()
        return TelemetrySample(
            timestamp_sec=timestamp_sec,
            window=window,
            host_ram_used_gb=host_used,
            process_rss_gb=process[1] if process else None,
            accelerator_memory_used_gb=vram[0] if vram else None,
            accelerator_memory_total_gb=vram[1] if vram else None,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            timestamp = time.monotonic() - self._started_at
            with self._lock:
                window = self._window
            try:
                sample = self._sample_fn(timestamp, window)
            except Exception:
                sample = TelemetrySample(timestamp, window)
                with self._lock:
                    self._failed_samples += 1
            with self._lock:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_sec)
