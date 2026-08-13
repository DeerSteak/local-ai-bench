"""Shared resource queries, sampling, and memory aggregation."""

from dataclasses import asdict, dataclass
import platform
import json
import os
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
    peak = case["accelerator_memory_used_gb"].peak_gb
    if peak is None:
        peak = case["host_ram_used_gb"].peak_gb
    return {
        "windows": [
            window_summary_dict(
                window, [sample for sample in samples if sample.window == window.name],
            )
            for window in windows
        ],
        "summary": {name: asdict(summary) for name, summary in case.items()},
        "headroom": asdict(calculate_headroom(peak, ceiling_gb)),
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
    elif platform.system() == "Darwin":
        accelerator = "unified-memory"
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
        for key, child in value.items():
            if key != "memory":
                visit(child, (*path, str(key)))

    visit(dict(sections), ())
    if not channels and tightest is None:
        return None
    return {"channels": channels, "tightest_headroom": tightest}


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
        self._channel_failures = {channel: 0 for channel in MEMORY_CHANNELS}
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

    @property
    def channel_failures(self) -> Mapping[str, int]:
        with self._lock:
            return dict(self._channel_failures)

    def set_window(self, name: str) -> None:
        if not name:
            raise ValueError("telemetry window must not be empty")
        with self._lock:
            self._window = name

    def mark_window(self, name: str) -> None:
        self.set_window(name)
        self.capture()

    def capture(self) -> TelemetrySample:
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
        self._thread = threading.Thread(target=self._run, name="memory-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> tuple[TelemetrySample, ...]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(3.0, self.interval_sec * 2))
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
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            timestamp = time.monotonic() - self._started_at
            with self._lock:
                window = self._window
            sample = self._capture(timestamp, window)
            with self._lock:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_sec)

    def _capture(self, timestamp: float, window: str) -> TelemetrySample:
        try:
            sample = self._sample_fn(timestamp, window)
        except Exception:
            with self._lock:
                self._failed_samples += 1
                for channel in MEMORY_CHANNELS:
                    self._channel_failures[channel] += 1
            return TelemetrySample(timestamp, window)
        with self._lock:
            for channel in MEMORY_CHANNELS:
                if getattr(sample, channel) is None:
                    self._channel_failures[channel] += 1
        return sample


class CaseTelemetry:
    def __init__(self, pid: int | None = None, interval_sec: float = config.TELEMETRY_INTERVAL_SEC,
                 sampler: TelemetrySampler | None = None, sources: Mapping[str, str] | None = None):
        self.sampler = sampler or TelemetrySampler(pid or os.getpid(), interval_sec)
        self.sources = dict(sources or default_memory_sources())
        self.ceiling_gb = memory_ceiling_gb(self.sources)
        self._cursor = 0
        self._failed_cursor = 0
        self._channel_failure_cursor = {channel: 0 for channel in MEMORY_CHANNELS}

    def start(self) -> "CaseTelemetry":
        self.sampler.start()
        self.sampler.mark_window("idle")
        self._cursor = 0
        self._failed_cursor = 0
        self._channel_failure_cursor = {channel: 0 for channel in MEMORY_CHANNELS}
        return self

    def stop(self) -> None:
        self.sampler.stop()

    def begin_model_load(self) -> None:
        self.sampler.mark_window("model_load")

    def begin_measured(self, subwindow: str = "measured") -> None:
        self.sampler.mark_window(subwindow)

    def finish_case(self, ceiling_gb: float | None = None) -> dict[str, Any]:
        self.sampler.capture()
        samples = self.sampler.samples[self._cursor:]
        self._cursor = len(self.sampler.samples)
        failed_samples = self.sampler.failed_samples - self._failed_cursor
        self._failed_cursor = self.sampler.failed_samples
        current_failures = self.sampler.channel_failures
        case_failures = {
            channel: current_failures[channel] - self._channel_failure_cursor[channel]
            for channel in MEMORY_CHANNELS
        }
        self._channel_failure_cursor = dict(current_failures)
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
        self.sampler.mark_window("idle")
        return block
