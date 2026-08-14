"""Resource sampling and presentation for benchmark GUI progress."""

from scripts.runtime import hardware
from scripts.runtime.telemetry import (
    PsutilLike, parse_gpu_process_memory, parse_gpu_usage, process_resource_usage,
    query_gpu_process_memory, query_gpu_usage, query_vram_usage, system_memory_usage,
)


def show_vram_usage(devices: list[dict]) -> bool:
    return any(
        device.get("vram_gb") is not None
        and (device.get("vendor") == "nvidia"
             or hardware.classify_gpu(str(device.get("name", ""))) == "discrete")
        for device in devices
    )


def resource_usage_rows(process_usage, system_usage, baseline_system_used: float,
                        gpu_usage: float | None, gpu_memory: float | None,
                        vram_usage: tuple[float, float] | None = None,
                        include_vram: bool = False) -> dict[str, str]:
    rows = {
        "CPU": "Unavailable" if process_usage is None else f"{process_usage[0]:.0f}%",
        "Process RAM": "Unavailable" if process_usage is None else f"{process_usage[1]:.1f} GB",
        "System RAM": "Unavailable",
        "GPU": "Unavailable" if gpu_usage is None else f"{gpu_usage:.0f}% utilization",
    }
    if system_usage is not None:
        delta = system_usage[0] - baseline_system_used
        rows["System RAM"] = (
            f"{system_usage[0]:.1f} / {system_usage[1]:.1f} GB (Δ {delta:+.1f} GB)"
        )
    if gpu_memory is not None:
        rows["GPU"] += f" · {gpu_memory:.1f} GB process memory"
    if include_vram:
        rows["VRAM"] = (
            "Unavailable" if vram_usage is None
            else f"{vram_usage[0]:.1f} / {vram_usage[1]:.1f} GB used"
        )
    return rows
