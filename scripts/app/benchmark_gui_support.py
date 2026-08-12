"""Pure data, process, progress, and resource helpers for the benchmark GUI."""

"""Single-screen Tk launcher for Local AI Bench."""

import os
import json
import platform
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from scripts.runtime import config, hardware
from scripts.runtime.shared import RUN_LOG_UTC_OFFSET_ENV
import psutil
from scripts.results.acceptance_policy import evaluate_policy, load_policy
from scripts.app.benchmark_frontend import (
    merge_model_inventories, models_runnable_by,
    parse_engine_selection, format_engine_selection,
    FRONTEND_STATE_PATH,
    GUI_OPTION_DEFAULTS,
    MenuEntry,
    LLM_BACKED_TESTS,
    MAX_PROMPT_TOKEN_OPTIONS,
    MAX_PROMPT_TOKEN_TESTS,
    TEST_DEFINITIONS,
    TEST_STAGE_LABELS,
    expand_selected_tests,
    TG_TOKEN_OPTIONS,
    TG_TOKEN_TESTS,
    apply_saved_model_selection,
    apply_saved_test_selection,
    build_benchmark_command,
    build_frontend_state,
    frontend_state_from_run_plan,
    frontend_state_availability_errors,
    build_model_entries,
    build_test_entries,
    engine_incompatible_tests,
    load_frontend_state,
    model_selection_error,
    save_frontend_state,
    validate_gui_options,
)
from scripts.app.benchmark_presets import (
    build_portable_preset, compare_portable_presets, load_portable_preset,
    save_portable_preset,
)
from scripts.app.benchmark_project import (
    PROJECT_WORKFLOWS, build_project, load_project, project_frontend_state, save_project,
)
from scripts.runtime.comfyui_installation import find_comfyui_installation, normalize_comfyui_dir
from scripts.results.decision_report import load_result, report_output_paths, write_html_report, write_pdf_report
from scripts.runtime.engines import engine_names, get_engine, installed_engine_names
from scripts.runtime.llamacpp_tools import find_llamacpp_tool
from scripts.results.run_plan import load_run_plan
from scripts.results.result_bundle import export_result_bundle, import_result_bundle, verify_result_bundle
from scripts.results.result_history import (
    delete_multiple_run_artifacts, discover_results, existing_run_artifacts, filter_results,
    load_result as load_history_result,
)
from scripts.results.recovery_inspector import inspect_recovery
from scripts.results.support_bundle import export_support_bundle, preview_support_bundle
from scripts.setup.model_inventory import build_model_inventory
from scripts.app.model_import_dialog import show_model_import_dialog
from scripts.app.engine_management import collect_engine_management, vllm_update_support
from scripts.setup.runtime_update import (
    RuntimeUpdateResult, detect_nvidia_max_cuda_version, fetch_llamacpp_release,
    fetch_llamacpp_release_tag, fetch_llamacpp_releases, rebuild_managed_llamacpp,
    update_macos_llamacpp, update_managed_vllm, update_windows_llamacpp,
)
from scripts.setup.model_compatibility import ModelCompatibility, probe_llamacpp_load
from scripts.stage_registry import STAGE_ORDER
from scripts.results.outbound_metadata import outbound_metadata_preview, prepare_outbound_result
from scripts.runtime.pause_control import PAUSE_CONTROL_ENV, create_pause_control, write_pause_state
from scripts.runtime.progress_events import PROGRESS_PREFIX
from scripts.runtime.crash_cache import clear_crash_caches, crash_cache_paths
from scripts.runtime.shared import Shared
from scripts.setup.setup_config import (
    available_gpu_split_modes, configured_comfyui_dir, configured_gpu_devices,
    load_setup_config,
)
from scripts.setup.vllm_install import fetch_vllm_versions, is_dgx_spark
from scripts.app.tk_utils import mousewheel_scroll_units, refresh_tk_layout
from scripts.results.vendor_diagnostic import write_vendor_diagnostic
from scripts.app.result_actions import (
    completed_result_paths, dashboard_launcher_command, record_result_path,
    result_paths_for_log, run_log_path, selected_result_paths, write_run_logs,
)
from scripts.app.recovery_actions import (
    fork_executor_command, fork_review_report, format_recovery_inspection,
    recovery_executor_command, recovery_progress_entries, retry_executor_command,
)
from scripts.app.benchmark_gui_screens.history import build_history_screen

GPU_SPLIT_MODE_LABELS = {
    "single": "Single GPU",
    "layer": "Layer split (recommended)",
    "tensor": "Tensor parallel (experimental)",
}


def gpu_split_mode_labels(modes) -> tuple[str, ...]:
    return tuple(GPU_SPLIT_MODE_LABELS[mode] for mode in modes)


def gpu_split_mode_value(label: str) -> str:
    for mode, candidate in GPU_SPLIT_MODE_LABELS.items():
        if candidate == label:
            return mode
    raise ValueError(f"Unknown GPU mode: {label}")


def effective_gui_options(state: dict | None) -> dict:
    options = state.get("gui_options") if state else None
    return dict(options) if options is not None else dict(GUI_OPTION_DEFAULTS)


PROCESS_EXIT_DRAIN_GRACE_SECONDS = 0.25


def should_finalize_process_exit(exit_code: int | None, reader_done: bool,
                                 exit_observed_at: float | None, last_output_at: float,
                                 now: float, grace: float = PROCESS_EXIT_DRAIN_GRACE_SECONDS) -> bool:
    """Wait for reader completion or a quiet drain period after parent exit."""
    if exit_code is None or exit_observed_at is None:
        return False
    return reader_done or now - max(exit_observed_at, last_output_at) >= grace


def open_path_command(path: Path, system: str) -> list[str]:
    if system == "Darwin":
        return ["open", str(path)]
    if system == "Windows":
        return ["explorer", str(path)]
    return ["xdg-open", str(path)]


def launch_controlled_process(command: list[str], *, creationflags: int = 0,
                              pause_path_factory=create_pause_control,
                              popen=subprocess.Popen,
                              utc_offset_fn=None,
                              ) -> tuple[subprocess.Popen, Path]:
    control_path = pause_path_factory()
    child_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
        "LOCAL_AI_BENCH_PROGRESS": "1",
        PAUSE_CONTROL_ENV: str(control_path),
    }
    utc_offset = (utc_offset_fn or windows_host_utc_offset_minutes)()
    if utc_offset is not None:
        child_env[RUN_LOG_UTC_OFFSET_ENV] = str(utc_offset)
    try:
        process = popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=creationflags, env=child_env,
        )
    except BaseException:
        control_path.unlink(missing_ok=True)
        raise
    return process, control_path


def windows_host_utc_offset_minutes(*, system=platform.system, release=platform.release,
                                    run=subprocess.run) -> int | None:
    """Read the Windows host's current UTC offset when the GUI runs under WSL."""
    if not hardware.detect_wsl(system(), release()):
        return None
    try:
        result = run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "[int][TimeZoneInfo]::Local.GetUtcOffset([DateTimeOffset]::UtcNow).TotalMinutes"],
            capture_output=True, text=True, timeout=10,
        )
        minutes = int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return minutes if result.returncode == 0 and -14 * 60 <= minutes <= 14 * 60 else None


def build_discovery_report(*, platform_name: str, architecture: str, ram_gb: float,
                           backend: str, tools: dict[str, str | None],
                           comfyui_dir: Path | None,
                           inventory: dict[str, list[dict]], free_storage_gb: float | None = None) -> dict:
    counts = {key: len(inventory.get(key, [])) for key in ("llm", "custom", "embedding", "image")}
    issues = []
    if not tools.get("llama-server"):
        issues.append("llama-server was not found; LLM-backed tests cannot start.")
    if not any(counts.values()):
        issues.append("No benchmark models were found; run Setup to add models.")
    if counts["image"] and comfyui_dir is None:
        issues.append("Image models are installed, but ComfyUI was not found.")
    installed_sizes = [model.get("size") for models in inventory.values() for model in models]
    installed_gb = sum(size for size in installed_sizes if isinstance(size, (int, float))) / 1e9
    largest_gb = max((size for size in installed_sizes if isinstance(size, (int, float))), default=0) / 1e9
    memory_risk = (f"Largest installed model is {largest_gb:.1f} GB before runtime overhead; "
                   f"{ram_gb:.1f} GB system RAM detected.")
    return {
        "system": f"{platform_name} {architecture} · {ram_gb:.1f} GB RAM · {backend}",
        "models": (f"{counts['llm']} LLM, {counts['custom']} custom LLM, "
                   f"{counts['embedding']} embedding, {counts['image']} image"),
        "runtime": ", ".join(
            f"{name}: {'found' if path else 'missing'}" for name, path in tools.items()
        ),
        "comfyui": str(comfyui_dir) if comfyui_dir else "Not found",
        "storage": (f"{installed_gb:.1f} GB installed models · {free_storage_gb:.1f} GB free"
                    if free_storage_gb is not None else f"{installed_gb:.1f} GB installed models"),
        "memory_risk": memory_risk,
        "issues": issues,
    }


def parse_progress_line(line: str) -> dict | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        event = json.loads(line.removeprefix(PROGRESS_PREFIX))
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or event.get("kind") not in {
            "stage", "model", "measurement", "result"}:
        return None
    statuses = (
        {"retrying", "valid", "invalid"} if event.get("kind") == "measurement"
        else {"complete"} if event.get("kind") == "result"
        else {"running", "complete", "skipped", "failed", "interrupted"}
    )
    if event.get("status") not in statuses:
        return None
    if not isinstance(event.get("stage"), str):
        return None
    if event["kind"] in {"model", "measurement"} and not isinstance(event.get("model"), str):
        return None
    if "model_id" in event and not isinstance(event["model_id"], str):
        return None
    if event["kind"] == "result" and not isinstance(event.get("path"), str):
        return None
    if "usable" in event and not isinstance(event["usable"], bool):
        return None
    return event


def progress_model_identity(event: dict) -> str:
    return event.get("model_id", event["model"])


def history_row_height(font_line_height: int, vertical_padding: int = 10) -> int:
    return max(38, font_line_height * 2 + vertical_padding)


def update_progress_metrics(metrics: dict, event: dict) -> dict:
    updated = dict(metrics)
    if event["kind"] == "measurement":
        key = {"retrying": "retries", "valid": "valid", "invalid": "invalid"}[event["status"]]
        updated[key] += 1
    elif event["kind"] == "model" and event["status"] in {
            "complete", "skipped", "failed", "interrupted"}:
        identity = (event["stage"], progress_model_identity(event))
        finished = set(updated["finished_models"])
        finished.add(identity)
        updated["finished_models"] = finished
        if event.get("usable"):
            usable = set(updated["usable_models"])
            usable.add(identity)
            updated["usable_models"] = usable
    return updated


def progress_summary_rows(metrics: dict) -> dict[str, str]:
    total = metrics["total_models"]
    return {
        "Finished models": f"{len(metrics['finished_models'])} / {total}",
        "Usable coverage": f"{len(metrics['usable_models'])} / {total}",
        "Invalid measurements": str(metrics["invalid"]),
        "Retries": str(metrics["retries"]),
    }


def estimate_remaining_seconds(elapsed: float, completed: int, total: int) -> int | None:
    if elapsed < 0 or completed <= 0 or total <= completed:
        return 0 if total > 0 and completed >= total else None
    return round((elapsed / completed) * (total - completed))


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
    """The subset of the psutil module this file actually calls. Read-only property
    declarations, not plain attributes — psutil's real return types are immutable
    namedtuple-style objects, which Protocol's mutable-attribute matching rejects."""
    Error: Any
    def Process(self, pid: int, /) -> _PsutilProcess: ...
    def virtual_memory(self) -> _PsutilMemory: ...


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
            [executable, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (psutil_module.Error, OSError, subprocess.SubprocessError):
        return None
    return parse_gpu_process_memory(result.stdout, process_ids) if result.returncode == 0 else None


def show_vram_usage(devices: list[dict]) -> bool:
    return any(
        device.get("vram_gb") is not None
        and (device.get("vendor") == "nvidia"
             or hardware.classify_gpu(str(device.get("name", ""))) == "discrete")
        for device in devices
    )


def query_vram_usage() -> tuple[float, float] | None:
    snapshot = Shared.sample_memory_gb()
    used = snapshot["gpu_vram_used_gb"]
    total = snapshot["gpu_vram_total_gb"]
    return (used, total) if used is not None and total is not None else None


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


def workload_preflight_errors(tests: list[str], tools: dict[str, str | None],
                              comfyui_available: bool) -> list[str]:
    errors = []
    server_tests = set(tests) - {"llamabench", "llamabenchconc", "img"}
    if server_tests and not tools.get("llama-server"):
        errors.append("llama-server is required for the selected tests. Run Setup or install it on PATH.")
    if "llamabench" in tests and not tools.get("llama-bench"):
        errors.append("llama-bench is required for llama-bench throughput. Run Setup or install it on PATH.")
    if "llamabenchconc" in tests and not tools.get("llama-batched-bench"):
        errors.append("llama-batched-bench is required for llama-bench concurrency. Run Setup or install it on PATH.")
    if "img" in tests and not comfyui_available:
        errors.append("ComfyUI is required for image generation. Run Setup or choose a valid ComfyUI path.")
    return errors


def format_run_outcome(exit_code: int) -> str:
    if exit_code == 0:
        return "Benchmark completed successfully. Results are ready to review."
    return (f"Benchmark stopped with exit code {exit_code}. Checkpointed measurements from completed "
            "models remain usable; pending work was not fabricated. Automatic cleanup was requested. "
            "Review the final Run Log message, correct the reported cause, then start a new run.")


def custom_option_defaults(comfyui_dir: Path) -> dict:
    return {**GUI_OPTION_DEFAULTS, "comfyui": str(comfyui_dir)}


def default_control_values(test_entries, model_entries, engine: str, comfyui_dir: Path) -> dict:
    return {
        "tests": {entry.value: entry.checked for entry in test_entries},
        "models": {entry.value: entry.checked for entry in model_entries},
        "engine": engine,
        "max_prompt_tokens": "No cap",
        "tg_tokens": set(config.LLAMABENCH_TG),
        "options": custom_option_defaults(comfyui_dir),
    }


BENCHMARK_PRESETS = {
    "Consumer guidance": {"tests": ["llm", "conv"], "max_prompt_tokens": 32768},
    "Vendor validation": {"tests": ["llm", "conv", "llamabench", "emb", "mcq", "math", "reasoning", "code", "tool", "img"]},
    "Neutral comparison": {"tests": ["llm", "conv", "emb", "img"]},
    "Platform optimized": {"tests": ["llm", "conv", "llamabench"]},
    "Offline / private": {"tests": ["llm", "conv", "emb"]},
    "Quick run": {"tests": ["llm", "emb"], "runs": 1, "max_prompt_tokens": 8192},
    "Full run": {"tests": [name for name, *_ in TEST_DEFINITIONS], "force_all": True},
    "Role: Orchestrator": {"tests": ["llm", "conv", "reasoning", "tool", "conc_chat"]},
    "Role: Agent / tool caller": {"tests": ["llm", "conv", "tool", "code", "conc_tool"], "max_prompt_tokens": 32768},
    "Role: Coding assistant": {"tests": ["llm", "conv", "code", "reasoning"], "max_prompt_tokens": 32768},
    "Role: Chat assistant": {"tests": ["llm", "conv", "mcq", "reasoning", "conc_chat"], "max_prompt_tokens": 8192},
    "Role: RAG / retrieval": {"tests": ["llm", "conv", "emb", "mcq"], "max_prompt_tokens": 32768},
}
CUSTOM_PRESET = "Custom"
DEFAULT_BENCHMARK_PRESET = "Consumer guidance"


def restored_preset_name(saved: dict | None) -> str:
    name = (DEFAULT_BENCHMARK_PRESET if saved is None
            else saved.get("selected_preset", CUSTOM_PRESET))
    return name if name in {*BENCHMARK_PRESETS, CUSTOM_PRESET} else CUSTOM_PRESET


def preset_after_control_change(current: str, applying_preset: bool) -> str:
    return current if applying_preset else CUSTOM_PRESET


def resolve_preset(name: str, available_tests: set[str]) -> dict:
    preset = BENCHMARK_PRESETS[name]
    return {
        "tests": [test for test in preset["tests"] if test in available_tests],
        "runs": preset.get("runs", config.N_RUNS),
        "max_prompt_tokens": preset.get("max_prompt_tokens"),
        "force_all": preset.get("force_all", False),
    }


def progress_event_engine(event: dict, progress_engines: list[str]) -> str | None:
    """Which engine's rows an event belongs to. An unnamed event is assumed to be the
    sole running engine, never guessed at — see docs/how-it-works.md."""
    named = event.get("engine")
    if named:
        return named if named in progress_engines else None
    return progress_engines[0] if len(progress_engines) == 1 else None


def preset_control_values(name: str, available_tests: set[str], defaults: dict) -> dict:
    """No "engine" key: presets describe tests and run settings, so applying one
    must leave the live engine selection alone."""
    preset = resolve_preset(name, available_tests)
    values = {
        "tests": {test: test in preset["tests"] for test in defaults["tests"]},
        "models": dict(defaults["models"]),
        "max_prompt_tokens": (str(preset["max_prompt_tokens"])
                              if preset["max_prompt_tokens"] else "No cap"),
        "tg_tokens": set(defaults["tg_tokens"]),
        "options": dict(defaults["options"]),
    }
    values["options"]["runs"] = preset["runs"]
    values["options"]["force_all"] = preset["force_all"]
    return values


def apply_hardware_model_defaults(entries, inventory: dict[str, list[dict]], ram_gb: float) -> None:
    usable_bytes = max(0.0, ram_gb - 8.0) * 1e9
    sizes = {
        (family, model.get("short") if family == "image" else model.get("tag")): model.get("size")
        for family, models in inventory.items() for model in models
    }
    for entry in entries:
        size = sizes.get((entry.kind, entry.value))
        if isinstance(size, (int, float)) and size * 1.2 > usable_bytes:
            entry.checked = False


def build_plan_preview(*, engine: str, tests: list[str], entries, options: dict,
                       max_prompt_tokens: int | None, tg_tokens: list[int] | None,
                       comfyui_dir: Path) -> str:
    selected = [entry for entry in entries if entry.checked]
    models = [entry.label for entry in selected]
    model_passes = 0
    for test in tests:
        if test == "emb":
            model_passes += sum(entry.kind == "embedding" for entry in selected)
        elif test == "img":
            model_passes += sum(entry.kind == "image" for entry in selected)
        elif test in LLM_BACKED_TESTS:
            model_passes += sum(entry.kind in {"llm", "custom"} for entry in selected)
    processes = []
    if set(tests) - {"llamabench", "llamabenchconc", "img"}:
        processes.append("llama-server")
    if "llamabench" in tests:
        processes.append("llama-bench")
    if "llamabenchconc" in tests:
        processes.append("llama-batched-bench")
    if "img" in tests:
        processes.append("ComfyUI")
    output = options.get("out") or "Automatic results/results_<host>_<time>.json"
    lines = [
        f"Engine: {engine}", f"Tests: {', '.join(tests)}", f"Models: {', '.join(models)}",
        f"Warmups: {options['warmup']}", f"Measured runs: {options['runs']}",
        f"Run timeout: {options['timeout']} seconds",
        f"Accuracy timeout: {options['acc_timeout']} seconds",
        f"Accuracy token budget: {options['acc_token_budget']} tokens",
        f"Prompt cap: {max_prompt_tokens or 'No cap'}",
        f"llama-bench generation sizes: {', '.join(map(str, tg_tokens)) if tg_tokens else 'Defaults'}",
        f"CPU only: {'Yes' if options['cpu_only'] else 'No'}",
        f"Offline: {'Yes' if options['offline'] else 'No'}",
        f"Force slow models: {'Yes' if options['force_all'] else 'No'}",
        f"Retry prior crashes: {'Yes' if options['retry_crashed_models'] else 'No'}",
        f"Broad cases: {model_passes} model-workload passes; contexts, questions, and levels expand within them.",
        f"Model loads: at least {model_passes}; context and concurrency workloads may reload models.",
        "Duration range: minutes to hours; this hardware has no calibrated estimate yet.",
        f"Processes: {', '.join(processes) if processes else 'None'}",
        f"Results: {output}", f"ComfyUI: {comfyui_dir}",
        "Disk use: results JSON plus accuracy sidecars and generated images when selected.",
        "Network use: none expected; all selected models must already be local.",
    ]
    return "\n".join(lines)


def plan_preview_sections(preview: str) -> list[tuple[str, list[str]]]:
    groups = (
        ("Selection", {"Engine", "Tests", "Models"}),
        ("Measurement settings", {
            "Warmups", "Measured runs", "Run timeout", "Accuracy timeout",
            "Accuracy token budget", "Prompt cap", "llama-bench generation sizes",
            "CPU only", "Offline", "Force slow models",
        }),
        ("Scope and duration", {"Broad cases", "Model loads", "Duration range", "Processes"}),
        ("Output and environment", {"Results", "ComfyUI", "Disk use", "Network use"}),
    )
    sections = [(title, []) for title, _ in groups]
    for line in preview.splitlines():
        label = line.partition(":")[0]
        index = next((i for i, (_, labels) in enumerate(groups) if label in labels), len(groups) - 1)
        sections[index][1].append(line)
    return [(title, lines) for title, lines in sections if lines]


def reconcile_imported_model_state(
    previous_values: set[str], previous_selected: set[str], previous_defaults: dict[str, bool],
    rebuilt: list[MenuEntry], imported_tag: str,
) -> tuple[set[str], set[str], set[str], dict[str, bool]]:
    current_values = {entry.value for entry in rebuilt}
    selected = previous_selected & current_values
    if imported_tag in current_values:
        selected.add(imported_tag)
    defaults = {
        entry.value: previous_defaults.get(entry.value, entry.checked) for entry in rebuilt
    }
    return selected, previous_values - current_values, current_values - previous_values, defaults
