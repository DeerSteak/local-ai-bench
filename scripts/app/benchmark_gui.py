#!/usr/bin/env python3
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
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Protocol, Sequence

from scripts.runtime import config, hardware
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
    delete_run_artifacts, discover_results, existing_run_artifacts, filter_results,
    load_result as load_history_result,
)
from scripts.results.recovery_inspector import inspect_recovery
from scripts.results.support_bundle import export_support_bundle, preview_support_bundle
from scripts.setup.model_inventory import build_model_inventory
from scripts.app.model_import_dialog import show_model_import_dialog
from scripts.app.engine_management import (
    build_engine_management_tab, collect_engine_management, vllm_update_support,
)
from scripts.setup.runtime_update import (
    RuntimeUpdateResult, detect_nvidia_max_cuda_version, rebuild_managed_llamacpp,
    update_homebrew_llamacpp, update_managed_vllm, update_windows_llamacpp,
)
from scripts.workloads.models import LLM_MODELS
from scripts.app.orchestration import STAGE_ORDER
from scripts.results.outbound_metadata import outbound_metadata_preview, prepare_outbound_result
from scripts.runtime.pause_control import PAUSE_CONTROL_ENV, create_pause_control, write_pause_state
from scripts.app.progress_events import PROGRESS_PREFIX
from scripts.runtime.shared import Shared
from scripts.setup.setup_config import (
    available_gpu_split_modes, configured_comfyui_dir, configured_gpu_devices,
    load_setup_config,
)
from scripts.app.tk_utils import mousewheel_scroll_units, refresh_tk_layout
from scripts.results.vendor_diagnostic import write_vendor_diagnostic


def effective_gui_options(state: dict | None) -> dict:
    options = state.get("gui_options") if state else None
    return dict(options) if options is not None else dict(GUI_OPTION_DEFAULTS)


def open_path_command(path: Path, system: str) -> list[str]:
    if system == "Darwin":
        return ["open", str(path)]
    if system == "Windows":
        return ["explorer", str(path)]
    return ["xdg-open", str(path)]


def selected_result_paths(selected_items, item_paths: dict, *, exact: int | None = None,
                          maximum: int | None = None) -> list[Path]:
    paths = [Path(item_paths[item]).resolve() for item in selected_items if item in item_paths]
    if exact is not None and len(paths) != exact:
        noun = "result" if exact == 1 else "results"
        raise ValueError(f"Select exactly {exact} {noun} first.")
    if not paths:
        raise ValueError("Select at least one result first.")
    if maximum is not None and len(paths) > maximum:
        raise ValueError(f"Select no more than {maximum} results.")
    return paths


def run_log_path(result_path: Path) -> Path:
    result_path = Path(result_path)
    stem = result_path.stem
    suffix = stem[len("results_"):] if stem.startswith("results_") else stem
    return result_path.with_name(f"log_{suffix}.txt")


def completed_result_paths(log: str) -> list[Path]:
    paths = []
    for line in log.splitlines():
        match = re.search(r"Results saved to:\s*(.+?)\s*$", line)
        if match:
            paths.append(Path(match.group(1)).expanduser().resolve())
    return list(dict.fromkeys(paths))


def write_run_logs(log: str, result_paths: list[Path]) -> list[Path]:
    written = []
    for result_path in result_paths:
        destination = run_log_path(result_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(log, encoding="utf-8")
        written.append(destination)
    return written


def dashboard_launcher_command(result_paths: list[Path], system: str,
                               repo_root: Path = config.SCRIPT_DIR) -> list[str]:
    root = Path(repo_root).resolve()
    launcher = root / ("launch_dashboard.bat" if system == "Windows"
                       else "launch_dashboard.sh")
    command = ["cmd", "/c", str(launcher)] if system == "Windows" else ["bash", str(launcher)]
    for result_path in result_paths:
        command.extend(("--result", str(Path(result_path).resolve())))
    return command


def launch_controlled_process(command: list[str], *, creationflags: int = 0,
                              pause_path_factory=create_pause_control,
                              popen=subprocess.Popen) -> tuple[subprocess.Popen, Path]:
    control_path = pause_path_factory()
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1", "LOCAL_AI_BENCH_PROGRESS": "1",
                 PAUSE_CONTROL_ENV: str(control_path)}
    try:
        process = popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=creationflags, env=child_env,
        )
    except BaseException:
        control_path.unlink(missing_ok=True)
        raise
    return process, control_path


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
    if not isinstance(event, dict) or event.get("kind") not in {"stage", "model", "measurement"}:
        return None
    statuses = ({"retrying", "valid", "invalid"} if event.get("kind") == "measurement"
                else {"running", "complete", "skipped", "failed", "interrupted"})
    if event.get("status") not in statuses:
        return None
    if not isinstance(event.get("stage"), str):
        return None
    if event["kind"] in {"model", "measurement"} and not isinstance(event.get("model"), str):
        return None
    if "usable" in event and not isinstance(event["usable"], bool):
        return None
    return event


def update_progress_metrics(metrics: dict, event: dict) -> dict:
    updated = dict(metrics)
    if event["kind"] == "measurement":
        key = {"retrying": "retries", "valid": "valid", "invalid": "invalid"}[event["status"]]
        updated[key] += 1
    elif event["kind"] == "model" and event["status"] in {
            "complete", "skipped", "failed", "interrupted"}:
        identity = (event["stage"], event["model"])
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


def recovery_executor_command(result_path: Path, python_executable=sys.executable) -> list[str]:
    return [python_executable, "-m", "scripts.results.recovery_executor",
            str(Path(result_path).resolve())]


def fork_executor_command(source_path: Path, output_path: Path,
                          python_executable=sys.executable) -> list[str]:
    return [
        python_executable, "-m", "scripts.results.fork_executor",
        str(Path(source_path).resolve()), str(Path(output_path).resolve()),
    ]


def retry_executor_command(result_path: Path, case_ids: list[str],
                           python_executable=sys.executable) -> list[str]:
    return [
        python_executable, "-m", "scripts.results.retry_executor",
        str(Path(result_path).resolve()), *case_ids,
    ]


def format_recovery_inspection(report: dict) -> str:
    lines = [
        f"Decision: {report['action'].upper()}",
        f"Plan: {report['plan_id']}",
        f"Interrupted attempts: {report['interrupted_attempts']}", "", "Stages:",
    ]
    lines += [f"  {stage}: {state}" for stage, state in report["stage_states"].items()]
    lines += ["", "Cases:"]
    lines += [f"  {state}: {count}" for state, count in report["case_counts"].items()]
    retryable = report.get("retryable_cases", [])
    if retryable:
        lines += ["", "Retry candidates:"]
        lines += [
            f"  {case['stage']}: {case['label']} ({case['state']})"
            for case in retryable
        ]
    if report["reasons"]:
        lines += ["", "Reasons:", *[f"  - {reason}" for reason in report["reasons"]]]
    return "\n".join(lines)


def fork_review_report(result_path: Path) -> dict:
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    plan = load_run_plan(result_path)
    run = data.get("run", {})
    return {
        "action": "fork", "can_resume": False, "plan_id": plan.plan_id,
        "interrupted_attempts": 0,
        "stage_states": {
            stage: run.get("stages", {}).get(stage, {}).get("status", "pending")
            for stage in plan.stage_order
        },
        "case_counts": {}, "retryable_cases": [],
        "reasons": ["fork creates a new run and leaves the source unchanged"],
    }


def recovery_progress_entries(plan, model_shorts=None) -> list:
    entries = []
    seen = set()
    labels = {model["tag"]: model["label"] for model in LLM_MODELS}
    for family, kind in (("llm", "llm"), ("concurrency", "llm")):
        for model in plan.models[family]:
            if model_shorts is not None and model.get("short") not in model_shorts:
                continue
            key = (kind, model.get("tag") or model.get("short"))
            if key in seen:
                continue
            seen.add(key)
            entries.append(SimpleNamespace(
                checked=True, kind=kind,
                label=labels.get(model.get("tag"), model.get("tag") or model.get("short")),
            ))
    return entries


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


def run_benchmark_gui() -> int:  # pragma: no cover — interactive desktop UI
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    saved = load_frontend_state(FRONTEND_STATE_PATH)
    setup = load_setup_config(config.SETUP_CONFIG_PATH)
    found_comfyui = find_comfyui_installation(
        saved_path=configured_comfyui_dir(setup), managed_dir=config.COMFYUI_DIR,
    )
    detected_comfyui = found_comfyui or config.COMFYUI_DIR
    available_engines = installed_engine_names()
    selected_engine = saved["engine"] if saved and saved["engine"] in available_engines else available_engines[0]
    # Every installed engine's models, so switching engines re-gates the list instead of
    # offering models the newly selected engine cannot run.
    engine_inventories = {
        name: build_model_inventory(get_engine(name), config.COMFYUI_MODELS_DIR)
        for name in available_engines
    }
    inventory, model_owners = merge_model_inventories(engine_inventories)
    detected_tools = {name: find_llamacpp_tool(name) for name in (
        "llama-server", "llama-bench", "llama-batched-bench",
    )}
    system_ram_gb = Shared.system_ram_gb()
    hardware_backend = Shared.detect_backend()
    runtime_backend = get_engine(selected_engine).runtime_backend(hardware_backend)
    gpu_split_modes = available_gpu_split_modes(setup, runtime_backend)
    discovery = build_discovery_report(
        platform_name=platform.system(), architecture=platform.machine(),
        ram_gb=system_ram_gb, backend=hardware_backend,
        tools=detected_tools,
        comfyui_dir=found_comfyui, inventory=inventory,
        free_storage_gb=shutil.disk_usage(config.SCRIPT_DIR).free / 1e9,
    )

    default_tests = build_test_entries(inventory)
    default_test_values = [entry.value for entry in default_tests if entry.checked]
    default_models = build_model_entries(inventory, default_test_values)
    apply_hardware_model_defaults(default_models, inventory, system_ram_gb)
    custom_tests = build_test_entries(inventory)
    custom_test_defaults = {entry.value: entry.checked for entry in custom_tests}
    apply_saved_test_selection(custom_tests, saved)
    custom_test_values = [entry.value for entry in custom_tests if entry.checked]
    custom_models = build_model_entries(inventory, [entry.value for entry in custom_tests if entry.available])
    apply_hardware_model_defaults(custom_models, inventory, system_ram_gb)
    custom_model_defaults = {entry.value: entry.checked for entry in custom_models}
    apply_saved_model_selection(custom_models, saved)

    root = tk.Tk()
    root.title(f"Local AI Bench v{config.VERSION}")
    root.geometry("1080x820")
    root.minsize(860, 650)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    style = ttk.Style(root)
    style.configure("Title.TLabel", font=("TkDefaultFont", 21, "bold"))
    style.configure("Section.TLabel", font=("TkDefaultFont", 12, "bold"))
    style.configure("Start.TButton", font=("TkDefaultFont", 12, "bold"), padding=(18, 9))

    advanced_var = tk.BooleanVar(value=False)
    engine_var = tk.StringVar(value=selected_engine)
    test_vars = {entry.value: tk.BooleanVar(value=entry.checked) for entry in custom_tests}
    model_vars = {entry.value: tk.BooleanVar(value=entry.checked) for entry in custom_models}
    cap_var = tk.StringVar(value=str(saved["max_prompt_tokens"]) if saved and saved["max_prompt_tokens"] else "No cap")
    saved_tg = set(saved["tg_tokens"] or config.LLAMABENCH_TG) if saved else set(config.LLAMABENCH_TG)
    tg_vars = {value: tk.BooleanVar(value=value in saved_tg) for value in TG_TOKEN_OPTIONS}
    options = effective_gui_options(saved)
    if options["gpu_split_mode"] not in gpu_split_modes:
        options["gpu_split_mode"] = "layer"
    option_vars: dict[str, tk.Variable] = {
        key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value)))
        for key, value in options.items()
    }
    if not option_vars["comfyui"].get():
        option_vars["comfyui"].set(str(detected_comfyui))

    def perform_vllm_update():
        snapshot = collect_engine_management(get_engine, hardware_backend)
        status = next(item for item in snapshot.statuses if item.engine == "vllm")
        support = vllm_update_support(status, setup, platform.machine())
        if support is None:
            return RuntimeUpdateResult(False, "This vLLM runtime is not app managed or updateable.")
        return update_managed_vllm(support, config.VLLM_VENV)

    def perform_llamacpp_update():
        snapshot = collect_engine_management(get_engine, hardware_backend)
        status = next(item for item in snapshot.statuses if item.engine == "llamacpp")
        if platform.system() == "Darwin":
            return update_homebrew_llamacpp(status.location)
        if platform.system() == "Windows":
            return update_windows_llamacpp(
                config.LLAMACPP_DIR, detect_nvidia_max_cuda_version(),
            )
        if not status.managed:
            return RuntimeUpdateResult(False, "This llama.cpp runtime is not app managed.")
        return rebuild_managed_llamacpp(config.LLAMACPP_DIR, status.backend)

    llamacpp_update_prompts = {
        "Darwin": "Ask Homebrew to update llama.cpp, then validate the installed tools?",
        "Windows": "Download and validate the latest compatible llama.cpp release, then replace the current one?",
        "Linux": "Clone and build the latest llama.cpp, then replace the current checkout?",
    }

    notebook = ttk.Notebook(root)
    notebook.grid(sticky="nsew")
    config_tab = ttk.Frame(notebook, padding=18)
    log_tab = ttk.Frame(notebook, padding=18)
    history_tab = ttk.Frame(notebook, padding=18)
    engines_tab = ttk.Frame(notebook, padding=18)
    notebook.add(config_tab, text="Configuration")
    notebook.add(log_tab, text="Run Log")
    notebook.add(history_tab, text="Result History")
    notebook.add(engines_tab, text="Engine Management")
    build_engine_management_tab(
        parent=engines_tab, root=root, tk=tk, ttk=ttk, messagebox=messagebox,
        status_loader=lambda: collect_engine_management(get_engine, hardware_backend),
        vllm_updater=perform_vllm_update,
        llamacpp_updater=perform_llamacpp_update,
        llamacpp_update_prompt=llamacpp_update_prompts.get(platform.system()),
        run_active=lambda: process is not None and process.poll() is None,
    )
    notebook.bind(
        "<<NotebookTabChanged>>", lambda _event: refresh_tk_layout(root), add="+",
    )
    config_tab.columnconfigure(0, weight=1)
    config_tab.rowconfigure(2, weight=1)

    ttk.Label(config_tab, text=f"Local AI Bench v{config.VERSION}", style="Title.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(
        config_tab,
        text="Choose a ready-made preset or adjust any setting to create a remembered Custom configuration.",
    ).grid(row=1, column=0, sticky="w", pady=(2, 12))

    canvas = tk.Canvas(config_tab, highlightthickness=0)
    scrollbar = ttk.Scrollbar(config_tab, orient="vertical", command=canvas.yview)
    form = ttk.Frame(canvas)
    form.columnconfigure(0, weight=1)
    form.columnconfigure(1, weight=1)
    window_id = canvas.create_window((0, 0), window=form, anchor="nw")
    form.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")

    discovery_box = ttk.LabelFrame(form, text="System inventory and preflight", padding=12)
    discovery_box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
    for row, (label, value) in enumerate((
        ("System", discovery["system"]), ("Installed models", discovery["models"]),
        ("Storage", discovery["storage"]), ("Memory-fit context", discovery["memory_risk"]),
        ("llama.cpp tools", discovery["runtime"]), ("ComfyUI", discovery["comfyui"]),
    )):
        ttk.Label(discovery_box, text=f"{label}:", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, sticky="nw", padx=(0, 10), pady=2,
        )
        ttk.Label(discovery_box, text=value, wraplength=780).grid(row=row, column=1, sticky="w", pady=2)
    issue_text = ("Ready to configure a benchmark." if not discovery["issues"] else
                  "\n".join(f"• {issue}" for issue in discovery["issues"]))
    ttk.Label(discovery_box, text=issue_text, wraplength=900).grid(
        row=6, column=0, columnspan=2, sticky="w", pady=(8, 0),
    )

    configuration_frame = ttk.Frame(form)
    configuration_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
    configuration_frame.columnconfigure(0, weight=1, uniform="configuration")
    configuration_frame.columnconfigure(1, weight=1, uniform="configuration")
    header_frame = ttk.Frame(configuration_frame)
    header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    # Engine choice sits above the preset row: it changes what a run produces, and
    # presets no longer carry it, so it has to be visible without scrolling.
    engine_box = ttk.LabelFrame(header_frame, text="Inference engines", padding=12)
    engine_box.pack(side="top", fill="x", pady=(0, 10))
    preset_row = ttk.Frame(header_frame)
    preset_row.pack(side="top", fill="x")
    preset_var = tk.StringVar(value=restored_preset_name(saved))
    ttk.Label(preset_row, text="Preset").pack(side="left")
    ttk.Combobox(
        preset_row, state="readonly", textvariable=preset_var,
        values=[*BENCHMARK_PRESETS, CUSTOM_PRESET], width=24,
    ).pack(side="left", padx=(8, 8))
    project_row = ttk.Frame(configuration_frame)
    project_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    active_project: dict[str, dict | None] = {"value": None}
    project_status = tk.StringVar(value="No project loaded")
    ttk.Label(project_row, textvariable=project_status).pack(side="left", padx=(0, 12))
    advanced_toggle = ttk.Checkbutton(
        configuration_frame, text="Show advanced execution and path settings", variable=advanced_var,
    )
    advanced_toggle.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

    tests_box = ttk.LabelFrame(configuration_frame, text="Tests", padding=12)
    tests_box.grid(row=3, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
    tests_box.columnconfigure(0, weight=1)
    test_widgets = {}
    test_labels = {}
    for row, (name, label, _, _) in enumerate(TEST_DEFINITIONS):
        entry = next(item for item in custom_tests if item.value == name)
        text = label if entry.available else f"{label} (model not installed)"
        option_row = ttk.Frame(tests_box)
        option_row.grid(row=row, column=0, sticky="ew", pady=2)
        option_row.columnconfigure(1, weight=1)
        widget = ttk.Checkbutton(option_row, variable=test_vars[name])
        widget.grid(row=0, column=0, sticky="nw")
        option_label = ttk.Label(option_row, text=text, wraplength=280)
        option_label.grid(row=0, column=1, sticky="w", padx=(2, 0))
        option_label.bind("<Button-1>", lambda _event, control=widget: control.invoke())
        ttk.Button(
            tests_box, text="Reset", width=6,
            command=lambda key=name: test_vars[key].set(custom_test_defaults[key]),
        ).grid(row=row, column=1, sticky="e", padx=(8, 0))
        test_widgets[name] = widget
        test_labels[name] = option_label
    ttk.Label(
        tests_box, text="Accuracy and concurrency add substantial runtime; native llama-bench tests require their matching tools.",
        wraplength=330,
    ).grid(row=len(TEST_DEFINITIONS), column=0, columnspan=2, sticky="w", pady=(8, 0))

    models_box = ttk.LabelFrame(configuration_frame, text="Installed models", padding=12)
    models_box.grid(row=3, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
    models_box.columnconfigure(0, weight=1)
    model_rows = ttk.Frame(models_box)
    model_rows.grid(row=0, column=0, columnspan=2, sticky="ew")
    model_rows.columnconfigure(0, weight=1)
    model_widgets = {}

    def render_model_rows():
        for child in model_rows.winfo_children():
            child.destroy()
        model_widgets.clear()
        previous = None
        row = 0
        for entry in custom_models:
            if entry.section != previous:
                ttk.Label(model_rows, text=entry.section, style="Section.TLabel").grid(
                    row=row, column=0, sticky="w", pady=(7, 2),
                )
                row += 1
                previous = entry.section
            option_row = ttk.Frame(model_rows)
            option_row.grid(row=row, column=0, sticky="ew", padx=(12, 0), pady=2)
            option_row.columnconfigure(1, weight=1)
            widget = ttk.Checkbutton(option_row, variable=model_vars[entry.value])
            widget.grid(row=0, column=0, sticky="nw")
            option_label = ttk.Label(option_row, text=entry.label, wraplength=280)
            option_label.grid(row=0, column=1, sticky="w", padx=(2, 0))
            option_label.bind("<Button-1>", lambda _event, control=widget: control.invoke())
            ttk.Button(
                model_rows, text="Reset", width=6,
                command=lambda key=entry.value: model_vars[key].set(custom_model_defaults[key]),
            ).grid(row=row, column=1, sticky="e", padx=(8, 0))
            model_widgets[entry.value] = widget
            row += 1

    render_model_rows()
    ttk.Label(
        models_box, text="Each checked model runs once through every applicable selected workload. Larger models may exceed memory.",
        wraplength=330,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

    workload_box = ttk.LabelFrame(configuration_frame, text="Workload sizes", padding=12)
    workload_box.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    ttk.Label(workload_box, text="Maximum prompt-processing size").grid(row=0, column=0, sticky="w")
    cap_combo = ttk.Combobox(workload_box, state="readonly", textvariable=cap_var,
                             values=["No cap", *[str(value) for value in MAX_PROMPT_TOKEN_OPTIONS]], width=18)
    cap_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))
    ttk.Button(workload_box, text="Reset", width=6, command=lambda: cap_var.set("No cap")).grid(
        row=0, column=2, padx=(8, 0),
    )
    ttk.Label(workload_box, text="llama-bench generation sizes").grid(row=1, column=0, sticky="w", pady=(10, 0))
    tg_frame = ttk.Frame(workload_box)
    tg_frame.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
    for column, value in enumerate(TG_TOKEN_OPTIONS):
        ttk.Checkbutton(tg_frame, text=str(value), variable=tg_vars[value]).grid(row=0, column=column, padx=(0, 8))
    ttk.Button(workload_box, text="Reset", width=6, command=lambda: [
        variable.set(value in config.LLAMABENCH_TG) for value, variable in tg_vars.items()
    ]).grid(row=1, column=2, padx=(8, 0), pady=(10, 0))
    ttk.Label(
        workload_box, text="Prompt and generation values are tokens. No cap tests every configured depth; larger values increase time and memory.",
        wraplength=430,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

    engine_check_vars = {
        name: tk.BooleanVar(value=name in parse_engine_selection(selected_engine))
        for name in available_engines
    }
    engine_note = tk.StringVar()
    # Guards apply_engine_availability's per-checkbox trace during a batch update — without
    # it, setting checkboxes one at a time passes through a momentary all-unchecked state
    # that trips the "never leave zero engines selected" safeguard and re-checks the wrong one.
    restoring_engines = [False]

    def set_selected_engines(names) -> None:
        """Point the checkboxes at `names`, ignoring any engine that isn't installed."""
        wanted = [name for name in names if name in engine_check_vars] or [available_engines[0]]
        restoring_engines[0] = True
        try:
            for name, variable in engine_check_vars.items():
                variable.set(name in wanted)
        finally:
            restoring_engines[0] = False
        apply_engine_availability()

    for index, name in enumerate(available_engines):
        ttk.Checkbutton(engine_box, text=name, variable=engine_check_vars[name]).grid(
            row=0, column=index, sticky="w", padx=(0, 16))
    ttk.Button(
        engine_box, text="Reset", width=6,
        command=lambda: set_selected_engines([available_engines[0]]),
    ).grid(row=0, column=len(available_engines), padx=(8, 0), sticky="w")
    engine_box.columnconfigure(len(available_engines) + 1, weight=1)
    ttk.Label(engine_box, textvariable=engine_note).grid(
        row=1, column=0, columnspan=len(available_engines) + 2, sticky="w", pady=(8, 0))

    def clear_all_crash_caches():
        if process is not None and process.poll() is None:
            messagebox.showerror("Benchmark active", "Stop the active process first.", parent=root)
            return
        caches = Shared.crash_cache_paths(config.SCRIPT_DIR)
        if not caches:
            messagebox.showinfo("Clear crash caches", "No crash caches were found.", parent=root)
            return
        names = "\n".join(f"  • {path.name}" for path in caches)
        if not messagebox.askyesno(
            "Clear crash caches",
            f"Delete all {len(caches)} crash cache file(s)?\n\n{names}\n\n"
            "Previously crashing models will be tried again on future runs. This cannot be undone.",
            parent=root,
        ):
            return
        removed, failures = Shared.clear_crash_caches(config.SCRIPT_DIR)
        if failures:
            detail = "\n".join(f"{path.name}: {reason}" for path, reason in failures.items())
            messagebox.showerror(
                "Crash-cache cleanup incomplete",
                f"Deleted {len(removed)} cache(s), but some could not be removed:\n\n{detail}",
                parent=root,
            )
            return
        messagebox.showinfo(
            "Crash caches cleared", f"Deleted {len(removed)} crash cache file(s).", parent=root,
        )

    execution_box = ttk.LabelFrame(configuration_frame, text="Execution", padding=12)
    execution_box.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    split_label = "Multi-GPU mode (tensor is experimental)" if "tensor" in gpu_split_modes else "Multi-GPU mode"
    ttk.Label(execution_box, text=split_label).grid(row=1, column=0, sticky="w", pady=2)
    ttk.Combobox(
        execution_box, state="readonly", textvariable=option_vars["gpu_split_mode"],
        values=gpu_split_modes, width=16,
    ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=2)
    ttk.Button(
        execution_box, text="Reset", width=6,
        command=lambda: option_vars["gpu_split_mode"].set("layer"),
    ).grid(row=1, column=2, padx=(8, 0))
    labels = (("warmup", f"Warmup runs (default {config.WARMUP_RUNS})"),
              ("runs", f"Measured runs (1–10; default {config.N_RUNS})"),
              ("timeout", f"Run timeout, seconds (default {config.RUN_TIMEOUT})"),
              ("acc_timeout", f"Accuracy timeout, seconds (default {config.ACC_TIMEOUT})"),
              ("acc_token_budget", f"Accuracy token budget (default {config.ACC_TOKEN_BUDGET})"))
    for row, (key, label) in enumerate(labels, 2):
        ttk.Label(execution_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(execution_box, textvariable=option_vars[key], width=12).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=2)
        ttk.Button(
            execution_box, text="Reset", width=6,
            command=lambda option=key: option_vars[option].set(str(GUI_OPTION_DEFAULTS[option])),
        ).grid(row=row, column=2, padx=(8, 0), pady=2)
    ttk.Checkbutton(execution_box, text="CPU-only inference", variable=option_vars["cpu_only"]).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Button(execution_box, text="Reset", width=6, command=lambda: option_vars["cpu_only"].set(False)).grid(row=8, column=2, padx=(8, 0))
    ttk.Checkbutton(execution_box, text="Run slow models instead of skipping", variable=option_vars["force_all"]).grid(row=9, column=0, columnspan=2, sticky="w")
    ttk.Button(execution_box, text="Reset", width=6, command=lambda: option_vars["force_all"].set(False)).grid(row=9, column=2, padx=(8, 0))
    ttk.Checkbutton(execution_box, text="Retry models that crashed previously", variable=option_vars["retry_crashed_models"]).grid(row=10, column=0, columnspan=2, sticky="w")
    ttk.Button(execution_box, text="Reset", width=6, command=lambda: option_vars["retry_crashed_models"].set(False)).grid(row=10, column=2, padx=(8, 0))
    ttk.Checkbutton(execution_box, text="Offline mode (loopback only)", variable=option_vars["offline"]).grid(row=11, column=0, columnspan=2, sticky="w")
    ttk.Button(execution_box, text="Reset", width=6, command=lambda: option_vars["offline"].set(False)).grid(row=11, column=2, padx=(8, 0))
    ttk.Label(
        execution_box, text="More warmups/runs improve repeatability but increase time. CPU-only changes the tested device; force-all can make runs much longer.",
        wraplength=430,
    ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Button(
        execution_box, text="Clear Crash Caches", command=clear_all_crash_caches,
    ).grid(row=14, column=0, sticky="w", pady=(8, 0))

    paths_box = ttk.LabelFrame(configuration_frame, text="Paths", padding=12)
    paths_box.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    paths_box.columnconfigure(1, weight=1)
    ttk.Label(paths_box, text="Results JSON (blank = automatic)").grid(row=0, column=0, sticky="w")
    ttk.Entry(paths_box, textvariable=option_vars["out"]).grid(row=0, column=1, sticky="ew", padx=10)
    ttk.Button(paths_box, text="Browse…", command=lambda: option_vars["out"].set(
        filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON results", "*.json")]) or option_vars["out"].get()
    )).grid(row=0, column=2)
    ttk.Button(paths_box, text="Reset", width=6, command=lambda: option_vars["out"].set("")).grid(row=0, column=3, padx=(8, 0))
    ttk.Label(paths_box, text="ComfyUI installation").grid(row=1, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(paths_box, textvariable=option_vars["comfyui"]).grid(row=1, column=1, sticky="ew", padx=10, pady=(8, 0))
    ttk.Button(paths_box, text="Browse…", command=lambda: option_vars["comfyui"].set(
        filedialog.askdirectory() or option_vars["comfyui"].get()
    )).grid(row=1, column=2, pady=(8, 0))
    ttk.Button(
        paths_box, text="Reset", width=6,
        command=lambda: option_vars["comfyui"].set(str(detected_comfyui)),
    ).grid(row=1, column=3, padx=(8, 0), pady=(8, 0))
    ttk.Label(
        paths_box, text="Blank output uses results/results_<host>_<time>.json. ComfyUI must identify a usable program installation, not its model folder.",
        wraplength=900,
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def reset_tests():
        defaults = {entry.value: entry.checked for entry in build_test_entries(inventory)}
        for name, variable in test_vars.items():
            variable.set(defaults[name])

    def reset_models():
        for name, variable in model_vars.items():
            variable.set(custom_model_defaults[name])

    def reset_workload():
        cap_var.set("No cap")
        for value, variable in tg_vars.items():
            variable.set(value in config.LLAMABENCH_TG)

    def reset_execution():
        set_selected_engines([available_engines[0]])
        defaults = custom_option_defaults(detected_comfyui)
        for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget", "gpu_split_mode",
                    "cpu_only", "force_all", "retry_crashed_models", "offline"):
            variable = option_vars[key]
            variable.set(defaults[key])

    def reset_paths():
        defaults = custom_option_defaults(detected_comfyui)
        option_vars["out"].set(defaults["out"])
        option_vars["comfyui"].set(defaults["comfyui"])

    def reset_all():
        reset_tests()
        reset_models()
        reset_workload()
        reset_execution()
        reset_paths()

    def current_custom_state():
        tests = expand_selected_tests(
            name for name, variable in test_vars.items() if variable.get())
        for entry in custom_models:
            entry.checked = model_vars[entry.value].get()
        options = collect_options()
        cap = None if cap_var.get() == "No cap" else int(cap_var.get())
        tg = [value for value, variable in tg_vars.items() if variable.get()]
        return build_frontend_state(
            engine_var.get(), tests, custom_models, max_prompt_tokens=cap,
            tg_tokens=tg, gui_options=options, selected_preset=preset_var.get(),
        )

    def export_preset(preset=None):
        if preset is None:
            name = simpledialog.askstring("Preset name", "Name this portable preset:", parent=root)
            if not name:
                return
            portable = build_portable_preset(name, current_custom_state())
        else:
            portable = preset
        path = filedialog.asksaveasfilename(
            title="Export benchmark preset", defaultextension=".json",
            filetypes=[("Benchmark preset", "*.json")],
        )
        if path:
            save_portable_preset(Path(path), portable)

    def apply_frontend_state(state):
        errors = frontend_state_availability_errors(
            state, available_engines, custom_tests, custom_models,
        )
        if errors:
            raise ValueError("\n".join(errors))
        requested_split = state.get("gui_options", {}).get("gpu_split_mode", "layer")
        if requested_split not in gpu_split_modes:
            raise ValueError("Tensor split is unavailable for the detected GPU runtime and topology.")
        applying_configuration[0] = True
        try:
            restored = [name for name in parse_engine_selection(state.get("engine", ""))
                        if name in available_engines]
            if restored:
                set_selected_engines(restored)
            selected_tests = set(state["tests"])
            for entry in custom_tests:
                test_vars[entry.value].set(entry.available and entry.value in selected_tests)
            selected_models = {
                *state["models"]["llm"], *state["models"]["embedding"],
                *state["models"]["image"],
            }
            for entry in custom_models:
                model_vars[entry.value].set(entry.value in selected_models)
            cap = state["max_prompt_tokens"]
            cap_var.set(str(cap) if cap else "No cap")
            selected_tg = set(state["tg_tokens"] or config.LLAMABENCH_TG)
            for value, variable in tg_vars.items():
                variable.set(value in selected_tg)
            for key, value in state.get("gui_options", {}).items():
                option_vars[key].set(value if isinstance(value, bool) else str(value))
        finally:
            applying_configuration[0] = False
        preset_var.set(CUSTOM_PRESET)

    def apply_portable_preset(portable):
        configuration = portable["configuration"]
        apply_frontend_state({
            "tests": configuration["tests"],
            "models": configuration["models"],
            "max_prompt_tokens": configuration["max_prompt_tokens"],
            "tg_tokens": configuration["tg_tokens"],
            "gui_options": configuration["options"],
        })

    def import_preset():
        path = filedialog.askopenfilename(
            title="Import benchmark preset", filetypes=[("Benchmark preset", "*.json")],
        )
        if not path:
            return
        try:
            apply_portable_preset(load_portable_preset(Path(path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Preset import failed", str(exc), parent=root)

    def compare_preset():
        path = filedialog.askopenfilename(
            title="Compare with preset", filetypes=[("Benchmark preset", "*.json")],
        )
        if not path:
            return
        try:
            saved_preset = load_portable_preset(Path(path))
            current = build_portable_preset("Current screen", current_custom_state())
            differences = compare_portable_presets(current, saved_preset)
            detail = ", ".join(differences) if differences else "No configuration differences."
            messagebox.showinfo("Preset comparison", detail, parent=root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Preset comparison failed", str(exc), parent=root)

    def import_run_plan():
        path = filedialog.askopenfilename(
            title="Import CLI run plan or results", filetypes=[("Benchmark JSON", "*.json")],
        )
        if not path:
            return
        try:
            local_options = collect_options()
            apply_frontend_state(frontend_state_from_run_plan(load_run_plan(Path(path)), local_options))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Run-plan import failed", str(exc), parent=root)

    def choose_project_workflow():
        selected: dict[str, str | None] = {"value": None}
        dialog = tk.Toplevel(root)
        dialog.title("Project workflow")
        dialog.transient(root)
        dialog.grab_set()
        workflow_var = tk.StringVar(value=next(iter(PROJECT_WORKFLOWS)))
        ttk.Label(dialog, text="What decision will this project support?").pack(
            anchor="w", padx=18, pady=(18, 8),
        )
        combo = ttk.Combobox(
            dialog, state="readonly", width=30,
            values=list(PROJECT_WORKFLOWS.values()),
        )
        combo.current(0)
        combo.pack(fill="x", padx=18)

        def accept():
            workflow_var.set(next(
                key for key, label in PROJECT_WORKFLOWS.items() if label == combo.get()
            ))
            selected["value"] = workflow_var.get()
            dialog.destroy()

        actions = ttk.Frame(dialog)
        actions.pack(fill="x", padx=18, pady=18)
        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(actions, text="Continue", command=accept).pack(side="right", padx=(0, 8))
        root.wait_window(dialog)
        return selected["value"]

    def save_current_project():
        name = simpledialog.askstring("New project", "Project name:", parent=root)
        if not name:
            return
        workflow = choose_project_workflow()
        if not workflow:
            return
        baseline = None
        if messagebox.askyesno("Baseline", "Attach an existing baseline result?", parent=root):
            baseline = filedialog.askopenfilename(
                title="Choose baseline result", initialdir=config.RESULTS_DIR,
                filetypes=[("Benchmark result", "*.json")],
            )
            if not baseline:
                return
        acceptance = None
        if messagebox.askyesno("Acceptance policy", "Attach an acceptance policy?", parent=root):
            policy_path = filedialog.askopenfilename(
                title="Choose acceptance policy", filetypes=[("Acceptance policy", "*.json")],
            )
            if not policy_path:
                return
            try:
                acceptance = load_policy(Path(policy_path))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                messagebox.showerror("Acceptance policy failed", str(exc), parent=root)
                return
        destination = filedialog.asksaveasfilename(
            title="Save benchmark project", defaultextension=".labproject",
            filetypes=[("Local AI Bench project", "*.labproject")],
        )
        if not destination:
            return
        try:
            project = build_project(
                name, workflow, current_custom_state(), baseline_result=baseline,
                acceptance_policy=acceptance,
            )
            save_project(Path(destination), project)
            active_project["value"] = project
            project_status.set(f"Project: {project['name']} ({PROJECT_WORKFLOWS[workflow]})")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Project creation failed", str(exc), parent=root)

    def open_project():
        path = filedialog.askopenfilename(
            title="Open benchmark project", filetypes=[("Local AI Bench project", "*.labproject")],
        )
        if not path:
            return
        try:
            project = load_project(Path(path))
            apply_frontend_state(project_frontend_state(project, collect_options()))
            active_project["value"] = project
            project_status.set(
                f"Project: {project['name']} ({PROJECT_WORKFLOWS[project['workflow']]})"
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Project open failed", str(exc), parent=root)

    ttk.Button(preset_row, text="Export", command=export_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Import", command=import_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Compare", command=compare_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Import CLI Plan", command=import_run_plan).pack(side="left", padx=(8, 0))
    ttk.Button(project_row, text="New Project", command=save_current_project).pack(side="left")
    ttk.Button(project_row, text="Open Project", command=open_project).pack(side="left", padx=(8, 0))

    ttk.Button(tests_box, text="Reset Tests", command=reset_tests).grid(
        row=len(TEST_DEFINITIONS) + 1, column=0, sticky="w", pady=(8, 0),
    )
    model_actions = ttk.Frame(models_box)
    model_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    ttk.Button(model_actions, text="Reset Models", command=reset_models).pack(side="left")
    ttk.Button(
        model_actions, text="Import Hugging Face Model",
        command=lambda: open_model_import_dialog(),
    ).pack(side="right")
    ttk.Button(workload_box, text="Reset Workload Sizes", command=reset_workload).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(8, 0),
    )
    ttk.Button(execution_box, text="Reset Execution", command=reset_execution).grid(
        row=13, column=0, columnspan=2, sticky="w", pady=(8, 0),
    )
    ttk.Button(paths_box, text="Reset Paths", command=reset_paths).grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(8, 0),
    )

    footer = ttk.Frame(config_tab)
    footer.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    status_var = tk.StringVar(value="Ready to benchmark.")
    ttk.Label(footer, textvariable=status_var).pack(side="left")
    start_button = ttk.Button(footer, text="Start Benchmark", style="Start.TButton")
    start_button.pack(side="right")
    ttk.Button(footer, text="Reset All Options", command=reset_all).pack(side="right", padx=(0, 10))

    log_tab.columnconfigure(0, weight=1)
    log_tab.rowconfigure(2, weight=1)
    ttk.Label(log_tab, text="Benchmark run", style="Title.TLabel").grid(row=0, column=0, sticky="w")
    run_status = tk.StringVar(value="No benchmark is running.")
    ttk.Label(log_tab, textvariable=run_status).grid(row=1, column=0, sticky="w", pady=(2, 10))
    log_text = tk.Text(log_tab, wrap="word", state="disabled", font=("TkFixedFont", 10))
    log_scroll = ttk.Scrollbar(log_tab, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.grid(row=2, column=0, sticky="nsew")
    log_scroll.grid(row=2, column=1, sticky="ns")
    log_actions = ttk.Frame(log_tab)
    log_actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    log_run_actions = ttk.Frame(log_actions)
    log_run_actions.pack(fill="x")
    log_result_actions = ttk.Frame(log_actions)
    log_result_actions.pack(fill="x", pady=(8, 0))
    stop_button = ttk.Button(log_run_actions, text="Stop Benchmark", state="disabled")
    stop_button.pack(side="right")
    pause_button = ttk.Button(log_run_actions, text="Pause", state="disabled")
    pause_button.pack(side="right", padx=(0, 8))
    ttk.Button(log_run_actions, text="Back to Configuration", command=lambda: notebook.select(config_tab)).pack(side="left")

    def current_log() -> str:
        return log_text.get("1.0", "end-1c")

    def export_log():
        log = current_log()
        if not log:
            messagebox.showinfo("Export Log", "The Run Log is empty.", parent=root)
            return
        known_results = completed_result_paths(log) or active_result_paths[:1]
        suggested = run_log_path(known_results[0]) if known_results else config.RESULTS_DIR / "run_log.txt"
        destination = filedialog.asksaveasfilename(
            title="Export Run Log", initialdir=str(suggested.parent),
            initialfile=suggested.name, defaultextension=".txt",
            filetypes=[("Text log", "*.txt"), ("All files", "*")],
        )
        if not destination:
            return
        try:
            Path(destination).write_text(log, encoding="utf-8")
            messagebox.showinfo("Log exported", f"Run Log saved to:\n{destination}", parent=root)
        except OSError as exc:
            messagebox.showerror("Log export failed", str(exc), parent=root)

    def open_results_folder():
        output = option_vars["out"].get().strip()
        folder = Path(output).expanduser().resolve().parent if output else config.RESULTS_DIR
        subprocess.Popen(open_path_command(folder, platform.system()))

    def review_outbound_metadata(result, purpose, *, allow_aliases=True):
        decision: dict[str, dict | None] = {"value": None}
        dialog = tk.Toplevel(root)
        dialog.title(f"Review metadata for {purpose}")
        dialog.geometry("760x600")
        dialog.transient(root)
        dialog.grab_set()
        ttk.Label(
            dialog,
            text="Review every identity field before it leaves this machine. Optional aliases replace exported names only; the source result is unchanged.",
            wraplength=710,
        ).pack(anchor="w", padx=16, pady=(16, 8))
        preview_frame = ttk.Frame(dialog)
        preview_frame.pack(fill="both", expand=True, padx=16)
        text_widget = tk.Text(preview_frame, wrap="none", height=20)
        scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scroll.set)
        text_widget.insert("1.0", "\n".join(
            f"{label}: {value}" for label, value in outbound_metadata_preview(result)
        ))
        text_widget.configure(state="disabled")
        text_widget.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        aliases = ttk.LabelFrame(dialog, text="Optional private aliases", padding=10)
        if allow_aliases:
            aliases.pack(fill="x", padx=16, pady=(10, 0))
        system_alias = tk.StringVar()
        hardware_alias = tk.StringVar()
        ttk.Label(aliases, text="System name").grid(row=0, column=0, sticky="w")
        ttk.Entry(aliases, textvariable=system_alias).grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ttk.Label(aliases, text="Hardware name").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(aliases, textvariable=hardware_alias).grid(
            row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0),
        )
        aliases.columnconfigure(1, weight=1)
        actions = ttk.Frame(dialog)
        actions.pack(fill="x", padx=16, pady=16)

        def approve():
            decision["value"] = {
                "system_alias": system_alias.get().strip() or None,
                "hardware_alias": hardware_alias.get().strip() or None,
            }
            dialog.destroy()

        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(actions, text="Approve Export", command=approve).pack(side="right", padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        root.wait_window(dialog)
        return decision["value"]

    def export_bundle():
        result = filedialog.askopenfilename(
            title="Choose result JSON", initialdir=config.RESULTS_DIR,
            filetypes=[("Benchmark result", "*.json")],
        )
        if not result:
            return
        try:
            source = load_result(Path(result))
            aliases = review_outbound_metadata(source, "result bundle")
            if aliases is None:
                return
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Bundle export failed", str(exc), parent=root)
            return
        bundle = filedialog.asksaveasfilename(
            title="Export verified result bundle", defaultextension=".labresult",
            filetypes=[("Local AI Bench result", "*.labresult")],
        )
        if not bundle:
            return
        try:
            export_result_bundle(Path(result), Path(bundle), **aliases)
            messagebox.showinfo("Bundle exported", f"Verified bundle saved to:\n{bundle}", parent=root)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Bundle export failed", str(exc), parent=root)

    def import_bundle():
        bundle = filedialog.askopenfilename(
            title="Import and verify result bundle",
            filetypes=[("Local AI Bench result", "*.labresult")],
        )
        if not bundle:
            return
        destination = filedialog.asksaveasfilename(
            title="Save verified result JSON", initialdir=config.RESULTS_DIR,
            defaultextension=".json", filetypes=[("Benchmark result", "*.json")],
        )
        if not destination:
            return
        try:
            verify_result_bundle(Path(bundle))
            import_result_bundle(Path(bundle), Path(destination), Path(destination).with_suffix(""))
            messagebox.showinfo(
                "Bundle verified and imported", f"Verified result saved to:\n{destination}", parent=root,
            )
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Bundle verification failed", str(exc), parent=root)

    def create_report():
        result_path = filedialog.askopenfilename(
            title="Choose result JSON", initialdir=config.RESULTS_DIR,
            filetypes=[("Benchmark result", "*.json")],
        )
        if not result_path:
            return
        try:
            source_result = load_result(Path(result_path))
            aliases = review_outbound_metadata(source_result, "decision report")
            if aliases is None:
                return
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Report creation failed", str(exc), parent=root)
            return
        destination = filedialog.asksaveasfilename(
            title="Save decision report", initialdir=config.RESULTS_DIR,
            defaultextension=".html", filetypes=[("Decision report", "*.html")],
        )
        if not destination:
            return
        try:
            html_path, pdf_path = report_output_paths(Path(destination))
            result = prepare_outbound_result(source_result, **aliases)
            project_policy = (active_project["value"] or {}).get("acceptance_policy")
            policy = project_policy
            if project_policy is None and messagebox.askyesno(
                    "Acceptance policy", "Apply an acceptance policy to this report?", parent=root):
                policy_path = filedialog.askopenfilename(
                    title="Choose acceptance policy", filetypes=[("Acceptance policy", "*.json")],
                )
                if not policy_path:
                    return
                policy = load_policy(Path(policy_path))
            write_html_report(result, html_path, policy)
            write_pdf_report(result, pdf_path, policy)
            messagebox.showinfo(
                "Decision report created",
                f"HTML and PDF reports saved to:\n{html_path.parent}", parent=root,
            )
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Report creation failed", str(exc), parent=root)

    def confirm_support_preview(preview):
        accepted = {"value": False}
        dialog = tk.Toplevel(root)
        dialog.title("Review redacted support bundle")
        dialog.geometry("720x520")
        dialog.transient(root)
        dialog.grab_set()
        ttk.Label(
            dialog, text="Review every file and field before export. Raw results and logs are not included.",
            wraplength=680,
        ).pack(anchor="w", padx=16, pady=(16, 8))
        text_widget = tk.Text(dialog, wrap="none", height=22)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side="left", fill="both", expand=True, padx=(16, 0), pady=(0, 16))
        scrollbar.pack(side="left", fill="y", pady=(0, 16))
        details = "Files:\n" + "\n".join(f"  {name}" for name in preview["files"])
        details += "\n\nFields:\n" + "\n".join(f"  {field}" for field in preview["fields"])
        text_widget.insert("1.0", details)
        text_widget.configure(state="disabled")
        actions = ttk.Frame(dialog, padding=(8, 16))
        actions.pack(side="right", fill="y")

        def accept():
            accepted["value"] = True
            dialog.destroy()

        ttk.Button(actions, text="Export", command=accept).pack(fill="x", pady=(0, 8))
        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(fill="x")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        root.wait_window(dialog)
        return accepted["value"]

    def export_support():
        result = filedialog.askopenfilename(
            title="Choose result for support bundle", initialdir=config.RESULTS_DIR,
            filetypes=[("Benchmark result", "*.json")],
        )
        if not result:
            return
        try:
            preview = preview_support_bundle(Path(result))
            if not confirm_support_preview(preview):
                return
            destination = filedialog.asksaveasfilename(
                title="Export redacted support bundle", defaultextension=".labsupport",
                filetypes=[("Local AI Bench support", "*.labsupport")],
            )
            if destination:
                export_support_bundle(Path(result), Path(destination))
                messagebox.showinfo("Support bundle exported", destination, parent=root)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Support bundle failed", str(exc), parent=root)

    ttk.Button(log_result_actions, text="Open Results Folder", command=open_results_folder).pack(
        side="left",
    )
    ttk.Button(log_result_actions, text="Export Log", command=export_log).pack(side="left", padx=(8, 0))
    ttk.Button(log_result_actions, text="Export Bundle", command=export_bundle).pack(side="left", padx=(8, 0))
    ttk.Button(log_result_actions, text="Import / Verify", command=import_bundle).pack(side="left", padx=(8, 0))
    ttk.Button(log_result_actions, text="Create Report", command=create_report).pack(side="left", padx=(8, 0))
    ttk.Button(log_result_actions, text="Support Bundle", command=export_support).pack(side="left", padx=(8, 0))

    history_tab.columnconfigure(0, weight=1)
    history_tab.rowconfigure(2, weight=1)
    ttk.Label(history_tab, text="Local result history", style="Title.TLabel").grid(
        row=0, column=0, sticky="w",
    )
    history_filters = ttk.Frame(history_tab)
    history_filters.grid(row=1, column=0, sticky="ew", pady=(8, 10))
    history_query = tk.StringVar()
    history_status_filter = tk.StringVar(value="all")
    history_engine_filter = tk.StringVar(value="all")
    ttk.Label(history_filters, text="Search").pack(side="left")
    ttk.Entry(history_filters, textvariable=history_query, width=26).pack(side="left", padx=(8, 14))
    ttk.Label(history_filters, text="Status").pack(side="left")
    ttk.Combobox(
        history_filters, state="readonly", width=12, textvariable=history_status_filter,
        values=("all", "complete", "partial", "interrupted", "failed", "running", "legacy"),
    ).pack(side="left", padx=(8, 14))
    ttk.Label(history_filters, text="Engine").pack(side="left")
    history_engine_combo = ttk.Combobox(
        history_filters, state="readonly", width=14, textvariable=history_engine_filter,
        values=("all",),
    )
    history_engine_combo.pack(side="left", padx=(8, 14))
    history_tree = ttk.Treeview(
        history_tab, columns=("date", "system", "status", "engine", "profile", "models"),
        show="headings", selectmode="extended",
    )
    for column, label, width in (
        ("date", "Started", 170), ("system", "System", 190), ("status", "Status", 95),
        ("engine", "Engine", 95), ("profile", "Profile", 110), ("models", "Models", 70),
    ):
        history_tree.heading(column, text=label)
        history_tree.column(column, width=width, anchor="w")
    history_scroll = ttk.Scrollbar(history_tab, orient="vertical", command=history_tree.yview)
    history_tree.configure(yscrollcommand=history_scroll.set)
    history_tree.grid(row=2, column=0, sticky="nsew")
    history_scroll.grid(row=2, column=1, sticky="ns")
    history_actions = ttk.Frame(history_tab)
    history_actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    history_review_actions = ttk.Frame(history_actions)
    history_review_actions.pack(fill="x")
    history_recovery_actions = ttk.Frame(history_actions)
    history_recovery_actions.pack(fill="x", pady=(8, 0))
    history_message = tk.StringVar(value="History has not been loaded.")
    ttk.Label(history_tab, textvariable=history_message).grid(row=4, column=0, sticky="w", pady=(8, 0))
    history_entries = {"all": [], "visible": []}
    history_item_paths = {}

    def selected_history_items():
        return sorted(history_tree.selection(), key=history_tree.index)

    def selected_history_path():
        return selected_result_paths(selected_history_items(), history_item_paths, exact=1)[0]

    def open_history_in_dashboard():
        try:
            paths = selected_result_paths(
                selected_history_items(), history_item_paths, maximum=6,
            )
            command = dashboard_launcher_command(paths, platform.system())
            subprocess.Popen(
                command, cwd=config.SCRIPT_DIR,
                creationflags=(getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                               if platform.system() == "Windows" else 0))
            history_message.set(
                f"Opening {len(paths)} selected result{'s' if len(paths) != 1 else ''} in the dashboard."
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dashboard launch failed", str(exc), parent=root)

    def delete_history_selection():
        if process is not None and process.poll() is None:
            messagebox.showerror("Benchmark active", "Stop the active process first.", parent=root)
            return
        try:
            result_path = selected_history_path()
            artifacts = existing_run_artifacts(result_path, config.RESULTS_DIR)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Delete run", str(exc), parent=root)
            return
        if not artifacts:
            refresh_history()
            messagebox.showinfo("Delete run", "The selected run no longer exists.", parent=root)
            return
        names = "\n".join(f"  • {path.name}" for path in artifacts)
        if not messagebox.askyesno(
            "Delete benchmark run",
            f"Permanently delete {result_path.name} and all {len(artifacts) - 1} "
            f"associated artifact(s)?\n\n{names}\n\nThis cannot be undone. Separately "
            "exported bundles and reports are not deleted.",
            parent=root,
        ):
            return
        removed, failures = delete_run_artifacts(result_path, config.RESULTS_DIR)
        refresh_history()
        if failures:
            detail = "\n".join(f"{path.name}: {reason}" for path, reason in failures.items())
            messagebox.showerror(
                "Run deletion incomplete",
                f"Deleted {len(removed)} artifact(s), but some could not be removed. "
                f"The main result was retained when possible so deletion can be retried.\n\n{detail}",
                parent=root,
            )
            return
        history_message.set(f"Deleted {result_path.name} and {len(removed) - 1} associated artifact(s).")

    def show_history_details(title, content):
        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.geometry("920x620")
        dialog.transient(root)
        text_widget = tk.Text(dialog, wrap="none", font=("TkFixedFont", 10))
        y_scroll = ttk.Scrollbar(dialog, orient="vertical", command=text_widget.yview)
        x_scroll = ttk.Scrollbar(dialog, orient="horizontal", command=text_widget.xview)
        text_widget.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")
        text_widget.grid(row=0, column=0, sticky="nsew", padx=(12, 0), pady=(12, 0))
        y_scroll.grid(row=0, column=1, sticky="ns", pady=(12, 0))
        x_scroll.grid(row=1, column=0, sticky="ew", padx=(12, 0))
        ttk.Button(dialog, text="Close", command=dialog.destroy).grid(
            row=2, column=0, columnspan=2, pady=12,
        )
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

    def apply_history_filters(*_):
        visible = filter_results(
            history_entries["all"], query=history_query.get(),
            status=history_status_filter.get(), engine=history_engine_filter.get(),
        )
        history_entries["visible"] = visible
        history_tree.delete(*history_tree.get_children())
        history_item_paths.clear()
        for entry in visible:
            item_id = history_tree.insert("", "end", values=(
                entry["started_at"], entry["system"], entry["status"], entry["engine"],
                entry["methodology_profile"], entry["models_with_results"],
            ))
            history_item_paths[item_id] = entry["path"]
        history_message.set(f"Showing {len(visible)} of {len(history_entries['all'])} local results.")

    def refresh_history():
        entries, skipped = discover_results(config.RESULTS_DIR)
        history_entries["all"] = entries
        engines = sorted({entry["engine"] for entry in entries})
        history_engine_combo.configure(values=("all", *engines))
        if history_engine_filter.get() not in {"all", *engines}:
            history_engine_filter.set("all")
        apply_history_filters()
        if skipped:
            history_message.set(
                f"Showing {len(history_entries['visible'])} results; ignored {len(skipped)} unreadable/non-result JSON files."
            )

    def evaluate_history_selection():
        try:
            result_path = selected_history_path()
            policy_path = filedialog.askopenfilename(
                title="Choose acceptance policy", filetypes=[("Acceptance policy", "*.json")],
            )
            if not policy_path:
                return
            evaluation = evaluate_policy(
                load_history_result(result_path), load_policy(Path(policy_path)),
            )
            lines = [f"Decision: {evaluation['decision'].upper()}", ""]
            lines.extend(
                f"{item['id']}: {item['status']} (actual={item['actual']}, threshold={item['threshold']}, evidence={item['evidence']})"
                for item in evaluation["rules"]
            )
            show_history_details("Acceptance evaluation", "\n".join(lines))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Acceptance evaluation failed", str(exc), parent=root)

    def export_history_diagnostic():
        try:
            baseline_path, candidate_path = selected_result_paths(
                selected_history_items(), history_item_paths, exact=2,
            )
            baseline = load_history_result(baseline_path)
            candidate = load_history_result(candidate_path)
            if review_outbound_metadata(
                    baseline, "diagnostic baseline", allow_aliases=False) is None:
                return
            if review_outbound_metadata(
                    candidate, "diagnostic candidate", allow_aliases=False) is None:
                return
            destination = filedialog.asksaveasfilename(
                title="Export vendor diagnostic", defaultextension=".labdiag",
                filetypes=[("Local AI Bench diagnostic", "*.labdiag")],
            )
            if not destination:
                return
            write_vendor_diagnostic(baseline_path, candidate_path, Path(destination))
            messagebox.showinfo("Vendor diagnostic created", destination, parent=root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Vendor diagnostic failed", str(exc), parent=root)

    def inspect_history_recovery(action="inspect"):
        try:
            result_path = selected_history_path()
        except ValueError as exc:
            messagebox.showerror("Recovery selection", str(exc), parent=root)
            return
        history_message.set(f"Verifying recovery identity for {result_path.name}…")

        def worker():
            try:
                report = inspect_recovery(result_path)
                error = None
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                if action == "fork":
                    try:
                        report, error = fork_review_report(result_path), None
                    except (OSError, KeyError, ValueError, json.JSONDecodeError) as fork_exc:
                        report, error = None, str(fork_exc)
                else:
                    report, error = None, str(exc)

            def finish():
                if error or report is None:
                    history_message.set("Recovery inspection failed.")
                    messagebox.showerror("Recovery inspection failed",
                                         error or "No recovery report was produced.", parent=root)
                    return
                history_message.set(
                    f"Recovery decision for {result_path.name}: {report['action']}"
                )
                if action == "inspect":
                    show_history_details("Recovery inspection", format_recovery_inspection(report))
                    return
                if action in {"resume", "retry"} and not report["can_resume"]:
                    show_history_details("Fork required", format_recovery_inspection(report))
                    return
                if action == "resume":
                    start_history_recovery(result_path, report)
                elif action == "retry":
                    start_history_retry(result_path, report)
                else:
                    start_history_fork(result_path, report)

            root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def start_history_recovery(result_path, report):
        nonlocal process, active_process_kind, active_result_paths
        if process is not None and process.poll() is None:
            messagebox.showerror("Benchmark active", "Stop the active process first.", parent=root)
            return
        plan = load_run_plan(result_path)
        unsupported = [stage for stage in plan.stage_order if stage not in {
            "llm", "conv", "llamabench", "conc_tool", "conc_chat",
        }]
        if unsupported:
            messagebox.showerror(
                "Recovery unavailable",
                "This saved plan contains stages without durable recovery: "
                + ", ".join(unsupported), parent=root,
            )
            return
        detail = format_recovery_inspection(report)
        if not messagebox.askyesno(
            "Resume stopped benchmark",
            f"{detail}\n\nResume the remaining journal-owned work in this result?",
            parent=root,
        ):
            return
        command = recovery_executor_command(result_path)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if platform.system() == "Windows" else 0
        try:
            process, control_path = launch_controlled_process(
                command, creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror("Recovery could not start", str(exc), parent=root)
            return
        begin_process_control(control_path)
        active_process_kind = "recovery"
        active_result_paths = [Path(result_path).resolve()]
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set("Recovery is running. Completed evidence is preserved.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(plan.stage_order, recovery_progress_entries(plan),
                             engines=[plan.engine_name])
        threading.Thread(target=read_process, args=(process,), daemon=True).start()

    def start_history_fork(source_path, report):
        nonlocal process, active_process_kind, pending_fork_source, active_result_paths
        if process is not None and process.poll() is None:
            messagebox.showerror("Benchmark active", "Stop the active process first.", parent=root)
            return
        plan = load_run_plan(source_path)
        unsupported = [stage for stage in plan.stage_order if stage not in {
            "llm", "conv", "llamabench", "conc_tool", "conc_chat",
        }]
        destination = filedialog.asksaveasfilename(
            title="Save forked benchmark", defaultextension=".json",
            initialdir=str(config.RESULTS_DIR),
            initialfile=f"{source_path.stem}_fork.json",
            filetypes=[("JSON results", "*.json")],
        )
        if not destination:
            return
        output_path = Path(destination).resolve()
        detail = format_recovery_inspection(report)
        if not messagebox.askyesno(
            "Fork benchmark plan",
            f"{detail}\n\nRun this saved plan from the beginning as a new result? "
            "The source result will not be changed.", parent=root,
        ):
            return
        if unsupported:
            try:
                state = frontend_state_from_run_plan(plan, collect_options())
                state["gui_options"]["out"] = str(output_path)
                apply_frontend_state(state)
            except (KeyError, ValueError) as exc:
                messagebox.showerror("Fork unavailable", str(exc), parent=root)
                return
            pending_fork_source = source_path
            notebook.select(config_tab)
            root.after(0, start_run)
            return
        command = fork_executor_command(source_path, output_path)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if platform.system() == "Windows" else 0
        try:
            process, control_path = launch_controlled_process(
                command, creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror("Fork could not start", str(exc), parent=root)
            return
        begin_process_control(control_path)
        active_process_kind = "fork"
        active_result_paths = [output_path]
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set("Forked run is active. The source evidence remains unchanged.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(plan.stage_order, recovery_progress_entries(plan),
                             engines=[plan.engine_name])
        threading.Thread(target=read_process, args=(process,), daemon=True).start()

    def choose_retry_cases(candidates):
        by_stage = {}
        for candidate in candidates:
            by_stage.setdefault(candidate["stage"], []).append(candidate)
        if not by_stage:
            messagebox.showinfo("Selected retry", "No cases are retry-eligible.", parent=root)
            return []
        dialog = tk.Toplevel(root)
        dialog.title("Select cases to retry")
        dialog.transient(root)
        dialog.grab_set()
        shell = ttk.Frame(dialog, padding=16)
        shell.pack(fill="both", expand=True)
        ttk.Label(
            shell, text="Retry only the chosen cases. Other incomplete evidence remains unchanged.",
            wraplength=520,
        ).pack(anchor="w", pady=(0, 8))
        stage_var = tk.StringVar(value=next(iter(by_stage)))
        stage_picker = ttk.Combobox(
            shell, textvariable=stage_var, values=list(by_stage), state="readonly",
        )
        stage_picker.pack(fill="x", pady=(0, 8))
        case_list = tk.Listbox(shell, selectmode="extended", width=72, height=12)
        case_list.pack(fill="both", expand=True)

        def refresh_cases(_event=None):
            case_list.delete(0, "end")
            for candidate in by_stage[stage_var.get()]:
                case_list.insert("end", f"{candidate['label']} — {candidate['state']}")

        selected = []

        def accept():
            selected.extend(
                by_stage[stage_var.get()][index] for index in case_list.curselection()
            )
            if not selected:
                messagebox.showerror("Selected retry", "Select at least one case.", parent=dialog)
                return
            dialog.destroy()

        stage_picker.bind("<<ComboboxSelected>>", refresh_cases)
        refresh_cases()
        buttons = ttk.Frame(shell)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="Retry Selected", command=accept).pack(side="right", padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        root.wait_window(dialog)
        return selected

    def start_history_retry(result_path, report):
        nonlocal process, active_process_kind, active_result_paths
        if process is not None and process.poll() is None:
            messagebox.showerror("Benchmark active", "Stop the active process first.", parent=root)
            return
        selected = choose_retry_cases(report.get("retryable_cases", []))
        if not selected:
            return
        if not messagebox.askyesno(
            "Retry selected cases",
            f"Retry {len(selected)} selected case(s)? Completed and unselected evidence will not rerun.",
            parent=root,
        ):
            return
        plan = load_run_plan(result_path)
        command = retry_executor_command(
            result_path, [candidate["case_id"] for candidate in selected],
        )
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if platform.system() == "Windows" else 0
        try:
            process, control_path = launch_controlled_process(
                command, creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror("Selected retry could not start", str(exc), parent=root)
            return
        begin_process_control(control_path)
        active_process_kind = "retry"
        active_result_paths = [Path(result_path).resolve()]
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set("Selected retry is running. Unselected evidence remains unchanged.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(
            [selected[0]["stage"]],
            recovery_progress_entries(plan, {candidate["model"] for candidate in selected}),
            engines=[plan.engine_name],
        )
        threading.Thread(target=read_process, args=(process,), daemon=True).start()

    ttk.Button(history_filters, text="Refresh", command=refresh_history).pack(side="right")
    ttk.Button(
        history_review_actions, text="Open in Dashboard", command=open_history_in_dashboard,
    ).pack(side="left")
    ttk.Button(
        history_review_actions, text="Delete", command=delete_history_selection,
    ).pack(side="left", padx=(8, 0))
    ttk.Button(history_review_actions, text="Evaluate Policy", command=evaluate_history_selection).pack(
        side="left", padx=(8, 0),
    )
    ttk.Button(history_review_actions, text="Export Diagnostic", command=export_history_diagnostic).pack(
        side="left", padx=(8, 0),
    )
    ttk.Button(
        history_recovery_actions, text="Inspect Recovery",
        command=lambda: inspect_history_recovery("inspect"),
    ).pack(side="left")
    ttk.Button(
        history_recovery_actions, text="Resume", command=lambda: inspect_history_recovery("resume"),
    ).pack(side="left", padx=(8, 0))
    ttk.Button(
        history_recovery_actions, text="Retry Cases", command=lambda: inspect_history_recovery("retry"),
    ).pack(side="left", padx=(8, 0))
    ttk.Button(
        history_recovery_actions, text="Fork", command=lambda: inspect_history_recovery("fork"),
    ).pack(side="left", padx=(8, 0))
    history_query.trace_add("write", apply_history_filters)
    history_status_filter.trace_add("write", apply_history_filters)
    history_engine_filter.trace_add("write", apply_history_filters)
    refresh_history()

    def apply_control_values(values: dict) -> None:
        for name, variable in test_vars.items():
            variable.set(values["tests"].get(name, False))
        for name, variable in model_vars.items():
            variable.set(values["models"].get(name, False))
        if values.get("engine"):
            set_selected_engines(parse_engine_selection(values["engine"]))
        cap_var.set(values["max_prompt_tokens"])
        for value, variable in tg_vars.items():
            variable.set(value in values["tg_tokens"])
        for key, value in values["options"].items():
            option_vars[key].set(value)

    defaults_for_display = default_control_values(
        default_tests, default_models, selected_engine, detected_comfyui,
    )
    applying_configuration = [False]

    def control_signature() -> tuple:
        return (
            tuple(variable.get() for variable in test_vars.values()),
            tuple(variable.get() for variable in model_vars.values()),
            engine_var.get(), cap_var.get(),
            tuple(variable.get() for variable in tg_vars.values()),
            tuple(variable.get() for variable in option_vars.values()),
        )

    last_control_signature = [control_signature()]

    def apply_named_preset(name: str) -> None:
        if name == CUSTOM_PRESET:
            return
        available = {entry.value for entry in custom_tests if entry.available}
        applying_configuration[0] = True
        try:
            apply_control_values(preset_control_values(name, available, defaults_for_display))
        finally:
            applying_configuration[0] = False

    def mark_custom(*_) -> None:
        signature = control_signature()
        changed = signature != last_control_signature[0]
        last_control_signature[0] = signature
        if not changed:
            return
        updated = preset_after_control_change(preset_var.get(), applying_configuration[0])
        if updated != preset_var.get():
            preset_var.set(updated)

    def select_preset(*_) -> None:
        apply_named_preset(preset_var.get())

    def update_advanced() -> None:
        visible = advanced_var.get()
        for box in (execution_box, paths_box):
            box.grid() if visible else box.grid_remove()

    def apply_engine_availability(*_) -> None:
        """Mirror the engine checkboxes into engine_var, then disable and uncheck any
        model none of the selected engines can run."""
        if restoring_engines[0]:  # a batch update is mid-flight — set_selected_engines re-runs this once done
            return
        chosen = [name for name in available_engines if engine_check_vars[name].get()]
        if not chosen:  # never leave a run with no engine at all
            chosen = [available_engines[0]]
            engine_check_vars[chosen[0]].set(True)
        engine_var.set(format_engine_selection(chosen))
        engine_note.set(
            f"Runs the full selection once per engine ({len(chosen)} passes, "
            f"{len(chosen)} results files)." if len(chosen) > 1
            else "Only installed engines are listed. Models this engine cannot run are disabled."
        )
        runnable = {}
        for name in chosen:
            for value, ok_here in models_runnable_by(custom_models, name, model_owners).items():
                runnable[value] = runnable.get(value, False) or ok_here
        for value, widget in model_widgets.items():
            available = runnable.get(value, True)
            widget.configure(state="normal" if available else "disabled")
            if not available and model_vars[value].get():
                model_vars[value].set(False)

    for _engine_var in engine_check_vars.values():
        _engine_var.trace_add("write", apply_engine_availability)
    apply_engine_availability()

    def refresh_imported_models(selected_tag: str) -> None:
        previous_values = set(model_vars)
        previous_selected = {name for name, variable in model_vars.items() if variable.get()}
        refreshed = {
            name: build_model_inventory(get_engine(name), config.COMFYUI_MODELS_DIR)
            for name in available_engines
        }
        merged, owners = merge_model_inventories(refreshed)
        engine_inventories.clear()
        engine_inventories.update(refreshed)
        inventory.clear()
        inventory.update(merged)
        model_owners.clear()
        model_owners.update(owners)
        refreshed_tests = {entry.value: entry for entry in build_test_entries(inventory)}
        for entry in custom_tests:
            entry.available = refreshed_tests[entry.value].available
            test_widgets[entry.value].configure(state="normal" if entry.available else "disabled")
            test_labels[entry.value].configure(
                text=entry.label if entry.available else f"{entry.label} (model not installed)",
            )
        rebuilt = build_model_entries(
            inventory, [entry.value for entry in custom_tests if entry.available],
        )
        custom_models[:] = rebuilt
        selected, dropped, added, defaults = reconcile_imported_model_state(
            previous_values, previous_selected, custom_model_defaults, rebuilt, selected_tag,
        )
        for value in dropped:
            model_vars.pop(value)
        custom_model_defaults.clear()
        custom_model_defaults.update(defaults)
        for entry in rebuilt:
            if entry.value in added:
                model_vars[entry.value] = tk.BooleanVar(value=False)
                model_vars[entry.value].trace_add("write", mark_custom)
            model_vars[entry.value].set(entry.value in selected)
        render_model_rows()
        apply_engine_availability()

    def open_model_import_dialog() -> None:
        show_model_import_dialog(
            root=root, tk=tk, ttk=ttk, messagebox=messagebox,
            available_engines=available_engines, engine_factory=get_engine,
            on_imported=refresh_imported_models,
        )
    preset_var.trace_add("write", select_preset)
    for variable in (
            *test_vars.values(), *model_vars.values(), engine_var, cap_var,
            *tg_vars.values(), *option_vars.values()):
        variable.trace_add("write", mark_custom)
    advanced_var.trace_add("write", lambda *_: update_advanced())
    for entry in custom_tests:
        if not entry.available:
            test_widgets[entry.value].configure(state="disabled")
    if preset_var.get() != CUSTOM_PRESET:
        apply_named_preset(preset_var.get())

    def scroll_form(event):
        widget = root.winfo_containing(root.winfo_pointerx(), root.winfo_pointery())
        current = widget
        while current is not None and current not in {canvas, form}:
            current = getattr(current, "master", None)
        if current is None:
            return None
        units = mousewheel_scroll_units(
            delta=getattr(event, "delta", 0), button=getattr(event, "num", 0),
            platform_name=root.tk.call("tk", "windowingsystem"),
        )
        if units:
            canvas.yview_scroll(units, "units")
        return "break"

    root.bind_all("<MouseWheel>", scroll_form)
    root.bind_all("<Button-4>", scroll_form)
    root.bind_all("<Button-5>", scroll_form)

    process = None
    active_process_kind = None
    active_result_paths: list[Path] = []
    pending_fork_source = None
    process_control_path = None
    process_paused = False

    def begin_process_control(control_path):
        nonlocal process_control_path, process_paused
        process_control_path = control_path
        process_paused = False
        pause_button.configure(text="Pause", state="normal")

    def finish_process_control():
        nonlocal process_control_path, process_paused
        if process_control_path is not None:
            try:
                process_control_path.unlink(missing_ok=True)
            except OSError:
                pass
        process_control_path = None
        process_paused = False
        pause_button.configure(text="Pause", state="disabled")
    output_queue = queue.Queue()
    progress_window = None
    stage_progress_vars = {}
    model_progress_vars = {}
    progress_engines = [""]
    progress_summary_vars = {}
    progress_resource_vars = {}
    progress_remaining_var = tk.StringVar(value="Remaining time: calibrating")
    progress_metrics = {}
    progress_started_at = None
    gpu_sample = {
        "usage": None, "memory": None, "vram": None, "next_at": 0.0,
        "running": False, "generation": 0,
    }
    system_memory_baseline = [0.0]
    has_discrete_vram = [False]

    def show_progress_window(tests, entries, engines=None):
        nonlocal progress_window, stage_progress_vars, model_progress_vars
        nonlocal progress_metrics, progress_started_at
        gpu_sample.update({
            "usage": None, "memory": None, "vram": None, "next_at": 0.0, "running": False,
            "generation": gpu_sample["generation"] + 1,
        })
        has_discrete_vram[0] = show_vram_usage(configured_gpu_devices(setup))
        system_memory_baseline[0] = system_memory_usage()[0]
        if progress_window is not None and progress_window.winfo_exists():
            progress_window.destroy()
        progress_window = tk.Toplevel(root)
        progress_window.title(f"Local AI Bench v{config.VERSION} Progress")
        progress_window.geometry("460x640")
        progress_window.minsize(380, 300)
        progress_window.attributes("-topmost", True)
        progress_window.protocol("WM_DELETE_WINDOW", progress_window.withdraw)
        shell = ttk.Frame(progress_window, padding=18)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Benchmark progress", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            shell, text="Live workload/model status, measurement quality, resources, and estimated remaining time.",
            wraplength=390,
        ).pack(anchor="w", pady=(2, 12))
        stage_progress_vars = {}
        model_progress_vars = {}
        labels = TEST_STAGE_LABELS
        selected = [entry for entry in entries if entry.checked]
        total_models = sum(
            1 for stage in STAGE_ORDER if stage in tests for entry in selected
            if ((stage == "emb" and entry.kind == "embedding")
                or (stage == "img" and entry.kind == "image")
                or (stage in LLM_BACKED_TESTS and entry.kind in {"llm", "custom"}))
        )
        progress_metrics = {
            "total_models": total_models, "finished_models": set(), "usable_models": set(),
            "retries": 0, "valid": 0, "invalid": 0,
        }
        progress_started_at = time.monotonic()
        summary_box = ttk.LabelFrame(shell, text="Run summary", padding=(10, 6))
        summary_box.pack(fill="x", pady=(0, 8))
        summary_box.columnconfigure(1, weight=1)
        progress_summary_vars.clear()
        for row, (label, value) in enumerate(progress_summary_rows(progress_metrics).items()):
            ttk.Label(summary_box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=1)
            variable = tk.StringVar(value=value)
            progress_summary_vars[label] = variable
            ttk.Label(summary_box, textvariable=variable).grid(row=row, column=1, sticky="w", pady=1)
        resource_box = ttk.LabelFrame(shell, text="Resources", padding=(10, 6))
        resource_box.pack(fill="x", pady=(0, 8))
        resource_box.columnconfigure(1, weight=1)
        progress_resource_vars.clear()
        resource_labels = ["CPU", "Process RAM", "System RAM", "GPU"]
        if has_discrete_vram[0]:
            resource_labels.append("VRAM")
        for row, label in enumerate(resource_labels):
            ttk.Label(resource_box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=1)
            variable = tk.StringVar(value="Starting…")
            progress_resource_vars[label] = variable
            ttk.Label(resource_box, textvariable=variable).grid(row=row, column=1, sticky="w", pady=1)
        progress_remaining_var.set("Remaining time: calibrating")
        ttk.Label(shell, textvariable=progress_remaining_var).pack(anchor="w", pady=(0, 8))
        status_shell = ttk.Frame(shell)
        status_shell.pack(fill="both", expand=True)
        status_canvas = tk.Canvas(status_shell, highlightthickness=0)
        status_scrollbar = ttk.Scrollbar(status_shell, orient="vertical", command=status_canvas.yview)
        status_list = ttk.Frame(status_canvas)
        status_window = status_canvas.create_window((0, 0), window=status_list, anchor="nw")
        status_list.bind(
            "<Configure>", lambda _event: status_canvas.configure(scrollregion=status_canvas.bbox("all")),
        )
        status_canvas.bind(
            "<Configure>", lambda event: status_canvas.itemconfigure(status_window, width=event.width),
        )
        status_canvas.configure(yscrollcommand=status_scrollbar.set)
        status_canvas.pack(side="left", fill="both", expand=True)
        status_scrollbar.pack(side="right", fill="y")

        def scroll_status(event):
            units = mousewheel_scroll_units(
                delta=getattr(event, "delta", 0), button=getattr(event, "num", 0),
                platform_name=root.tk.call("tk", "windowingsystem"),
            )
            if units:
                status_canvas.yview_scroll(units, "units")
            return "break"

        progress_window.bind("<MouseWheel>", scroll_status)
        progress_window.bind("<Button-4>", scroll_status)
        progress_window.bind("<Button-5>", scroll_status)
        # One full section set per engine, in the order the run executes them.
        run_engines = list(engines or parse_engine_selection(engine_var.get()))
        progress_engines[:] = run_engines
        for engine_index, engine_name in enumerate(run_engines):
            skipped_here = set(engine_incompatible_tests(tests, engine_name))
            for stage in (key for key in STAGE_ORDER if key in tests):
                # Images don't depend on the engine — benchmark.py runs them on the first pass only.
                if stage == "img" and engine_index > 0:
                    continue
                # llamabench/vllmbench shell out to one specific engine's own binary.
                if stage in skipped_here:
                    continue
                row = ttk.Frame(status_list)
                row.pack(fill="x", pady=(10, 2))
                heading = labels.get(stage, stage)
                if len(run_engines) > 1:
                    heading = f"{heading} — {engine_name}"
                ttk.Label(row, text=heading, font=("TkDefaultFont", 10, "bold")).pack(
                    side="left", anchor="w",
                )
                stage_progress_vars[(engine_name, stage)] = tk.StringVar(value="○ Queued")
                ttk.Label(row, textvariable=stage_progress_vars[(engine_name, stage)]).pack(
                    side="right", anchor="e")
                if stage == "emb":
                    stage_models = [entry for entry in selected if entry.kind == "embedding"]
                elif stage == "img":
                    stage_models = [entry for entry in selected if entry.kind == "image"]
                elif stage in LLM_BACKED_TESTS:
                    stage_models = [entry for entry in selected if entry.kind in {"llm", "custom"}]
                else:
                    stage_models = []
                for entry in stage_models:
                    model_row = ttk.Frame(status_list)
                    model_row.pack(fill="x", padx=(14, 0), pady=2)
                    ttk.Label(model_row, text=entry.label, wraplength=270).pack(
                        side="left", anchor="w", fill="x", expand=True,
                    )
                    variable = tk.StringVar(value="○ Queued")
                    model_progress_vars[(engine_name, stage, entry.label)] = variable
                    ttk.Label(model_row, textvariable=variable).pack(side="right", anchor="e")
        progress_window.lift()

    def update_progress(event):
        nonlocal progress_metrics
        progress_metrics = update_progress_metrics(progress_metrics, event)
        completed = len(progress_metrics["finished_models"])
        for label, value in progress_summary_rows(progress_metrics).items():
            progress_summary_vars[label].set(value)
        if event["kind"] == "measurement":
            return
        engine_name = progress_event_engine(event, progress_engines)
        if engine_name is None:
            return
        if event["kind"] == "model":
            variable = model_progress_vars.get((engine_name, event["stage"], event["model"]))
        else:
            variable = stage_progress_vars.get((engine_name, event["stage"]))
        if variable is None:
            return
        variable.set({
            "running": "▶ Running", "complete": "✓ Complete",
            "skipped": "— Skipped",
            "failed": "✕ Failed", "interrupted": "■ Interrupted",
        }[event["status"]])
        if event["kind"] == "stage" and event["status"] in {"complete", "failed", "interrupted"}:
            for (row_engine, stage, _), model_var in model_progress_vars.items():
                if (row_engine == engine_name and stage == event["stage"]
                        and model_var.get() in {"○ Queued", "▶ Running"}):
                    model_var.set("— Not run" if event["status"] != "interrupted" else "■ Interrupted")

    def append_log(text):
        log_text.configure(state="normal")
        log_text.insert("end", text)
        log_text.see("end")
        log_text.configure(state="disabled")

    def poll_output():
        nonlocal process, active_process_kind
        try:
            while True:
                kind, value = output_queue.get_nowait()
                if kind == "line":
                    append_log(value)
                elif kind == "progress":
                    update_progress(value)
                else:
                    process = None
                    if value == 0:
                        try:
                            write_run_logs(
                                current_log(), completed_result_paths(current_log()) or active_result_paths,
                            )
                        except OSError as exc:
                            append_log(f"\nCould not save Run Log: {exc}\n")
                    finish_process_control()
                    stop_button.configure(state="disabled")
                    start_button.configure(state="normal")
                    if active_process_kind in {"recovery", "retry", "fork"}:
                        label = {
                            "recovery": "Recovery", "retry": "Selected retry",
                            "fork": "Forked run",
                        }[active_process_kind]
                        run_status.set(
                            f"{label} completed successfully. Results are ready to review."
                            if value == 0 else
                            f"{label} stopped with exit code {value}. Preserved evidence remains available."
                        )
                    else:
                        run_status.set(format_run_outcome(value))
                    active_process_kind = None
                    refresh_history()
                    for variable in stage_progress_vars.values():
                        if variable.get() in {"○ Queued", "▶ Running"}:
                            variable.set("— Not run" if value else "✓ Complete")
                    for variable in model_progress_vars.values():
                        if variable.get() in {"○ Queued", "▶ Running"}:
                            variable.set("— Not run")
        except queue.Empty:
            pass
        if process is not None and process.poll() is None and progress_started_at is not None:
            now = time.monotonic()
            elapsed = now - progress_started_at
            completed = len(progress_metrics.get("finished_models", ()))
            remaining = estimate_remaining_seconds(
                elapsed, completed, progress_metrics.get("total_models", 0),
            )
            estimate = "calibrating" if remaining is None else f"about {remaining // 60}m {remaining % 60}s"
            usage = process_resource_usage(process.pid)
            if now >= gpu_sample["next_at"] and not gpu_sample["running"]:
                gpu_sample["running"] = True
                gpu_sample["next_at"] = now + 2.0
                generation = gpu_sample["generation"]
                sampled_pid = process.pid

                def sample_gpu():
                    value = query_gpu_usage()
                    memory = query_gpu_process_memory(sampled_pid)
                    vram = query_vram_usage() if has_discrete_vram[0] else None
                    if gpu_sample["generation"] == generation:
                        gpu_sample["usage"] = value
                        gpu_sample["memory"] = memory
                        gpu_sample["vram"] = vram
                        gpu_sample["running"] = False

                threading.Thread(target=sample_gpu, daemon=True).start()
            resources = resource_usage_rows(
                usage, system_memory_usage(), system_memory_baseline[0],
                gpu_sample["usage"], gpu_sample["memory"], gpu_sample["vram"],
                has_discrete_vram[0],
            )
            for label, value in resources.items():
                progress_resource_vars[label].set(value)
            progress_remaining_var.set(f"Remaining time: {estimate}")
        root.after(100, poll_output)

    def read_process(proc):
        for line in proc.stdout:
            progress = parse_progress_line(line)
            output_queue.put(("progress", progress) if progress else ("line", line))
        output_queue.put(("done", proc.wait()))

    def collect_options():
        values = dict(GUI_OPTION_DEFAULTS)
        for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget"):
            try:
                values[key] = int(option_vars[key].get())
            except ValueError:
                values[key] = option_vars[key].get()
        values["gpu_split_mode"] = option_vars["gpu_split_mode"].get()
        for key in ("cpu_only", "force_all", "retry_crashed_models", "offline"):
            values[key] = option_vars[key].get()
        for key in ("out", "comfyui"):
            values[key] = option_vars[key].get().strip()
        return values

    def confirm_plan_preview(preview: str) -> bool:
        dialog = tk.Toplevel(root)
        dialog.title("Review benchmark plan")
        dialog.geometry("760x620")
        dialog.minsize(620, 460)
        dialog.transient(root)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)
        header = ttk.Frame(dialog)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 12))
        ttk.Label(header, text="Review benchmark plan", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header, text="Confirm the resolved workload, measurement settings, and output before starting.",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(dialog)
        body.grid(row=1, column=0, sticky="nsew", padx=20)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        text_widget = tk.Text(body, wrap="word", padx=14, pady=12, borderwidth=1, relief="solid")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        text_widget.tag_configure("heading", font=("TkDefaultFont", 12, "bold"), spacing1=10, spacing3=4)
        text_widget.tag_configure("label", font=("TkDefaultFont", 10, "bold"))
        text_widget.tag_configure("value", lmargin1=12, lmargin2=12, spacing3=5)
        for title, lines in plan_preview_sections(preview):
            text_widget.insert("end", f"{title}\n", "heading")
            for line in lines:
                label, separator, value = line.partition(":")
                if separator:
                    text_widget.insert("end", f"{label}: ", "label")
                    text_widget.insert("end", f"{value.strip()}\n", "value")
                else:
                    text_widget.insert("end", f"{line}\n", "value")
        text_widget.configure(state="disabled")

        confirmed = [False]

        def finish(value: bool) -> None:
            confirmed[0] = value
            dialog.destroy()

        actions = ttk.Frame(dialog)
        actions.grid(row=2, column=0, sticky="e", padx=20, pady=18)
        ttk.Button(actions, text="Cancel", command=lambda: finish(False)).pack(side="left")
        start = ttk.Button(actions, text="Start Benchmark", style="Start.TButton",
                           command=lambda: finish(True))
        start.pack(side="left", padx=(10, 0))
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        dialog.bind("<Escape>", lambda _event: finish(False))
        dialog.bind("<Return>", lambda _event: finish(True))
        dialog.grab_set()
        start.focus_set()
        dialog.lift()
        root.wait_window(dialog)
        return confirmed[0]

    def start_run():
        nonlocal process, active_process_kind, pending_fork_source, active_result_paths
        tests = expand_selected_tests(
            name for name, variable in test_vars.items() if variable.get())
        entries = custom_models
        for entry in entries:
            entry.checked = model_vars[entry.value].get()
        max_prompt = None if cap_var.get() == "No cap" else int(cap_var.get())
        tg_tokens = [value for value, variable in tg_vars.items() if variable.get()]
        gui_options = collect_options()
        errors = validate_gui_options(gui_options)
        if not tests:
            errors.append("Select at least one benchmark test.")
        selection_error = model_selection_error(entries, tests)
        if selection_error:
            errors.append(selection_error)
        if TG_TOKEN_TESTS & set(tests) and not tg_tokens:
            errors.append("Select at least one llama-bench generation size.")
        custom_comfyui = (normalize_comfyui_dir(Path(gui_options["comfyui"]))
                          if gui_options["comfyui"] else found_comfyui)
        errors.extend(workload_preflight_errors(tests, detected_tools, custom_comfyui is not None))
        if errors:
            messagebox.showerror("Check benchmark options", "\n".join(errors), parent=root)
            return
        effective_options = gui_options or custom_option_defaults(detected_comfyui)
        preview = build_plan_preview(
            engine=engine_var.get(), tests=tests, entries=entries, options=effective_options,
            max_prompt_tokens=max_prompt, tg_tokens=tg_tokens,
            comfyui_dir=custom_comfyui or detected_comfyui,
        )
        if not confirm_plan_preview(preview):
            pending_fork_source = None
            return
        state = build_frontend_state(
            engine_var.get(), tests, entries, max_prompt_tokens=max_prompt,
            tg_tokens=tg_tokens if TG_TOKEN_TESTS & set(tests) else None,
            gui_options=gui_options, selected_preset=preset_var.get(),
        )
        if not save_frontend_state(state, FRONTEND_STATE_PATH):
            if not messagebox.askyesno("Settings not saved", "The configuration could not be saved. Run it anyway?", parent=root):
                return
        command = build_benchmark_command(
            engine_var.get(), detected_comfyui, tests, entries,
            max_prompt_tokens=max_prompt if MAX_PROMPT_TOKEN_TESTS & set(tests) else None,
            tg_tokens=tg_tokens if TG_TOKEN_TESTS & set(tests) else None,
            gui_options=gui_options,
        )
        if pending_fork_source is not None:
            command.extend(["--fork-plan", str(pending_fork_source)])
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if platform.system() == "Windows" else 0
        try:
            process, control_path = launch_controlled_process(
                command, creationflags=creationflags,
            )
        except OSError as exc:
            pending_fork_source = None
            messagebox.showerror("Benchmark could not start", str(exc), parent=root)
            return
        begin_process_control(control_path)
        pending_fork_source = None
        active_process_kind = "benchmark"
        explicit_output = gui_options.get("out", "").strip()
        active_result_paths = [Path(explicit_output).expanduser().resolve()] if explicit_output else []
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set("Benchmark is running. Results are checkpointed throughout the run.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(tests, entries, engines=parse_engine_selection(engine_var.get()))
        threading.Thread(target=read_process, args=(process,), daemon=True).start()

    def stop_run():
        if process is None or process.poll() is not None:
            return
        if process_control_path is not None:
            try:
                write_pause_state(process_control_path, "running")
            except OSError:
                pass
        run_status.set("Stopping benchmark safely…")
        if platform.system() == "Windows":
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
        else:
            process.send_signal(signal.SIGINT)

    def toggle_pause():
        nonlocal process_paused
        if process is None or process.poll() is not None or process_control_path is None:
            return
        next_paused = not process_paused
        try:
            write_pause_state(process_control_path, "paused" if next_paused else "running")
        except OSError as exc:
            messagebox.showerror("Pause unavailable", str(exc), parent=root)
            return
        process_paused = next_paused
        pause_button.configure(text="Resume" if process_paused else "Pause")
        run_status.set(
            "Pause requested; the benchmark will pause at the next safe case boundary."
            if process_paused else "Benchmark resumed. Results remain checkpointed throughout the run."
        )

    def close_window():
        if process is not None and process.poll() is None:
            if not messagebox.askyesno("Benchmark running", "Stop the benchmark and close?", parent=root):
                return
            stop_run()
        root.destroy()

    start_button.configure(command=start_run)
    stop_button.configure(command=stop_run)
    pause_button.configure(command=toggle_pause)
    root.protocol("WM_DELETE_WINDOW", close_window)
    update_advanced()
    root.after(100, poll_output)
    root.after(150, lambda: (root.lift(), root.attributes("-topmost", True), root.focus_force(),
                             root.after(400, lambda: root.attributes("-topmost", False))))
    root.mainloop()
    return 0


def main():  # pragma: no cover — desktop entrypoint
    raise SystemExit(run_benchmark_gui())


if __name__ == "__main__":
    main()
