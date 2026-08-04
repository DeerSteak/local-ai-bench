#!/usr/bin/env python3
"""Single-screen Tk launcher for Local AI Bench."""

import os
import json
import platform
import queue
import signal
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import config
import psutil
from acceptance_policy import evaluate_policy, load_policy
from benchmark_frontend import (
    FRONTEND_STATE_PATH,
    GUI_OPTION_DEFAULTS,
    LLM_BACKED_TESTS,
    MAX_PROMPT_TOKEN_OPTIONS,
    MAX_PROMPT_TOKEN_TESTS,
    TEST_DEFINITIONS,
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
    load_frontend_state,
    model_selection_error,
    save_frontend_state,
    validate_gui_options,
)
from benchmark_presets import (
    build_portable_preset, compare_portable_presets, duplicate_portable_preset,
    load_portable_preset, save_portable_preset,
)
from benchmark_project import (
    PROJECT_WORKFLOWS, build_project, load_project, project_frontend_state, save_project,
)
from comfyui_installation import find_comfyui_installation, normalize_comfyui_dir
from decision_report import load_result, report_output_paths, write_html_report, write_pdf_report
from engines import engine_names, get_engine
from llamacpp_tools import find_llamacpp_tool
from run_plan import load_run_plan
from result_bundle import export_result_bundle, import_result_bundle, verify_result_bundle
from result_history import compare_results, discover_results, filter_results, load_result as load_history_result
from recovery_inspector import inspect_recovery
from support_bundle import export_support_bundle, preview_support_bundle
from model_inventory import build_model_inventory
from models import LLM_MODELS
from orchestration import STAGE_ORDER
from outbound_metadata import outbound_metadata_preview, prepare_outbound_result
from pause_control import PAUSE_CONTROL_ENV, create_pause_control, write_pause_state
from progress_events import PROGRESS_PREFIX
from shared import Shared
from setup_config import configured_comfyui_dir, load_setup_config
from tk_utils import mousewheel_scroll_units
from vendor_diagnostic import write_vendor_diagnostic


def effective_gui_options(state: dict | None) -> dict:
    options = state.get("gui_options") if state else None
    return dict(options) if options is not None else dict(GUI_OPTION_DEFAULTS)


def open_path_command(path: Path, system: str) -> list[str]:
    if system == "Darwin":
        return ["open", str(path)]
    if system == "Windows":
        return ["explorer", str(path)]
    return ["xdg-open", str(path)]


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
                else {"running", "complete", "failed", "interrupted"})
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
    elif event["kind"] == "model" and event["status"] in {"complete", "failed", "interrupted"}:
        identity = (event["stage"], event["model"])
        finished = set(updated["finished_models"])
        finished.add(identity)
        updated["finished_models"] = finished
        if event.get("usable"):
            usable = set(updated["usable_models"])
            usable.add(identity)
            updated["usable_models"] = usable
    return updated


def estimate_remaining_seconds(elapsed: float, completed: int, total: int) -> int | None:
    if elapsed < 0 or completed <= 0 or total <= completed:
        return 0 if total > 0 and completed >= total else None
    return round((elapsed / completed) * (total - completed))


def process_resource_usage(pid: int, psutil_module=psutil) -> tuple[float, float] | None:
    try:
        parent = psutil_module.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
        cpu = sum(item.cpu_percent(interval=None) for item in processes)
        memory_gb = sum(item.memory_info().rss for item in processes) / (1024 ** 3)
        return cpu, memory_gb
    except (psutil_module.Error, OSError):
        return None


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
    return [python_executable, str(config.SCRIPT_DIR / "scripts" / "recovery_executor.py"),
            str(Path(result_path).resolve())]


def fork_executor_command(source_path: Path, output_path: Path,
                          python_executable=sys.executable) -> list[str]:
    return [
        python_executable, str(config.SCRIPT_DIR / "scripts" / "fork_executor.py"),
        str(Path(source_path).resolve()), str(Path(output_path).resolve()),
    ]


def retry_executor_command(result_path: Path, case_ids: list[str],
                           python_executable=sys.executable) -> list[str]:
    return [
        python_executable, str(config.SCRIPT_DIR / "scripts" / "retry_executor.py"),
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


def advanced_controls_visible(mode: str, requested: bool) -> bool:
    return mode == "custom" and requested


BENCHMARK_PRESETS = {
    "Consumer guidance": {"tests": ["llm", "conv"], "max_prompt_tokens": 32768},
    "Vendor validation": {"tests": ["llm", "conv", "llamabench", "emb", "mcq", "math", "reasoning", "code", "tool", "img"]},
    "Neutral comparison": {"tests": ["llm", "conv", "emb", "img"]},
    "Platform optimized": {"tests": ["llm", "conv", "llamabench", "llamabenchconc"]},
    "Offline / private": {"tests": ["llm", "conv", "emb"]},
    "Quick run": {"tests": ["llm", "emb"], "runs": 1, "max_prompt_tokens": 8192},
    "Full run": {"tests": [name for name, *_ in TEST_DEFINITIONS], "force_all": True},
}


def resolve_preset(name: str, available_tests: set[str]) -> dict:
    preset = BENCHMARK_PRESETS[name]
    return {
        "tests": [test for test in preset["tests"] if test in available_tests],
        "runs": preset.get("runs", config.N_RUNS),
        "max_prompt_tokens": preset.get("max_prompt_tokens"),
        "force_all": preset.get("force_all", False),
    }


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
        f"Broad cases: {model_passes} model-workload passes; contexts, questions, and levels expand within them.",
        f"Model loads: at least {model_passes}; context and concurrency workloads may reload models.",
        "Duration range: minutes to hours; this hardware has no calibrated estimate yet.",
        f"Processes: {', '.join(processes) if processes else 'None'}",
        f"Results: {output}", f"ComfyUI: {comfyui_dir}",
        "Disk use: results JSON plus accuracy sidecars and generated images when selected.",
        "Network use: none expected; all selected models must already be local.",
    ]
    return "\n".join(lines)


def run_benchmark_gui() -> int:  # pragma: no cover — interactive desktop UI
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    saved = load_frontend_state(FRONTEND_STATE_PATH)
    setup = load_setup_config(config.SETUP_CONFIG_PATH)
    found_comfyui = find_comfyui_installation(
        saved_path=configured_comfyui_dir(setup), managed_dir=config.COMFYUI_DIR,
    )
    detected_comfyui = found_comfyui or config.COMFYUI_DIR
    available_engines = engine_names()
    selected_engine = saved["engine"] if saved and saved["engine"] in available_engines else available_engines[0]
    inventory = build_model_inventory(get_engine(selected_engine), config.COMFYUI_MODELS_DIR)
    detected_tools = {name: find_llamacpp_tool(name) for name in (
        "llama-server", "llama-bench", "llama-batched-bench",
    )}
    system_ram_gb = Shared.system_ram_gb()
    discovery = build_discovery_report(
        platform_name=platform.system(), architecture=platform.machine(),
        ram_gb=system_ram_gb, backend=Shared.detect_backend(),
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
    root.title("Local AI Bench")
    root.geometry("1080x820")
    root.minsize(860, 650)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    style = ttk.Style(root)
    style.configure("Title.TLabel", font=("TkDefaultFont", 21, "bold"))
    style.configure("Section.TLabel", font=("TkDefaultFont", 12, "bold"))
    style.configure("Mode.TRadiobutton", font=("TkDefaultFont", 12, "bold"))
    style.configure("Start.TButton", font=("TkDefaultFont", 12, "bold"), padding=(18, 9))

    mode_var = tk.StringVar(value="default")
    advanced_var = tk.BooleanVar(value=False)
    engine_var = tk.StringVar(value=selected_engine)
    test_vars = {entry.value: tk.BooleanVar(value=entry.checked) for entry in custom_tests}
    model_vars = {entry.value: tk.BooleanVar(value=entry.checked) for entry in custom_models}
    cap_var = tk.StringVar(value=str(saved["max_prompt_tokens"]) if saved and saved["max_prompt_tokens"] else "No cap")
    saved_tg = set(saved["tg_tokens"] or config.LLAMABENCH_TG) if saved else set(config.LLAMABENCH_TG)
    tg_vars = {value: tk.BooleanVar(value=value in saved_tg) for value in TG_TOKEN_OPTIONS}
    options = effective_gui_options(saved)
    option_vars = {
        key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value)))
        for key, value in options.items()
    }
    if not option_vars["comfyui"].get():
        option_vars["comfyui"].set(str(detected_comfyui))

    notebook = ttk.Notebook(root)
    notebook.grid(sticky="nsew")
    config_tab = ttk.Frame(notebook, padding=18)
    log_tab = ttk.Frame(notebook, padding=18)
    history_tab = ttk.Frame(notebook, padding=18)
    notebook.add(config_tab, text="Configuration")
    notebook.add(log_tab, text="Run Log")
    notebook.add(history_tab, text="Result History")
    config_tab.columnconfigure(0, weight=1)
    config_tab.rowconfigure(2, weight=1)

    ttk.Label(config_tab, text="Local AI Bench", style="Title.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(
        config_tab,
        text="Choose Default for a reliable standard run, or Custom to configure every practical benchmark option.",
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

    mode_box = ttk.LabelFrame(form, text="Configuration mode", padding=12)
    mode_box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
    ttk.Radiobutton(mode_box, text="Default", value="default", variable=mode_var,
                    style="Mode.TRadiobutton").grid(row=0, column=0, sticky="w", padx=(0, 28))
    ttk.Radiobutton(mode_box, text="Custom", value="custom", variable=mode_var,
                    style="Mode.TRadiobutton").grid(row=0, column=1, sticky="w")
    mode_note = ttk.Label(mode_box, text="Uses the recommended installed-model selection and standard execution settings.")
    mode_note.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

    discovery_box = ttk.LabelFrame(form, text="System inventory and preflight", padding=12)
    discovery_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
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

    custom_frame = ttk.Frame(form)
    custom_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
    custom_frame.columnconfigure(0, weight=1)
    custom_frame.columnconfigure(1, weight=1)
    preset_row = ttk.Frame(custom_frame)
    preset_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    preset_var = tk.StringVar(value="Consumer guidance")
    ttk.Label(preset_row, text="Preset").pack(side="left")
    ttk.Combobox(
        preset_row, state="readonly", textvariable=preset_var,
        values=list(BENCHMARK_PRESETS), width=24,
    ).pack(side="left", padx=(8, 8))
    project_row = ttk.Frame(custom_frame)
    project_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    active_project = {"value": None}
    project_status = tk.StringVar(value="No project loaded")
    ttk.Label(project_row, textvariable=project_status).pack(side="left", padx=(0, 12))
    advanced_toggle = ttk.Checkbutton(
        custom_frame, text="Show advanced execution and path settings", variable=advanced_var,
    )
    advanced_toggle.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

    tests_box = ttk.LabelFrame(custom_frame, text="Tests", padding=12)
    tests_box.grid(row=3, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
    test_widgets = {}
    for row, (name, label, _, _) in enumerate(TEST_DEFINITIONS):
        entry = next(item for item in custom_tests if item.value == name)
        text = label if entry.available else f"{label} (model not installed)"
        widget = ttk.Checkbutton(tests_box, text=text, variable=test_vars[name])
        widget.grid(row=row, column=0, sticky="w")
        ttk.Button(
            tests_box, text="Reset", width=6,
            command=lambda key=name: test_vars[key].set(custom_test_defaults[key]),
        ).grid(row=row, column=1, sticky="e", padx=(8, 0))
        test_widgets[name] = widget
    ttk.Label(
        tests_box, text="Accuracy and concurrency add substantial runtime; native llama-bench tests require their matching tools.",
        wraplength=430,
    ).grid(row=len(TEST_DEFINITIONS), column=0, sticky="w", pady=(8, 0))

    models_box = ttk.LabelFrame(custom_frame, text="Installed models", padding=12)
    models_box.grid(row=3, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
    previous = None
    model_widgets = {}
    row = 0
    for entry in custom_models:
        if entry.section != previous:
            ttk.Label(models_box, text=entry.section, style="Section.TLabel").grid(row=row, column=0, sticky="w", pady=(7, 2))
            row += 1
            previous = entry.section
        widget = ttk.Checkbutton(models_box, text=entry.label, variable=model_vars[entry.value])
        widget.grid(row=row, column=0, sticky="w", padx=(12, 0))
        ttk.Button(
            models_box, text="Reset", width=6,
            command=lambda key=entry.value: model_vars[key].set(custom_model_defaults[key]),
        ).grid(row=row, column=1, sticky="e", padx=(8, 0))
        model_widgets[entry.value] = widget
        row += 1
    model_end_row = row
    ttk.Label(
        models_box, text="Each checked model runs once through every applicable selected workload. Larger models may exceed memory.",
        wraplength=430,
    ).grid(row=row, column=0, sticky="w", pady=(8, 0))

    workload_box = ttk.LabelFrame(custom_frame, text="Workload sizes", padding=12)
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

    execution_box = ttk.LabelFrame(custom_frame, text="Execution", padding=12)
    execution_box.grid(row=5, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
    ttk.Label(execution_box, text="Inference engine").grid(row=0, column=0, sticky="w", pady=2)
    engine_combo = ttk.Combobox(execution_box, state="readonly", textvariable=engine_var,
                                values=available_engines, width=16)
    engine_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=2)
    ttk.Button(
        execution_box, text="Reset", width=6,
        command=lambda: engine_var.set(available_engines[0]),
    ).grid(row=0, column=2, padx=(8, 0))
    labels = (("warmup", f"Warmup runs (default {config.WARMUP_RUNS})"),
              ("runs", f"Measured runs (1–10; default {config.N_RUNS})"),
              ("timeout", f"Run timeout, seconds (default {config.RUN_TIMEOUT})"),
              ("acc_timeout", f"Accuracy timeout, seconds (default {config.ACC_TIMEOUT})"),
              ("acc_token_budget", f"Accuracy token budget (default {config.ACC_TOKEN_BUDGET})"))
    for row, (key, label) in enumerate(labels, 1):
        ttk.Label(execution_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(execution_box, textvariable=option_vars[key], width=12).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=2)
        ttk.Button(
            execution_box, text="Reset", width=6,
            command=lambda option=key: option_vars[option].set(str(GUI_OPTION_DEFAULTS[option])),
        ).grid(row=row, column=2, padx=(8, 0), pady=2)
    ttk.Checkbutton(execution_box, text="CPU-only inference", variable=option_vars["cpu_only"]).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Button(execution_box, text="Reset", width=6, command=lambda: option_vars["cpu_only"].set(False)).grid(row=7, column=2, padx=(8, 0))
    ttk.Checkbutton(execution_box, text="Run slow models instead of skipping", variable=option_vars["force_all"]).grid(row=8, column=0, columnspan=2, sticky="w")
    ttk.Button(execution_box, text="Reset", width=6, command=lambda: option_vars["force_all"].set(False)).grid(row=8, column=2, padx=(8, 0))
    ttk.Checkbutton(execution_box, text="Offline mode (loopback only)", variable=option_vars["offline"]).grid(row=9, column=0, columnspan=2, sticky="w")
    ttk.Button(execution_box, text="Reset", width=6, command=lambda: option_vars["offline"].set(False)).grid(row=9, column=2, padx=(8, 0))
    ttk.Label(
        execution_box, text="More warmups/runs improve repeatability but increase time. CPU-only changes the tested device; force-all can make runs much longer.",
        wraplength=430,
    ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 0))

    paths_box = ttk.LabelFrame(custom_frame, text="Paths", padding=12)
    paths_box.grid(row=5, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
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
        engine_var.set(available_engines[0])
        defaults = custom_option_defaults(detected_comfyui)
        for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget",
                    "cpu_only", "force_all", "offline"):
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

    def apply_preset():
        available = {entry.value for entry in custom_tests if entry.available}
        preset = resolve_preset(preset_var.get(), available)
        for name, variable in test_vars.items():
            variable.set(name in preset["tests"])
        option_vars["runs"].set(str(preset["runs"]))
        option_vars["force_all"].set(preset["force_all"])
        cap_var.set(str(preset["max_prompt_tokens"]) if preset["max_prompt_tokens"] else "No cap")

    def current_custom_state():
        tests = [name for name, variable in test_vars.items() if variable.get()]
        for entry in custom_models:
            entry.checked = model_vars[entry.value].get()
        options = collect_options()
        cap = None if cap_var.get() == "No cap" else int(cap_var.get())
        tg = [value for value, variable in tg_vars.items() if variable.get()]
        return build_frontend_state(
            engine_var.get(), tests, custom_models, max_prompt_tokens=cap,
            tg_tokens=tg, gui_options=options,
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
        mode_var.set("custom")
        if state["engine"] in available_engines:
            engine_var.set(state["engine"])
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

    def apply_portable_preset(portable):
        configuration = portable["configuration"]
        apply_frontend_state({
            "engine": configuration["engine"], "tests": configuration["tests"],
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

    def duplicate_preset():
        path = filedialog.askopenfilename(
            title="Choose preset to duplicate", filetypes=[("Benchmark preset", "*.json")],
        )
        if not path:
            return
        try:
            source = load_portable_preset(Path(path))
            name = simpledialog.askstring("Duplicate preset", "Name the duplicate:", parent=root)
            if name:
                export_preset(duplicate_portable_preset(source, name))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Preset duplication failed", str(exc), parent=root)

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
        selected = {"value": None}
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

    ttk.Button(preset_row, text="Apply", command=apply_preset).pack(side="left")
    ttk.Button(preset_row, text="Export", command=export_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Import", command=import_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Duplicate", command=duplicate_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Compare", command=compare_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Import CLI Plan", command=import_run_plan).pack(side="left", padx=(8, 0))
    ttk.Button(project_row, text="New Project", command=save_current_project).pack(side="left")
    ttk.Button(project_row, text="Open Project", command=open_project).pack(side="left", padx=(8, 0))

    ttk.Button(tests_box, text="Reset Tests", command=reset_tests).grid(
        row=len(TEST_DEFINITIONS) + 1, column=0, sticky="w", pady=(8, 0),
    )
    ttk.Button(models_box, text="Reset Models", command=reset_models).grid(
        row=model_end_row + 1, column=0, sticky="w", pady=(8, 0),
    )
    ttk.Button(workload_box, text="Reset Workload Sizes", command=reset_workload).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(8, 0),
    )
    ttk.Button(execution_box, text="Reset Execution", command=reset_execution).grid(
        row=11, column=0, columnspan=2, sticky="w", pady=(8, 0),
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
    ttk.Button(footer, text="Reset All Custom Options", command=reset_all).pack(side="right", padx=(0, 10))

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
    stop_button = ttk.Button(log_actions, text="Stop Benchmark", state="disabled")
    stop_button.pack(side="right")
    pause_button = ttk.Button(log_actions, text="Pause", state="disabled")
    pause_button.pack(side="right", padx=(0, 8))
    ttk.Button(log_actions, text="Back to Configuration", command=lambda: notebook.select(config_tab)).pack(side="left")
    def open_results_folder():
        output = option_vars["out"].get().strip() if mode_var.get() == "custom" else ""
        folder = Path(output).expanduser().resolve().parent if output else config.RESULTS_DIR
        subprocess.Popen(open_path_command(folder, platform.system()))

    def review_outbound_metadata(result, purpose, *, allow_aliases=True):
        decision = {"value": None}
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

    ttk.Button(log_actions, text="Open Results Folder", command=open_results_folder).pack(
        side="left", padx=(10, 0),
    )
    ttk.Button(log_actions, text="Export Bundle", command=export_bundle).pack(side="left", padx=(10, 0))
    ttk.Button(log_actions, text="Import / Verify", command=import_bundle).pack(side="left", padx=(10, 0))
    ttk.Button(log_actions, text="Create Report", command=create_report).pack(side="left", padx=(10, 0))
    ttk.Button(log_actions, text="Support Bundle", command=export_support).pack(side="left", padx=(10, 0))

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
        show="headings", selectmode="browse",
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
    history_message = tk.StringVar(value="History has not been loaded.")
    ttk.Label(history_tab, textvariable=history_message).grid(row=4, column=0, sticky="w", pady=(8, 0))
    history_entries = {"all": [], "visible": []}
    history_item_paths = {}
    baseline_path = {"value": None}

    def selected_history_path():
        selected = history_tree.selection()
        if not selected:
            raise ValueError("Select one result first.")
        return Path(history_item_paths[selected[0]])

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

    def set_history_baseline():
        try:
            baseline_path["value"] = selected_history_path()
            history_message.set(f"Baseline: {baseline_path['value'].name}")
        except ValueError as exc:
            messagebox.showerror("Baseline selection", str(exc), parent=root)

    def compare_history_selection():
        try:
            candidate_path = selected_history_path()
            if baseline_path["value"] is None:
                raise ValueError("Set a baseline result first.")
            comparison = compare_results(
                load_history_result(baseline_path["value"]), load_history_result(candidate_path),
            )
            lines = [
                "Compatible comparison" if comparison["compatible"] else
                "Blocked comparison: " + ", ".join(comparison["incompatible_fields"]), "",
            ]
            for row in comparison["rows"]:
                before = "missing" if row["baseline"] is None else f"{row['baseline']:.4g}"
                after = "missing" if row["candidate"] is None else f"{row['candidate']:.4g}"
                delta = "—" if row["percent_change"] is None else f"{row['percent_change']:+.2f}%"
                lines.append(f"{row['metric']}: {before} → {after} ({delta})")
            show_history_details("Baseline comparison", "\n".join(lines))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Baseline comparison failed", str(exc), parent=root)

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
            candidate_path = selected_history_path()
            if baseline_path["value"] is None:
                raise ValueError("Set a baseline result first.")
            baseline = load_history_result(baseline_path["value"])
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
            write_vendor_diagnostic(baseline_path["value"], candidate_path, Path(destination))
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
                if error:
                    history_message.set("Recovery inspection failed.")
                    messagebox.showerror("Recovery inspection failed", error, parent=root)
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
        nonlocal process, active_process_kind
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
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
        child_env = begin_process_control()
        process = subprocess.Popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=creationflags, env=child_env,
        )
        active_process_kind = "recovery"
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set("Recovery is running. Completed evidence is preserved.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(plan.stage_order, recovery_progress_entries(plan))
        threading.Thread(target=read_process, args=(process,), daemon=True).start()

    def start_history_fork(source_path, report):
        nonlocal process, active_process_kind, pending_fork_source
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
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
        child_env = begin_process_control()
        process = subprocess.Popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=creationflags, env=child_env,
        )
        active_process_kind = "fork"
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set("Forked run is active. The source evidence remains unchanged.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(plan.stage_order, recovery_progress_entries(plan))
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
        nonlocal process, active_process_kind
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
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
        child_env = begin_process_control()
        process = subprocess.Popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=creationflags, env=child_env,
        )
        active_process_kind = "retry"
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
        )
        threading.Thread(target=read_process, args=(process,), daemon=True).start()

    ttk.Button(history_filters, text="Refresh", command=refresh_history).pack(side="right")
    ttk.Button(history_actions, text="Set Baseline", command=set_history_baseline).pack(side="left")
    ttk.Button(history_actions, text="Compare to Baseline", command=compare_history_selection).pack(
        side="left", padx=(8, 0),
    )
    ttk.Button(history_actions, text="Evaluate Policy", command=evaluate_history_selection).pack(
        side="left", padx=(8, 0),
    )
    ttk.Button(history_actions, text="Export Diagnostic", command=export_history_diagnostic).pack(
        side="left", padx=(8, 0),
    )
    ttk.Button(
        history_actions, text="Inspect Recovery",
        command=lambda: inspect_history_recovery("inspect"),
    ).pack(side="left", padx=(8, 0))
    ttk.Button(
        history_actions, text="Resume", command=lambda: inspect_history_recovery("resume"),
    ).pack(side="left", padx=(8, 0))
    ttk.Button(
        history_actions, text="Retry Cases", command=lambda: inspect_history_recovery("retry"),
    ).pack(side="left", padx=(8, 0))
    ttk.Button(
        history_actions, text="Fork", command=lambda: inspect_history_recovery("fork"),
    ).pack(side="left", padx=(8, 0))
    history_query.trace_add("write", apply_history_filters)
    history_status_filter.trace_add("write", apply_history_filters)
    history_engine_filter.trace_add("write", apply_history_filters)
    refresh_history()

    def walk_widgets(parent):
        for child in parent.winfo_children():
            yield child
            yield from walk_widgets(child)

    def update_mode() -> None:
        custom = mode_var.get() == "custom"
        mode_note.configure(text=(
            "Restores and saves your selections in .benchmark_frontend_state.json."
            if custom else "Uses the recommended installed-model selection and standard execution settings."
        ))
        for widget in walk_widgets(custom_frame):
            try:
                widget.configure(state="normal" if custom else "disabled")
            except tk.TclError:
                pass
        if custom:
            cap_combo.configure(state="readonly")
            engine_combo.configure(state="readonly")
            for entry in custom_tests:
                if not entry.available:
                    test_widgets[entry.value].configure(state="disabled")

    def update_advanced() -> None:
        visible = advanced_controls_visible(mode_var.get(), advanced_var.get())
        for box in (execution_box, paths_box):
            box.grid() if visible else box.grid_remove()

    mode_var.trace_add("write", lambda *_: (update_mode(), update_advanced()))
    advanced_var.trace_add("write", lambda *_: update_advanced())

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
    pending_fork_source = None
    process_control_path = None
    process_paused = False

    def begin_process_control():
        nonlocal process_control_path, process_paused
        process_control_path = create_pause_control(config.RESULTS_DIR)
        process_paused = False
        pause_button.configure(text="Pause", state="normal")
        return {**os.environ, "LOCAL_AI_BENCH_PROGRESS": "1",
                PAUSE_CONTROL_ENV: str(process_control_path)}

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
    progress_summary_var = tk.StringVar(value="")
    progress_resource_var = tk.StringVar(value="")
    progress_metrics = {}
    progress_started_at = None

    def show_progress_window(tests, entries):
        nonlocal progress_window, stage_progress_vars, model_progress_vars
        nonlocal progress_metrics, progress_started_at
        if progress_window is not None and progress_window.winfo_exists():
            progress_window.destroy()
        progress_window = tk.Toplevel(root)
        progress_window.title("Local AI Bench Progress")
        progress_window.geometry("430x520")
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
        labels = {name: label for name, label, _, _ in TEST_DEFINITIONS}
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
        progress_summary_var.set(
            f"Finished: 0/{total_models} models · Usable coverage: 0/{total_models} · Invalid: 0 · Retries: 0"
        )
        progress_resource_var.set("Resources: starting · Remaining time: calibrating")
        ttk.Label(shell, textvariable=progress_summary_var, wraplength=390).pack(anchor="w")
        ttk.Label(shell, textvariable=progress_resource_var, wraplength=390).pack(
            anchor="w", pady=(2, 8),
        )
        for stage in (key for key in STAGE_ORDER if key in tests):
            row = ttk.Frame(shell)
            row.pack(fill="x", pady=(6, 1))
            ttk.Label(row, text=labels.get(stage, stage), font=("TkDefaultFont", 10, "bold")).pack(
                side="left", anchor="w",
            )
            stage_progress_vars[stage] = tk.StringVar(value="○ Queued")
            ttk.Label(row, textvariable=stage_progress_vars[stage]).pack(side="right", anchor="e")
            if stage == "emb":
                stage_models = [entry for entry in selected if entry.kind == "embedding"]
            elif stage == "img":
                stage_models = [entry for entry in selected if entry.kind == "image"]
            elif stage in LLM_BACKED_TESTS:
                stage_models = [entry for entry in selected if entry.kind in {"llm", "custom"}]
            else:
                stage_models = []
            for entry in stage_models:
                model_row = ttk.Frame(shell)
                model_row.pack(fill="x", padx=(14, 0), pady=1)
                ttk.Label(model_row, text=entry.label, width=32).pack(side="left", anchor="w")
                variable = tk.StringVar(value="○ Queued")
                model_progress_vars[(stage, entry.label)] = variable
                ttk.Label(model_row, textvariable=variable).pack(side="right", anchor="e")
        progress_window.lift()

    def update_progress(event):
        nonlocal progress_metrics
        progress_metrics = update_progress_metrics(progress_metrics, event)
        completed = len(progress_metrics["finished_models"])
        progress_summary_var.set(
            f"Finished: {completed}/{progress_metrics['total_models']} models · "
            f"Usable coverage: {len(progress_metrics['usable_models'])}/{progress_metrics['total_models']} · "
            f"Invalid: {progress_metrics['invalid']} · "
            f"Retries: {progress_metrics['retries']}"
        )
        if event["kind"] == "measurement":
            return
        if event["kind"] == "model":
            variable = model_progress_vars.get((event["stage"], event["model"]))
        else:
            variable = stage_progress_vars.get(event["stage"])
        if variable is None:
            return
        variable.set({
            "running": "▶ Running", "complete": "✓ Complete",
            "failed": "✕ Failed", "interrupted": "■ Interrupted",
        }[event["status"]])
        if event["kind"] == "stage" and event["status"] in {"complete", "failed", "interrupted"}:
            for (stage, _), model_var in model_progress_vars.items():
                if stage == event["stage"] and model_var.get() in {"○ Queued", "▶ Running"}:
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
            elapsed = time.monotonic() - progress_started_at
            completed = len(progress_metrics.get("finished_models", ()))
            remaining = estimate_remaining_seconds(
                elapsed, completed, progress_metrics.get("total_models", 0),
            )
            estimate = "calibrating" if remaining is None else f"about {remaining // 60}m {remaining % 60}s"
            usage = process_resource_usage(process.pid)
            resources = ("unavailable" if usage is None
                         else f"{usage[0]:.0f}% CPU · {usage[1]:.1f} GB RAM")
            progress_resource_var.set(f"Resources: {resources} · Remaining time: {estimate}")
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
        for key in ("cpu_only", "force_all", "offline"):
            values[key] = option_vars[key].get()
        for key in ("out", "comfyui"):
            values[key] = option_vars[key].get().strip()
        return values

    def start_run():
        nonlocal process, active_process_kind, pending_fork_source
        custom = mode_var.get() == "custom"
        if custom:
            tests = [name for name, variable in test_vars.items() if variable.get()]
            entries = custom_models
            for entry in entries:
                entry.checked = model_vars[entry.value].get()
            max_prompt = None if cap_var.get() == "No cap" else int(cap_var.get())
            tg_tokens = [value for value, variable in tg_vars.items() if variable.get()]
            gui_options = collect_options()
            errors = validate_gui_options(gui_options)
        else:
            tests = default_test_values
            entries = default_models
            max_prompt = None
            tg_tokens = None
            gui_options = None
            errors = []
        if not tests:
            errors.append("Select at least one benchmark test.")
        selection_error = model_selection_error(entries, tests)
        if selection_error:
            errors.append(selection_error)
        if custom and TG_TOKEN_TESTS & set(tests) and not tg_tokens:
            errors.append("Select at least one llama-bench generation size.")
        custom_comfyui = (normalize_comfyui_dir(Path(gui_options["comfyui"]))
                          if custom and gui_options["comfyui"] else found_comfyui)
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
        if not messagebox.askyesno(
            "Review benchmark plan", f"{preview}\n\nStart this benchmark?", parent=root,
        ):
            pending_fork_source = None
            return
        state = build_frontend_state(
            engine_var.get(), tests, entries, max_prompt_tokens=max_prompt,
            tg_tokens=tg_tokens if TG_TOKEN_TESTS & set(tests) else None,
            gui_options=gui_options,
        )
        should_save = custom or saved is None
        if should_save and not save_frontend_state(state, FRONTEND_STATE_PATH):
            if not messagebox.askyesno("Settings not saved", "The configuration could not be saved. Run it anyway?", parent=root):
                return
        command = build_benchmark_command(
            engine_var.get(), detected_comfyui, tests, entries,
            max_prompt_tokens=max_prompt if MAX_PROMPT_TOKEN_TESTS & set(tests) else None,
            tg_tokens=tg_tokens if custom and TG_TOKEN_TESTS & set(tests) else None,
            gui_options=gui_options,
        )
        if pending_fork_source is not None:
            command.extend(["--fork-plan", str(pending_fork_source)])
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
        child_env = begin_process_control()
        process = subprocess.Popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=creationflags, env=child_env,
        )
        pending_fork_source = None
        active_process_kind = "benchmark"
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set("Benchmark is running. Results are checkpointed throughout the run.")
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(tests, entries)
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
            process.send_signal(signal.CTRL_BREAK_EVENT)
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
    update_mode()
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
