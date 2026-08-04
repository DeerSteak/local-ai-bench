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
from pathlib import Path

import config
import psutil
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
from comfyui_installation import find_comfyui_installation, normalize_comfyui_dir
from engines import engine_names, get_engine
from llamacpp_tools import find_llamacpp_tool
from run_plan import load_run_plan
from result_bundle import export_result_bundle, import_result_bundle, verify_result_bundle
from model_inventory import build_model_inventory
from orchestration import STAGE_ORDER
from progress_events import PROGRESS_PREFIX
from shared import Shared
from setup_config import configured_comfyui_dir, load_setup_config
from tk_utils import mousewheel_scroll_units


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
    notebook.add(config_tab, text="Configuration")
    notebook.add(log_tab, text="Run Log")
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
    advanced_toggle = ttk.Checkbutton(
        custom_frame, text="Show advanced execution and path settings", variable=advanced_var,
    )
    advanced_toggle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

    tests_box = ttk.LabelFrame(custom_frame, text="Tests", padding=12)
    tests_box.grid(row=2, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
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
    models_box.grid(row=2, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
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
    workload_box.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
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
    execution_box.grid(row=4, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
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
    ttk.Label(
        execution_box, text="More warmups/runs improve repeatability but increase time. CPU-only changes the tested device; force-all can make runs much longer.",
        wraplength=430,
    ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))

    paths_box = ttk.LabelFrame(custom_frame, text="Paths", padding=12)
    paths_box.grid(row=4, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
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
                    "cpu_only", "force_all"):
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

    ttk.Button(preset_row, text="Apply", command=apply_preset).pack(side="left")
    ttk.Button(preset_row, text="Export", command=export_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Import", command=import_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Duplicate", command=duplicate_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Compare", command=compare_preset).pack(side="left", padx=(8, 0))
    ttk.Button(preset_row, text="Import CLI Plan", command=import_run_plan).pack(side="left", padx=(8, 0))

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
        row=10, column=0, columnspan=2, sticky="w", pady=(8, 0),
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
    ttk.Button(log_actions, text="Back to Configuration", command=lambda: notebook.select(config_tab)).pack(side="left")
    def open_results_folder():
        output = option_vars["out"].get().strip() if mode_var.get() == "custom" else ""
        folder = Path(output).expanduser().resolve().parent if output else config.RESULTS_DIR
        subprocess.Popen(open_path_command(folder, platform.system()))

    def export_bundle():
        result = filedialog.askopenfilename(
            title="Choose result JSON", initialdir=config.RESULTS_DIR,
            filetypes=[("Benchmark result", "*.json")],
        )
        if not result:
            return
        bundle = filedialog.asksaveasfilename(
            title="Export verified result bundle", defaultextension=".labresult",
            filetypes=[("Local AI Bench result", "*.labresult")],
        )
        if not bundle:
            return
        try:
            export_result_bundle(Path(result), Path(bundle))
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

    ttk.Button(log_actions, text="Open Results Folder", command=open_results_folder).pack(
        side="left", padx=(10, 0),
    )
    ttk.Button(log_actions, text="Export Bundle", command=export_bundle).pack(side="left", padx=(10, 0))
    ttk.Button(log_actions, text="Import / Verify", command=import_bundle).pack(side="left", padx=(10, 0))

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
        nonlocal process
        try:
            while True:
                kind, value = output_queue.get_nowait()
                if kind == "line":
                    append_log(value)
                elif kind == "progress":
                    update_progress(value)
                else:
                    process = None
                    stop_button.configure(state="disabled")
                    start_button.configure(state="normal")
                    run_status.set(format_run_outcome(value))
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
        for key in ("cpu_only", "force_all"):
            values[key] = option_vars[key].get()
        for key in ("out", "comfyui"):
            values[key] = option_vars[key].get().strip()
        return values

    def start_run():
        nonlocal process
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
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
        child_env = {**os.environ, "LOCAL_AI_BENCH_PROGRESS": "1"}
        process = subprocess.Popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=creationflags, env=child_env,
        )
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
        run_status.set("Stopping benchmark safely…")
        if platform.system() == "Windows":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)

    def close_window():
        if process is not None and process.poll() is None:
            if not messagebox.askyesno("Benchmark running", "Stop the benchmark and close?", parent=root):
                return
            stop_run()
        root.destroy()

    start_button.configure(command=start_run)
    stop_button.configure(command=stop_run)
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
