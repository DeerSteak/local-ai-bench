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
from pathlib import Path

import config
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
    build_model_entries,
    build_test_entries,
    load_frontend_state,
    model_selection_error,
    save_frontend_state,
    validate_gui_options,
)
from comfyui_installation import find_comfyui_installation, normalize_comfyui_dir
from engines import engine_names, get_engine
from llamacpp_tools import find_llamacpp_tool
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
    if not isinstance(event, dict) or event.get("kind") not in {"stage", "model"}:
        return None
    if event.get("status") not in {"running", "complete", "failed", "interrupted"}:
        return None
    if not isinstance(event.get("stage"), str):
        return None
    if event["kind"] == "model" and not isinstance(event.get("model"), str):
        return None
    return event


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


def run_benchmark_gui() -> int:  # pragma: no cover — interactive desktop UI
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

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
    discovery = build_discovery_report(
        platform_name=platform.system(), architecture=platform.machine(),
        ram_gb=Shared.system_ram_gb(), backend=Shared.detect_backend(),
        tools=detected_tools,
        comfyui_dir=found_comfyui, inventory=inventory,
        free_storage_gb=shutil.disk_usage(config.SCRIPT_DIR).free / 1e9,
    )

    default_tests = build_test_entries(inventory)
    default_test_values = [entry.value for entry in default_tests if entry.checked]
    default_models = build_model_entries(inventory, default_test_values)
    custom_tests = build_test_entries(inventory)
    apply_saved_test_selection(custom_tests, saved)
    custom_test_values = [entry.value for entry in custom_tests if entry.checked]
    custom_models = build_model_entries(inventory, [entry.value for entry in custom_tests if entry.available])
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

    tests_box = ttk.LabelFrame(custom_frame, text="Tests", padding=12)
    tests_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
    test_widgets = {}
    for row, (name, label, _, _) in enumerate(TEST_DEFINITIONS):
        entry = next(item for item in custom_tests if item.value == name)
        text = label if entry.available else f"{label} (model not installed)"
        widget = ttk.Checkbutton(tests_box, text=text, variable=test_vars[name])
        widget.grid(row=row, column=0, sticky="w")
        test_widgets[name] = widget
    ttk.Label(
        tests_box, text="Accuracy and concurrency add substantial runtime; native llama-bench tests require their matching tools.",
        wraplength=430,
    ).grid(row=len(TEST_DEFINITIONS), column=0, sticky="w", pady=(8, 0))

    models_box = ttk.LabelFrame(custom_frame, text="Installed models", padding=12)
    models_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
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
        model_widgets[entry.value] = widget
        row += 1
    ttk.Label(
        models_box, text="Each checked model runs once through every applicable selected workload. Larger models may exceed memory.",
        wraplength=430,
    ).grid(row=row, column=0, sticky="w", pady=(8, 0))

    workload_box = ttk.LabelFrame(custom_frame, text="Workload sizes", padding=12)
    workload_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
    ttk.Label(workload_box, text="Maximum prompt-processing size").grid(row=0, column=0, sticky="w")
    cap_combo = ttk.Combobox(workload_box, state="readonly", textvariable=cap_var,
                             values=["No cap", *[str(value) for value in MAX_PROMPT_TOKEN_OPTIONS]], width=18)
    cap_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))
    ttk.Label(workload_box, text="llama-bench generation sizes").grid(row=1, column=0, sticky="w", pady=(10, 0))
    tg_frame = ttk.Frame(workload_box)
    tg_frame.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
    for column, value in enumerate(TG_TOKEN_OPTIONS):
        ttk.Checkbutton(tg_frame, text=str(value), variable=tg_vars[value]).grid(row=0, column=column, padx=(0, 8))
    ttk.Label(
        workload_box, text="Prompt and generation values are tokens. No cap tests every configured depth; larger values increase time and memory.",
        wraplength=430,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

    execution_box = ttk.LabelFrame(custom_frame, text="Execution", padding=12)
    execution_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
    ttk.Label(execution_box, text="Inference engine").grid(row=0, column=0, sticky="w", pady=2)
    engine_combo = ttk.Combobox(execution_box, state="readonly", textvariable=engine_var,
                                values=available_engines, width=16)
    engine_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=2)
    labels = (("warmup", f"Warmup runs (default {config.WARMUP_RUNS})"),
              ("runs", f"Measured runs (1–10; default {config.N_RUNS})"),
              ("timeout", f"Run timeout, seconds (default {config.RUN_TIMEOUT})"),
              ("acc_timeout", f"Accuracy timeout, seconds (default {config.ACC_TIMEOUT})"),
              ("acc_token_budget", f"Accuracy token budget (default {config.ACC_TOKEN_BUDGET})"))
    for row, (key, label) in enumerate(labels, 1):
        ttk.Label(execution_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(execution_box, textvariable=option_vars[key], width=12).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=2)
    ttk.Checkbutton(execution_box, text="CPU-only inference", variable=option_vars["cpu_only"]).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Checkbutton(execution_box, text="Run slow models instead of skipping", variable=option_vars["force_all"]).grid(row=8, column=0, columnspan=2, sticky="w")
    ttk.Label(
        execution_box, text="More warmups/runs improve repeatability but increase time. CPU-only changes the tested device; force-all can make runs much longer.",
        wraplength=430,
    ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))

    paths_box = ttk.LabelFrame(custom_frame, text="Paths", padding=12)
    paths_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    paths_box.columnconfigure(1, weight=1)
    ttk.Label(paths_box, text="Results JSON (blank = automatic)").grid(row=0, column=0, sticky="w")
    ttk.Entry(paths_box, textvariable=option_vars["out"]).grid(row=0, column=1, sticky="ew", padx=10)
    ttk.Button(paths_box, text="Browse…", command=lambda: option_vars["out"].set(
        filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON results", "*.json")]) or option_vars["out"].get()
    )).grid(row=0, column=2)
    ttk.Label(paths_box, text="ComfyUI installation").grid(row=1, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(paths_box, textvariable=option_vars["comfyui"]).grid(row=1, column=1, sticky="ew", padx=10, pady=(8, 0))
    ttk.Button(paths_box, text="Browse…", command=lambda: option_vars["comfyui"].set(
        filedialog.askdirectory() or option_vars["comfyui"].get()
    )).grid(row=1, column=2, pady=(8, 0))
    ttk.Label(
        paths_box, text="Blank output uses results/results_<host>_<time>.json. ComfyUI must identify a usable program installation, not its model folder.",
        wraplength=900,
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

    footer = ttk.Frame(config_tab)
    footer.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    status_var = tk.StringVar(value="Ready to benchmark.")
    ttk.Label(footer, textvariable=status_var).pack(side="left")
    start_button = ttk.Button(footer, text="Start Benchmark", style="Start.TButton")
    start_button.pack(side="right")

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

    ttk.Button(log_actions, text="Open Results Folder", command=open_results_folder).pack(
        side="left", padx=(10, 0),
    )

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

    mode_var.trace_add("write", lambda *_: update_mode())

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

    def show_progress_window(tests, entries):
        nonlocal progress_window, stage_progress_vars, model_progress_vars
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
            shell, text="Broad stage status. Detailed output remains in the Run Log.",
            wraplength=390,
        ).pack(anchor="w", pady=(2, 12))
        stage_progress_vars = {}
        model_progress_vars = {}
        labels = {name: label for name, label, _, _ in TEST_DEFINITIONS}
        selected = [entry for entry in entries if entry.checked]
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
    root.after(100, poll_output)
    root.after(150, lambda: (root.lift(), root.attributes("-topmost", True), root.focus_force(),
                             root.after(400, lambda: root.attributes("-topmost", False))))
    root.mainloop()
    return 0


def main():  # pragma: no cover — desktop entrypoint
    raise SystemExit(run_benchmark_gui())


if __name__ == "__main__":
    main()
