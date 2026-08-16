#!/usr/bin/env python3
"""Single-screen Tk launcher for Local AI Bench."""

import os
import platform
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from scripts.runtime import config, hardware
from scripts.runtime.shared import RUN_LOG_UTC_OFFSET_ENV
import psutil
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
from scripts.runtime.comfyui_installation import find_comfyui_installation, normalize_comfyui_dir
from scripts.runtime.engines import engine_names, get_engine, installed_engine_names
from scripts.runtime.llamacpp_tools import find_llamacpp_tool
from scripts.setup.model_inventory import build_model_inventory
from scripts.app.engine_management import collect_engine_management
from scripts.setup.runtime_update import (
    fetch_llamacpp_releases,
)
from scripts.stage_registry import STAGE_ORDER
from scripts.runtime.pause_control import PAUSE_CONTROL_ENV, create_pause_control, write_pause_state
from scripts.runtime.progress_events import PROGRESS_PREFIX
from scripts.runtime.crash_cache import clear_crash_caches, crash_cache_paths
from scripts.runtime.shared import Shared
from scripts.setup.setup_config import (
    available_gpu_split_modes, configured_comfyui_dir, configured_gpu_devices,
    load_setup_config,
)
from scripts.setup.vllm_install import fetch_vllm_versions, is_dgx_spark
from scripts.app.tk_utils import refresh_tk_layout
from scripts.app.result_actions import (
    completed_result_paths, record_result_path, result_paths_for_log, write_run_logs,
)
from scripts.app.benchmark_gui_screens.history import build_history_screen
from scripts.app.benchmark_gui_screens.history_actions import HistoryActions
from scripts.app.benchmark_gui_screens.history_process import HistoryProcessActions
from scripts.app.benchmark_gui_screens.run_log import build_run_log_screen
from scripts.app.benchmark_gui_screens.run_log_actions import RunLogActions
from scripts.app.benchmark_gui_screens.engines import EngineUpdateActions, build_engine_screen
from scripts.app.benchmark_gui_screens.configuration import (
    build_configuration_screen, confirm_plan_preview as show_plan_preview,
)
from scripts.app.benchmark_gui_screens.configuration_files import ConfigurationFileActions
from scripts.app.benchmark_gui_screens.configuration_state import ConfigurationStateController
from scripts.app.benchmark_gui_screens.progress import ProgressScreen
from scripts.app.benchmark_gui_process import (
    launch_controlled_process as _launch_controlled_process,
    open_path_command, should_finalize_process_exit,
    windows_host_utc_offset_minutes,
)
from scripts.app.benchmark_gui_resources import (
    PsutilLike, parse_gpu_process_memory, parse_gpu_usage, process_resource_usage,
    query_gpu_process_memory, query_gpu_usage, query_vram_usage,
    resource_usage_rows, show_vram_usage, system_memory_usage,
)

from scripts.app.benchmark_gui_support import (
    BENCHMARK_PRESETS,
    CUSTOM_PRESET,
    GPU_SPLIT_MODE_LABELS,
    apply_hardware_model_defaults,
    build_discovery_report,
    build_plan_preview,
    custom_option_defaults,
    default_control_values,
    effective_gui_options,
    estimate_remaining_seconds,
    format_run_outcome,
    gpu_split_mode_labels,
    gpu_split_mode_value,
    history_row_height,
    parse_progress_line,
    plan_preview_sections,
    preset_after_control_change,
    preset_control_values,
    progress_event_engine,
    progress_model_identity,
    progress_summary_rows,
    reconcile_imported_model_state,
    resolve_preset,
    restored_preset_name,
    update_progress_metrics,
    workload_preflight_errors,
)


def launch_controlled_process(command: list[str], **kwargs):
    return _launch_controlled_process(
        command, utc_offset_fn=windows_host_utc_offset_minutes, **kwargs,
    )


@dataclass(frozen=True)
class BenchmarkLaunchError:
    errors: list[str]


@dataclass(frozen=True)
class BenchmarkLaunchReady:
    preview: str
    state: dict
    command: list[str]


@dataclass(frozen=True)
class EngineSelectionState:
    engines: list[str]
    value: str
    note: str
    model_availability: dict[str, bool]


@dataclass(frozen=True)
class ProcessCompletionState:
    status: str
    write_run_log: bool


def resolve_engine_names(selected: list[str], available: list[str]) -> list[str]:
    return [name for name in available if name in selected] or [available[0]]


def resolve_engine_selection(selected: list[str], available: list[str],
                             models: list[MenuEntry],
                             model_owners: dict[str, set[str]]) -> EngineSelectionState:
    chosen = resolve_engine_names(selected, available)
    note = (
        f"Runs the full selection once per engine ({len(chosen)} passes, "
        f"{len(chosen)} results files)." if len(chosen) > 1
        else "Only installed engines are listed. Models this engine cannot run are disabled."
    )
    runnable: dict[str, bool] = {}
    for name in chosen:
        for value, supported in models_runnable_by(models, name, model_owners).items():
            runnable[value] = runnable.get(value, False) or supported
    return EngineSelectionState(chosen, format_engine_selection(chosen), note, runnable)


def process_completion_state(kind: str | None, exit_code: int) -> ProcessCompletionState:
    if kind in {"recovery", "retry", "fork"}:
        label = {"recovery": "Recovery", "retry": "Selected retry", "fork": "Forked run"}[kind]
        status = (
            f"{label} completed successfully. Results are ready to review."
            if exit_code == 0 else
            f"{label} stopped with exit code {exit_code}. Preserved evidence remains available."
        )
    else:
        status = format_run_outcome(exit_code)
    return ProcessCompletionState(status, exit_code == 0)


def normalize_gui_option_values(values: dict[str, Any]) -> dict[str, Any]:
    options = dict(GUI_OPTION_DEFAULTS)
    for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget",
                "sustained_duration"):
        try:
            options[key] = int(values[key])
        except (TypeError, ValueError):
            options[key] = values[key]
    try:
        raw_ambient = str(values["ambient_temp_c"]).strip()
        options["ambient_temp_c"] = float(raw_ambient) if raw_ambient else None
    except (TypeError, ValueError):
        options["ambient_temp_c"] = values["ambient_temp_c"]
    options["gpu_split_mode"] = gpu_split_mode_value(values["gpu_split_mode"])
    for key in ("cpu_only", "force_all", "retry_crashed_models", "offline", "memory_telemetry",
                "power_telemetry", "llamacpp_no_repack"):
        options[key] = values[key]
    for key in ("out", "comfyui"):
        options[key] = str(values[key]).strip()
    return options


def restored_tg_tokens(state: dict[str, Any] | None) -> set[int]:
    if state is None or state["tg_tokens"] is None:
        return set(config.LLAMABENCH_TG)
    return set(state["tg_tokens"])


def prepare_benchmark_launch(*, engine: str, tests: list[str], entries: list[MenuEntry],
                             max_prompt_tokens: int | None, tg_tokens: list[int],
                             gui_options: dict[str, Any], selected_preset: str,
                             detected_tools: dict[str, str | None],
                             found_comfyui: Path | None,
                             detected_comfyui: Path) -> BenchmarkLaunchError | BenchmarkLaunchReady:
    errors = validate_gui_options(gui_options)
    if not tests:
        errors.append("Select at least one benchmark test.")
    selection_error = model_selection_error(entries, tests)
    if selection_error:
        errors.append(selection_error)
    if TG_TOKEN_TESTS & set(tests) and not tg_tokens:
        errors.append("Select at least one llama-bench generation size.")
    custom_comfyui = (
        normalize_comfyui_dir(Path(gui_options["comfyui"]))
        if gui_options.get("comfyui") else found_comfyui
    )
    errors.extend(workload_preflight_errors(tests, detected_tools, custom_comfyui is not None))
    if errors:
        return BenchmarkLaunchError(errors)
    preview = build_plan_preview(
        engine=engine, tests=tests, entries=entries, options=gui_options,
        max_prompt_tokens=max_prompt_tokens, tg_tokens=tg_tokens,
        comfyui_dir=custom_comfyui or detected_comfyui,
    )
    state = build_frontend_state(
        engine, tests, entries, max_prompt_tokens=max_prompt_tokens,
        tg_tokens=tg_tokens,
        gui_options=gui_options, selected_preset=selected_preset,
    )
    command = build_benchmark_command(
        engine, detected_comfyui, tests, entries,
        max_prompt_tokens=(max_prompt_tokens
                           if MAX_PROMPT_TOKEN_TESTS & set(tests) else None),
        tg_tokens=tg_tokens if TG_TOKEN_TESTS & set(tests) else None,
        gui_options=gui_options,
    )
    return BenchmarkLaunchReady(preview, state, command)


def run_benchmark_gui() -> int:  # pragma: no cover — interactive desktop UI
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk

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
    history_font = tkfont.nametofont("TkDefaultFont")
    style.configure(
        "History.Treeview",
        rowheight=history_row_height(history_font.metrics("linespace")),
    )

    advanced_var = tk.BooleanVar(value=False)
    engine_var = tk.StringVar(value=selected_engine)
    test_vars = {entry.value: tk.BooleanVar(value=entry.checked) for entry in custom_tests}
    model_vars = {entry.value: tk.BooleanVar(value=entry.checked) for entry in custom_models}
    cap_var = tk.StringVar(value=str(saved["max_prompt_tokens"]) if saved and saved["max_prompt_tokens"] else "No cap")
    saved_tg = restored_tg_tokens(saved)
    tg_vars = {value: tk.BooleanVar(value=value in saved_tg) for value in TG_TOKEN_OPTIONS}
    options = effective_gui_options(saved)
    if options["gpu_split_mode"] not in gpu_split_modes:
        options["gpu_split_mode"] = "layer"
    option_vars: dict[str, tk.Variable] = {
        key: (tk.BooleanVar(value=value) if isinstance(value, bool)
              else tk.StringVar(value="" if value is None else str(value)))
        for key, value in options.items()
    }
    option_vars["gpu_split_mode"].set(GPU_SPLIT_MODE_LABELS[options["gpu_split_mode"]])
    if not option_vars["comfyui"].get():
        option_vars["comfyui"].set(str(detected_comfyui))
    preset_var = tk.StringVar(value=restored_preset_name(saved))
    active_project: dict[str, dict | None] = {"value": None}
    project_status = tk.StringVar(value="No project loaded")

    engine_updates = EngineUpdateActions(setup, hardware_backend)
    llamacpp_update_prompts = {
        "Darwin": "Download and validate the latest official macOS llama.cpp release, then replace the current one?",
        "Windows": "Download and validate the latest compatible llama.cpp release, then replace the current one?",
        "Linux": "Clone and build the latest llama.cpp, then replace the current checkout?",
    }

    notebook = ttk.Notebook(root)
    notebook.grid(sticky="nsew")
    configuration_screen = build_configuration_screen(
        notebook, tk=tk, ttk=ttk, discovery=discovery, advanced_var=advanced_var,
        preset_var=preset_var, project_status=project_status,
        preset_names=[*BENCHMARK_PRESETS, CUSTOM_PRESET], test_vars=test_vars,
        test_defaults=custom_test_defaults, custom_tests=custom_tests,
        model_vars=model_vars, model_defaults=custom_model_defaults,
        custom_models=custom_models, cap_var=cap_var, tg_vars=tg_vars,
    )
    config_tab = configuration_screen.frame
    run_log_screen = build_run_log_screen(
        notebook, tk=tk, ttk=ttk, configuration_frame=config_tab,
    )
    history_screen = build_history_screen(notebook, tk=tk, ttk=ttk)
    log_tab = run_log_screen.frame
    history_tab = history_screen.frame
    engines_tab, engine_management = build_engine_screen(
        notebook, root=root, tk=tk, ttk=ttk, messagebox=messagebox,
        status_loader=lambda: collect_engine_management(get_engine, hardware_backend),
        vllm_updater=engine_updates.update_vllm,
        vllm_version_loader=(
            None if is_dgx_spark(
                platform.machine(),
                [str(device.get("name", "")) for device in configured_gpu_devices(setup)],
            ) else fetch_vllm_versions
        ),
        vllm_version_updater=(
            None if is_dgx_spark(
                platform.machine(),
                [str(device.get("name", "")) for device in configured_gpu_devices(setup)],
            ) else engine_updates.update_vllm_version
        ),
        llamacpp_updater=engine_updates.update_llamacpp,
        llamacpp_update_prompt=llamacpp_update_prompts.get(platform.system()),
        llamacpp_release_loader=fetch_llamacpp_releases,
        llamacpp_version_updater=engine_updates.update_llamacpp_version,
        llamacpp_model_probe=engine_updates.probe_llamacpp_model,
        run_active=lambda: process is not None and process.poll() is None,
    )
    notebook.bind(
        "<<NotebookTabChanged>>", lambda _event: refresh_tk_layout(root), add="+",
    )
    form = configuration_screen.form
    canvas = configuration_screen.canvas
    configuration_frame = configuration_screen.configuration_frame
    engine_box = configuration_screen.engine_box
    preset_row = configuration_screen.preset_row
    project_row = configuration_screen.project_row
    advanced_toggle = configuration_screen.advanced_toggle
    tests_box = configuration_screen.tests_box
    test_widgets = configuration_screen.test_widgets
    test_labels = configuration_screen.test_labels
    models_box = configuration_screen.models_box
    model_rows = configuration_screen.model_rows
    model_widgets = configuration_screen.model_widgets
    workload_box = configuration_screen.workload_box

    def render_model_rows():
        configuration_screen.render_model_rows()

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
        wanted = resolve_engine_names(list(names), available_engines)
        restoring_engines[0] = True
        try:
            for name, variable in engine_check_vars.items():
                variable.set(name in wanted)
        finally:
            restoring_engines[0] = False
        apply_engine_availability()

    def apply_engine_availability(*_) -> None:
        if restoring_engines[0]:
            return
        selection = resolve_engine_selection(
            [name for name in available_engines if engine_check_vars[name].get()],
            available_engines, custom_models, model_owners,
        )
        if not any(engine_check_vars[name].get() for name in selection.engines):
            engine_check_vars[selection.engines[0]].set(True)
        engine_var.set(selection.value)
        engine_note.set(selection.note)
        for value, widget in model_widgets.items():
            available = selection.model_availability.get(value, True)
            widget.configure(state="normal" if available else "disabled")
            if not available and model_vars[value].get():
                model_vars[value].set(False)

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
    for _engine_var in engine_check_vars.values():
        _engine_var.trace_add("write", apply_engine_availability)
    apply_engine_availability()

    def clear_all_crash_caches():
        if process is not None and process.poll() is None:
            messagebox.showerror("Benchmark active", "Stop the active process first.", parent=root)
            return
        caches = crash_cache_paths(config.SCRIPT_DIR)
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
        removed, failures = clear_crash_caches(config.SCRIPT_DIR)
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

    def execution_row(pady: Any = 0):
        frame = ttk.Frame(execution_box)
        frame.pack(fill="x", pady=pady)
        frame.columnconfigure(0, minsize=330)
        frame.columnconfigure(1, weight=1)
        return frame

    split_labels = gpu_split_mode_labels(gpu_split_modes)
    gpu_mode_row = execution_row(pady=2)
    ttk.Label(gpu_mode_row, text="GPU mode").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        gpu_mode_row, state="readonly", textvariable=option_vars["gpu_split_mode"],
        values=split_labels, width=30,
    ).grid(row=0, column=1, sticky="w", padx=(10, 0))
    ttk.Button(
        gpu_mode_row, text="Reset", width=6,
        command=lambda: option_vars["gpu_split_mode"].set(GPU_SPLIT_MODE_LABELS["layer"]),
    ).grid(row=0, column=2, padx=(8, 0))
    labels = (("warmup", f"Warmup runs (default {config.WARMUP_RUNS})"),
              ("runs", f"Measured runs (1–10; default {config.N_RUNS})"),
              ("timeout", f"Run timeout, seconds (default {config.RUN_TIMEOUT})"),
              ("acc_timeout", f"Accuracy timeout, seconds (default {config.ACC_TIMEOUT})"),
              ("acc_token_budget", f"Accuracy token budget (default {config.ACC_TOKEN_BUDGET})"),
              ("sustained_duration", f"Sustained soak, seconds (default {config.SUSTAINED_DURATION_SEC})"),
              ("ambient_temp_c", "Ambient temperature, °C (optional)"))
    for key, label in labels:
        row = execution_row(pady=2)
        ttk.Label(row, text=label).grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=option_vars[key], width=12).grid(
            row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Button(
            row, text="Reset", width=6,
            command=lambda option=key: option_vars[option].set(
                "" if GUI_OPTION_DEFAULTS[option] is None
                else str(GUI_OPTION_DEFAULTS[option])
            ),
        ).grid(row=0, column=2, padx=(8, 0))
    checkboxes = (
        ("cpu_only", "CPU-only inference", (8, 0)),
        ("force_all", "Run slow models instead of skipping", 0),
        ("retry_crashed_models", "Retry models that crashed previously", 0),
        ("offline", "Offline mode (loopback only)", 0),
        ("memory_telemetry", "Memory telemetry", 0),
        ("power_telemetry", "Power and energy telemetry (requires permission)", 0),
    )
    for key, text, pady in checkboxes:
        row = execution_row(pady=pady)
        ttk.Checkbutton(row, text=text, variable=option_vars[key]).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(
            row, text="Reset", width=6,
            command=lambda option=key: option_vars[option].set(GUI_OPTION_DEFAULTS[option]),
        ).grid(row=0, column=2, padx=(8, 0))
    repack_row = execution_row()
    ttk.Checkbutton(
        repack_row, text="Disable llama.cpp weight repacking (-nr)",
        variable=option_vars["llamacpp_no_repack"],
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Button(
        repack_row, text="Reset", width=6,
        command=lambda: option_vars["llamacpp_no_repack"].set(False),
    ).grid(row=0, column=2, padx=(8, 0))
    ttk.Label(
        execution_row(pady=(8, 0)), text="More warmups/runs improve repeatability but increase time. CPU-only changes the tested device; force-all can make runs much longer.",
        wraplength=430,
    ).pack(anchor="w")
    reset_execution_row = execution_row(pady=(8, 0))
    ttk.Button(
        execution_row(pady=(8, 0)), text="Clear Crash Caches", command=clear_all_crash_caches,
    ).pack(anchor="w")

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
        for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget",
                    "sustained_duration", "ambient_temp_c", "gpu_split_mode",
                    "cpu_only", "force_all", "retry_crashed_models", "offline", "memory_telemetry",
                    "power_telemetry", "llamacpp_no_repack"):
            variable = option_vars[key]
            value = GPU_SPLIT_MODE_LABELS[defaults[key]] if key == "gpu_split_mode" else defaults[key]
            variable.set(value)

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

    defaults_for_display = default_control_values(
        default_tests, default_models, selected_engine, detected_comfyui,
    )
    applying_configuration = [False]

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
                if key == "gpu_split_mode":
                    option_vars[key].set(GPU_SPLIT_MODE_LABELS[value])
                else:
                    option_vars[key].set(value if isinstance(value, bool) else str(value))
        finally:
            applying_configuration[0] = False
        preset_var.set(CUSTOM_PRESET)

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
    ttk.Button(reset_execution_row, text="Reset Execution", command=reset_execution).pack(anchor="w")
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

    run_status = run_log_screen.status
    log_text = run_log_screen.text
    stop_button = run_log_screen.stop_button
    pause_button = run_log_screen.pause_button

    run_log_actions = RunLogActions(
        run_log_screen, root=root, tk=tk, ttk=ttk, filedialog=filedialog,
        messagebox=messagebox, option_vars=option_vars, active_project=active_project,
        active_result_paths=lambda: active_result_paths,
    )
    run_log_actions.bind()
    current_log = run_log_actions.current_log
    review_outbound_metadata = run_log_actions.review_outbound_metadata

    def launch_history_process(
            command, kind, result_paths, status, stages, entries, engines, error_title):
        nonlocal process, active_process_kind, active_result_paths
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if platform.system() == "Windows" else 0
        )
        try:
            process, control_path = launch_controlled_process(
                command, creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror(error_title, str(exc), parent=root)
            return
        begin_process_control(control_path)
        active_process_kind = kind
        active_result_paths = result_paths
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")
        run_status.set(status)
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        notebook.select(log_tab)
        show_progress_window(stages, entries, engines=engines)
        threading.Thread(target=read_process, args=(process,), daemon=True).start()

    def fallback_history_fork(source_path, output_path, plan):
        nonlocal pending_fork_source
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

    history_process = HistoryProcessActions(
        root=root, filedialog=filedialog, messagebox=messagebox,
        process_active=lambda: process is not None and process.poll() is None,
        launch=launch_history_process, fallback_fork=fallback_history_fork,
    )
    history_actions = HistoryActions(
        history_screen, root=root, tk=tk, ttk=ttk, filedialog=filedialog,
        messagebox=messagebox,
        process_active=lambda: process is not None and process.poll() is None,
        review_outbound_metadata=review_outbound_metadata,
        start_recovery=history_process.start,
    )
    history_actions.bind()
    refresh_history = history_actions.refresh

    configuration_state = ConfigurationStateController(
        configuration_screen, root=root, tk=tk, ttk=ttk, messagebox=messagebox,
        advanced_var=advanced_var, engine_var=engine_var, test_vars=test_vars,
        model_vars=model_vars, cap_var=cap_var, tg_vars=tg_vars,
        option_vars=option_vars, preset_var=preset_var,
        available_engines=available_engines, custom_tests=custom_tests,
        custom_models=custom_models, defaults_for_display=defaults_for_display,
        applying_configuration=applying_configuration,
        engine_inventories=engine_inventories, inventory=inventory,
        model_owners=model_owners, custom_model_defaults=custom_model_defaults,
        set_selected_engines=set_selected_engines,
        apply_engine_availability=apply_engine_availability,
        execution_box=execution_box, paths_box=paths_box,
    )
    configuration_state.bind()
    open_model_import_dialog = configuration_state.open_model_import_dialog
    update_advanced = configuration_state.update_advanced

    process = None
    active_process_kind = None
    active_result_paths: list[Path] = []
    pending_fork_source = None
    process_control_path = None
    process_paused = False
    process_exit_observed_at = None
    process_output_activity_at = time.monotonic()

    def begin_process_control(control_path):
        nonlocal process_control_path, process_paused
        nonlocal process_exit_observed_at, process_output_activity_at
        process_control_path = control_path
        process_paused = False
        process_exit_observed_at = None
        process_output_activity_at = time.monotonic()
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
    progress_screen = ProgressScreen(
        root, tk, ttk, update_progress_metrics, progress_summary_rows,
        progress_event_engine, progress_model_identity,
    )
    progress_metrics = {}
    progress_started_at = None
    gpu_sample = {
        "usage": None, "memory": None, "vram": None, "next_at": 0.0,
        "running": False, "generation": 0,
    }
    system_memory_baseline = [0.0]
    has_discrete_vram = [False]

    def show_progress_window(tests, entries, engines=None):
        nonlocal progress_metrics, progress_started_at
        gpu_sample.update({
            "usage": None, "memory": None, "vram": None, "next_at": 0.0,
            "running": False, "generation": gpu_sample["generation"] + 1,
        })
        has_discrete_vram[0] = show_vram_usage(configured_gpu_devices(setup))
        system_memory_baseline[0] = system_memory_usage()[0]
        run_engines = list(engines or parse_engine_selection(engine_var.get()))
        progress_screen.show(tests, entries, run_engines, show_vram=has_discrete_vram[0])
        progress_metrics = progress_screen.metrics
        progress_started_at = progress_screen.started_at

    def update_progress(event):
        nonlocal progress_metrics
        progress_screen.update(event)
        progress_metrics = progress_screen.metrics

    def append_log(text):
        log_text.configure(state="normal")
        log_text.insert("end", text)
        log_text.see("end")
        log_text.configure(state="disabled")

    def finish_active_process(exit_code):
        nonlocal process, active_process_kind
        completion = process_completion_state(active_process_kind, exit_code)
        process = None
        if completion.write_run_log:
            try:
                write_run_logs(
                    current_log(), result_paths_for_log(current_log(), active_result_paths),
                )
            except OSError as exc:
                append_log(f"\nCould not save Run Log: {exc}\n")
        finish_process_control()
        stop_button.configure(state="disabled")
        start_button.configure(state="normal")
        run_status.set(completion.status)
        active_process_kind = None
        refresh_history()
        progress_screen.finish_pending(exit_code)

    def poll_output():
        nonlocal process_exit_observed_at, process_output_activity_at
        try:
            while True:
                kind, value, source = output_queue.get_nowait()
                if source is not process:
                    continue
                if kind == "line":
                    process_output_activity_at = time.monotonic()
                    append_log(value)
                elif kind == "progress":
                    process_output_activity_at = time.monotonic()
                    if value["kind"] == "result":
                        active_result_paths[:] = record_result_path(
                            active_result_paths, value["path"],
                        )
                    else:
                        update_progress(value)
                elif process is not None:
                    finish_active_process(value)
        except queue.Empty:
            pass
        if process is not None:
            exit_code = process.poll()
            now = time.monotonic()
            if exit_code is not None and process_exit_observed_at is None:
                process_exit_observed_at = now
            if should_finalize_process_exit(
                    exit_code, False, process_exit_observed_at,
                    process_output_activity_at, now):
                finish_active_process(exit_code)
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
            progress_screen.set_resources(resources, estimate)
        root.after(100, poll_output)

    def read_process(proc):
        for line in proc.stdout:
            progress = parse_progress_line(line)
            kind, value = ("progress", progress) if progress else ("line", line)
            output_queue.put((kind, value, proc))
        output_queue.put(("done", proc.wait(), proc))

    def collect_options():
        return normalize_gui_option_values({
            key: variable.get() for key, variable in option_vars.items()
        })

    configuration_files = ConfigurationFileActions(
        configuration_screen, root=root, tk=tk, ttk=ttk, filedialog=filedialog,
        simpledialog=simpledialog, messagebox=messagebox, active_project=active_project,
        project_status=project_status, current_state=current_custom_state,
        apply_state=apply_frontend_state, collect_options=collect_options,
    )
    configuration_files.bind()

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
        preparation = prepare_benchmark_launch(
            engine=engine_var.get(), tests=tests, entries=entries,
            max_prompt_tokens=max_prompt, tg_tokens=tg_tokens,
            gui_options=gui_options, selected_preset=preset_var.get(),
            detected_tools=detected_tools, found_comfyui=found_comfyui,
            detected_comfyui=detected_comfyui,
        )
        if isinstance(preparation, BenchmarkLaunchError):
            messagebox.showerror(
                "Check benchmark options", "\n".join(preparation.errors), parent=root,
            )
            return
        if not show_plan_preview(root, tk, ttk, preparation.preview):
            pending_fork_source = None
            return
        if not save_frontend_state(preparation.state, FRONTEND_STATE_PATH):
            if not messagebox.askyesno("Settings not saved", "The configuration could not be saved. Run it anyway?", parent=root):
                return
        command = preparation.command
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
        active_result_paths = []
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
        if engine_management.busy():
            if messagebox.askyesno(
                    "Runtime update active",
                    "Cancel the active runtime update? Keep this window open until cleanup finishes.",
                    parent=root):
                engine_management.cancel()
            return
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
