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
from scripts.runtime.engines import engine_names, get_engine, installed_engine_names
from scripts.runtime.llamacpp_tools import find_llamacpp_tool
from scripts.results.run_plan import load_run_plan
from scripts.results.result_history import (
    delete_multiple_run_artifacts, discover_results, existing_run_artifacts, filter_results,
    load_result as load_history_result,
)
from scripts.results.recovery_inspector import inspect_recovery
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
from scripts.app.benchmark_gui_screens.history_actions import HistoryActions
from scripts.app.benchmark_gui_screens.run_log import build_run_log_screen
from scripts.app.benchmark_gui_screens.run_log_actions import RunLogActions
from scripts.app.benchmark_gui_screens.engines import build_engine_screen
from scripts.app.benchmark_gui_screens.configuration import build_configuration_screen
from scripts.app.benchmark_gui_screens.progress import ProgressScreen

from scripts.app.benchmark_gui_support import (
    BENCHMARK_PRESETS,
    CUSTOM_PRESET,
    GPU_SPLIT_MODE_LABELS,
    PsutilLike,
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
    launch_controlled_process as _launch_controlled_process,
    open_path_command,
    parse_gpu_process_memory,
    parse_gpu_usage,
    parse_progress_line,
    plan_preview_sections,
    preset_after_control_change,
    preset_control_values,
    process_resource_usage,
    progress_event_engine,
    progress_model_identity,
    progress_summary_rows,
    query_gpu_process_memory,
    query_gpu_usage,
    query_vram_usage,
    reconcile_imported_model_state,
    resolve_preset,
    resource_usage_rows,
    restored_preset_name,
    should_finalize_process_exit,
    show_vram_usage,
    system_memory_usage,
    update_progress_metrics,
    windows_host_utc_offset_minutes,
    workload_preflight_errors,
)


def launch_controlled_process(command: list[str], **kwargs):
    return _launch_controlled_process(
        command, utc_offset_fn=windows_host_utc_offset_minutes, **kwargs,
    )


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
    saved_tg = set(saved["tg_tokens"] or config.LLAMABENCH_TG) if saved else set(config.LLAMABENCH_TG)
    tg_vars = {value: tk.BooleanVar(value=value in saved_tg) for value in TG_TOKEN_OPTIONS}
    options = effective_gui_options(saved)
    if options["gpu_split_mode"] not in gpu_split_modes:
        options["gpu_split_mode"] = "layer"
    option_vars: dict[str, tk.Variable] = {
        key: (tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value)))
        for key, value in options.items()
    }
    option_vars["gpu_split_mode"].set(GPU_SPLIT_MODE_LABELS[options["gpu_split_mode"]])
    if not option_vars["comfyui"].get():
        option_vars["comfyui"].set(str(detected_comfyui))
    preset_var = tk.StringVar(value=restored_preset_name(saved))
    active_project: dict[str, dict | None] = {"value": None}
    project_status = tk.StringVar(value="No project loaded")

    def perform_vllm_update(control):
        return perform_vllm_version_update(None, control)

    def perform_vllm_version_update(version, control):
        snapshot = collect_engine_management(get_engine, hardware_backend)
        status = next(item for item in snapshot.statuses if item.engine == "vllm")
        support = vllm_update_support(status, setup, platform.machine())
        if support is None:
            return RuntimeUpdateResult(False, "This vLLM runtime is not app managed or updateable.")
        return update_managed_vllm(
            support, config.VLLM_VENV, control=control, log=control.log, version=version,
        )

    def perform_llamacpp_update(control):
        return perform_llamacpp_version_update(None, control)

    def perform_llamacpp_version_update(tag, control):
        release_fetcher = (lambda: fetch_llamacpp_release_tag(tag)) \
            if tag else fetch_llamacpp_release
        snapshot = collect_engine_management(get_engine, hardware_backend)
        status = next(item for item in snapshot.statuses if item.engine == "llamacpp")
        if not status.managed and platform.system() != "Darwin":
            return RuntimeUpdateResult(False, "This llama.cpp runtime is not app managed.")
        if platform.system() == "Darwin":
            return update_macos_llamacpp(
                config.LLAMACPP_DIR, platform.machine(), control=control,
                release_fetcher=release_fetcher,
            )
        if platform.system() == "Windows":
            return update_windows_llamacpp(
                config.LLAMACPP_DIR, detect_nvidia_max_cuda_version(), control=control,
                release_fetcher=release_fetcher,
            )
        return rebuild_managed_llamacpp(
            config.LLAMACPP_DIR, status.backend, control=control, log=control.log,
            release_fetcher=release_fetcher,
        )

    def perform_llamacpp_model_probe(tag, control):
        engine = get_engine("llamacpp")
        paths = getattr(engine, "model_paths", lambda _tag: ())(tag)
        if not paths:
            return ModelCompatibility(
                "llamacpp", tag, None, "unavailable", f"Model files for {tag} were not found.",
            )
        return probe_llamacpp_load(
            tag, paths[0], getattr(engine, "runtime_location", lambda: None)(), control=control,
        )

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
        vllm_updater=perform_vllm_update,
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
            ) else perform_vllm_version_update
        ),
        llamacpp_updater=perform_llamacpp_update,
        llamacpp_update_prompt=llamacpp_update_prompts.get(platform.system()),
        llamacpp_release_loader=fetch_llamacpp_releases,
        llamacpp_version_updater=perform_llamacpp_version_update,
        llamacpp_model_probe=perform_llamacpp_model_probe,
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
    split_labels = gpu_split_mode_labels(gpu_split_modes)
    ttk.Label(execution_box, text="GPU mode").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Combobox(
        execution_box, state="readonly", textvariable=option_vars["gpu_split_mode"],
        values=split_labels, width=30,
    ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=2)
    ttk.Button(
        execution_box, text="Reset", width=6,
        command=lambda: option_vars["gpu_split_mode"].set(GPU_SPLIT_MODE_LABELS["layer"]),
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
    ttk.Checkbutton(
        execution_box, text="Disable llama.cpp weight repacking (-nr)",
        variable=option_vars["llamacpp_no_repack"],
    ).grid(row=12, column=0, columnspan=2, sticky="w")
    ttk.Button(
        execution_box, text="Reset", width=6,
        command=lambda: option_vars["llamacpp_no_repack"].set(False),
    ).grid(row=12, column=2, padx=(8, 0))
    ttk.Label(
        execution_box, text="More warmups/runs improve repeatability but increase time. CPU-only changes the tested device; force-all can make runs much longer.",
        wraplength=430,
    ).grid(row=13, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Button(
        execution_box, text="Clear Crash Caches", command=clear_all_crash_caches,
    ).grid(row=16, column=0, sticky="w", pady=(8, 0))

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
                    "cpu_only", "force_all", "retry_crashed_models", "offline",
                    "llamacpp_no_repack"):
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
                if key == "gpu_split_mode":
                    option_vars[key].set(GPU_SPLIT_MODE_LABELS[value])
                else:
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
        row=14, column=0, columnspan=2, sticky="w", pady=(8, 0),
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

    def start_history_retry(result_path, report, selected):
        nonlocal process, active_process_kind, active_result_paths
        if process is not None and process.poll() is None:
            messagebox.showerror("Benchmark active", "Stop the active process first.", parent=root)
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

    def start_history_action(action, result_path, report, selected):
        if action == "resume":
            start_history_recovery(result_path, report)
        elif action == "retry":
            start_history_retry(result_path, report, selected)
        else:
            start_history_fork(result_path, report)

    history_actions = HistoryActions(
        history_screen, root=root, tk=tk, ttk=ttk, filedialog=filedialog,
        messagebox=messagebox,
        process_active=lambda: process is not None and process.poll() is None,
        review_outbound_metadata=review_outbound_metadata,
        start_recovery=start_history_action,
    )
    history_actions.bind()
    refresh_history = history_actions.refresh

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
        process = None
        if exit_code == 0:
            try:
                write_run_logs(
                    current_log(), result_paths_for_log(current_log(), active_result_paths),
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
                if exit_code == 0 else
                f"{label} stopped with exit code {exit_code}. Preserved evidence remains available."
            )
        else:
            run_status.set(format_run_outcome(exit_code))
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
        values = dict(GUI_OPTION_DEFAULTS)
        for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget"):
            try:
                values[key] = int(option_vars[key].get())
            except ValueError:
                values[key] = option_vars[key].get()
        selected_split_label = option_vars["gpu_split_mode"].get()
        values["gpu_split_mode"] = gpu_split_mode_value(selected_split_label)
        for key in ("cpu_only", "force_all", "retry_crashed_models", "offline",
                    "llamacpp_no_repack"):
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
