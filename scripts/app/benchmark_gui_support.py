"""Pure data, process, progress, and resource helpers for the benchmark GUI."""

import json
from pathlib import Path

from scripts.runtime import config
from scripts.app.benchmark_frontend import (
    GUI_OPTION_DEFAULTS, LLM_BACKED_TESTS, TEST_DEFINITIONS, MenuEntry,
)
from scripts.runtime.progress_events import PROGRESS_PREFIX
GPU_SPLIT_MODE_LABELS = {
    "single": "Single GPU",
    "layer": "Layer split (recommended)",
    "tensor": "Tensor parallel (experimental)",
}
MTP_MODE_LABELS = {"off": "Off", "on": "On", "both": "Both"}

def gpu_split_mode_labels(modes) -> tuple[str, ...]:
    return tuple(GPU_SPLIT_MODE_LABELS[mode] for mode in modes)


def gpu_split_mode_value(label: str) -> str:
    if label in GPU_SPLIT_MODE_LABELS:
        return label
    for mode, candidate in GPU_SPLIT_MODE_LABELS.items():
        if candidate == label:
            return mode
    raise ValueError(f"Unknown GPU mode: {label}")


def mtp_mode_value(label: str) -> str:
    if label in MTP_MODE_LABELS:
        return label
    for mode, candidate in MTP_MODE_LABELS.items():
        if candidate == label:
            return mode
    raise ValueError(f"Unknown MTP mode: {label}")


def effective_gui_options(state: dict | None) -> dict:
    options = state.get("gui_options") if state else None
    effective = dict(options) if options is not None else dict(GUI_OPTION_DEFAULTS)
    try:
        effective["gpu_split_mode"] = gpu_split_mode_value(effective["gpu_split_mode"])
    except (KeyError, TypeError, ValueError):
        effective["gpu_split_mode"] = "layer"
    try:
        effective["mtp"] = mtp_mode_value(effective["mtp"])
    except (KeyError, TypeError, ValueError):
        effective["mtp"] = "off"
    return effective


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
        newly_finished = identity not in finished
        finished.add(identity)
        updated["finished_models"] = finished
        if newly_finished and isinstance(event.get("elapsed_seconds"), (int, float)):
            updated["last_completion_elapsed"] = event["elapsed_seconds"]
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


def estimate_remaining_seconds(elapsed: float, completed: int, total: int,
                               last_completion_elapsed: float | None = None) -> int | None:
    if elapsed < 0 or completed <= 0 or total <= completed:
        return 0 if total > 0 and completed >= total else None
    calibrated_at = elapsed if last_completion_elapsed is None else last_completion_elapsed
    if calibrated_at < 0 or calibrated_at > elapsed:
        return None
    projected_remaining = (calibrated_at / completed) * (total - completed)
    return max(0, round(projected_remaining - (elapsed - calibrated_at)))


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
    "Consumer guidance": {"tests": ["llm", "llm_cached", "conv"], "max_prompt_tokens": 32768},
    "Vendor validation": {"tests": ["llm", "llm_cached", "conv", "llamabench", "emb", "mcq", "math", "reasoning", "code", "tool", "img"]},
    "Neutral comparison": {"tests": ["llm", "llm_cached", "conv", "emb", "img"]},
    "Platform optimized": {"tests": ["llm", "llm_cached", "conv", "llamabench"]},
    "Offline / private": {"tests": ["llm", "llm_cached", "conv", "emb"]},
    "Quick run": {"tests": ["llm", "emb"], "runs": 1, "max_prompt_tokens": 8192},
    "Full run": {"tests": [name for name, *_ in TEST_DEFINITIONS], "force_all": True},
    "Role: Orchestrator": {"tests": ["llm", "llm_cached", "conv", "reasoning", "tool", "conc_chat"]},
    "Role: Agent / tool caller": {"tests": ["llm", "llm_cached", "conv", "tool", "code", "conc_tool"], "max_prompt_tokens": 32768},
    "Role: Coding assistant": {"tests": ["llm", "llm_cached", "conv", "code", "reasoning"], "max_prompt_tokens": 32768},
    "Role: Chat assistant": {"tests": ["llm", "llm_cached", "conv", "mcq", "reasoning", "conc_chat"], "max_prompt_tokens": 8192},
    "Role: RAG / retrieval": {"tests": ["llm", "llm_cached", "conv", "emb", "mcq"], "max_prompt_tokens": 32768},
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
        f"Native MTP: {MTP_MODE_LABELS[options['mtp']]}",
        f"Offline: {'Yes' if options['offline'] else 'No'}",
        f"Memory telemetry: {'On' if options['memory_telemetry'] else 'Off'}",
        f"Power telemetry: {'On' if options['power_telemetry'] else 'Off'}",
        f"Sustained soak: {options['sustained_duration']} seconds per model",
        f"Ambient temperature: {options['ambient_temp_c'] if options['ambient_temp_c'] is not None else 'Not recorded'}",
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
            "CPU only", "Native MTP", "Offline", "Force slow models",
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
