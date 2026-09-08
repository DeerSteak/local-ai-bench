"""Local result discovery, filtering, and methodology-safe baseline comparison."""

import json
import shutil
import statistics
from datetime import datetime
from pathlib import Path

from scripts.results.local_execution_context import local_execution_path
from scripts.results.result_store import as_dict, validate_json_data
from scripts.results.significance import compare_metric, metric_evidence
from scripts.stage_registry import ACCURACY_TESTS


PERFORMANCE_METRICS = {
    "llm": ("tps_mean", "ttft_mean_sec"),
    "llm_conversation": ("tps_mean", "client_ttft_mean_sec", "server_prompt_mean_sec"),
    "llm_cached": ("tps_mean", "prefill_tps_mean", "server_prompt_mean_sec"),
    "concurrency_tool": ("aggregate_tps", "ttft_mean_sec"),
    "concurrency_chat": ("aggregate_tps", "ttft_mean_sec"),
}
DISPERSION_KEYS = {
    "tps_mean": "tps_stdev",
    "ttft_mean_sec": "ttft_stdev_sec",
    "client_ttft_mean_sec": "client_ttft_stdev_sec",
    "server_prompt_mean_sec": "server_prompt_stdev_sec",
    "chunks_per_sec_mean": "chunks_per_sec_stdev",
    "sec_per_image_mean": "sec_per_image_stdev",
}
ACCURACY_SECTIONS = tuple(ACCURACY_TESTS)
ETA_MATCH_KEYS = (
    "runs", "warmup_runs", "run_timeout_seconds", "accuracy_timeout_seconds",
    "accuracy_token_budget", "cpu_only", "force_all", "max_prompt_tokens",
    "context_lengths", "llamabench_pp", "llamabench_tg", "sample_size",
    "concurrency_tool_levels", "concurrency_chat_levels",
    "concurrency_tool_context", "concurrency_chat_context",
    "concurrency_chat_soft_exit_floor",
    "mtp_enabled",
)
SUSTAINED_ETA_KEYS = (
    "sustained_duration_sec", "sustained_window_sec", "sustained_context_tokens",
)
HARDWARE_IDENTITY_KEYS = ("hostname", "os", "arch", "ram_gb", "backend", "wsl")
BACKEND_LABELS = {
    "cuda": "CUDA", "rocm": "ROCm", "metal": "Metal",
    "vulkan": "Vulkan", "xpu": "XPU", "cpu": "CPU",
}


def hardware_identity(profile: dict) -> dict:
    """Stable hardware fields used to scope historical duration estimates."""
    values = dict(profile)
    values["backend"] = profile.get("hardware_backend", profile.get("backend"))
    values["wsl"] = bool(profile.get("wsl", False))
    return {key: values.get(key) for key in HARDWARE_IDENTITY_KEYS}


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def completed_run_duration_seconds(result: dict) -> float | None:
    run = as_dict(result.get("run"))
    if run.get("status") != "complete":
        return None
    started, finished = _timestamp(run.get("started_at")), _timestamp(run.get("finished_at"))
    if started is None or finished is None:
        return None
    try:
        seconds = (finished - started).total_seconds()
    except TypeError:
        return None
    return seconds if seconds > 0 else None


def estimate_matching_plan_seconds(directory: Path, engine: str, tests: list[str],
                                   models: dict[str, list[dict]],
                                   effective_config: dict, profile: dict) -> float | None:
    """Median duration for exact local plan matches; unmatched history is not an ETA."""
    expected_models = {
        family: sorted(str(model.get("short")) for model in entries)
        for family, entries in models.items()
    }
    durations = []
    for path in Path(directory).glob("*.json") if Path(directory).exists() else ():
        try:
            result = load_result(path)
        except (OSError, ValueError):
            continue
        run = as_dict(result.get("run"))
        plan = as_dict(run.get("plan"))
        actual_models = {
            family: sorted(str(model.get("short")) for model in entries)
            for family, entries in as_dict(plan.get("models")).items()
            if isinstance(entries, list)
        }
        recorded_tests = plan.get("requested_tests")
        recorded_config = as_dict(plan.get("effective_config"))
        recorded_profile = as_dict(result.get("profile"))
        if ((result.get("engine") or run.get("engine")) != engine
                or not isinstance(recorded_tests, list)
                or sorted(recorded_tests) != sorted(tests)
                or actual_models != expected_models
                or any(key not in recorded_config
                       or recorded_config.get(key) != effective_config.get(key)
                       for key in (*ETA_MATCH_KEYS,
                                   *(SUSTAINED_ETA_KEYS if "sustained" in tests else ())))
                or any(key not in recorded_profile for key in HARDWARE_IDENTITY_KEYS[:-1])
                or hardware_identity(recorded_profile) != hardware_identity(profile)):
            continue
        duration = completed_run_duration_seconds(result)
        if duration is not None:
            durations.append(duration)
    return statistics.median(durations) if durations else None


def run_artifact_paths(result_path: Path, results_dir: Path) -> tuple[Path, ...]:
    """Exact repository-owned artifacts derived from one history result."""
    result_path, results_dir = Path(result_path), Path(results_dir)
    if result_path.parent.resolve() != results_dir.resolve():
        raise ValueError("history result is outside the results directory")
    stem = result_path.stem
    regraded_standard = stem.startswith("regraded_results_")
    regraded_custom = (
        stem.startswith("regraded_")
        and (results_dir / result_path.name[len("regraded_"):]).is_file()
    )
    regraded = regraded_standard or regraded_custom
    if regraded_standard:
        suffix = stem[len("regraded_results_"):]
    elif regraded_custom:
        suffix = stem[len("regraded_"):]
    else:
        suffix = stem[len("results_"):] if stem.startswith("results_") else stem
    prefix = "regraded_" if regraded else ""
    artifacts = [
        *(results_dir / f"{prefix}answers_{workload}_{suffix}.json"
          for workload in ACCURACY_SECTIONS),
    ]
    if not regraded:
        event_path = result_path.with_suffix(".events.sqlite3")
        artifacts.extend([
            results_dir / f"log_{suffix}.txt",
            results_dir / f"images_{suffix}",
            event_path,
            local_execution_path(event_path),
            *(Path(f"{event_path}{suffix}") for suffix in ("-wal", "-shm", "-journal")),
            *(results_dir / f"regraded_answers_{workload}_{suffix}.json"
              for workload in ACCURACY_SECTIONS),
            results_dir / f"regraded_{result_path.name}",
        ])
    artifacts.append(result_path)
    return tuple(dict.fromkeys(artifacts))


def existing_run_artifacts(result_path: Path, results_dir: Path) -> list[Path]:
    return [path for path in run_artifact_paths(result_path, results_dir)
            if path.exists() or path.is_symlink()]


def delete_run_artifacts(result_path: Path, results_dir: Path) -> tuple[list[Path], dict[Path, str]]:
    """Delete one run's exact artifacts, leaving the main JSON until last for retry."""
    removed = []
    failures = {}
    for path in existing_run_artifacts(result_path, results_dir):
        if path == result_path and failures:
            break
        try:
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
            removed.append(path)
        except OSError as exc:
            failures[path] = str(exc)
    return removed, failures


def delete_multiple_run_artifacts(result_paths: list[Path], results_dir: Path) -> \
        tuple[list[Path], dict[Path, str]]:
    """Delete selected runs independently so one failure does not block the rest."""
    removed = []
    failures = {}
    for result_path in result_paths:
        run_removed, run_failures = delete_run_artifacts(result_path, results_dir)
        removed.extend(run_removed)
        failures.update(run_failures)
    return removed, failures


def _run_settings(result: dict) -> dict:
    run = as_dict(result.get("run"))
    plan = as_dict(run.get("plan"))
    settings = as_dict(plan.get("effective_config"))
    return settings


def history_backend_label(profile: dict) -> str:
    backend = profile.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        return "Not recorded"
    normalized = backend.strip().casefold()
    return BACKEND_LABELS.get(normalized, backend.strip())


def history_mtp_label(settings: dict) -> str:
    enabled = settings.get("mtp_enabled")
    if enabled is True:
        return "On"
    if enabled is False:
        return "Off"
    return "Not recorded"


def summarize_result(result: dict, path: Path | str) -> dict:
    if not isinstance(result, dict) or not isinstance(result.get("profile"), dict):
        raise ValueError("not a benchmark result")
    validate_json_data(result)
    run = as_dict(result.get("run"))
    profile = result["profile"]
    stages = as_dict(run.get("stages"))
    settings = _run_settings(result)
    return {
        "path": str(Path(path).resolve()),
        "started_at": run.get("started_at") or "Not recorded",
        "system": str(profile.get("hostname") or "Unnamed system"),
        "status": str(run.get("status") or "legacy"),
        "engine": str(result.get("engine") or run.get("engine") or "Not recorded"),
        "runtime_backend": history_backend_label(profile),
        "mtp": history_mtp_label(settings),
        "version": str(result.get("version") or "Not recorded"),
        "methodology_profile": str(settings.get("methodology_profile") or "unrecorded"),
        "models_with_results": sum(
            int(stage.get("models_with_results") or 0)
            for stage in stages.values() if isinstance(stage, dict)
        ),
    }


def discover_results(directory: Path) -> tuple[list[dict], list[dict]]:
    entries = []
    skipped = []
    directory = Path(directory)
    if not directory.exists():
        return entries, skipped
    for path in sorted(directory.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            entries.append(summarize_result(result, path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            skipped.append({"path": str(path.resolve()), "reason": str(exc)})
    entries.sort(key=lambda item: (item["started_at"], item["path"]), reverse=True)
    return entries, skipped


def filter_results(entries, *, query: str = "", status: str = "all", engine: str = "all") -> list[dict]:
    needle = query.strip().casefold()
    return [entry for entry in entries if (
        (status == "all" or entry["status"] == status)
        and (engine == "all" or entry["engine"] == engine)
        and (not needle or needle in " ".join((
            entry["system"], entry["path"], entry["version"], entry["methodology_profile"],
            entry["runtime_backend"], f"MTP {entry['mtp']}",
        )).casefold())
    )]


def load_result(path: Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    return result


def extract_comparable_metrics(result: dict) -> dict[str, dict]:
    metrics = {}
    for section, names in PERFORMANCE_METRICS.items():
        section_data = result.get(section)
        if not isinstance(section_data, dict):
            continue
        for model, cases in section_data.items():
            if not isinstance(cases, dict):
                continue
            for case, values in cases.items():
                if not isinstance(values, dict):
                    continue
                for metric in names:
                    evidence = metric_evidence(values, metric, DISPERSION_KEYS.get(metric))
                    if evidence["value"] is not None:
                        metrics[f"{section}/{model}/{case}/{metric}"] = evidence
    embeddings = result.get("embeddings")
    if isinstance(embeddings, dict):
        for model, values in embeddings.items():
            if isinstance(values, dict):
                evidence = metric_evidence(values, "chunks_per_sec_mean", "chunks_per_sec_stdev")
                if evidence["value"] is not None:
                    metrics[f"embeddings/{model}/chunks_per_sec_mean"] = evidence
    images = result.get("images")
    if isinstance(images, dict):
        for model, values in images.items():
            resolutions = values.get("resolutions") if isinstance(values, dict) else None
            if not isinstance(resolutions, dict):
                continue
            for resolution, evidence in resolutions.items():
                if isinstance(evidence, dict):
                    normalized = metric_evidence(
                        evidence, "sec_per_image_mean", "sec_per_image_stdev")
                    if normalized["value"] is not None:
                        metrics[f"images/{model}/{resolution}/sec_per_image_mean"] = normalized
    for section in ACCURACY_SECTIONS:
        section_data = result.get(section)
        if not isinstance(section_data, dict):
            continue
        for model, evidence in section_data.items():
            if isinstance(evidence, dict):
                normalized = metric_evidence(evidence, "accuracy_pct", None)
                if normalized["value"] is not None:
                    metrics[f"{section}/{model}/accuracy_pct"] = normalized
    return metrics


def _comparison_settings(settings: dict) -> dict:
    comparable = dict(settings)
    if not comparable.get("memory_telemetry") \
            or comparable.get("memory_telemetry_interval_sec") == 0.5:
        comparable.pop("memory_telemetry", None)
        comparable.pop("memory_telemetry_interval_sec", None)
    return comparable


def compare_results(baseline: dict, candidate: dict) -> dict:
    baseline_settings = _run_settings(baseline)
    candidate_settings = _run_settings(candidate)
    baseline_profile = baseline_settings.get("methodology_profile")
    candidate_profile = candidate_settings.get("methodology_profile")
    identity = {
        "version": (baseline.get("version"), candidate.get("version")),
        "engine": (baseline.get("engine"), candidate.get("engine")),
        "methodology_profile": (baseline_profile, candidate_profile),
        "effective_config": (
            _comparison_settings(baseline_settings), _comparison_settings(candidate_settings),
        ),
    }
    incompatible = [key for key, values in identity.items() if values[0] != values[1]]
    if baseline_profile is None or candidate_profile is None:
        incompatible.append("unrecorded_methodology")
    baseline_metrics = extract_comparable_metrics(baseline)
    candidate_metrics = extract_comparable_metrics(candidate)
    rows = []
    for key in sorted(set(baseline_metrics) | set(candidate_metrics)):
        metric = key.rsplit("/", 1)[-1]
        rows.append({"metric": key, **compare_metric(
            metric, baseline_metrics.get(key), candidate_metrics.get(key))})
    return {"compatible": not incompatible, "incompatible_fields": sorted(set(incompatible)),
            "rows": rows}
