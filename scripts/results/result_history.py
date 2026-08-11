"""Local result discovery, filtering, and methodology-safe baseline comparison."""

import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path

from scripts.results.result_store import as_dict, validate_json_data


PERFORMANCE_METRICS = {
    "llm": ("tps_mean", "ttft_mean_sec"),
    "llm_conversation": ("tps_mean", "client_ttft_mean_sec", "server_prompt_mean_sec"),
    "concurrency_tool": ("aggregate_tps", "ttft_mean_sec"),
    "concurrency_chat": ("aggregate_tps", "ttft_mean_sec"),
}
ACCURACY_SECTIONS = ("mcq", "math", "reasoning", "code", "tool")


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
                                   models: dict[str, list[dict]]) -> float | None:
    """Median duration for exact local plan matches; unmatched history is not an ETA."""
    expected_models = {
        family: sorted(str(model.get("short")) for model in entries)
        for family, entries in models.items()
    }
    durations = []
    for path in Path(directory).glob("*.json") if Path(directory).exists() else ():
        try:
            result = load_result(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        run = as_dict(result.get("run"))
        plan = as_dict(run.get("plan"))
        actual_models = {
            family: sorted(str(model.get("short")) for model in entries)
            for family, entries in as_dict(plan.get("models")).items()
            if isinstance(entries, list)
        }
        if ((result.get("engine") or run.get("engine")) != engine
                or plan.get("requested_tests") != tests
                or actual_models != expected_models):
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


def _run_settings(result: dict) -> dict:
    run = as_dict(result.get("run"))
    plan = as_dict(run.get("plan"))
    settings = as_dict(plan.get("effective_config"))
    return settings


def summarize_result(result: dict, path: Path | str) -> dict:
    if not isinstance(result, dict) or not isinstance(result.get("profile"), dict):
        raise ValueError("not a benchmark result")
    validate_json_data(result)
    run = as_dict(result.get("run"))
    profile = result["profile"]
    stages = as_dict(run.get("stages"))
    return {
        "path": str(Path(path).resolve()),
        "started_at": run.get("started_at") or "Not recorded",
        "system": str(profile.get("hostname") or "Unnamed system"),
        "status": str(run.get("status") or "legacy"),
        "engine": str(result.get("engine") or run.get("engine") or "Not recorded"),
        "version": str(result.get("version") or "Not recorded"),
        "methodology_profile": str(_run_settings(result).get("methodology_profile") or "unrecorded"),
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
        )).casefold())
    )]


def load_result(path: Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    return result


def extract_comparable_metrics(result: dict) -> dict[str, float]:
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
                    value = values.get(metric)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        metrics[f"{section}/{model}/{case}/{metric}"] = float(value)
    embeddings = result.get("embeddings")
    if isinstance(embeddings, dict):
        for model, values in embeddings.items():
            value = values.get("chunks_per_sec_mean") if isinstance(values, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                metrics[f"embeddings/{model}/chunks_per_sec_mean"] = float(value)
    images = result.get("images")
    if isinstance(images, dict):
        for model, values in images.items():
            resolutions = values.get("resolutions") if isinstance(values, dict) else None
            if not isinstance(resolutions, dict):
                continue
            for resolution, evidence in resolutions.items():
                value = evidence.get("sec_per_image_mean") if isinstance(evidence, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    metrics[f"images/{model}/{resolution}/sec_per_image_mean"] = float(value)
    for section in ACCURACY_SECTIONS:
        section_data = result.get(section)
        if not isinstance(section_data, dict):
            continue
        for model, evidence in section_data.items():
            value = evidence.get("accuracy_pct") if isinstance(evidence, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                metrics[f"{section}/{model}/accuracy_pct"] = float(value)
    return metrics


def compare_results(baseline: dict, candidate: dict) -> dict:
    baseline_settings = _run_settings(baseline)
    candidate_settings = _run_settings(candidate)
    baseline_profile = baseline_settings.get("methodology_profile")
    candidate_profile = candidate_settings.get("methodology_profile")
    identity = {
        "version": (baseline.get("version"), candidate.get("version")),
        "engine": (baseline.get("engine"), candidate.get("engine")),
        "methodology_profile": (baseline_profile, candidate_profile),
        "effective_config": (baseline_settings, candidate_settings),
    }
    incompatible = [key for key, values in identity.items() if values[0] != values[1]]
    if baseline_profile is None or candidate_profile is None:
        incompatible.append("unrecorded_methodology")
    baseline_metrics = extract_comparable_metrics(baseline)
    candidate_metrics = extract_comparable_metrics(candidate)
    rows = []
    for key in sorted(set(baseline_metrics) | set(candidate_metrics)):
        before = baseline_metrics.get(key)
        after = candidate_metrics.get(key)
        delta = after - before if before is not None and after is not None else None
        percent = (delta / before * 100) if delta is not None and before else None
        rows.append({"metric": key, "baseline": before, "candidate": after,
                     "delta": delta, "percent_change": percent})
    return {"compatible": not incompatible, "incompatible_fields": sorted(set(incompatible)),
            "rows": rows}
