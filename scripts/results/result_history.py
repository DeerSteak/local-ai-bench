"""Local result discovery, filtering, and methodology-safe baseline comparison."""

import json
import math
from pathlib import Path

from scripts.results.result_store import validate_json_data


PERFORMANCE_METRICS = {
    "llm": ("tps_mean", "ttft_mean_sec"),
    "llm_conversation": ("tps_mean", "client_ttft_mean_sec", "server_prompt_mean_sec"),
    "concurrency_tool": ("aggregate_tps", "ttft_mean_sec"),
    "concurrency_chat": ("aggregate_tps", "ttft_mean_sec"),
}
ACCURACY_SECTIONS = ("mcq", "math", "reasoning", "code", "tool")


def _run_settings(result: dict) -> dict:
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
    settings = plan.get("effective_config") if isinstance(plan.get("effective_config"), dict) else {}
    return settings


def summarize_result(result: dict, path: Path) -> dict:
    if not isinstance(result, dict) or not isinstance(result.get("profile"), dict):
        raise ValueError("not a benchmark result")
    validate_json_data(result)
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    profile = result["profile"]
    stages = run.get("stages") if isinstance(run.get("stages"), dict) else {}
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
        percent = (delta / before * 100) if delta is not None and before != 0 else None
        rows.append({"metric": key, "baseline": before, "candidate": after,
                     "delta": delta, "percent_change": percent})
    return {"compatible": not incompatible, "incompatible_fields": sorted(set(incompatible)),
            "rows": rows}
