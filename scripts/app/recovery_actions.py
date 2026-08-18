"""Recovery command construction and review presentation for the GUI."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.results.run_plan import load_run_plan
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


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
        f"Decision: {report['action'].upper()}", f"Plan: {report['plan_id']}",
        f"Interrupted attempts: {report['interrupted_attempts']}", "", "Stages:",
    ]
    lines += [f"  {stage}: {state}" for stage, state in report["stage_states"].items()]
    lines += ["", "Cases:"]
    lines += [f"  {state}: {count}" for state, count in report["case_counts"].items()]
    retryable = report.get("retryable_cases", [])
    if retryable:
        lines += ["", "Retry candidates:"]
        lines += [f"  {case['stage']}: {case['label']} ({case['state']})" for case in retryable]
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
                checked=True, kind=kind, value=key[1],
                label=labels.get(model.get("tag"), model.get("tag") or model.get("short")),
            ))
    catalogs = {
        "embeddings": ("embedding", {model["short"]: model["label"] for model in EMBED_MODELS}),
        "images": ("image", {model["short"]: model["label"] for model in IMAGE_MODELS}),
    }
    for family, (kind, family_labels) in catalogs.items():
        for model in plan.models[family]:
            short = model["short"]
            if model_shorts is not None and short not in model_shorts:
                continue
            entries.append(SimpleNamespace(
                checked=True, kind=kind, value=short,
                label=family_labels.get(short, short),
            ))
    return entries
