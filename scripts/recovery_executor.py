"""State-changing recovery for plans composed entirely of journal-owned stages."""

import json
import sys
from pathlib import Path

from benchmark import run_supervised_stage
from llm_event_stage import event_store_path
from recovery_inspector import JOURNAL_STAGES, current_resume_identity, inspect_recovery
from result_store import ResultStore
from run_plan import load_run_plan


SECTION_BY_STAGE = {
    "llm": "llm", "conv": "llm_conversation", "llamabench": "llamabench",
    "conc_tool": "concurrency_tool", "conc_chat": "concurrency_chat",
}
FAMILY_BY_STAGE = {
    "llm": "llm", "conv": "llm", "llamabench": "llm",
    "conc_tool": "concurrency", "conc_chat": "concurrency",
}


def _run_stage(plan, journal_path, stage, save, identity, resume):
    return run_supervised_stage(
        plan, journal_path, stage, save, resume_identity=identity, resume=resume,
    )


def resume_journal_run(result_path, *, identity_builder=current_resume_identity,
                       stage_runner=_run_stage):
    """Resume a stopped journal-only result after a read-only eligibility decision."""
    result_path = Path(result_path).resolve()
    plan = load_run_plan(result_path)
    unsupported = [stage for stage in plan.stage_order if stage not in JOURNAL_STAGES]
    if unsupported:
        raise ValueError(
            "saved plan contains stages without durable recovery: " + ", ".join(unsupported)
        )
    identity = identity_builder(plan)
    inspection = inspect_recovery(result_path, lambda _plan: identity)
    if not inspection["can_resume"]:
        raise ValueError("fork required: " + "; ".join(inspection["reasons"]))
    data = json.loads(result_path.read_text(encoding="utf-8"))
    store = ResultStore(result_path, data)
    store.begin_recovery()
    journal_path = event_store_path(result_path)
    try:
        for stage in plan.stage_order:
            record = data["run"]["stages"].get(stage)
            if record and record["status"] == "complete":
                continue
            selected_models = len(plan.models[FAMILY_BY_STAGE[stage]])
            if record is None:
                store.start_stage(stage, selected_models)
            else:
                store.resume_stage(stage, selected_models)
            section = SECTION_BY_STAGE[stage]
            resume = inspection["stage_states"].get(stage) != "pending"
            result = stage_runner(
                plan, journal_path, stage,
                lambda value, section=section, stage=stage:
                    store.update_section(section, value, stage),
                identity, resume,
            )
            store.update_section(section, result, stage)
            store.complete_stage(stage, section)
        store.finish("complete")
    except BaseException as exc:
        active = next((
            key for key, value in data["run"]["stages"].items()
            if value.get("status") == "running"
        ), None)
        if active:
            store.complete_stage(
                active, SECTION_BY_STAGE[active], status="failed",
                reason=type(exc).__name__,
            )
        store.finish("failed", type(exc).__name__)
        raise
    return data


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/recovery_executor.py RESULT.json")
    try:
        recovered = resume_journal_run(Path(sys.argv[1]))
    except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "error": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({
        "schema_version": 1, "status": recovered["run"]["status"],
        "result": str(Path(sys.argv[1]).resolve()),
    }, indent=2))
