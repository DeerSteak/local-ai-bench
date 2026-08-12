"""State-changing recovery for plans composed entirely of journal-owned stages."""

import json
import sys
from pathlib import Path

from scripts.app.supervised_stage import run_supervised_stage
from scripts.results.event_store import EventStore
from scripts.results.llm_event_stage import event_store_path
from scripts.runtime.pause_control import apply_pause_evidence
from scripts.results.recovery_inspector import (
    JOURNAL_STAGES, SELECTED_RETRY_STAGES, current_resume_identity, inspect_recovery,
)
from scripts.results.result_store import ResultStore, build_run_manifest
from scripts.results.run_plan import RunPlan, load_run_plan
from scripts.stage_registry import stage_spec


SECTION_BY_STAGE = {key: stage_spec(key).section for key in JOURNAL_STAGES}
FAMILY_BY_STAGE = {key: stage_spec(key).model_family for key in JOURNAL_STAGES}


def _finish_result(store, data, status, reason=None):
    apply_pause_evidence(data["run"])
    store.finish(status, reason)


def _run_stage(plan, journal_path, stage, save, identity, resume, selected_case_ids=None):
    return run_supervised_stage(
        plan, journal_path, stage, save, resume_identity=identity, resume=resume,
        selected_case_ids=selected_case_ids,
    )


def _finish_journal_job(journal_path, plan, state, reason=None, preserve_terminal=False):
    journal = EventStore(journal_path)
    try:
        current = journal.rebuild(plan.job_id)["jobs"][plan.job_id]["state"]
        if preserve_terminal and current in {"complete", "invalid", "skipped", "timed_out",
                                             "interrupted", "failed"}:
            return
        journal.finish_job(plan.job_id, state, reason)
    finally:
        journal.close()


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
    journal_path = event_store_path(result_path)
    journal = EventStore(journal_path)
    try:
        journal.resume_job(plan.job_id)
    finally:
        journal.close()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    store = ResultStore(result_path, data)
    store.begin_recovery()
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
        _finish_journal_job(journal_path, plan, "complete")
        _finish_result(store, data, "complete")
    except BaseException as exc:
        terminal_status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        active = next((
            key for key, value in data["run"]["stages"].items()
            if value.get("status") == "running"
        ), None)
        if active:
            store.complete_stage(
                active, SECTION_BY_STAGE[active], status=terminal_status,
                reason=type(exc).__name__,
            )
        _finish_journal_job(
            journal_path, plan, terminal_status, type(exc).__name__, preserve_terminal=True,
        )
        _finish_result(store, data, terminal_status, type(exc).__name__)
        raise
    return data


def retry_selected_cases(result_path, case_ids, *, identity_builder=current_resume_identity,
                         stage_runner=_run_stage):
    """Retry an explicit eligible subset from one stopped journal stage."""
    result_path = Path(result_path).resolve()
    requested = tuple(dict.fromkeys(case_ids))
    if not requested:
        raise ValueError("select at least one retry-eligible case")
    plan = load_run_plan(result_path)
    identity = identity_builder(plan)
    inspection = inspect_recovery(result_path, lambda _plan: identity)
    if not inspection["can_resume"]:
        raise ValueError("fork required: " + "; ".join(inspection["reasons"]))
    candidates = {case["case_id"]: case for case in inspection["retryable_cases"]}
    unknown = set(requested) - set(candidates)
    if unknown:
        raise ValueError("cases are not retry-eligible: " + ", ".join(sorted(unknown)))
    stages = {candidates[case_id]["stage"] for case_id in requested}
    if len(stages) != 1:
        raise ValueError("selected retry cases must belong to one stage")
    stage = stages.pop()
    if stage not in SELECTED_RETRY_STAGES:
        raise ValueError(f"stage does not support selected retry: {stage}")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    store = ResultStore(result_path, data)
    store.begin_recovery()
    selected_models = len(plan.models[FAMILY_BY_STAGE[stage]])
    store.resume_stage(stage, selected_models)
    section = SECTION_BY_STAGE[stage]
    journal_path = event_store_path(result_path)
    try:
        result = stage_runner(
            plan, journal_path, stage,
            lambda value: store.update_section(section, value, stage),
            identity, True, list(requested),
        )
        store.update_section(section, result, stage)
        journal = EventStore(journal_path)
        try:
            journal_state = journal.rebuild(plan.job_id)["stages"][plan.stage_id(stage)]["state"]
        finally:
            journal.close()
        status = "complete" if journal_state == "complete" else "failed"
        reason = None if status == "complete" else "selected_retry_has_remaining_cases"
        store.complete_stage(stage, section, status=status, reason=reason)
        _finish_journal_job(journal_path, plan, status, reason)
        _finish_result(store, data, status, reason)
    except BaseException as exc:
        terminal_status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        store.complete_stage(
            stage, section, status=terminal_status, reason=type(exc).__name__,
        )
        _finish_journal_job(
            journal_path, plan, terminal_status, type(exc).__name__, preserve_terminal=True,
        )
        _finish_result(store, data, terminal_status, type(exc).__name__)
        raise
    return data


def fork_journal_run(source_path, output_path, *, identity_builder=current_resume_identity,
                     stage_runner=_run_stage):
    """Execute a saved journal-only plan under a new durable job and result identity."""
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    source_plan = load_run_plan(source_path)
    unsupported = [stage for stage in source_plan.stage_order if stage not in JOURNAL_STAGES]
    if unsupported:
        raise ValueError(
            "saved plan contains stages without durable recovery: " + ", ".join(unsupported)
        )
    journal_path = event_store_path(output_path)
    if output_path.exists() or journal_path.exists():
        raise ValueError("fork output or event journal already exists")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_run = source.get("run", {})
    plan = RunPlan.create(
        application_version=source_plan.application_version,
        engine_name=source_plan.engine_name, tests=source_plan.tests,
        stage_order=source_plan.stage_order, models=source_plan.models,
        effective_config=source_plan.effective_config,
    )
    identity = identity_builder(plan)
    data = {"run": build_run_manifest(plan=plan, repo_root=Path(__file__).parents[2])}
    data["run"]["forked_from"] = {
        "run_id": source_run.get("run_id"), "job_id": source_plan.job_id,
        "plan_id": source_plan.plan_id,
    }
    for section in SECTION_BY_STAGE.values():
        data[section] = {}
    store = ResultStore(output_path, data)
    apply_pause_evidence(data["run"])
    store.checkpoint()
    try:
        for stage in plan.stage_order:
            selected_models = len(plan.models[FAMILY_BY_STAGE[stage]])
            store.start_stage(stage, selected_models)
            section = SECTION_BY_STAGE[stage]
            result = stage_runner(
                plan, journal_path, stage,
                lambda value, section=section, stage=stage:
                    store.update_section(section, value, stage),
                identity, False,
            )
            store.update_section(section, result, stage)
            store.complete_stage(stage, section)
        _finish_journal_job(journal_path, plan, "complete")
        _finish_result(store, data, "complete")
    except BaseException as exc:
        terminal_status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        active = next((
            key for key, value in data["run"]["stages"].items()
            if value.get("status") == "running"
        ), None)
        if active:
            store.complete_stage(
                active, SECTION_BY_STAGE[active], status=terminal_status,
                reason=type(exc).__name__,
            )
        _finish_journal_job(
            journal_path, plan, terminal_status, type(exc).__name__, preserve_terminal=True,
        )
        _finish_result(store, data, terminal_status, type(exc).__name__)
        raise
    return data


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.results.recovery_executor RESULT.json")
    try:
        recovered = resume_journal_run(Path(sys.argv[1]))
    except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "error": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({
        "schema_version": 1, "status": recovered["run"]["status"],
        "result": str(Path(sys.argv[1]).resolve()),
    }, indent=2))
