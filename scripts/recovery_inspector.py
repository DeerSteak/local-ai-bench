"""Read-only recovery eligibility and durable coverage inspection."""

import json
import sys
from pathlib import Path

import config
from engines import get_engine
from event_store import EventStore
from llamacpp_tools import find_llamacpp_tool
from llm_event_stage import event_store_path
from resume_policy import assess_resume, build_engine_resume_identity
from run_plan import load_run_plan
from shared import Shared


JOURNAL_STAGES = {"llm", "conv", "llamabench", "conc_tool", "conc_chat"}


def current_resume_identity(plan, *, profile=None, engine=None, tool_finder=find_llamacpp_tool,
                            digest_cache_path=config.RESUME_DIGEST_CACHE_PATH):
    """Discover the current local identities needed by the plan's journal stages."""
    engine = engine or get_engine(plan.engine_name)
    stages = set(plan.stage_order) & JOURNAL_STAGES
    families = []
    if stages & {"llm", "conv", "llamabench"}:
        families.append("llm")
    if stages & {"conc_tool", "conc_chat"}:
        families.append("concurrency")
    extra = {}
    if "llamabench" in stages:
        binary = tool_finder("llama-bench")
        if not binary:
            raise ValueError("llama-bench runtime required by the saved plan was not found")
        extra["llama-bench"] = Path(binary).resolve()
    return build_engine_resume_identity(
        plan, engine, model_families=families,
        include_engine_runtime=bool(stages - {"llamabench"}), extra_runtimes=extra,
        digest_cache_path=digest_cache_path, environment=profile or Shared.build_profile(),
    )


def inspect_recovery(result_path, identity_builder=current_resume_identity):
    """Return a redacted recovery decision without changing the result or journal."""
    result_path = Path(result_path).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    plan = load_run_plan(result_path)
    journal_path = event_store_path(result_path)
    if not journal_path.is_file():
        raise ValueError("result has no durable event journal")
    store = EventStore(journal_path)
    try:
        store.verify(plan.job_id)
        projection = store.rebuild(plan.job_id)
        decision = assess_resume(
            store.resume_identity(plan.job_id), identity_builder(plan), projection,
            list(projection["cases"]),
        )
    finally:
        store.close()
    stage_states = {
        stage: projection["stages"].get(plan.stage_id(stage), {}).get("state", "pending")
        for stage in plan.stage_order if stage in JOURNAL_STAGES
    }
    case_states = {}
    for case in projection["cases"].values():
        case_states[case["state"]] = case_states.get(case["state"], 0) + 1
    reasons = list(decision.reasons)
    if result.get("run", {}).get("status") == "complete":
        reasons.append("result is already complete")
    can_resume = not reasons
    return {
        "schema_version": 1, "action": "resume" if can_resume else "fork",
        "can_resume": can_resume, "reasons": reasons,
        "job_id": plan.job_id, "plan_id": plan.plan_id,
        "stage_states": stage_states, "case_counts": dict(sorted(case_states.items())),
        "interrupted_attempts": len(decision.interrupted_attempts),
    }


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/recovery_inspector.py RESULT.json")
    try:
        report = inspect_recovery(Path(sys.argv[1]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "error": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["can_resume"] else 1)
