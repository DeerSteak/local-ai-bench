"""Read-only recovery eligibility and durable coverage inspection."""

import json
import sys
from pathlib import Path

from scripts.runtime import config
from scripts.runtime.engines import get_engine
from scripts.results.event_store import EventStore
from scripts.runtime.llamacpp_tools import find_llamacpp_tool
from scripts.results.llm_event_stage import event_store_path
from scripts.results.local_execution_context import load_local_execution_context
from scripts.results.resume_policy import assess_resume, build_engine_resume_identity
from scripts.results.canonical_json import sha256_json
from scripts.results.run_plan import load_run_plan
from scripts.runtime.shared import Shared
from scripts.stage_registry import JOURNAL_STAGES, SELECTED_RETRY_STAGES
from scripts.stage_registry import ACCURACY_TESTS
from scripts.workloads.accuracy_registry import accuracy_spec
from scripts.workloads.embedding_benchmark import EmbeddingBenchmark
from scripts.workloads.image_benchmark import image_resume_artifacts, image_resume_runtimes
from scripts.workloads.models import IMAGE_MODELS


RETRYABLE_CASE_STATES = {"running", "failed", "interrupted", "invalid", "timed_out"}
META_CASE_KINDS = {"model_plan", "model_state", "model_evidence", "model_complete"}
ENGINE_BACKED_STAGES = {
    "llm", "conv", "llamabench", "llamabenchconc", "emb", "mcq", "math", "reasoning",
    "code", "tool", "conc_tool", "conc_chat", "sustained",
}


def retryable_case_records(plan, projection):
    stages = {
        plan.stage_id(stage): stage for stage in plan.stage_order
        if stage in SELECTED_RETRY_STAGES
    }
    records = []
    for case_id, case in projection["cases"].items():
        if (case.get("state") not in RETRYABLE_CASE_STATES
                or case.get("parent_id") not in stages
                or case.get("case_kind") not in {
                    "context", "sustained", "accuracy_question", "embedding_batch",
                    "image_resolution", "vllm_bench_case",
                }):
            continue
        details = []
        if case.get("context_label"):
            details.append(str(case["context_label"]))
        elif case.get("question_id"):
            details.append(str(case["question_id"]))
        elif case.get("case_kind") == "embedding_batch":
            details.append("input batch")
        elif case.get("case_kind") == "image_resolution":
            details.append(f"{case['width']}x{case['height']}")
        elif case.get("case_kind") == "vllm_bench_case":
            details.append(f"{case['kind']} in{case['input_len']}/out{case['output_len']}")
        elif case.get("case_kind"):
            details.append(str(case["case_kind"]).replace("_", " "))
        records.append({
            "case_id": case_id, "stage": stages[case["parent_id"]],
            "state": case["state"], "model": case.get("model_short", "unknown"),
            "label": " · ".join([case.get("model_short", "unknown"), *details]),
        })
    return sorted(records, key=lambda item: (plan.stage_order.index(item["stage"]), item["label"],
                                              item["case_id"]))


def workload_case_counts(plan, projection, stages) -> dict:
    """Count durable measurement states per workload while excluding model metadata."""
    stage_ids = {plan.stage_id(stage): stage for stage in stages}
    counts = {stage: {} for stage in stages}
    for case in projection["cases"].values():
        stage = stage_ids.get(case.get("parent_id"))
        if stage is None or case.get("case_kind") in META_CASE_KINDS:
            continue
        state_counts = counts[stage]
        state_counts[case["state"]] = state_counts.get(case["state"], 0) + 1
    return {stage: dict(sorted(values.items())) for stage, values in counts.items()}


def recovery_environment_profile(plan, engine) -> dict:
    """Reconstruct the run profile without its volatile timestamp."""
    hardware = Shared.build_profile()
    hardware_backend = hardware["backend"]
    backend = (engine.runtime_backend(hardware_backend, cpu_only=plan.cpu_only)
               if set(plan.tests) & ENGINE_BACKED_STAGES else hardware_backend)
    return {**hardware, "hardware_backend": hardware_backend, "backend": backend}


def legacy_environment_identity(current_profile: dict, saved_profile) -> dict | None:
    """Rebuild pre-fix identity using only the saved run timestamp."""
    if not isinstance(saved_profile, dict) or not isinstance(saved_profile.get("timestamp"), str):
        return None
    profile = {**current_profile, "timestamp": saved_profile["timestamp"]}
    return {"profile_sha256": sha256_json(profile)}


def current_resume_identity(plan, *, profile=None, engine=None, tool_finder=find_llamacpp_tool,
                            digest_cache_path=config.RESUME_DIGEST_CACHE_PATH,
                            event_path: Path | None = None):
    """Discover the current local identities needed by the plan's journal stages."""
    engine = engine or get_engine(plan.engine_name)
    stages = set(plan.stage_order) & JOURNAL_STAGES
    families = []
    accuracy_stages = stages & set(ACCURACY_TESTS)
    if stages & {"llm", "conv", "llamabench", "llamabenchconc", "vllmbench", "sustained", *ACCURACY_TESTS}:
        families.append("llm")
    if stages & {"conc_tool", "conc_chat"}:
        families.append("concurrency")
    if "emb" in stages:
        families.append("embeddings")
    extra = {}
    extra_artifacts = {
        f"bank:{stage}": accuracy_spec(stage).data_path for stage in accuracy_stages
    }
    if "emb" in stages:
        extra_artifacts["corpus:embeddings"] = EmbeddingBenchmark.EMBED_DOCUMENT_PATH
    if "img" in stages:
        if event_path is None:
            raise ValueError("image recovery requires its private local execution context")
        context = load_local_execution_context(event_path, plan.job_id)
        catalog = {model["short"]: model for model in IMAGE_MODELS}
        models = []
        for identity in plan.models["images"]:
            model = catalog.get(identity["short"])
            if model is None:
                raise ValueError(f"image model is absent from the catalog: {identity['short']}")
            models.append(model)
        extra_artifacts.update(image_resume_artifacts(models))
        extra.update(image_resume_runtimes(context.comfyui_dir))
    if "llamabench" in stages:
        binary = tool_finder("llama-bench")
        if not binary:
            raise ValueError("llama-bench runtime required by the saved plan was not found")
        extra["llama-bench"] = Path(binary).resolve()
    if "llamabenchconc" in stages:
        binary = tool_finder("llama-batched-bench")
        if not binary:
            raise ValueError("llama-batched-bench runtime required by the saved plan was not found")
        extra["llama-batched-bench"] = Path(binary).resolve()
    if "vllmbench" in stages:
        bench_executable = getattr(engine, "bench_executable", None)
        binary = bench_executable() if callable(bench_executable) else None
        if not isinstance(binary, (str, Path)):
            raise ValueError("vLLM bench runtime required by the saved plan was not found")
        extra["vllm-bench"] = Path(binary).resolve()
    profile = profile or recovery_environment_profile(plan, engine)
    return build_engine_resume_identity(
        plan, engine, model_families=families,
        include_engine_runtime=bool(stages & {
            "llm", "conv", "vllmbench", "sustained", "emb", "conc_tool", "conc_chat",
            *ACCURACY_TESTS,
        }), extra_runtimes=extra,
        extra_artifacts=extra_artifacts,
        digest_cache_path=digest_cache_path, environment=profile,
        use_digest_cache=False,
    )


def inspect_recovery(result_path, identity_builder=None):
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
        if identity_builder is None:
            engine = get_engine(plan.engine_name)
            current_profile = recovery_environment_profile(plan, engine)
            current_identity = current_resume_identity(
                plan, profile=current_profile, engine=engine, event_path=journal_path,
            )
            saved_identity = store.resume_identity(plan.job_id)
            legacy_environment = legacy_environment_identity(
                current_profile, result.get("profile"),
            )
            if (saved_identity.get("environment") != current_identity.get("environment")
                    and legacy_environment is not None
                    and saved_identity.get("environment") == legacy_environment):
                current_identity = {**current_identity, "environment": legacy_environment}
        else:
            current_identity = identity_builder(plan)
        decision = assess_resume(
            store.resume_identity(plan.job_id), current_identity, projection,
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
        "stage_case_counts": workload_case_counts(plan, projection, stage_states),
        "retryable_cases": retryable_case_records(plan, projection),
        "interrupted_attempts": len(decision.interrupted_attempts),
    }


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.results.recovery_inspector RESULT.json")
    try:
        report = inspect_recovery(Path(sys.argv[1]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "error": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["can_resume"] else 1)
