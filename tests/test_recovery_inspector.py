import json

from scripts.results import recovery_inspector
from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.recovery_inspector import current_resume_identity, inspect_recovery, retryable_case_records
from scripts.results.run_plan import RunPlan


def make_result(tmp_path):
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llm"],
        stage_order=["llm"], models={
            "llm": [{"tag": "model:4b", "short": "model"}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"run": {"plan": plan.to_dict()}}), encoding="utf-8")
    journal = result.with_suffix(".events.sqlite3")
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    store = EventStore(journal)
    store.start_stage(plan, "llm", identity)
    model_id = plan.model_id("llm", plan.models["llm"][0])
    case_id = plan.case_id("llm", model_id, {"context_tokens": 512})
    attempt_id = plan.attempt_id(case_id, 1)
    store.append(plan.job_id, [
        JournalEvent("case", case_id, "running", {
            "case_kind": "context", "model_short": "model", "context_label": "512",
        }, parent_id=plan.stage_id("llm")),
        JournalEvent("attempt", attempt_id, "running", {"number": 1}, parent_id=case_id),
    ])
    store.close()
    return result, plan, identity


def test_recovery_inspector_reports_durable_coverage_without_mutation(tmp_path):
    result, plan, identity = make_result(tmp_path)
    before = result.with_suffix(".events.sqlite3").read_bytes()
    report = inspect_recovery(result, lambda _plan: identity)
    assert report["action"] == "resume" and report["can_resume"] is True
    assert report["stage_states"] == {"llm": "running"}
    assert report["case_counts"] == {"running": 1}
    assert len(report["retryable_cases"]) == 1
    assert report["retryable_cases"][0] == {
        "case_id": plan.case_id(
            "llm", plan.model_id("llm", plan.models["llm"][0]), {"context_tokens": 512},
        ),
        "stage": "llm", "state": "running", "model": "model", "label": "model · 512",
    }
    assert report["interrupted_attempts"] == 1
    assert result.with_suffix(".events.sqlite3").read_bytes() == before


def test_recovery_inspector_requires_fork_when_current_identity_changes(tmp_path):
    result, _, identity = make_result(tmp_path)
    changed = {**identity, "environment": {"profile_sha256": "different"}}
    report = inspect_recovery(result, lambda _plan: changed)
    assert report["action"] == "fork" and report["can_resume"] is False
    assert report["reasons"] == ["execution environment identity changed"]


def test_current_recovery_identity_bypasses_persistent_digest_cache(monkeypatch, tmp_path):
    _, plan, _ = make_result(tmp_path)
    seen = {}

    def build(*args, **kwargs):
        seen.update(kwargs)
        return {"identity": "fresh"}

    monkeypatch.setattr(recovery_inspector, "build_engine_resume_identity", build)
    identity = current_resume_identity(
        plan, profile={"os": "test"}, engine=object(), digest_cache_path=tmp_path / "cache",
    )
    assert identity == {"identity": "fresh"}
    assert seen["use_digest_cache"] is False


def test_recovery_inspector_rejects_result_without_journal(tmp_path):
    result, _, _ = make_result(tmp_path)
    result.with_suffix(".events.sqlite3").unlink()
    try:
        inspect_recovery(result, lambda _plan: {})
    except ValueError as exc:
        assert "no durable event journal" in str(exc)
    else:
        raise AssertionError("missing journal was accepted")


def test_recovery_inspector_never_reopens_a_complete_portable_result(tmp_path):
    result, _, identity = make_result(tmp_path)
    value = json.loads(result.read_text())
    value["run"]["status"] = "complete"
    result.write_text(json.dumps(value), encoding="utf-8")
    report = inspect_recovery(result, lambda _plan: identity)
    assert report["action"] == "fork" and report["can_resume"] is False
    assert report["reasons"] == ["result is already complete"]


def test_retryable_case_records_are_ordered_and_exclude_completed_cases(tmp_path):
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["conv", "llm"],
        stage_order=["conv", "llm"], models={
            "llm": [{"tag": "model:4b", "short": "model"}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )
    projection = {"cases": {
        "case_complete": {
            "state": "complete", "parent_id": plan.stage_id("conv"),
            "model_short": "model", "context_label": "2K", "case_kind": "context",
        },
        "case_llm": {
            "state": "invalid", "parent_id": plan.stage_id("llm"),
            "model_short": "model", "context_label": "8K", "case_kind": "context",
        },
        "case_conv": {
            "state": "timed_out", "parent_id": plan.stage_id("conv"),
            "model_short": "model", "context_label": "4K", "case_kind": "context",
        },
        "case_model_state": {
            "state": "failed", "parent_id": plan.stage_id("llm"),
            "model_short": "model", "case_kind": "model_state",
        },
    }}
    assert retryable_case_records(plan, projection) == [
        {"case_id": "case_conv", "stage": "conv", "state": "timed_out",
         "model": "model", "label": "model · 4K"},
        {"case_id": "case_llm", "stage": "llm", "state": "invalid",
         "model": "model", "label": "model · 8K"},
    ]
