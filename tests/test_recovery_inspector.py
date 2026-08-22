import json
from types import SimpleNamespace

from scripts.results import recovery_inspector
from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.recovery_inspector import (
    compatible_environment_identity, current_resume_identity, inspect_recovery,
    legacy_environment_identity,
    retryable_case_records, workload_case_counts,
)
from scripts.results.run_plan import RunPlan
from scripts.stage_registry import STAGE_ORDER


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
    assert report["stage_case_counts"] == {"llm": {"running": 1}}
    assert len(report["retryable_cases"]) == 1
    assert report["retryable_cases"][0] == {
        "case_id": plan.case_id(
            "llm", plan.model_id("llm", plan.models["llm"][0]), {"context_tokens": 512},
        ),
        "stage": "llm", "state": "running", "model": "model", "label": "model · 512",
    }
    assert report["interrupted_attempts"] == 1
    assert result.with_suffix(".events.sqlite3").read_bytes() == before


def test_workload_case_coverage_is_reported_for_every_registered_stage():
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=list(STAGE_ORDER),
        stage_order=list(STAGE_ORDER), models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )
    cases = {}
    for stage in STAGE_ORDER:
        cases[f"{stage}-complete"] = {
            "parent_id": plan.stage_id(stage), "case_kind": "measurement", "state": "complete",
        }
        cases[f"{stage}-failed"] = {
            "parent_id": plan.stage_id(stage), "case_kind": "measurement", "state": "failed",
        }
        cases[f"{stage}-metadata"] = {
            "parent_id": plan.stage_id(stage), "case_kind": "model_state", "state": "skipped",
        }
    assert workload_case_counts(plan, {"cases": cases}, STAGE_ORDER) == {
        stage: {"complete": 1, "failed": 1} for stage in STAGE_ORDER
    }


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


def test_legacy_environment_identity_reuses_only_saved_timestamp():
    current = {"hostname": "host", "backend": "metal", "timestamp": "now"}
    first = legacy_environment_identity(current, {"timestamp": "then"})
    second = legacy_environment_identity(
        {**current, "timestamp": "later"}, {"timestamp": "then"},
    )
    assert first == second
    assert legacy_environment_identity(current, {}) is None


def test_legacy_environment_identity_omits_new_support_profile_for_old_result():
    current = {
        "hostname": "host", "timestamp": "now",
        "engine_support": {"support_level": "unverified"},
    }
    assert legacy_environment_identity(current, {"timestamp": "then"}) == {
        "profile_sha256": recovery_inspector.sha256_json({
            "hostname": "host", "timestamp": "then",
        }),
    }


def test_compatible_environment_identity_returns_saved_legacy_hash_only_on_exact_match():
    profile = {"hostname": "host", "backend": "metal", "timestamp": "now"}
    current = {"environment": {"profile_sha256": "stable"}, "plan_id": "plan"}
    legacy = legacy_environment_identity(profile, {"timestamp": "then"})
    saved = {"environment": legacy}
    assert compatible_environment_identity(
        current, profile, saved, {"timestamp": "then"},
    )["environment"] == legacy
    assert compatible_environment_identity(
        current, profile, {"environment": {"profile_sha256": "other"}},
        {"timestamp": "then"},
    ) == current


def test_current_embedding_identity_includes_model_family_and_corpus(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("stable embedding input", encoding="utf-8")
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=["emb"],
        stage_order=["emb"], models={
            "llm": [], "concurrency": [],
            "embeddings": [{"tag": "embed", "short": "embed"}], "images": [],
        }, effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                             "force_all": False},
    )
    seen = {}

    def build(*args, **kwargs):
        seen.update(kwargs)
        return {"identity": "embedding"}

    monkeypatch.setattr(recovery_inspector, "build_engine_resume_identity", build)
    monkeypatch.setattr(recovery_inspector.EmbeddingBenchmark, "EMBED_DOCUMENT_PATH", corpus)
    identity = current_resume_identity(plan, profile={"os": "test"}, engine=object())
    assert identity == {"identity": "embedding"}
    assert seen["model_families"] == ["embeddings"]
    assert seen["extra_artifacts"] == {"corpus:embeddings": corpus}


def test_current_image_identity_uses_private_runtime_and_model_assets(monkeypatch, tmp_path):
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=["img"],
        stage_order=["img"], models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [{"short": "sdxl"}],
        }, effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                             "force_all": False},
    )
    event_path = tmp_path / "events.sqlite3"
    asset = tmp_path / "model.safetensors"
    runtime = tmp_path / "main.py"
    seen = {}

    def build(*args, **kwargs):
        seen.update(kwargs)
        return {"identity": "image"}

    monkeypatch.setattr(recovery_inspector, "build_engine_resume_identity", build)
    monkeypatch.setattr(
        recovery_inspector, "load_local_execution_context",
        lambda _path, _job: SimpleNamespace(comfyui_dir=tmp_path),
    )
    monkeypatch.setattr(recovery_inspector, "image_resume_artifacts",
                        lambda _models: {"image:sdxl:checkpoint": asset})
    monkeypatch.setattr(recovery_inspector, "image_resume_runtimes",
                        lambda _path: {"comfyui-main": runtime})
    identity = current_resume_identity(
        plan, profile={"os": "test"}, engine=object(), event_path=event_path,
    )
    assert identity == {"identity": "image"}
    assert seen["include_engine_runtime"] is False
    assert seen["extra_artifacts"] == {"image:sdxl:checkpoint": asset}
    assert seen["extra_runtimes"] == {"comfyui-main": runtime}


def test_retryable_image_resolution_has_resolution_label():
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=["img"],
        stage_order=["img"], models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [{"short": "sdxl"}],
        }, effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                             "force_all": False},
    )
    assert retryable_case_records(plan, {"cases": {"case_image": {
        "state": "timed_out", "parent_id": plan.stage_id("img"),
        "case_kind": "image_resolution", "model_short": "sdxl",
        "width": 1024, "height": 1024,
    }}}) == [{
        "case_id": "case_image", "stage": "img", "state": "timed_out",
        "model": "sdxl", "label": "sdxl · 1024x1024",
    }]


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
