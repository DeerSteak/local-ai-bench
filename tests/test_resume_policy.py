import json

import pytest

from resume_policy import assess_resume, build_resume_identity, file_identity
from run_plan import RunPlan


def make_plan():
    return RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llm"],
        stage_order=["llm"], models={
            "llm": [{"tag": "model:4b", "short": "model"}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 1, "cpu_only": False, "force_all": False},
    )


def test_resume_identity_hashes_content_without_serializing_paths(tmp_path):
    model = tmp_path / "private" / "model.gguf"
    runtime = tmp_path / "bin" / "llama-server"
    model.parent.mkdir()
    runtime.parent.mkdir()
    model.write_bytes(b"model-content")
    runtime.write_bytes(b"runtime-content")
    identity = build_resume_identity(
        make_plan(), artifacts={"model_id": model}, runtimes={"llama-server": runtime},
        methodology={"llm": "4.1-neutral"},
    )
    encoded = json.dumps(identity)
    assert str(tmp_path) not in encoded
    assert identity["artifacts"]["model_id"] == file_identity(model)
    assert identity["runtimes"]["llama-server"] == file_identity(runtime)


def test_resume_identity_rejects_path_bearing_logical_names(tmp_path):
    artifact = tmp_path / "model"
    artifact.write_bytes(b"x")
    with pytest.raises(ValueError, match="logical identifiers"):
        build_resume_identity(
            make_plan(), artifacts={"/private/model": artifact}, runtimes={}, methodology={},
        )


def test_exact_identity_resumes_only_remaining_cases_and_advances_attempt_number():
    identity = {"plan_id": "p", "artifacts": {}, "runtimes": {}, "methodology": {}}
    projection = {
        "jobs": {"job_p": {"state": "interrupted"}},
        "cases": {"case_a": {"state": "complete"}, "case_b": {"state": "running"}},
        "attempts": {
            "attempt_1": {"state": "failed", "parent_id": "case_b", "number": 1},
            "attempt_2": {"state": "running", "parent_id": "case_b", "number": 2},
        },
    }
    decision = assess_resume(identity, identity, projection, ["case_a", "case_b", "case_c"])
    assert decision.can_resume and decision.action == "resume"
    assert decision.completed_cases == ("case_a",)
    assert decision.remaining_cases == ("case_b", "case_c")
    assert decision.interrupted_attempts == ("attempt_2",)
    assert decision.next_attempts == (("case_b", 3), ("case_c", 1))


@pytest.mark.parametrize(("key", "reason"), [
    ("plan_id", "plan identity changed"),
    ("artifacts", "model artifacts identity changed"),
    ("runtimes", "runtime binaries identity changed"),
    ("methodology", "methodology identity changed"),
])
def test_identity_change_requires_fork(key, reason):
    saved = {"plan_id": "p", "artifacts": {}, "runtimes": {}, "methodology": {}}
    current = dict(saved)
    current[key] = "changed"
    decision = assess_resume(saved, current, {"jobs": {}, "cases": {}, "attempts": {}}, [])
    assert not decision.can_resume and decision.action == "fork"
    assert decision.reasons == (reason,)


def test_complete_job_or_unknown_case_requires_fork():
    identity = {"plan_id": "p", "artifacts": {}, "runtimes": {}, "methodology": {}}
    projection = {
        "jobs": {"job_p": {"state": "complete"}},
        "cases": {"case_unplanned": {"state": "failed"}}, "attempts": {},
    }
    decision = assess_resume(identity, identity, projection, ["case_planned"])
    assert decision.reasons == (
        "job is already complete", "journal contains cases outside the current plan",
    )
