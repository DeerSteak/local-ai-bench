import json

import pytest

from scripts.results.resume_policy import (
    assess_resume, build_engine_resume_identity, build_resume_identity,
    cached_file_identity, file_identity, load_digest_cache, stable_environment,
)
from scripts.results.run_plan import RunPlan


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


def test_engine_resume_identity_covers_selected_models_runtime_and_methodology(tmp_path):
    plan = make_plan()
    model = tmp_path / "model.gguf"
    runtime = tmp_path / "llama-server"
    extra = tmp_path / "llama-bench"
    for path, content in ((model, b"model"), (runtime, b"runtime"), (extra, b"bench")):
        path.write_bytes(content)

    class Engine:
        @staticmethod
        def model_pulled(_tag):
            return True

        def resume_artifact_paths(self, tag):
            assert tag == "model:4b"
            return (model,)

        @staticmethod
        def resume_runtime_paths():
            return {"llama-server": runtime}

    identity = build_engine_resume_identity(
        plan, Engine(), model_families=["llm"], extra_runtimes={"llama-bench": extra},
        extra_artifacts={"bank:mcq": extra},
        environment={"os": "ExampleOS", "backend": "metal"},
    )
    assert identity["artifacts"]["model:model:4b:part1"] == file_identity(model)
    assert identity["artifacts"]["bank:mcq"] == file_identity(extra)
    assert identity["runtimes"]["llama-server"] == file_identity(runtime)
    assert identity["runtimes"]["llama-bench"] == file_identity(extra)
    assert len(identity["methodology"]["execution"]) == 64
    assert len(identity["environment"]["profile_sha256"]) == 64


def test_resume_environment_excludes_only_the_volatile_run_timestamp():
    first = {"os": "Darwin", "backend": "metal", "timestamp": "first"}
    second = {"os": "Darwin", "backend": "metal", "timestamp": "second"}
    assert stable_environment(first) == {"os": "Darwin", "backend": "metal"}
    assert stable_environment(first) == stable_environment(second)


def test_native_only_identity_does_not_require_server_runtime(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    class Engine:
        @staticmethod
        def model_pulled(_tag):
            return True

        @staticmethod
        def resume_artifact_paths(_tag):
            return (model,)

        @staticmethod
        def resume_runtime_paths():
            raise AssertionError("server runtime should not be requested")

    identity = build_engine_resume_identity(
        make_plan(), Engine(), model_families=["llm"], include_engine_runtime=False,
    )
    assert identity["runtimes"] == {}


def test_digest_cache_avoids_rehash_until_file_metadata_changes(tmp_path, monkeypatch):
    from scripts.results import resume_policy

    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"first")
    cache = {}
    first = cached_file_identity(artifact, cache)
    monkeypatch.setattr(resume_policy, "file_identity", lambda _path: (_ for _ in ()).throw(
        AssertionError("unchanged file should use cached digest")
    ))
    assert cached_file_identity(artifact, cache) == first
    artifact.write_bytes(b"second-longer")
    with pytest.raises(AssertionError, match="cached digest"):
        cached_file_identity(artifact, cache)


def test_engine_identity_persists_private_digest_cache_but_not_paths_in_identity(tmp_path):
    plan = make_plan()
    model = tmp_path / "private" / "model.gguf"
    model.parent.mkdir()
    model.write_bytes(b"model")
    cache_path = tmp_path / "cache.json"

    class Engine:
        @staticmethod
        def model_pulled(_tag):
            return True

        @staticmethod
        def resume_artifact_paths(_tag):
            return (model,)

        @staticmethod
        def resume_runtime_paths():
            return {}

    identity = build_engine_resume_identity(
        plan, Engine(), model_families=["llm"], digest_cache_path=cache_path,
    )
    assert str(tmp_path) not in json.dumps(identity)
    assert str(model.resolve()) in load_digest_cache(cache_path)


def test_recovery_identity_can_bypass_same_metadata_digest_cache(tmp_path):
    import os

    plan = make_plan()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"first")
    original_stat = model.stat()
    cache_path = tmp_path / "cache.json"

    class Engine:
        @staticmethod
        def model_pulled(_tag):
            return True

        @staticmethod
        def resume_artifact_paths(_tag):
            return (model,)

        @staticmethod
        def resume_runtime_paths():
            return {}

    original = build_engine_resume_identity(
        plan, Engine(), model_families=["llm"], digest_cache_path=cache_path,
    )
    model.write_bytes(b"other")
    os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    cached = build_engine_resume_identity(
        plan, Engine(), model_families=["llm"], digest_cache_path=cache_path,
    )
    fresh = build_engine_resume_identity(
        plan, Engine(), model_families=["llm"], digest_cache_path=cache_path,
        use_digest_cache=False,
    )
    key = "model:model:4b:part1"
    assert cached["artifacts"][key] == original["artifacts"][key]
    assert fresh["artifacts"][key] == file_identity(model)
    assert fresh["artifacts"][key] != original["artifacts"][key]


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
    ("environment", "execution environment identity changed"),
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


def test_identity_skips_models_the_engine_does_not_have(tmp_path):
    """A partial download must not abort the run: workloads skip missing models, so
    resume identity has nothing to protect for them."""
    plan = make_plan()
    runtime = tmp_path / "vllm"
    runtime.write_bytes(b"runtime")

    class Engine:
        @staticmethod
        def model_pulled(_tag):
            return False

        @staticmethod
        def resume_artifact_paths(tag):
            raise AssertionError(f"must not hash a model that is not installed: {tag}")

        @staticmethod
        def resume_runtime_paths():
            return {"vllm": runtime}

    identity = build_engine_resume_identity(plan, Engine(), model_families=["llm"])
    assert identity["artifacts"] == {}
    assert identity["runtimes"]["vllm"] == file_identity(runtime)


def test_installing_a_model_later_changes_the_identity(tmp_path):
    """Resume must be refused once a previously absent model becomes measurable."""
    plan = make_plan()
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"weights")
    runtime = tmp_path / "vllm"
    runtime.write_bytes(b"runtime")

    class Engine:
        def __init__(self, installed):
            self.installed = installed

        def model_pulled(self, _tag):
            return self.installed

        @staticmethod
        def resume_artifact_paths(_tag):
            return (model,)

        @staticmethod
        def resume_runtime_paths():
            return {"vllm": runtime}

    before = build_engine_resume_identity(plan, Engine(False), model_families=["llm"])
    after = build_engine_resume_identity(plan, Engine(True), model_families=["llm"])
    assert before["artifacts"] != after["artifacts"]
