import pytest

from scripts.results.run_plan import RunPlan
from scripts.results.vllm_bench_event_stage import VllmBenchEventStage, export_vllm_bench


MODEL = {"tag": "model", "short": "model", "label": "Model"}


def make_plan():
    return RunPlan.create(
        application_version="6.0-pre7", engine_name="vllm", tests=["vllmbench"],
        stage_order=["vllmbench"], models={
            "llm": [{"tag": "model", "short": "model"}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )


def test_vllm_cases_project_compatible_buckets_and_telemetry(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = VllmBenchEventStage(path, plan, lambda _: None)
    latency = {"input_len": 512, "output_len": 128, "avg_latency_sec": 2.0,
               "memory": {"summary": {}}}
    throughput = {"input_len": 512, "output_len": 128, "requests_per_sec": 4.0,
                  "power": {"status": "unavailable"}}
    stage.record_case(MODEL, "latency", 512, 128, latency, 2)
    stage.record_case(MODEL, "throughput", 512, 128, throughput, 2)
    stage.finish()
    expected = {
        "latency_entries": [latency], "throughput_entries": [throughput],
        "requested_cases": 2, "completed_cases": 2,
    }
    assert stage.export()["model"] == expected
    stage.close()
    assert export_vllm_bench(path, plan.job_id)["model"] == expected


def test_vllm_timeout_retains_error_and_resumes_at_next_attempt(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    stage = VllmBenchEventStage(path, plan, lambda _: None, resume_identity=identity)
    stage.record_case(MODEL, "latency", 512, 128, None, 2, "timed_out",
                      error="no result within 60s")
    result = stage.export()["model"]
    assert result["timed_out_at"] == "latency in512"
    assert result["error"] == "no result within 60s"
    stage.close()
    resumed = VllmBenchEventStage(
        path, plan, lambda _: None, resume=True, resume_identity=identity,
    )
    assert resumed.next_attempt(MODEL, "latency", 512, 128) == 2
    resumed.record_case(
        MODEL, "latency", 512, 128,
        {"input_len": 512, "output_len": 128, "avg_latency_sec": 1.0}, 2,
        attempt_number=2,
    )
    result = resumed.export()["model"]
    assert "timed_out" not in result and "error" not in result
    resumed.close()


def test_vllm_completed_case_is_skipped_and_duplicate_rejected(tmp_path):
    stage = VllmBenchEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    entry = {"input_len": 512, "output_len": 128, "avg_latency_sec": 1.0}
    stage.record_case(MODEL, "latency", 512, 128, entry, 2)
    assert stage.next_attempt(MODEL, "latency", 512, 128) is None
    with pytest.raises(ValueError, match="already completed"):
        stage.record_case(MODEL, "latency", 512, 128, entry, 2)
    stage.close()


def test_vllm_model_state_projects_exact_payload(tmp_path):
    stage = VllmBenchEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    stage.record_model_state(MODEL, "failed", {"error": "no vllm_repo for tag"})
    assert stage.export()["model"] == {"error": "no vllm_repo for tag"}
    stage.close()


@pytest.mark.parametrize("completed", [0, 1, 3])
def test_vllm_resume_skips_every_case_committed_before_interruption(tmp_path, completed):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    cases = [("latency", 512, 128), ("throughput", 512, 128),
             ("latency", 2048, 128), ("throughput", 2048, 128)]
    stage = VllmBenchEventStage(path, plan, lambda _: None, resume_identity=identity)
    for kind, input_len, output_len in cases[:completed]:
        stage.record_case(MODEL, kind, input_len, output_len, {
            "input_len": input_len, "output_len": output_len, "value": input_len,
        }, len(cases))
    stage.close()

    resumed = VllmBenchEventStage(
        path, plan, lambda _: None, resume=True, resume_identity=identity,
    )
    attempts = [resumed.next_attempt(MODEL, *case) for case in cases]
    assert attempts[:completed] == [None] * completed
    assert attempts[completed:] == [1] * (len(cases) - completed)
    resumed.close()
