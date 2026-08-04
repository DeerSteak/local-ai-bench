import json
from pathlib import Path

from engines.base import GenerationMeasurement
from event_store import EventStore
from llm_event_stage import (
    LLMEventStage, event_store_path, export_llm_section, measurement_from_payload,
    measurement_payload,
)
from run_plan import RunPlan


MODEL = {"tag": "model:4b", "short": "model", "label": "Model 4B"}


def make_plan():
    return RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llm"],
        stage_order=["llm"], models={
            "llm": [{"tag": MODEL["tag"], "short": MODEL["short"]}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={
            "runs": 2, "warmup_runs": 1, "cpu_only": False, "force_all": False,
        },
    )


def measurement(ttft, tokens, tps, *, implausible=False):
    decode = tokens / tps
    return GenerationMeasurement(
        client_ttft_sec=ttft, generated_tokens=tokens, tokens_per_sec=tps,
        client_wall_sec=ttft + decode, decode_sec=decode,
        server_prompt_sec=ttft * 0.8, finish_reason="length",
        server_tps_implausible=implausible,
    )


def test_measurement_event_payload_excludes_response_content_and_round_trips():
    original = measurement(0.2, 100, 50)
    payload = measurement_payload(original)
    assert "response_text" not in payload
    assert measurement_from_payload(payload) == original


def test_event_store_path_is_predictable_beside_result(tmp_path):
    assert event_store_path(tmp_path / "results.json") == tmp_path / "results.events.sqlite3"


def test_existing_runner_stage_and_independent_export_reuse_the_journal_job(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    owner = LLMEventStage(path, plan, lambda _: None)
    owner.close()
    runner = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        runner.record_case(MODEL, 2048, "2K", [measurement(0.2, 100, 50)], "ok", 1)
    finally:
        runner.close()
    assert export_llm_section(path, plan.job_id)["model"]["2K"]["tps_mean"] == 50


def test_llm_journal_rebuilds_compatible_aggregate_and_checkpoints_projection(tmp_path):
    snapshots = []
    plan = make_plan()
    stage = LLMEventStage(tmp_path / "events.sqlite3", plan, snapshots.append)
    try:
        stage.record_case(
            MODEL, 2048, "2K",
            [measurement(0.2, 100, 50), measurement(0.4, 120, 60)],
            "ok", 2,
        )
        result = stage.export()["model"]["2K"]
        assert result["ttft_mean_sec"] == 0.3
        assert result["tps_mean"] == 55
        assert result["completed_runs"] == result["valid_runs"] == 2
        assert result["invalid_runs"] == []
        assert snapshots[-1] == stage.export()
        stage.finish()
    finally:
        stage.close()

    reopened = EventStore(tmp_path / "events.sqlite3")
    try:
        reopened.verify(plan.job_id)
        projection = reopened.rebuild(plan.job_id)
        assert next(iter(projection["stages"].values()))["state"] == "complete"
    finally:
        reopened.close()


def test_invalid_and_timed_out_samples_remain_visible_but_do_not_affect_mean(tmp_path):
    stage = LLMEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    try:
        stage.record_case(
            MODEL, 2048, "2K",
            [measurement(0.2, 100, 50), measurement(0.1, 10, 20, implausible=True)],
            "timed_out", 3, {"timed_out": "2K"},
        )
        model = stage.export()["model"]
        result = model["2K"]
        assert (result["requested_runs"], result["completed_runs"], result["valid_runs"]) == (3, 2, 1)
        assert result["tps_mean"] == 50
        assert result["invalid_runs"] == [{"run": 2, "errors": ["implausible_server_tps"]}]
        assert model["timed_out"] == "2K"
    finally:
        stage.close()


def test_model_skip_is_journal_owned_and_exported(tmp_path):
    stage = LLMEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    try:
        skipped = {"label": "Model 4B", "skipped": True, "skip_reason": "known_crash"}
        stage.record_model_state(MODEL, "skipped", skipped)
        assert stage.export() == {"model": skipped}
    finally:
        stage.close()


def test_journal_rejects_model_not_present_in_plan(tmp_path):
    stage = LLMEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    try:
        missing = {"tag": "missing", "short": "missing", "label": "Missing"}
        try:
            stage.record_case(missing, 2048, "2K", [], "failed", 2)
        except ValueError as exc:
            assert "absent from run plan" in str(exc)
        else:
            raise AssertionError("missing model was accepted")
    finally:
        stage.close()


def test_failed_json_export_does_not_undo_committed_measurement(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()

    def fail_export(_section):
        raise OSError("output volume is read-only")

    stage = LLMEventStage(path, plan, fail_export)
    try:
        try:
            stage.record_case(MODEL, 2048, "2K", [measurement(0.2, 100, 50)], "ok", 1)
        except OSError:
            pass
        else:
            raise AssertionError("failed export was accepted")
    finally:
        stage.close()
    assert export_llm_section(path, plan.job_id)["model"]["2K"]["tps_mean"] == 50


def test_journal_export_preserves_schema_three_golden_llm_fields(tmp_path):
    fixture_path = Path(__file__).parent / "fixtures" / "results_v4_1_schema3_plan.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    plan = RunPlan.from_dict(fixture["run"]["plan"])
    model = {"tag": "golden:model", "short": "golden", "label": "Golden"}
    expected = fixture["llm"]["golden"]["2K"]
    sample = measurement_from_payload(expected["valid_samples"][0])
    stage = LLMEventStage(tmp_path / "events.sqlite3", plan, lambda _: None)
    try:
        stage.record_case(model, 2048, "2K", [sample], "ok", 1)
        actual = stage.export()["golden"]["2K"]
        for key, value in expected.items():
            assert actual[key] == value
    finally:
        stage.close()
