import json
from pathlib import Path

from scripts.runtime.engines.base import GenerationMeasurement
from scripts.results.event_store import EventStore
from scripts.results.llm_event_stage import (
    LLMEventStage, event_store_path, export_llm_section, measurement_from_payload,
    measurement_payload,
)
from scripts.results.run_plan import RunPlan


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


def test_case_telemetry_survives_journal_reopen_and_projection(tmp_path):
    memory = {
        "windows": [{"name": "measured", "sample_count": 1}],
        "summary": {"process_rss_gb": {"peak_gb": 4.0}},
        "headroom": {"absolute_gb": 8.0, "fraction": 0.5, "state": "comfortable"},
        "provenance": {"interval_sec": 1.0, "failed_samples": 0},
    }
    power = {
        "status": "recorded", "source": "powermetrics", "scope": "processor_package",
        "energy_joules": 12.5, "idle_baseline_watts": 4.0,
    }

    class Telemetry:
        last_power = power

        def begin_model_load(self):
            pass

        def begin_measured(self, subwindow="measured"):
            pass

        def finish_case(self, ceiling_gb=None):
            return memory

    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = LLMEventStage(path, plan, lambda _: None, telemetry=Telemetry())
    stage.record_case(MODEL, 2048, "2K", [measurement(0.2, 100, 50)], "ok", 1)
    stage.close()
    projected = export_llm_section(path, plan.job_id)["model"]["2K"]["memory"]
    assert {key: projected[key] for key in memory} == memory
    assert projected["case_id"].startswith("case_")
    projected_power = export_llm_section(path, plan.job_id)["model"]["2K"]["power"]
    assert {key: projected_power[key] for key in power} == power
    assert projected_power["case_id"] == projected["case_id"]


def test_conversation_stage_shares_job_but_projects_only_its_cases(tmp_path):
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llm", "conv"],
        stage_order=["llm", "conv"], models=make_plan().models,
        effective_config=make_plan().effective_config,
    )
    path = tmp_path / "events.sqlite3"
    llm = LLMEventStage(path, plan, lambda _: None)
    llm.record_case(MODEL, 2048, "2K", [measurement(0.2, 100, 50)], "ok", 1)
    llm.finish()
    llm.close()
    conversation = LLMEventStage(
        path, plan, lambda _: None, stage_name="conv",
    )
    try:
        conversation.record_case(
            MODEL, 0, "0K", [measurement(0.1, 96, 48)], "ok", 1,
            depth_tokens=400,
        )
        conversation.finish()
    finally:
        conversation.close()
    assert set(export_llm_section(path, plan.job_id)) == {"model"}
    projected = export_llm_section(path, plan.job_id, "conv")["model"]
    assert set(projected) == {"0K"}
    assert projected["0K"]["depth_tokens"] == 400
    assert projected["0K"]["client_ttft_mean_sec"] == 0.1


def test_concurrency_stage_projects_batch_metrics_and_invalid_samples(tmp_path):
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["conc_chat"],
        stage_order=["conc_chat"], models={
            "llm": [],
            "concurrency": [{"tag": MODEL["tag"], "short": MODEL["short"]}],
            "embeddings": [], "images": [],
        }, effective_config=make_plan().effective_config,
    )
    path = tmp_path / "events.sqlite3"
    stage = LLMEventStage(
        path, plan, lambda _: None, stage_name="conc_chat", model_family="concurrency",
    )
    try:
        stage.record_case(
            MODEL, 2, "2", [measurement(0.2, 100, 50), measurement(0.1, 10, 20, implausible=True)],
            "ok", 2, result_fields={
                "aggregate_tps": 55.0, "total_tokens": 110,
                "batch_elapsed_sec": 2.0, "memory": {"system_ram_used_gb": 8.0},
            },
        )
        stage.record_model_state(MODEL, "complete", {"stopped_at": "slow"})
        stage.finish()
    finally:
        stage.close()
    result = export_llm_section(
        path, plan.job_id, "conc_chat", "concurrency",
    )["model"]
    assert result["2"]["valid_runs"] == 1
    assert result["2"]["invalid_runs"][0]["errors"] == ["implausible_server_tps"]
    assert result["2"]["aggregate_tps"] == 55
    assert result["2"]["memory"]["system_ram_used_gb"] == 8
    assert result["stopped_at"] == "slow"


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
        stage.record_model_state(MODEL, "skipped", {"skip_reason": "different"})
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


def test_resume_preserves_prior_attempt_but_aggregates_latest_attempt_only(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {"execution": "same"}}
    first = LLMEventStage(path, plan, lambda _: None, resume_identity=identity)
    first.record_case(
        MODEL, 2048, "2K", [measurement(0.2, 100, 50)], "timed_out", 2,
        {"timed_out": "2K"},
    )
    first.close()

    owner = LLMEventStage(
        path, plan, lambda _: None, resume_identity=identity, resume=True,
    )
    owner.close()
    resumed = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        assert resumed.next_context_attempt(MODEL, 2048) == 2
        resumed.record_case(
            MODEL, 2048, "2K", [measurement(0.1, 120, 60)], "ok", 2,
            attempt_number=2,
        )
        assert resumed.next_context_attempt(MODEL, 2048) is None
        result = resumed.export()["model"]["2K"]
        assert result["completed_runs"] == result["valid_runs"] == 1
        assert result["tps_mean"] == 60
    finally:
        resumed.close()
    store = EventStore(path)
    projection = store.rebuild(plan.job_id)
    assert sorted(attempt["number"] for attempt in projection["attempts"].values()) == [1, 2]
    assert len(projection["samples"]) == 2
    store.close()


def test_selected_retry_runs_only_chosen_case_and_retains_incomplete_stage(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    first = LLMEventStage(path, plan, lambda _: None, resume_identity=identity)
    first.record_case(MODEL, 512, "512", [], "timed_out", 1)
    first.record_case(MODEL, 2048, "2K", [], "timed_out", 1)
    first.close()
    model_id = plan.model_id("llm", plan.models["llm"][0])
    selected = plan.case_id("llm", model_id, {"context_tokens": 512})
    owner = LLMEventStage(
        path, plan, lambda _: None, resume_identity=identity, resume=True,
        selected_case_ids=[selected],
    )
    owner.close()
    runner = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        assert runner.next_context_attempt(MODEL, 512) == 2
        assert runner.next_context_attempt(MODEL, 2048) is None
        runner.record_case(
            MODEL, 512, "512", [measurement(0.1, 100, 50)], "ok", 1,
            attempt_number=2,
        )
        runner.finish()
    finally:
        runner.close()
    store = EventStore(path)
    projection = store.rebuild(plan.job_id)
    assert projection["cases"][selected]["state"] == "complete"
    assert projection["stages"][plan.stage_id("llm")]["state"] == "failed"
    store.close()


def test_recovered_model_state_can_be_replaced_without_duplicate_start_transition(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    first = LLMEventStage(path, plan, lambda _: None, resume_identity=identity)
    first.record_model_state(MODEL, "timed_out", {"timed_out": "2K"})
    first.close()
    owner = LLMEventStage(path, plan, lambda _: None, resume_identity=identity, resume=True)
    owner.close()
    runner = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        runner.record_model_state(MODEL, "timed_out", {"timed_out": "8K"})
        assert runner.export()["model"]["timed_out"] == "8K"
    finally:
        runner.close()


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
            if key == "valid_samples":
                # Later schemas may add sample fields; the schema-3 ones must survive intact.
                for expected_sample, actual_sample in zip(value, actual[key], strict=True):
                    assert {k: actual_sample[k] for k in expected_sample} == expected_sample
                continue
            assert actual[key] == value
    finally:
        stage.close()
