import pytest

from scripts.results.image_event_stage import ImageEventStage, export_images
from scripts.results.run_plan import RunPlan


MODEL = {
    "short": "sdxl", "label": "SDXL", "checkpoint": "sdxl.safetensors", "steps": 20,
}


def make_plan():
    return RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=["img"],
        stage_order=["img"], models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [{"short": "sdxl"}],
        }, effective_config={"runs": 3, "warmup_runs": 1, "cpu_only": False,
                             "force_all": False},
    )


def test_image_resolution_projects_compatible_result_and_ignores_artifact(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = ImageEventStage(path, plan, lambda _: None)
    artifact = {"sha256": "a" * 64, "size": 100, "media_type": "image/png"}
    stage.record_resolution(MODEL, 1024, 1024, [2.0, 4.0, 3.0], 3, artifact=artifact)
    stage.record_model_evidence(MODEL, {"summary": "memory"}, {"status": "unavailable"})
    stage.finish()
    expected = {
        "label": "SDXL", "checkpoint": "sdxl.safetensors", "steps": 20,
        "resolutions": {"1024x1024": {
            "sec_per_image_mean": 3.0, "sec_per_image_stdev": 1.0,
            "n_runs": 3, "runs": [2.0, 4.0, 3.0],
        }},
        "memory": {"summary": "memory"}, "power": {"status": "unavailable"},
    }
    assert stage.export()["sdxl"] == expected
    stage.close()
    assert export_images(path, plan.job_id)["sdxl"] == expected


def test_completed_resolution_is_not_pending_and_duplicate_is_rejected(tmp_path):
    stage = ImageEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    stage.record_resolution(MODEL, 512, 512, [1.0], 1)
    assert stage.pending_resolutions(MODEL, [(512, 512), (768, 768)]) == [(768, 768, 1)]
    with pytest.raises(ValueError, match="already completed"):
        stage.record_resolution(MODEL, 512, 512, [1.0], 1)
    stage.close()


def test_timed_out_resolution_retains_partial_times_and_resumes_next_attempt(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    stage = ImageEventStage(path, plan, lambda _: None, resume_identity=identity)
    stage.record_resolution(MODEL, 1024, 1024, [2.0], 3, "timed_out")
    assert stage.export()["sdxl"]["timed_out"] == "1024x1024"
    stage.close()
    resumed = ImageEventStage(
        path, plan, lambda _: None, resume=True, resume_identity=identity,
    )
    assert resumed.next_attempt(MODEL, 1024, 1024) == 2
    resumed.record_resolution(MODEL, 1024, 1024, [1.5, 1.6, 1.4], 3,
                              attempt_number=2)
    result = resumed.export()["sdxl"]
    assert result["resolutions"]["1024x1024"]["runs"] == [1.5, 1.6, 1.4]
    assert "timed_out" not in result
    resumed.close()


def test_selected_retry_leaves_unselected_failed_resolution_pending(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    first = ImageEventStage(path, plan, lambda _: None, resume_identity=identity)
    first.record_resolution(MODEL, 512, 512, [], 3, "failed")
    first.record_resolution(MODEL, 768, 768, [], 3, "failed")
    projection = first.store.rebuild(plan.job_id)
    selected = next(case_id for case_id, case in projection["cases"].items()
                    if case.get("width") == 512)
    first.close()
    resumed = ImageEventStage(
        path, plan, lambda _: None, resume=True, resume_identity=identity,
        selected_case_ids=[selected],
    )
    assert resumed.next_attempt(MODEL, 512, 512) == 2
    assert resumed.next_attempt(MODEL, 768, 768) is None
    resumed.close()


def test_model_skip_projects_legacy_shape(tmp_path):
    stage = ImageEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    stage.record_resolution(MODEL, 512, 512, [1.0], 1)
    stage.record_model_state(MODEL, "skipped", {
        "skipped": True, "skip_reason": "checkpoint_not_found",
    })
    assert stage.export()["sdxl"] == {
        "label": "SDXL", "skipped": True, "skip_reason": "checkpoint_not_found",
    }
    stage.close()


def test_image_artifact_reference_is_strictly_validated(tmp_path):
    stage = ImageEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    with pytest.raises(ValueError, match="artifact reference"):
        stage.record_resolution(
            MODEL, 512, 512, [1.0], 1,
            artifact={"sha256": "not-a-digest", "size": 1, "media_type": "image/png"},
        )
    stage.close()
