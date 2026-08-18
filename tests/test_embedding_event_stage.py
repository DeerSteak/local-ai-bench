from scripts.results.embedding_event_stage import EmbeddingEventStage, export_embeddings
from scripts.results.resume_policy import build_resume_identity
from scripts.results.run_plan import RunPlan
from scripts.runtime.engines.base import EmbeddingMeasurement


MODELS = [
    {"tag": f"embed:{number}", "short": f"e{number}", "label": f"Embed {number}"}
    for number in range(1, 4)
]


def make_plan():
    return RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=["emb"],
        stage_order=["emb"], models={
            "llm": [], "concurrency": [],
            "embeddings": [{"tag": model["tag"], "short": model["short"]}
                           for model in MODELS],
            "images": [],
        }, effective_config={
            "runs": 3, "warmup_runs": 1, "cpu_only": False, "force_all": False,
        },
    )


def test_embedding_batch_projects_existing_aggregate_without_vector_payload(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("corpus")
    plan = make_plan()
    identity = build_resume_identity(
        plan, artifacts={"corpus:embeddings": corpus}, runtimes={}, methodology={},
    )
    corpus_hash = identity["artifacts"]["corpus:embeddings"]["sha256"]
    path = tmp_path / "events.sqlite3"
    stage = EmbeddingEventStage(
        path, plan, corpus_hash, lambda _: None, resume_identity=identity,
    )
    samples = [
        EmbeddingMeasurement([[1.0], [2.0]], 0.5),
        EmbeddingMeasurement([[3.0], [4.0]], 1.0),
        EmbeddingMeasurement([], float("nan")),
    ]
    try:
        stage.record_batch(
            MODELS[0], samples, "ok", 2, 3,
            memory={"summary": {"process_rss_gb": {"peak_gb": 2.0}}},
            power={"status": "unavailable"},
        )
        result = stage.export()["e1"]
    finally:
        stage.close()
    assert result["chunks_per_sec_mean"] == 3.0
    assert result["chunks_per_sec_stdev"] == 1.4
    assert result["client_wall_runs_sec"] == [0.5, 1.0]
    assert result["runs"] == [4.0, 2.0]
    assert result["invalid_runs"] == [{"run": 3, "errors": ["client_wall_sec"]}]
    assert result["memory"]["summary"]["process_rss_gb"]["peak_gb"] == 2.0
    projected = export_embeddings(path, plan.job_id)["e1"]
    assert projected == result
    assert "embeddings" not in str(projected)


def test_completed_embedding_batch_is_skipped_and_duplicate_is_rejected(tmp_path):
    stage = EmbeddingEventStage(tmp_path / "events.sqlite3", make_plan(), "corpus", lambda _: None)
    try:
        stage.record_batch(MODELS[0], [EmbeddingMeasurement([], 1.0)], "ok", 4, 1)
        assert stage.next_attempt(MODELS[0]) is None
        assert stage.next_attempt(MODELS[1]) == 1
        try:
            stage.record_batch(MODELS[0], [EmbeddingMeasurement([], 1.0)], "ok", 4, 1)
        except ValueError as exc:
            assert "already completed" in str(exc)
        else:
            raise AssertionError("duplicate embedding batch was accepted")
    finally:
        stage.close()


def test_failed_embedding_batch_resumes_with_next_attempt(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = EmbeddingEventStage(path, plan, "corpus", lambda _: None)
    stage.record_batch(MODELS[0], [], "crashed", 4, 3, crash_detail="crashed")
    stage.close()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    resumed = EmbeddingEventStage(
        path, plan, "corpus", lambda _: None, resume=True, resume_identity=identity,
    )
    try:
        assert resumed.next_attempt(MODELS[0]) == 2
        resumed.record_batch(
            MODELS[0], [EmbeddingMeasurement([], 0.5)], "ok", 4, 3,
            attempt_number=2,
        )
        assert resumed.export()["e1"]["chunks_per_sec_mean"] == 8.0
    finally:
        resumed.close()


def test_unexpected_failure_projection_is_retryable_and_preserves_result(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = EmbeddingEventStage(path, plan, "corpus", lambda _: None)
    failure = {"label": "Embed 1", "skipped": True, "skip_reason": "unexpected_error",
               "skip_detail": "boom"}
    stage.record_batch(MODELS[0], [], "failed", 4, 3, failure_result=failure)
    assert stage.export()["e1"] == failure
    stage.close()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    resumed = EmbeddingEventStage(
        path, plan, "corpus", lambda _: None, resume=True, resume_identity=identity,
    )
    try:
        assert resumed.next_attempt(MODELS[0]) == 2
    finally:
        resumed.close()


def test_embedding_model_skip_is_journal_owned(tmp_path):
    stage = EmbeddingEventStage(tmp_path / "events.sqlite3", make_plan(), "corpus", lambda _: None)
    try:
        stage.record_model_state(MODELS[0], "skipped", {
            "skipped": True, "skip_reason": "model_not_downloaded",
        })
        assert stage.export()["e1"] == {
            "label": "Embed 1", "skipped": True,
            "skip_reason": "model_not_downloaded",
        }
    finally:
        stage.close()
