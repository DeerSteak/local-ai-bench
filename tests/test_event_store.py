import sqlite3

import pytest

from event_store import EventStore, JournalEvent
from run_plan import RunPlan


def make_plan():
    return RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llm"],
        stage_order=["llm"],
        models={
            "llm": [{"tag": "model:4b", "short": "model"}],
            "concurrency": [], "embeddings": [], "images": [],
        },
        effective_config={"warmup_runs": 1, "cpu_only": False, "force_all": False},
    )


def ids(plan):
    model_id = plan.model_id("llm", plan.models["llm"][0])
    case_id = plan.case_id("llm", model_id, {"context_tokens": 2048})
    attempt_id = plan.attempt_id(case_id, 1)
    return plan.stage_id("llm"), model_id, case_id, attempt_id, plan.sample_id(attempt_id, 1)


def test_event_store_uses_wal_and_rebuilds_complete_projection(tmp_path):
    plan = make_plan()
    stage_id, _, case_id, attempt_id, sample_id = ids(plan)
    invalid_sample_id = plan.sample_id(attempt_id, 2)
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        identity = {
            "plan_id": plan.plan_id, "artifacts": {"model": {"sha256": "abc", "size": 3}},
            "runtimes": {}, "methodology": {"llm": "4.1"},
        }
        assert store.create_job(plan, identity) == plan.job_id
        assert store.resume_identity(plan.job_id) == identity
        store.append(plan.job_id, [
            JournalEvent("job", plan.job_id, "running", {}),
            JournalEvent("stage", stage_id, "running", {"stage": "llm"}, parent_id=plan.job_id),
            JournalEvent("case", case_id, "running", {"context_tokens": 2048}, parent_id=stage_id),
            JournalEvent("attempt", attempt_id, "running", {"number": 1}, parent_id=case_id),
            JournalEvent("sample", sample_id, "recorded", {
                "valid": True, "measurement": {"tokens_per_sec": 50.0}, "errors": [],
            }, parent_id=attempt_id),
            JournalEvent("sample", invalid_sample_id, "recorded", {
                "valid": False, "measurement": {"tokens_per_sec": 999.0},
                "errors": ["implausible_server_tps"],
            }, parent_id=attempt_id),
            JournalEvent("attempt", attempt_id, "complete", {}, parent_id=case_id),
            JournalEvent("case", case_id, "complete", {"valid_samples": 1}, parent_id=stage_id),
            JournalEvent("stage", stage_id, "complete", {"models_with_results": 1}, parent_id=plan.job_id),
            JournalEvent("job", plan.job_id, "complete", {}),
        ])
        projection = store.rebuild(plan.job_id)
        assert projection["jobs"][plan.job_id]["state"] == "complete"
        assert projection["cases"][case_id] == {
            "state": "complete", "parent_id": stage_id,
            "context_tokens": 2048, "valid_samples": 1,
        }
        assert projection["attempts"][attempt_id] == {
            "state": "complete", "parent_id": case_id, "number": 1,
        }
        assert projection["samples"][sample_id] == {
            "parent_id": attempt_id, "valid": True,
            "measurement": {"tokens_per_sec": 50.0}, "errors": [],
        }
        assert store.case_aggregate(plan.job_id, case_id) == {
            "attempts": 1, "samples": 2, "valid_samples": 1, "invalid_samples": 1,
            "measurement_means": {"tokens_per_sec": 50.0},
        }
        store.verify(plan.job_id)
    finally:
        store.close()


def test_illegal_transition_rejects_whole_batch(tmp_path):
    plan = make_plan()
    stage_id, _, _, _, _ = ids(plan)
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        store.create_job(plan)
        with pytest.raises(ValueError, match="illegal stage transition"):
            store.append(plan.job_id, [
                JournalEvent("job", plan.job_id, "running", {}),
                JournalEvent("stage", stage_id, "running", {}, parent_id=plan.job_id),
                JournalEvent("stage", stage_id, "pending", {}, parent_id=plan.job_id),
            ])
        assert store.events(plan.job_id) == []
    finally:
        store.close()


def test_unique_failure_rolls_back_every_event_in_transaction(tmp_path):
    plan = make_plan()
    stage_id, _, case_id, _, _ = ids(plan)
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        store.create_job(plan)
        duplicate = "event-fixed"
        with pytest.raises(sqlite3.IntegrityError):
            store.append(plan.job_id, [
                JournalEvent("job", plan.job_id, "running", {}, duplicate),
                JournalEvent("stage", stage_id, "running", {}, duplicate, parent_id=plan.job_id),
            ])
        assert store.events(plan.job_id) == []
    finally:
        store.close()


def test_database_triggers_prohibit_event_and_job_mutation(tmp_path):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        store.create_job(plan)
        store.append(plan.job_id, [JournalEvent("job", plan.job_id, "running", {})])
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute("UPDATE events SET state = 'failed'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store.connection.execute("DELETE FROM jobs")
    finally:
        store.close()


def test_event_store_migrates_pre_identity_job_table(tmp_path):
    path = tmp_path / "events.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY, plan_json TEXT NOT NULL,
        plan_id TEXT NOT NULL, schema_version INTEGER NOT NULL
    )""")
    connection.commit()
    connection.close()
    store = EventStore(path)
    try:
        columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(jobs)")}
        assert "resume_identity_json" in columns
    finally:
        store.close()


def test_verify_detects_digest_tampering_even_if_trigger_is_removed(tmp_path):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        store.create_job(plan)
        store.append(plan.job_id, [JournalEvent("job", plan.job_id, "running", {})])
        store.connection.execute("DROP TRIGGER events_no_update")
        store.connection.execute("UPDATE events SET payload_json = '{\"tampered\":true}'")
        store.connection.commit()
        with pytest.raises(ValueError, match="digest chain"):
            store.verify(plan.job_id)
    finally:
        store.close()


def test_sqlite_write_failure_preserves_prior_committed_events(tmp_path):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        store.create_job(plan)
        store.append(plan.job_id, [JournalEvent("job", plan.job_id, "running", {})])
        store.connection.execute("""
            CREATE TRIGGER simulate_disk_full BEFORE INSERT ON events
            BEGIN SELECT RAISE(ABORT, 'database or disk is full'); END
        """)
        with pytest.raises(sqlite3.IntegrityError, match="disk is full"):
            store.append(plan.job_id, [
                JournalEvent("stage", plan.stage_id("llm"), "running", {"stage": "llm"},
                             parent_id=plan.job_id),
            ])
        assert [(event.entity_type, event.state) for event in store.events(plan.job_id)] == [
            ("job", "running"),
        ]
        store.verify(plan.job_id)
    finally:
        store.close()


def test_child_event_requires_existing_parent(tmp_path):
    plan = make_plan()
    stage_id, _, case_id, _, _ = ids(plan)
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        store.create_job(plan)
        store.append(plan.job_id, [JournalEvent("job", plan.job_id, "running", {})])
        with pytest.raises(ValueError, match="missing parent"):
            store.append(plan.job_id, [
                JournalEvent("case", case_id, "running", {}, parent_id=stage_id),
            ])
    finally:
        store.close()


@pytest.mark.parametrize("event", [
    JournalEvent("unknown", "unknown_x", "running", {}),
    JournalEvent("case", "wrong", "running", {}),
    JournalEvent("case", "case_x", "running", {}, parent_id="wrong"),
    JournalEvent("sample", "sample_x", "complete", {}),
    JournalEvent("sample", "sample_x", "recorded", {
        "valid": False, "measurement": {}, "errors": [],
    }, parent_id="attempt_x"),
    JournalEvent("case", "case_x", "recorded", {}),
    JournalEvent("case", "case_x", "running", {"value": float("nan")}),
])
def test_event_validation_rejects_unknown_identity_state_and_nonfinite_payload(tmp_path, event):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    try:
        store.create_job(plan)
        with pytest.raises(ValueError):
            store.append(plan.job_id, [event])
        assert store.events(plan.job_id) == []
    finally:
        store.close()


def test_recovery_abandons_running_attempt_and_reopens_only_incomplete_case(tmp_path):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    store.start_stage(plan, "llm")
    model_id = plan.model_id("llm", plan.models["llm"][0])
    complete_case = plan.case_id("llm", model_id, {"context_tokens": 512})
    running_case = plan.case_id("llm", model_id, {"context_tokens": 2048})
    complete_attempt = plan.attempt_id(complete_case, 1)
    running_attempt = plan.attempt_id(running_case, 1)
    stage_id = plan.stage_id("llm")
    store.append(plan.job_id, [
        JournalEvent("case", complete_case, "running", {}, parent_id=stage_id),
        JournalEvent("attempt", complete_attempt, "running", {"number": 1},
                     parent_id=complete_case),
        JournalEvent("attempt", complete_attempt, "complete", {}, parent_id=complete_case),
        JournalEvent("case", complete_case, "complete", {}, parent_id=stage_id),
        JournalEvent("case", running_case, "running", {}, parent_id=stage_id),
        JournalEvent("attempt", running_attempt, "running", {"number": 1},
                     parent_id=running_case),
    ])
    assert store.prepare_recovery(plan.job_id, "llm") == {running_case: 2}
    projection = store.rebuild(plan.job_id)
    assert projection["cases"][complete_case]["state"] == "complete"
    assert projection["cases"][running_case]["state"] == "running"
    assert projection["attempts"][running_attempt]["state"] == "interrupted"
    store.close()


def test_recovery_rejects_completed_stage(tmp_path):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    store.start_stage(plan, "llm")
    store.append(plan.job_id, [JournalEvent(
        "stage", plan.stage_id("llm"), "complete", {}, parent_id=plan.job_id,
    )])
    with pytest.raises(ValueError, match="create a fork"):
        store.prepare_recovery(plan.job_id, "llm")
    store.close()


def test_terminal_transition_requires_explicit_recovery_payload(tmp_path):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    store.start_stage(plan, "llm")
    stage_id = plan.stage_id("llm")
    store.append(plan.job_id, [
        JournalEvent("stage", stage_id, "failed", {}, parent_id=plan.job_id),
    ])
    with pytest.raises(ValueError, match="illegal stage transition"):
        store.append(plan.job_id, [
            JournalEvent("stage", stage_id, "running", {}, parent_id=plan.job_id),
        ])
    store.append(plan.job_id, [
        JournalEvent("stage", stage_id, "running", {"recovery": "resume"},
                     parent_id=plan.job_id),
    ])
    store.close()


def test_later_stage_must_match_jobs_saved_resume_identity(tmp_path):
    plan = make_plan()
    store = EventStore(tmp_path / "events.sqlite3")
    identity = {"plan_id": plan.plan_id, "artifacts": {"a": {"sha256": "x", "size": 1}},
                "runtimes": {}, "methodology": {}}
    store.start_stage(plan, "llm", identity)
    with pytest.raises(ValueError, match="resume identity"):
        store.start_stage(plan, "llm", {**identity, "artifacts": {}})
    store.close()
