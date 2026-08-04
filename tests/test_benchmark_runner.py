from benchmark import run_supervised_llm, run_supervised_stage
from engines.base import GenerationMeasurement
from llm_event_stage import LLMEventStage
from run_plan import RunPlan


def make_plan():
    return RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["llm"], stage_order=["llm"],
        models={
            "llm": [{"tag": "fake:model", "short": "fake"}],
            "concurrency": [], "embeddings": [], "images": [],
        },
        effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                          "force_all": False},
    )


def test_supervised_llm_checkpoints_commits_and_requires_clean_terminal(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"
    saved = []
    cancelled = []

    class Supervisor:
        def __init__(self, spec):
            assert spec.job_id == plan.job_id

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            sample = GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [sample], "ok", 1,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            cancelled.append(True)

    result = run_supervised_llm(plan, path, saved.append, Supervisor)
    assert result == saved[-1]
    assert saved[0]["fake"]["512"]["tps_mean"] == 50
    assert cancelled == [True]


def test_runner_failure_preserves_committed_case_for_parent_recovery(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"

    class CrashedSupervisor:
        def __init__(self, _spec):
            pass

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)], "ok", 1,
            )
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "failed"})
            return 1

        @staticmethod
        def cancel():
            pass

    saved = []
    try:
        run_supervised_llm(plan, path, saved.append, CrashedSupervisor)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed runner was accepted")
    assert saved[-1]["fake"]["512"]["tps_mean"] == 50


def test_parent_export_failure_keeps_child_commit(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"

    class InterruptedSupervisor:
        def __init__(self, _spec):
            pass

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)], "ok", 1,
            )
            stage.close()
            callback({"kind": "event"})
            raise KeyboardInterrupt

        @staticmethod
        def cancel():
            pass

    def read_only_export(_section):
        raise OSError("read-only output")

    try:
        run_supervised_llm(plan, path, read_only_export, InterruptedSupervisor)
    except (KeyboardInterrupt, OSError):
        pass
    else:
        raise AssertionError("coordinator failure was swallowed")

    reopened = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        assert reopened.export()["fake"]["512"]["tps_mean"] == 50
    finally:
        reopened.close()


def test_parent_interruption_keeps_child_commit(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"

    class InterruptedSupervisor:
        def __init__(self, _spec):
            pass

        def run(self, _callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)], "ok", 1,
            )
            stage.close()
            raise KeyboardInterrupt

        @staticmethod
        def cancel():
            pass

    try:
        run_supervised_llm(plan, path, lambda _: None, InterruptedSupervisor)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("interruption was swallowed")
    reopened = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        assert reopened.export()["fake"]["512"]["tps_mean"] == 50
    finally:
        reopened.close()


def test_generic_supervisor_projects_conversation_stage(tmp_path):
    base = make_plan()
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["conv"],
        stage_order=["conv"], models=base.models,
        effective_config=base.effective_config,
    )
    path = tmp_path / "events.sqlite3"

    class Supervisor:
        def __init__(self, spec):
            assert spec.stage == "conv"

        def run(self, callback):
            stage = LLMEventStage(
                path.resolve(), plan, lambda _: None, stage_name="conv", initialize=False,
            )
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                0, "0K", [GenerationMeasurement(0.1, 96, 48, 2.1, 2.0)], "ok", 1,
                depth_tokens=400,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            pass

    result = run_supervised_stage(plan, path, "conv", lambda _: None, Supervisor)
    assert result["fake"]["0K"]["depth_tokens"] == 400
