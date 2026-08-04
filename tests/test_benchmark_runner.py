from benchmark import run_supervised_llm
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
