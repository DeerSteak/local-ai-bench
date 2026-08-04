from pathlib import Path
from dataclasses import FrozenInstanceError

import pytest

from orchestration import (
    LifecycleCoordinator, RunContext, RunPaths, StageDefinition, StageExecutionError,
    execute_stages, execute_with_final_cleanup, ordered_stage_keys, select_stages,
)
from result_store import ResultStore
from run_plan import RunPlan


def _store(tmp_path):
    data = {"run": {"status": "running", "stages": {}}, "a": {}, "b": {}}
    return ResultStore(tmp_path / "result.json", data, writer=lambda *_: None)


def _plan(tests=("a",), cpu_only=False):
    return RunPlan.create(
        application_version="4.1", engine_name="fake", tests=tests,
        stage_order=tests, models={"llm": [], "concurrency": [], "embeddings": [], "images": []},
        effective_config={"warmup_runs": 1, "cpu_only": cpu_only, "force_all": False},
    )


def test_run_plan_is_immutable():
    spec = _plan(("llm",))
    with pytest.raises(FrozenInstanceError):
        spec.engine_name = "other"
    assert spec.tests == ("llm",)


def test_execute_stages_preserves_registry_order_and_transitions(tmp_path):
    events = []
    store = _store(tmp_path)
    context = RunContext(
        _plan(("a", "b")), RunPaths(Path("out")), object(), store, object(),
    )
    stages = [
        StageDefinition("a", "a", 1, lambda _: events.append("run-a") or {"m": {"x": 1}},
                        prepare=lambda _: events.append("prepare-a"),
                        cleanup=lambda _: events.append("cleanup-a")),
        StageDefinition("b", "b", 0, lambda _: events.append("run-b") or {}),
    ]
    execute_stages(context, stages)
    assert events == ["prepare-a", "run-a", "cleanup-a", "run-b"]
    assert list(store.data["run"]["stages"]) == ["a", "b"]
    assert all(stage["status"] == "complete" for stage in store.data["run"]["stages"].values())


def test_execute_stages_prepares_only_engine_required_stages(tmp_path):
    class Lifecycle:
        def __init__(self): self.calls = []
        def ensure_engine(self, cpu_only): self.calls.append(cpu_only)

    lifecycle = Lifecycle()
    context = RunContext(
        _plan(("llm", "img"), cpu_only=True), RunPaths(Path("out")),
        object(), _store(tmp_path), lifecycle,
    )
    stages = [
        StageDefinition("llm", "a", 0, lambda _: {}, requires_engine=True),
        StageDefinition("img", "b", 0, lambda _: {}),
    ]
    execute_stages(context, stages)
    assert lifecycle.calls == [True]


def test_select_stages_uses_fixed_order_not_cli_or_registry_order():
    noop = lambda _: {}
    registry = [
        StageDefinition("img", "images", 0, noop),
        StageDefinition("llm", "llm", 0, noop),
        StageDefinition("emb", "embeddings", 0, noop),
    ]
    selected = select_stages(registry, ("img", "emb", "llm"))
    assert [stage.key for stage in selected] == ["llm", "emb", "img"]


def test_select_stages_rejects_unknown_and_duplicate_keys():
    stage = StageDefinition("llm", "llm", 0, lambda _: {})
    with pytest.raises(ValueError, match="unknown"):
        select_stages([stage], ("conv",))
    with pytest.raises(ValueError, match="duplicate"):
        select_stages([stage, stage], ("llm",))


def test_select_stages_rejects_registry_key_missing_from_fixed_order():
    registry = [
        StageDefinition("llm", "llm", 0, lambda _: {}),
        StageDefinition("newstage", "newstage", 0, lambda _: {}),
    ]
    with pytest.raises(ValueError, match="registered stages missing from STAGE_ORDER"):
        select_stages(registry, ("llm", "newstage"))


def test_ordered_stage_keys_rejects_unknown_and_duplicate_selections():
    with pytest.raises(ValueError, match="missing from STAGE_ORDER"):
        ordered_stage_keys(("llm", "newstage"))
    with pytest.raises(ValueError, match="duplicate selected"):
        ordered_stage_keys(("llm", "llm"))


def test_execute_stages_always_cleans_up_and_leaves_failed_stage_running(tmp_path):
    events = []
    context = RunContext(
        _plan(), RunPaths(Path("out")), object(), _store(tmp_path), object(),
    )
    stage = StageDefinition(
        "a", "a", 1, lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        cleanup=lambda _: events.append("cleanup"),
    )
    with pytest.raises(StageExecutionError, match="execution failed: boom"):
        execute_stages(context, [stage])
    assert events == ["cleanup"]
    assert context.store.data["run"]["stages"]["a"]["status"] == "running"


@pytest.mark.parametrize("phase", ["preparation", "execution"])
def test_execute_stages_records_secondary_cleanup_failure(tmp_path, phase):
    def fail(_): raise RuntimeError("primary")
    def cleanup(_): raise OSError("cleanup")
    context = RunContext(
        _plan(), RunPaths(Path("out")), object(), _store(tmp_path), object(),
    )
    stage = StageDefinition(
        "a", "a", 0, fail if phase == "execution" else lambda _: {},
        prepare=fail if phase == "preparation" else lambda _: None, cleanup=cleanup,
    )
    with pytest.raises(StageExecutionError) as exc_info:
        execute_stages(context, [stage])
    assert exc_info.value.phase == phase
    assert context.store.data["run"]["stages"]["a"]["cleanup_failure"] == {
        "reason": "stage_cleanup_failed", "error_type": "OSError",
    }


def test_execute_stages_preserves_system_exit_and_still_cleans_up(tmp_path):
    events = []
    context = RunContext(
        _plan(), RunPaths(Path("out")), object(), _store(tmp_path), object(),
    )
    stage = StageDefinition(
        "a", "a", 0, lambda _: (_ for _ in ()).throw(SystemExit(0)),
        cleanup=lambda _: events.append("cleanup"),
    )
    with pytest.raises(SystemExit) as exc_info:
        execute_stages(context, [stage])
    assert exc_info.value.code == 0
    assert events == ["cleanup"]


def test_execute_with_final_cleanup_cleans_before_propagating_failure():
    events = []
    lifecycle = type("Lifecycle", (), {"cleanup": lambda _: events.append("cleanup")})()

    def fail():
        events.append("execute")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        execute_with_final_cleanup(fail, lifecycle)
    assert events == ["execute", "cleanup"]


def test_execute_with_final_cleanup_classifies_cleanup_only_failure():
    lifecycle = type(
        "Lifecycle", (),
        {"cleanup": lambda _: (_ for _ in ()).throw(OSError("cleanup"))},
    )()
    with pytest.raises(StageExecutionError) as exc_info:
        execute_with_final_cleanup(lambda: None, lifecycle)
    assert exc_info.value.phase == "cleanup"


@pytest.mark.parametrize("phase", ["preparation", "cleanup"])
def test_execute_stages_classifies_hook_failures(tmp_path, phase):
    def fail(_): raise RuntimeError("hook")
    context = RunContext(
        _plan(), RunPaths(Path("out")), object(), _store(tmp_path), object(),
    )
    stage = StageDefinition(
        "a", "a", 0, lambda _: {},
        prepare=fail if phase == "preparation" else lambda _: None,
        cleanup=fail if phase == "cleanup" else lambda _: None,
    )
    with pytest.raises(StageExecutionError) as exc_info:
        execute_stages(context, [stage])
    assert exc_info.value.phase == phase


class FakeEngine:
    def __init__(self, available=True):
        self.is_available = available
        self.calls = []
        self._cpu_only_active = False

    def available(self): return self.is_available
    def stop(self): self.calls.append("stop")
    def start(self, gpu_visible=True):
        self.calls.append(("start", gpu_visible))
        self._cpu_only_active = not gpu_visible
        return True
    def ensure_running(self): self.calls.append("ensure"); return True
    def unload_all(self): self.calls.append("unload_all")


def test_lifecycle_stops_competitors_and_restores_cpu_mode():
    active = FakeEngine()
    other = FakeEngine()
    cleaned = []
    lifecycle = LifecycleCoordinator(
        active, "active", ["active", "other"], lambda _: other, lambda: cleaned.append(True),
    )
    assert lifecycle.prepare_engine(cpu_only=True)
    assert other.calls == ["stop"]
    assert active.calls == ["stop", ("start", False)]
    lifecycle.restore_gpu()
    lifecycle.cleanup()
    assert active.calls[-3:] == ["stop", ("start", True), "unload_all"]
    assert cleaned == [True]


def test_lifecycle_reestablishes_cpu_mode_after_native_stage_stopped_engine():
    engine = FakeEngine(available=False)
    lifecycle = LifecycleCoordinator(
        engine, "active", ["active"], lambda _: None, lambda: None,
    )
    assert lifecycle.ensure_engine(cpu_only=True)
    assert engine.calls == ["stop", ("start", False)]


def test_lifecycle_handles_normal_gpu_availability_paths():
    engine = FakeEngine(available=True)
    lifecycle = LifecycleCoordinator(
        engine, "active", ["active"], lambda _: None, lambda: None,
    )
    assert lifecycle.prepare_engine(cpu_only=False)
    assert lifecycle.ensure_engine(cpu_only=False)
    lifecycle.stop_engine()
    assert engine.calls == ["ensure", "stop"]

    stopped = FakeEngine(available=False)
    lifecycle = LifecycleCoordinator(
        stopped, "active", ["active"], lambda _: None, lambda: None,
    )
    assert lifecycle.ensure_engine(cpu_only=False)
    assert stopped.calls == ["ensure"]


def test_lifecycle_attempts_all_cleanup_after_failure():
    class FailingEngine(FakeEngine):
        def unload_all(self): raise RuntimeError("unload")

    freed = []
    stopped = []
    lifecycle = LifecycleCoordinator(
        FailingEngine(), "active", ["active"], lambda _: None,
        lambda: stopped.append(True), lambda: True, lambda: freed.append(True),
    )
    with pytest.raises(RuntimeError, match="unload"):
        lifecycle.cleanup()
    assert freed == [True]
    assert stopped == [True]
