"""Small, testable benchmark stage and lifecycle orchestration boundaries."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from result_store import ResultStore
from run_plan import RunPlan
from shared import Shared


STAGE_ORDER = (
    "llm", "conv", "llamabench", "llamabenchconc", "emb",
    "mcq", "math", "reasoning", "code", "tool", "conc_tool", "conc_chat", "img",
)


@dataclass(frozen=True)
class RunPaths:
    output_path: Path
    comfyui_dir: Path | None = None


@dataclass
class RunContext:
    plan: RunPlan
    paths: RunPaths
    engine: object
    store: ResultStore
    lifecycle: "LifecycleCoordinator"
    profile: dict | None = None


def _noop(_context: RunContext) -> None:
    pass


@dataclass(frozen=True)
class StageDefinition:
    key: str
    section: str
    selected_models: int
    runner: Callable[[RunContext], dict]
    requires_engine: bool = False
    prepare: Callable[[RunContext], None] = field(default=_noop)
    cleanup: Callable[[RunContext], None] = field(default=_noop)


def ordered_stage_keys(selected: tuple[str, ...]) -> list[str]:
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate selected stage key")
    missing = set(selected) - set(STAGE_ORDER)
    if missing:
        raise ValueError(f"stages missing from STAGE_ORDER: {sorted(missing)}")
    return [key for key in STAGE_ORDER if key in selected]


def select_stages(registry: list[StageDefinition], selected: tuple[str, ...]) -> list[StageDefinition]:
    by_key = {stage.key: stage for stage in registry}
    if len(by_key) != len(registry):
        raise ValueError("duplicate stage key")
    unordered = set(by_key) - set(STAGE_ORDER)
    if unordered:
        raise ValueError(f"registered stages missing from STAGE_ORDER: {sorted(unordered)}")
    missing = set(selected) - set(by_key)
    if missing:
        raise ValueError(f"unknown selected stages: {sorted(missing)}")
    return [by_key[key] for key in ordered_stage_keys(selected)]


class StageExecutionError(RuntimeError):
    def __init__(self, stage: str, phase: str, cause: BaseException):
        super().__init__(f"{stage} {phase} failed: {cause}")
        self.stage = stage
        self.phase = phase
        self.__cause__ = cause


def execute_stages(context: RunContext, stages: list[StageDefinition]) -> None:
    for stage in stages:
        context.store.start_stage(stage.key, stage.selected_models)
        try:
            if stage.requires_engine:
                context.lifecycle.ensure_engine(context.plan.cpu_only)
            stage.prepare(context)
        except BaseException as exc:
            try:
                stage.cleanup(context)
            except Exception as cleanup_exc:
                context.store.record_cleanup_failure(stage.key, cleanup_exc)
                Shared.warn(f"{stage.key} cleanup also failed: {cleanup_exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise StageExecutionError(stage.key, "preparation", exc) from exc
        try:
            result = stage.runner(context)
        except BaseException as exc:
            try:
                stage.cleanup(context)
            except Exception as cleanup_exc:
                context.store.record_cleanup_failure(stage.key, cleanup_exc)
                Shared.warn(f"{stage.key} cleanup also failed: {cleanup_exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise StageExecutionError(stage.key, "execution", exc) from exc
        context.store.update_section(stage.section, result, stage.key)
        try:
            stage.cleanup(context)
        except BaseException as exc:
            raise StageExecutionError(stage.key, "cleanup", exc) from exc
        context.store.complete_stage(stage.key, stage.section)


def execute_with_final_cleanup(action: Callable[[], None], lifecycle: "LifecycleCoordinator") -> None:
    terminal_exc = None
    try:
        action()
    except BaseException as exc:
        terminal_exc = exc
    finally:
        try:
            lifecycle.cleanup()
        except Exception as cleanup_exc:
            if terminal_exc is None:
                terminal_exc = StageExecutionError("run", "cleanup", cleanup_exc)
            else:
                Shared.warn(f"Final cleanup also failed: {cleanup_exc}")
    if terminal_exc is not None:
        raise terminal_exc


class LifecycleCoordinator:
    """Coordinates engine exclusivity, mode changes, and final cleanup."""

    def __init__(self, engine, engine_name: str, engine_names: list[str], engine_factory,
                 shutdown_managed: Callable[[], None], comfyui_available=lambda: False,
                 free_comfyui=lambda: None):
        self.engine = engine
        self.engine_name = engine_name
        self.engine_names = engine_names
        self.engine_factory = engine_factory
        self.shutdown_managed = shutdown_managed
        self.comfyui_available = comfyui_available
        self.free_comfyui = free_comfyui

    def prepare_engine(self, cpu_only: bool) -> bool:
        for name in self.engine_names:
            if name == self.engine_name:
                continue
            other = self.engine_factory(name)
            if other.available():
                other.stop()
        if cpu_only:
            self.engine.stop()
            return self.engine.start(gpu_visible=False)
        return self.engine.ensure_running()

    def stop_engine(self) -> None:
        if self.engine.available():
            self.engine.stop()

    def ensure_engine(self, cpu_only: bool) -> bool:
        if cpu_only and (not getattr(self.engine, "_cpu_only_active", False)
                         or not self.engine.available()):
            self.engine.stop()
            return self.engine.start(gpu_visible=False)
        if not self.engine.available():
            return self.engine.ensure_running()
        return True

    def restore_gpu(self) -> None:
        if getattr(self.engine, "_cpu_only_active", False):
            self.engine.stop()
            self.engine.start()

    def cleanup(self) -> None:
        error = None
        try:
            if self.engine.available():
                self.engine.unload_all()
        except Exception as exc:
            error = exc
        try:
            if self.comfyui_available():
                self.free_comfyui()
        except Exception as exc:
            error = error or exc
        try:
            self.shutdown_managed()
        except Exception as exc:
            error = error or exc
        if error:
            raise error
