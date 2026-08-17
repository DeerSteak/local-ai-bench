"""Supervised journal-stage execution shared by benchmark and recovery flows."""

import sys
from pathlib import Path
from typing import Callable, Protocol

from scripts.runtime.progress_events import PROGRESS_PREFIX
from scripts.results.llm_event_stage import LLMEventStage, export_llm_section
from scripts.results.native_bench_event_stage import NativeBenchEventStage, export_native_bench_section
from scripts.results.sustained_event_stage import SustainedEventStage, export_sustained_section
from scripts.results.run_plan import RunPlan
from scripts.runtime.log_redaction import redact_log_text
from scripts.runtime.runner_supervisor import RunnerSpec, RunnerSupervisor
from scripts.runtime.telemetry import PowerAvailability, TemperatureAvailability


class RunnerLike(Protocol):
    def run(self, on_event, /) -> int | None: ...
    def cancel(self) -> None: ...


def relay_runner_log(text: str) -> None:
    """Relay runner timestamps unchanged while redacting ordinary log lines."""
    if text.startswith(PROGRESS_PREFIX):
        sys.stdout.write(text if text.endswith("\n") else f"{text}\n")
        sys.stdout.flush()
        return
    sys.stdout.write(f"{redact_log_text(text.rstrip())}\n")
    sys.stdout.flush()


def run_supervised_stage(plan: RunPlan, event_path: Path, stage_name: str, save_fn,
                         supervisor_factory: Callable[..., RunnerLike] = RunnerSupervisor,
                         resume_identity=None, resume=False,
                         selected_case_ids=None,
                         power_availability: PowerAvailability | None = None,
                         temperature_availability: TemperatureAvailability | None = None) -> dict:
    event_path = Path(event_path).resolve()
    if stage_name == "llamabench":
        journal = NativeBenchEventStage(
            event_path, plan, lambda _: None, resume_identity=resume_identity, resume=resume,
        )
        project = lambda: export_native_bench_section(event_path, plan.job_id)
    elif stage_name == "sustained":
        journal = SustainedEventStage(
            event_path, plan, lambda _: None, resume_identity=resume_identity, resume=resume,
            selected_case_ids=selected_case_ids,
        )
        project = lambda: export_sustained_section(event_path, plan.job_id)
    else:
        model_family = "concurrency" if stage_name in {"conc_tool", "conc_chat"} else "llm"
        journal = LLMEventStage(
            event_path, plan, lambda _: None, stage_name=stage_name,
            model_family=model_family, resume_identity=resume_identity, resume=resume,
            selected_case_ids=selected_case_ids,
        )
        project = lambda: export_llm_section(event_path, plan.job_id, stage_name, model_family)
    journal.close()
    supervisor = supervisor_factory(RunnerSpec(
        plan.job_id, stage_name, event_path, power_availability,
        temperature_availability,
    ))
    terminal = []

    def on_runner_event(event):
        if event["kind"] == "event":
            save_fn(project())
        elif event["kind"] == "terminal":
            terminal.append(event["status"])
        elif event["kind"] == "log":
            relay_runner_log(event["text"])

    try:
        return_code = supervisor.run(on_runner_event)
    finally:
        supervisor.cancel()
    section = project()
    save_fn(section)
    if return_code or terminal != ["complete"]:
        raise RuntimeError(f"{stage_name} runner failed with exit code {return_code}")
    return section


def run_supervised_llm(plan: RunPlan, event_path: Path, save_fn,
                       supervisor_factory: Callable[..., RunnerLike] = RunnerSupervisor,
                       resume_identity=None,
                       power_availability: PowerAvailability | None = None,
                       temperature_availability: TemperatureAvailability | None = None) -> dict:
    return run_supervised_stage(
        plan, event_path, "llm", save_fn, supervisor_factory, resume_identity,
        power_availability=power_availability,
        temperature_availability=temperature_availability,
    )
