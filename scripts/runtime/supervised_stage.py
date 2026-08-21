"""Supervised journal-stage execution shared by benchmark and recovery flows."""

import json
import sys
from pathlib import Path
from typing import Callable, Protocol

from scripts.runtime.progress_events import PROGRESS_PREFIX
from scripts.results.llm_event_stage import (
    LLMEventStage, export_llm_section, result_path_from_event_store,
)
from scripts.results.accuracy_event_stage import AccuracyEventStage, export_accuracy
from scripts.results.embedding_event_stage import (
    EmbeddingEventStage, embedding_corpus_hash, export_embeddings,
)
from scripts.results.image_event_stage import ImageEventStage, export_images
from scripts.results.native_bench_event_stage import NativeBenchEventStage, export_native_bench_section
from scripts.results.native_concurrency_event_stage import (
    NativeConcurrencyEventStage, export_native_concurrency,
)
from scripts.results.sustained_event_stage import SustainedEventStage, export_sustained_section
from scripts.results.vllm_bench_event_stage import VllmBenchEventStage, export_vllm_bench
from scripts.results.run_plan import RunPlan
from scripts.runtime.log_redaction import redact_log_text
from scripts.runtime.runner_supervisor import RunnerSpec, RunnerSupervisor
from scripts.runtime.telemetry import PowerAvailability, TemperatureAvailability
from scripts.runtime.shared import Shared, _console_safe_text
from scripts.results.regrade import answer_sidecar_path
from scripts.results.result_store import atomic_write_json
from scripts.stage_registry import ACCURACY_TESTS
from scripts.workloads.accuracy_registry import accuracy_spec, selected_questions


class RunnerLike(Protocol):
    def run(self, on_event, /) -> int | None: ...
    def cancel(self) -> None: ...


def _redact_progress_value(value):
    if isinstance(value, str):
        return redact_log_text(value)
    if isinstance(value, list):
        return [_redact_progress_value(item) for item in value]
    if isinstance(value, dict):
        return {
            redact_log_text(key): _redact_progress_value(item)
            for key, item in value.items()
        }
    return value


def _redact_runner_text(text: str) -> str:
    stripped = text.rstrip()
    if not stripped.startswith(PROGRESS_PREFIX):
        return redact_log_text(stripped)
    try:
        payload = json.loads(stripped.removeprefix(PROGRESS_PREFIX))
    except json.JSONDecodeError:
        return redact_log_text(stripped)
    redacted = _redact_progress_value(payload)
    return f"{PROGRESS_PREFIX}{json.dumps(redacted, separators=(',', ':'))}"


def relay_runner_log(text: str) -> None:
    """Relay runner timestamps unchanged after redacting every output shape."""
    output = _console_safe_text(f"{_redact_runner_text(text)}\n")
    # codeql[py/clear-text-logging-sensitive-data]
    sys.stdout.write(output)
    sys.stdout.flush()


def run_supervised_stage(plan: RunPlan, event_path: Path, stage_name: str, save_fn,
                         supervisor_factory: Callable[..., RunnerLike] = RunnerSupervisor,
                         resume_identity=None, resume=False,
                         selected_case_ids=None,
                         power_availability: PowerAvailability | None = None,
                         temperature_availability: TemperatureAvailability | None = None) -> dict:
    event_path = Path(event_path).resolve()
    project_answers: Callable[[], dict] | None = None
    if stage_name == "img":
        journal = ImageEventStage(
            event_path, plan, lambda _: None, resume_identity=resume_identity,
            resume=resume, selected_case_ids=selected_case_ids,
        )
        project = lambda: export_images(event_path, plan.job_id)
    elif stage_name == "emb":
        journal = EmbeddingEventStage(
            event_path, plan, embedding_corpus_hash(resume_identity or {}),
            lambda _: None, resume_identity=resume_identity, resume=resume,
            selected_case_ids=selected_case_ids,
        )
        project = lambda: export_embeddings(event_path, plan.job_id)
    elif stage_name in ACCURACY_TESTS:
        spec = accuracy_spec(stage_name)
        questions = selected_questions(stage_name, plan.effective_config.get("sample_size"))
        bank_hash = Shared.file_hash(spec.data_path)
        journal = AccuracyEventStage(
            event_path, plan, stage_name, questions, bank_hash, spec.benchmark.score,
            lambda _results, _answers: None, resume_identity=resume_identity, resume=resume,
            selected_case_ids=selected_case_ids,
        )
        project = lambda: export_accuracy(
            event_path, plan.job_id, stage_name, questions, bank_hash, spec.benchmark.score,
        )[0]
        project_answers = lambda: export_accuracy(
            event_path, plan.job_id, stage_name, questions, bank_hash, spec.benchmark.score,
        )[1]
    elif stage_name == "llamabench":
        journal = NativeBenchEventStage(
            event_path, plan, lambda _: None, resume_identity=resume_identity, resume=resume,
        )
        project = lambda: export_native_bench_section(event_path, plan.job_id)
    elif stage_name == "llamabenchconc":
        journal = NativeConcurrencyEventStage(
            event_path, plan, lambda _: None, resume_identity=resume_identity, resume=resume,
        )
        project = lambda: export_native_concurrency(event_path, plan.job_id)
    elif stage_name == "vllmbench":
        journal = VllmBenchEventStage(
            event_path, plan, lambda _: None, resume_identity=resume_identity, resume=resume,
            selected_case_ids=selected_case_ids,
        )
        project = lambda: export_vllm_bench(event_path, plan.job_id)
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
            if project_answers:
                atomic_write_json(
                    answer_sidecar_path(result_path_from_event_store(event_path), stage_name),
                    project_answers(),
                )
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
    if project_answers:
        atomic_write_json(
            answer_sidecar_path(result_path_from_event_store(event_path), stage_name),
            project_answers(),
        )
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
