#!/usr/bin/env python3
"""Internal workload-runner entrypoint; not a general command interface."""

import argparse
import json
import os
import sys
import threading
import time

import config
from engines import get_engine
from event_store import EventStore
from conversation_selection import conv_skip_entry
from llm_event_stage import LLMEventStage, export_llm_section
from llm_conversation_benchmark import LLMConversationBenchmark
from llm_prefill_benchmark import LLMPrefillBenchmark
from llamabench_benchmark import LlamaBenchBenchmark
from models import LLM_MODELS
from native_bench_event_stage import NativeBenchEventStage
from runner_supervisor import RUNNER_EVENT_PREFIX, SUPPORTED_RUNNER_STAGES
from shared import Shared


_emit_lock = threading.Lock()


def emit(kind: str, **details) -> None:
    payload = {
        "ownership_token": os.environ.get("LOCAL_AI_BENCH_RUNNER_TOKEN"),
        "kind": kind, "timestamp": time.time(), **details,
    }
    with _emit_lock:
        sys.stdout.write(f"{RUNNER_EVENT_PREFIX}{json.dumps(payload, separators=(',', ':'))}\n")
        sys.stdout.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Internal Local AI Bench workload runner")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(SUPPORTED_RUNNER_STAGES))
    parser.add_argument("--event-store", required=True)
    return parser.parse_args(argv)


def load_runner_plan(path, job_id):
    store = EventStore(path)
    try:
        return store.load_plan(job_id)
    finally:
        store.close()


def execute_llm_job(path, job_id, *, engine_factory=get_engine,
                    benchmark_factory=LLMPrefillBenchmark) -> None:
    plan = load_runner_plan(path, job_id)
    plan.validate_for_execution()
    if "llm" not in plan.tests:
        raise ValueError("runner job does not include the LLM stage")
    settings = plan.effective_config
    config.N_RUNS = settings["runs"]
    config.RUN_TIMEOUT = settings["run_timeout_seconds"]
    catalog = {model["tag"]: model for model in LLM_MODELS}
    models = [
        {**identity, "label": catalog.get(identity["tag"], identity).get("label", identity["tag"])}
        for identity in plan.models["llm"]
    ]
    engine = engine_factory(plan.engine_name)
    Shared._active_engine = engine

    def notify(_section):
        store = EventStore(path)
        try:
            sequence = store.last_sequence(job_id)
        finally:
            store.close()
        emit("event", sequence=sequence, event={"stage": "llm", "committed": True})

    journal = None
    try:
        if not engine.start(gpu_visible=not plan.cpu_only):
            raise RuntimeError("runner could not prepare the inference engine")
        journal = LLMEventStage(path, plan, notify, initialize=False)
        benchmark_factory().run(
            engine=engine, models=models, context_lengths=settings["context_lengths"],
            warmup_runs=plan.warmup_runs, force_all=plan.force_all, journal=journal,
        )
    finally:
        if journal is not None:
            journal.close()
        if engine.available():
            engine.unload_all()
        Shared.shutdown_managed()


def execute_conversation_job(path, job_id, *, engine_factory=get_engine,
                             benchmark_factory=LLMConversationBenchmark) -> None:
    plan = load_runner_plan(path, job_id)
    plan.validate_for_execution()
    if "conv" not in plan.tests:
        raise ValueError("runner job does not include the conversation stage")
    settings = plan.effective_config
    config.RUN_TIMEOUT = settings["run_timeout_seconds"]
    catalog = {model["tag"]: model for model in LLM_MODELS}
    models = [
        {**identity, "label": catalog.get(identity["tag"], identity).get("label", identity["tag"])}
        for identity in plan.models["llm"]
    ]
    engine = engine_factory(plan.engine_name)
    Shared._active_engine = engine

    def notify(_section):
        store = EventStore(path)
        try:
            sequence = store.last_sequence(job_id)
        finally:
            store.close()
        emit("event", sequence=sequence, event={"stage": "conv", "committed": True})

    journal = None
    try:
        if not engine.start(gpu_visible=not plan.cpu_only):
            raise RuntimeError("runner could not prepare the inference engine")
        journal = LLMEventStage(
            path, plan, notify, stage_name="conv", initialize=False,
        )
        if "llm" in plan.tests:
            llm_results = export_llm_section(path, job_id)
            first_ctx = Shared.context_label(settings["context_lengths"][0])
            selected = []
            for model in models:
                skip = conv_skip_entry(
                    model, llm_results.get(model["short"]), first_ctx, plan.force_all,
                )
                if skip is None:
                    selected.append(model)
                else:
                    journal.record_model_state(model, "skipped", skip)
            models = selected
        benchmark_factory().run(
            engine=engine, models=models, warmup_runs=plan.warmup_runs,
            force_all=plan.force_all, max_prompt_tokens=settings["max_prompt_tokens"],
            journal=journal,
        )
    finally:
        if journal is not None:
            journal.close()
        if engine.available():
            engine.unload_all()
        Shared.shutdown_managed()


def execute_llamabench_job(path, job_id, *, engine_factory=get_engine,
                           benchmark_factory=LlamaBenchBenchmark) -> None:
    plan = load_runner_plan(path, job_id)
    plan.validate_for_execution()
    if "llamabench" not in plan.tests:
        raise ValueError("runner job does not include the native llama-bench stage")
    settings = plan.effective_config
    config.LLAMABENCH_PP = settings["llamabench_pp"]
    config.LLAMABENCH_TG = settings["llamabench_tg"]
    catalog = {model["tag"]: model for model in LLM_MODELS}
    models = [
        {**identity, "label": catalog.get(identity["tag"], identity).get("label", identity["tag"])}
        for identity in plan.models["llm"]
    ]
    engine = engine_factory(plan.engine_name)

    def notify(_section):
        store = EventStore(path)
        try:
            sequence = store.last_sequence(job_id)
        finally:
            store.close()
        emit("event", sequence=sequence, event={"stage": "llamabench", "committed": True})

    journal = NativeBenchEventStage(path, plan, notify, initialize=False)
    try:
        benchmark_factory().run(
            engine=engine, models=models, reps=settings["runs"],
            cpu_only=plan.cpu_only, journal=journal,
        )
    finally:
        journal.close()
        Shared.shutdown_managed()


def heartbeat(stop_event: threading.Event) -> None:
    while not stop_event.wait(5):
        emit("heartbeat")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.environ.get("LOCAL_AI_BENCH_RUNNER_TOKEN"):
        sys.stderr.write("Runner ownership token is required.\n")
        return 2
    stop_event = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(stop_event,), daemon=True)
    thread.start()
    try:
        if args.stage == "llm":
            execute_llm_job(args.event_store, args.job_id)
        elif args.stage == "conv":
            execute_conversation_job(args.event_store, args.job_id)
        else:
            execute_llamabench_job(args.event_store, args.job_id)
    except BaseException as exc:
        sys.stderr.write(f"Runner failed: {type(exc).__name__}: {exc}\n")
        emit("terminal", status="failed", job_id=args.job_id, stage=args.stage)
        return 1
    finally:
        stop_event.set()
        thread.join(timeout=1)
    emit("terminal", status="complete", job_id=args.job_id, stage=args.stage)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
