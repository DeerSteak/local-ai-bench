"""Opt-in structured progress events shared by workloads and frontends."""

import json
import os
import sys

from scripts.results.result_store import model_counts


PROGRESS_PREFIX = "::local-ai-bench-progress::"
_current_engine: str | None = None


def set_progress_engine(name: str | None) -> None:
    """Attach the active engine name to later progress events."""
    global _current_engine
    _current_engine = name


def emit_progress(kind: str, stage: str, status: str, model: str | None = None,
                  **details) -> None:
    if os.environ.get("LOCAL_AI_BENCH_PROGRESS") != "1":
        return
    payload = {"kind": kind, "stage": stage, "status": status}
    if _current_engine is not None:
        payload["engine"] = _current_engine
    if model is not None:
        payload["model"] = model
    payload.update(details)
    sys.stdout.write(f"{PROGRESS_PREFIX}{json.dumps(payload, separators=(',', ':'))}\n")
    sys.stdout.flush()


def emit_model_finished(stage: str, model: str, result: dict | None = None,
                        model_id: str | None = None) -> None:
    exc_type = sys.exc_info()[0]
    if exc_type is None:
        status = "complete"
    elif issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        status = "interrupted"
    else:
        status = "failed"
    details = {}
    if result is not None:
        counts = model_counts({"model": result})
        details["usable"] = counts["models_with_results"] == 1
    if model_id is not None:
        details["model_id"] = model_id
    emit_progress("model", stage, status, model, **details)


def emit_result_saved(path) -> None:
    emit_progress("result", "run", "complete", path=str(path))
