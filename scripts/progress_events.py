"""Opt-in structured progress events for the graphical launcher."""

import json
import os
import sys


PROGRESS_PREFIX = "::local-ai-bench-progress::"


def emit_progress(kind: str, stage: str, status: str, model: str | None = None) -> None:
    if os.environ.get("LOCAL_AI_BENCH_PROGRESS") != "1":
        return
    payload = {"kind": kind, "stage": stage, "status": status}
    if model is not None:
        payload["model"] = model
    sys.stdout.write(f"{PROGRESS_PREFIX}{json.dumps(payload, separators=(',', ':'))}\n")
    sys.stdout.flush()


def emit_model_finished(stage: str, model: str) -> None:
    exc_type = sys.exc_info()[0]
    if exc_type is None:
        status = "complete"
    elif issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        status = "interrupted"
    else:
        status = "failed"
    emit_progress("model", stage, status, model)
