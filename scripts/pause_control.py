"""Cooperative pause state shared by benchmark parent and workload children."""

import json
import os
import time
import uuid
from pathlib import Path

from result_store import atomic_write_json


PAUSE_CONTROL_ENV = "LOCAL_AI_BENCH_PAUSE_CONTROL"
PAUSE_STATES = {"running", "paused"}


def create_pause_control(directory: Path) -> Path:
    path = Path(directory).resolve() / f".benchmark-control-{uuid.uuid4().hex}.json"
    write_pause_state(path, "running")
    return path


def write_pause_state(path: Path, state: str) -> None:
    if state not in PAUSE_STATES:
        raise ValueError(f"invalid pause state: {state}")
    atomic_write_json(Path(path), {"schema_version": 1, "state": state})


def read_pause_state(path: Path) -> str:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"benchmark pause control is unavailable: {exc}") from exc
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("state") not in PAUSE_STATES):
        raise RuntimeError("benchmark pause control is malformed")
    return value["state"]


def wait_if_paused(*, environ=os.environ, sleep=time.sleep, poll_seconds=0.2) -> bool:
    path = environ.get(PAUSE_CONTROL_ENV)
    if not path:
        return False
    paused = False
    while read_pause_state(Path(path)) == "paused":
        paused = True
        sleep(poll_seconds)
    return paused
