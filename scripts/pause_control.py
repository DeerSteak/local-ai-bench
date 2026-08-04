"""Cooperative pause state shared by benchmark parent and workload children."""

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from result_store import atomic_write_json


PAUSE_CONTROL_ENV = "LOCAL_AI_BENCH_PAUSE_CONTROL"
PAUSE_STATES = {"running", "paused"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def create_pause_control(directory: Path | None = None) -> Path:
    root = Path(directory or Path(tempfile.gettempdir()) / "local-ai-bench").resolve()
    path = root / f"benchmark-control-{uuid.uuid4().hex}.json"
    write_pause_state(path, "running")
    return path


def write_pause_state(path: Path, state: str, *, now=utc_now) -> None:
    if state not in PAUSE_STATES:
        raise ValueError(f"invalid pause state: {state}")
    path = Path(path)
    transitions = []
    if path.is_file():
        try:
            current = _read_pause_control(path)
            transitions = list(current.get("transitions", []))
            if current["state"] == state:
                return
        except RuntimeError:
            transitions = []
    transitions.append({"state": state, "at": now()})
    atomic_write_json(path, {"schema_version": 2, "state": state, "transitions": transitions})


def _read_pause_control(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"benchmark pause control is unavailable: {exc}") from exc
    if not isinstance(value, dict) or value.get("state") not in PAUSE_STATES:
        raise RuntimeError("benchmark pause control is malformed")
    if value.get("schema_version") == 1:
        return value
    transitions = value.get("transitions")
    if (value.get("schema_version") != 2 or not isinstance(transitions, list)
            or any(not isinstance(item, dict) or item.get("state") not in PAUSE_STATES
                   or not isinstance(item.get("at"), str) for item in transitions)):
        raise RuntimeError("benchmark pause control is malformed")
    return value


def read_pause_state(path: Path) -> str:
    return _read_pause_control(path)["state"]


def pause_evidence(path: Path) -> dict | None:
    control = _read_pause_control(path)
    transitions = control.get("transitions", [])
    pauses = [item for item in transitions if item["state"] == "paused"]
    if not pauses:
        return None
    return {"pause_requests": len(pauses), "control_transitions": transitions}


def apply_pause_evidence(run: dict, *, environ=os.environ) -> bool:
    path = environ.get(PAUSE_CONTROL_ENV)
    if not path:
        return False
    try:
        evidence = pause_evidence(Path(path))
    except RuntimeError:
        return False
    if evidence is None:
        return False
    run["pause"] = evidence
    return True


def wait_if_paused(*, environ=os.environ, sleep=time.sleep, poll_seconds=0.2) -> bool:
    path = environ.get(PAUSE_CONTROL_ENV)
    if not path:
        return False
    paused = False
    while True:
        try:
            state = read_pause_state(Path(path))
        except RuntimeError:
            return paused
        if state != "paused":
            return paused
        paused = True
        sleep(poll_seconds)
