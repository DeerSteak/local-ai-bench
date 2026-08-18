"""Resumable execution plans for platform qualification."""

import json
import hashlib
import os
import signal
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from scripts.release.qualification import QUALIFICATION_LIFECYCLE
from scripts.runtime import config


RECIPE_KEYS = {"target", "coverage", "environment", "steps"}
TARGET_KEYS = {
    "id", "platform", "architecture", "runtime", "runtime_version", "backend", "accelerator",
}
COVERAGE_KEYS = {"workloads", "models", "notes"}
STEP_KEYS = {
    "command", "timeout_seconds", "expected_exit_codes", "interrupt_when_log_contains",
}
QUALIFICATION_ENV_KEYS = {
    "HF_HOME", "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES", "LOCAL_AI_BENCH_PROGRESS",
}


def validate_qualification_recipe(recipe: dict) -> None:
    if not isinstance(recipe, dict) or set(recipe) != RECIPE_KEYS:
        raise ValueError("qualification recipe has missing or unknown fields")
    target = recipe["target"]
    if not isinstance(target, dict) or set(target) != TARGET_KEYS:
        raise ValueError("qualification recipe target has missing or unknown fields")
    if any(not isinstance(target[key], str) or not target[key].strip() for key in TARGET_KEYS):
        raise ValueError("qualification recipe target fields must be non-empty text")
    coverage = recipe["coverage"]
    if not isinstance(coverage, dict) or set(coverage) != COVERAGE_KEYS:
        raise ValueError("qualification recipe coverage has missing or unknown fields")
    for key in ("workloads", "models"):
        if (not isinstance(coverage[key], list) or not coverage[key]
                or any(not isinstance(item, str) or not item.strip() for item in coverage[key])):
            raise ValueError(f"qualification recipe coverage requires non-empty {key}")
    if not isinstance(coverage["notes"], str):
        raise ValueError("qualification recipe coverage notes must be text")
    environment = recipe["environment"]
    if (not isinstance(environment, dict)
            or any(key not in QUALIFICATION_ENV_KEYS
                   or not isinstance(value, str) for key, value in environment.items())):
        raise ValueError("qualification environment contains an unsafe or unknown field")
    steps = recipe["steps"]
    if not isinstance(steps, dict) or set(steps) != set(QUALIFICATION_LIFECYCLE):
        raise ValueError("qualification recipe must define every lifecycle step")
    for name, step in steps.items():
        if not isinstance(step, dict) or set(step) != STEP_KEYS:
            raise ValueError(f"qualification step {name} has missing or unknown fields")
        command = step["command"]
        if (not isinstance(command, list) or not command
                or any(not isinstance(arg, str) or not arg for arg in command)):
            raise ValueError(f"qualification step {name} requires an argv command")
        timeout = step["timeout_seconds"]
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
            raise ValueError(f"qualification step {name} requires a positive timeout")
        exit_codes = step["expected_exit_codes"]
        if (not isinstance(exit_codes, list) or not exit_codes
                or any(not isinstance(code, int) or isinstance(code, bool) for code in exit_codes)):
            raise ValueError(f"qualification step {name} requires expected exit codes")
        interrupt = step["interrupt_when_log_contains"]
        if interrupt is not None and (not isinstance(interrupt, str) or not interrupt):
            raise ValueError(f"qualification step {name} has an invalid interrupt marker")
        if (name == "cancellation") != (interrupt is not None):
            raise ValueError("only the cancellation step may define an interrupt marker")


def load_qualification_recipe(path: Path) -> dict:
    try:
        recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read qualification recipe: {exc}") from None
    validate_qualification_recipe(recipe)
    return recipe


def qualification_preview(recipe: dict, output_dir: Path) -> dict:
    validate_qualification_recipe(recipe)
    output_dir = Path(output_dir).resolve()
    return {
        "mode": "preview",
        "target": dict(recipe["target"]),
        "coverage": dict(recipe["coverage"]),
        "environment": dict(recipe["environment"]),
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / "qualification-state.json"),
        "steps": [
            {
                "name": name,
                "command": list(recipe["steps"][name]["command"]),
                "timeout_seconds": recipe["steps"][name]["timeout_seconds"],
                "interrupt_when_log_contains":
                    recipe["steps"][name]["interrupt_when_log_contains"],
            }
            for name in QUALIFICATION_LIFECYCLE
        ],
    }


def recipe_digest(recipe: dict) -> str:
    validate_qualification_recipe(recipe)
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def execution_recipe_gaps(recipe: dict) -> list[str]:
    validate_qualification_recipe(recipe)
    gaps = []
    for key, value in recipe["target"].items():
        if "REPLACE_" in value:
            gaps.append(f"target.{key}")
    for key in ("workloads", "models"):
        if any("REPLACE_" in value for value in recipe["coverage"][key]):
            gaps.append(f"coverage.{key}")
    for name, step in recipe["steps"].items():
        if any("REPLACE_" in argument for argument in step["command"]):
            gaps.append(f"steps.{name}.command")
    return gaps


def initial_run_state(recipe: dict) -> dict:
    return {
        "schema": "qualification-run-v1", "recipe_digest": recipe_digest(recipe),
        "target": dict(recipe["target"]), "coverage": dict(recipe["coverage"]),
        "steps": {
            name: {"status": "pending", "exit_code": None, "detail": None,
                   "started_at": None, "finished_at": None, "log": None}
            for name in QUALIFICATION_LIFECYCLE
        },
    }


def next_qualification_step(state: dict) -> str | None:
    for name in QUALIFICATION_LIFECYCLE:
        if state["steps"][name]["status"] != "passed":
            return name
    return None


def qualification_entry_from_run(state: dict, suite_version: str, evidence_path: str) -> dict:
    lifecycle = {
        name: "passed" if state["steps"][name]["status"] == "passed" else "failed"
        for name in QUALIFICATION_LIFECYCLE
    }
    failures = [
        {
            "step": name,
            "detail": state["steps"][name].get("detail") or "step was not completed",
        }
        for name in QUALIFICATION_LIFECYCLE if lifecycle[name] != "passed"
    ]
    target = state["target"]
    return {
        "id": target["id"], "platform": target["platform"],
        "architecture": target["architecture"], "runtime": target["runtime"],
        "runtime_version": target["runtime_version"], "backend": target["backend"],
        "accelerator": target["accelerator"],
        "qualified_at": date.today().isoformat(), "suite_version": suite_version,
        "lifecycle": lifecycle, "known_failures": failures,
        "evidence": [evidence_path] if any(value == "passed" for value in lifecycle.values()) else [],
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_contains_marker(path: Path, marker: str) -> bool:
    try:
        return marker in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def execute_qualification_step(step: dict, log_path: Path, environment: dict,
                               ) -> tuple[int, str]:  # pragma: no cover
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    start_new_session = os.name != "nt"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            step["command"], stdout=log, stderr=subprocess.STDOUT, text=True,
            creationflags=creationflags, start_new_session=start_new_session,
            env={**os.environ, **environment},
        )
        try:
            interrupt = step["interrupt_when_log_contains"]
            if interrupt is not None:
                deadline = time.monotonic() + step["timeout_seconds"]
                while not log_contains_marker(log_path, interrupt):
                    exit_code = process.poll()
                    if exit_code is not None:
                        return exit_code, f"process exited before interruption with code {exit_code}"
                    if time.monotonic() >= deadline:
                        process.kill()
                        process.wait()
                        return -1, "interruption marker was not observed before timeout"
                    time.sleep(0.2)
                if os.name == "nt":
                    process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
                else:
                    os.killpg(process.pid, signal.SIGINT)
            exit_code = process.wait(timeout=step["timeout_seconds"])
            return exit_code, f"process exited with code {exit_code}"
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return -1, f"step exceeded {step['timeout_seconds']} seconds"


def run_qualification(recipe: dict, output_dir: Path) -> dict:  # pragma: no cover
    validate_qualification_recipe(recipe)
    gaps = execution_recipe_gaps(recipe)
    if gaps:
        raise ValueError("qualification recipe has unresolved placeholders: " + ", ".join(gaps))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "qualification-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("recipe_digest") != recipe_digest(recipe):
            raise ValueError("qualification recipe changed since this run was checkpointed")
    else:
        state = initial_run_state(recipe)
        _write_json(state_path, state)
    while (name := next_qualification_step(state)) is not None:
        record = state["steps"][name]
        record.update({"status": "running", "started_at": _timestamp(), "finished_at": None})
        log_path = output_dir / f"{QUALIFICATION_LIFECYCLE.index(name) + 1:02d}-{name}.log"
        record["log"] = log_path.name
        _write_json(state_path, state)
        exit_code, detail = execute_qualification_step(
            recipe["steps"][name], log_path, recipe["environment"],
        )
        record.update({
            "status": "passed" if exit_code in recipe["steps"][name]["expected_exit_codes"]
            else "failed",
            "exit_code": exit_code, "detail": detail, "finished_at": _timestamp(),
        })
        _write_json(state_path, state)
        if record["status"] == "failed":
            break
    evidence = qualification_entry_from_run(state, config.VERSION, state_path.name)
    _write_json(output_dir / "qualification-entry.json", evidence)
    return state


def main(argv=None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Preview or run a qualification recipe")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Run commands; without this flag only print the plan")
    args = parser.parse_args(argv)
    recipe = load_qualification_recipe(args.recipe)
    if not args.execute:
        print(json.dumps(qualification_preview(recipe, args.output), indent=2))
        return 0
    state = run_qualification(recipe, args.output)
    print(json.dumps(state, indent=2))
    return 0 if next_qualification_step(state) is None else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
