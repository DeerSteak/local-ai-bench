"""Run and grade the ordinary benchmark as platform qualification evidence."""

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from scripts.release.qualification_coverage import (
    qualification_arguments, qualification_workloads,
    workload_coverage_errors, workload_failure_details,
)
from scripts.release.qualification_targets import (
    TARGET_ENGINES, qualification_target, qualification_target_errors, target_engine,
)
from scripts.runtime.shared import Shared
from scripts.workloads.models import qualification_llm_model


def benchmark_wrapper(root: Path) -> list[str]:
    if sys.platform == "win32":
        return ["cmd", "/d", "/c", str(root / "run_bench.bat")]
    return [str(root / "run_bench.sh")]


def default_result_path(root: Path, target_id: str) -> Path:
    return root / "qualification-evidence" / target_id / f"results_qualification_{target_id}.json"


def qualification_failure_summary(result: dict, engine: str) -> str:
    stages = result.get("run", {}).get("stages", {})
    non_image = [workload for workload in qualification_workloads(engine) if workload != "img"]
    probe = deepcopy(result)
    probe.setdefault("run", {})["status"] = "complete"
    if engine == "llamacpp" and stages.get("img", {}).get("status") == "failed" \
            and not workload_coverage_errors(probe, non_image):
        return "llama.cpp workloads passed; ComfyUI image generation did not pass"
    failed = [stage for stage, state in stages.items() if state.get("status") == "failed"]
    return f"benchmark failed during {', '.join(failed)}" if failed else "benchmark exited before qualification completed"


def run_qualification(target_id: str, root: Path, result: Path) -> None:  # pragma: no cover
    target = qualification_target(target_id)
    identity_errors = qualification_target_errors(target, Shared.build_profile())
    if identity_errors:
        raise ValueError(f"target {target_id} " + "; ".join(identity_errors))
    result.parent.mkdir(parents=True, exist_ok=True)
    engine = target_engine(target_id)
    model = qualification_llm_model(engine)["tag"]
    command = [
        *benchmark_wrapper(root),
        *qualification_arguments(engine, model, result),
    ]
    completed = subprocess.run(command, cwd=root)
    if completed.returncode:
        if result.is_file():
            data = json.loads(result.read_text(encoding="utf-8"))
            raise RuntimeError(qualification_failure_summary(data, engine))
        raise RuntimeError(f"benchmark exited with code {completed.returncode}")
    data = json.loads(result.read_text(encoding="utf-8"))
    errors = workload_coverage_errors(data, qualification_workloads(engine))
    if errors:
        details = workload_failure_details(data, qualification_workloads(engine))
        raise ValueError("; ".join([*errors, *details]))


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run an ordinary benchmark for qualification")
    parser.add_argument("target", nargs="?", choices=tuple(TARGET_ENGINES))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--result", type=Path)
    parser.add_argument("--list-targets", action="store_true")
    args = parser.parse_args(argv)
    if args.list_targets:
        print("\n".join(TARGET_ENGINES))
        return 0
    if not args.target:
        parser.error("target is required")
    root = args.root.resolve()
    result = (args.result or default_result_path(root, args.target)).resolve()
    try:
        run_qualification(args.target, root, result)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"qualification failed: {exc}", file=sys.stderr)
        return 1
    print(f"qualification passed: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
