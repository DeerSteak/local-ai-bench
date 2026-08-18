"""Generate concrete qualification recipes for supported platform targets."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from scripts.release.qualification_automation import validate_qualification_recipe


SMOKE_MODEL = "gemma3:1b-it-q4_K_M"
INTERRUPT_MARKER = '"kind":"model","stage":"llm","status":"running"'
TARGETS = {
    "macos-m5-pro-llamacpp-metal": ("macos", "arm64", "llamacpp", "metal"),
    "geforce-windows-llamacpp-cuda": ("windows", "x86_64", "llamacpp", "cuda"),
    "radeon-windows-llamacpp-vulkan": ("windows", "x86_64", "llamacpp", "vulkan"),
    "intel-arc-windows-llamacpp-vulkan": ("windows", "x86_64", "llamacpp", "vulkan"),
    "geforce-wsl2-llamacpp-cuda": ("wsl2", "x86_64", "llamacpp", "cuda"),
    "geforce-wsl2-vllm-cuda": ("wsl2", "x86_64", "vllm", "cuda"),
    "nvidia-linux-llamacpp-cuda": ("linux", "x86_64", "llamacpp", "cuda"),
    "nvidia-linux-vllm-cuda": ("linux", "x86_64", "vllm", "cuda"),
    "ryzen-ai-halo-llamacpp-rocm": ("linux", "x86_64", "llamacpp", "rocm"),
    "ryzen-ai-halo-vllm-rocm": ("linux", "x86_64", "vllm", "rocm"),
    "dgx-spark-llamacpp-cuda": ("linux", "aarch64", "llamacpp", "cuda"),
    "dgx-spark-vllm-cuda": ("linux", "aarch64", "vllm", "cuda"),
}


def step(command, timeout=3600, exit_codes=(0,), interrupt=None):
    return {
        "command": [str(item) for item in command], "timeout_seconds": timeout,
        "expected_exit_codes": list(exit_codes), "interrupt_when_log_contains": interrupt,
    }


def build_recipe(*, target_id: str, root: Path, output: Path, baseline_version: str,
                 target_version: str, python_executable: str = sys.executable,
                 model: str = SMOKE_MODEL) -> dict:
    if target_id not in TARGETS:
        raise ValueError(f"unknown qualification target: {target_id}")
    if not baseline_version or not target_version or baseline_version == target_version:
        raise ValueError("baseline and target runtime versions must be distinct")
    root, output = Path(root).resolve(), Path(output).resolve()
    platform_name, architecture, engine, backend = TARGETS[target_id]
    py = str(Path(python_executable).resolve())
    result = output / "smoke-result.json"
    interrupted = output / "interrupted-result.json"
    lifecycle = [py, "-m", "scripts.release.qualification_runtime"]
    install = lifecycle + ["install", "--root", root, "--engine", engine,
                           "--model", model, "--version", baseline_version]
    benchmark = [py, "-m", "scripts.app.benchmark", "--quick", "--engine", engine]
    if engine == "vllm":
        benchmark.append("--ack-experimental-engine")
    environment = {"LOCAL_AI_BENCH_PROGRESS": "1"}
    if engine == "vllm":
        environment["HF_HOME"] = str(root / "qualification-vllm-cache")
    recipe = {
        "target": {
            "id": target_id, "platform": platform_name, "architecture": architecture,
            "runtime": engine, "runtime_version": target_version, "backend": backend,
        },
        "coverage": {
            "workloads": ["llm"], "models": [model],
            "notes": "One-model lifecycle smoke; not full-catalog performance qualification.",
        },
        "environment": environment,
        "steps": {
            "install": step(install),
            "discovery": step(lifecycle + ["discover", "--root", root, "--engine", engine]),
            "first_valid_run": step(benchmark + ["--out", result]),
            "cancellation": step(
                benchmark + ["--out", interrupted], exit_codes=(-2, 130, -1073741510),
                interrupt=INTERRUPT_MARKER,
            ),
            "resume": step([py, "-m", "scripts.results.recovery_executor", interrupted]),
            "report_generation": step([
                py, "-m", "scripts.results.decision_report_cli", result,
                "--html", output / "smoke-report.html", "--reviewed-metadata",
                "--system-alias", target_id, "--hardware-alias", target_id,
            ]),
            "bundle_export": step([
                py, "-m", "scripts.results.result_bundle_cli", "export", result,
                output / "smoke-result.lab.zip", "--reviewed-metadata",
                "--system-alias", target_id, "--hardware-alias", target_id,
            ]),
            "upgrade": step(lifecycle + [
                "upgrade", "--root", root, "--engine", engine,
                "--model", model, "--version", target_version,
            ]),
            "rollback": step(lifecycle + ["rollback", "--root", root, "--engine", engine]),
            "uninstall": step(lifecycle + ["uninstall", "--root", root, "--engine", engine]),
        },
    }
    validate_qualification_recipe(recipe)
    return recipe


def write_recipe(path: Path, recipe: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate a concrete platform qualification recipe")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--target", choices=tuple(TARGETS))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-version")
    parser.add_argument("--target-version")
    parser.add_argument("--model", default=SMOKE_MODEL)
    args = parser.parse_args(argv)
    if args.list_targets:
        print("\n".join(TARGETS))
        return 0
    if not all((args.target, args.output, args.baseline_version, args.target_version)):
        parser.error("generation requires --target, --output, --baseline-version, and --target-version")
    recipe = build_recipe(
        target_id=args.target, root=args.root, output=args.output,
        baseline_version=args.baseline_version, target_version=args.target_version,
        model=args.model,
    )
    recipe_path = args.output / "qualification-recipe.json"
    write_recipe(recipe_path, recipe)
    print(recipe_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
