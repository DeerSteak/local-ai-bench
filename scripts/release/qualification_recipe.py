"""Generate concrete qualification recipes for supported platform targets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from scripts.release.qualification_coverage import (
    SMALLEST_EMBEDDING_MODEL, SMALLEST_IMAGE_MODEL, SMALLEST_LLM_MODEL,
    qualification_workloads,
)

SMOKE_MODEL = SMALLEST_LLM_MODEL
INTERRUPT_MARKER = '"kind":"model","stage":"llm","status":"running"'
TARGETS = {
    "macos-m5-pro-llamacpp-metal": ("macos", "arm64", "llamacpp", "metal"),
    "geforce-windows-llamacpp-cuda": ("windows", "x86_64", "llamacpp", "cuda"),
    "radeon-windows-llamacpp-vulkan": ("windows", "x86_64", "llamacpp", "vulkan"),
    "intel-arc-windows-llamacpp-vulkan": ("windows", "x86_64", "llamacpp", "vulkan"),
    "geforce-wsl2-llamacpp-cuda": ("wsl2", "x86_64", "llamacpp", "cuda"),
    "geforce-wsl2-vllm-cuda": ("wsl2", "x86_64", "vllm", "cuda"),
    "radeon-wsl2-llamacpp-rocm": ("wsl2", "x86_64", "llamacpp", "rocm"),
    "radeon-wsl2-vllm-rocm": ("wsl2", "x86_64", "vllm", "rocm"),
    "nvidia-linux-llamacpp-cuda": ("linux", "x86_64", "llamacpp", "cuda"),
    "nvidia-linux-vllm-cuda": ("linux", "x86_64", "vllm", "cuda"),
    "ryzen-ai-halo-llamacpp-rocm": ("linux", "x86_64", "llamacpp", "rocm"),
    "ryzen-ai-halo-vllm-rocm": ("linux", "x86_64", "vllm", "rocm"),
    "dgx-spark-llamacpp-cuda": ("linux", "aarch64", "llamacpp", "cuda"),
    "dgx-spark-vllm-cuda": ("linux", "aarch64", "vllm", "cuda"),
}
TARGET_ACCELERATORS = {
    "macos-m5-pro-llamacpp-metal": "M5 Pro",
    "geforce-windows-llamacpp-cuda": "NVIDIA GeForce",
    "radeon-windows-llamacpp-vulkan": "AMD Radeon",
    "intel-arc-windows-llamacpp-vulkan": "Intel Arc Pro B65",
    "geforce-wsl2-llamacpp-cuda": "NVIDIA GeForce",
    "geforce-wsl2-vllm-cuda": "NVIDIA GeForce",
    "radeon-wsl2-llamacpp-rocm": "Radeon RX 9060 XT",
    "radeon-wsl2-vllm-rocm": "Radeon RX 9060 XT",
    "nvidia-linux-llamacpp-cuda": "NVIDIA",
    "nvidia-linux-vllm-cuda": "NVIDIA",
    "ryzen-ai-halo-llamacpp-rocm": "Radeon 8060S",
    "ryzen-ai-halo-vllm-rocm": "Radeon 8060S",
    "dgx-spark-llamacpp-cuda": "NVIDIA GB10",
    "dgx-spark-vllm-cuda": "NVIDIA GB10",
}


def step(command, timeout=3600, exit_codes=(0,), interrupt=None):
    return {
        "command": [str(item) for item in command], "timeout_seconds": timeout,
        "expected_exit_codes": list(exit_codes), "interrupt_when_log_contains": interrupt,
    }


def build_recipe(*, target_id: str, root: Path, output: Path, baseline_version: str,
                 target_version: str, python_executable: str = sys.executable,
                 model: str = SMOKE_MODEL, accelerator_identity: str | None = None) -> dict:
    if target_id not in TARGETS:
        raise ValueError(f"unknown qualification target: {target_id}")
    from scripts.release.qualification_automation import validate_qualification_recipe
    if not baseline_version or not target_version:
        raise ValueError("baseline and target runtime versions are required")
    root, output = Path(root).resolve(), Path(output).resolve()
    platform_name, architecture, engine, backend = TARGETS[target_id]
    if accelerator_identity is None:
        from scripts.runtime.shared import Shared
        accelerator_identity = Shared.get_hostname()
    expected = TARGET_ACCELERATORS[target_id]
    if expected.lower() not in accelerator_identity.lower():
        raise ValueError(
            f"target {target_id} requires accelerator identity containing {expected!r}; "
            f"detected {accelerator_identity!r}"
        )
    py = os.path.abspath(python_executable)
    result = output / "smoke-result.json"
    interrupted = output / "interrupted-result.json"
    lifecycle = [py, "-m", "scripts.release.qualification_runtime"]
    install = lifecycle + ["install", "--root", root, "--engine", engine,
                           "--model", model, "--version", baseline_version,
                           "--inventory", output / "baseline-installation.json"]
    lifecycle_smoke = [py, "-m", "scripts.app.benchmark", "--quick", "--engine", engine]
    workload_tests = qualification_workloads(engine)
    qualification_run = [
        py, "-m", "scripts.release.qualification_coverage", "--engine", engine,
        "--model", model, "--result", result,
    ]
    if engine == "llamacpp":
        qualification_run += [
            "--comfyui", root / "qualification-comfyui-runtime" / "ComfyUI",
        ]
    if engine == "vllm":
        lifecycle_smoke.append("--ack-experimental-engine")
    environment = {
        "LOCAL_AI_BENCH_PROGRESS": "1", "LOCAL_AI_BENCH_QUALIFICATION": "1",
        "PYTHONUTF8": "1",
    }
    if engine == "vllm":
        environment["HF_HOME"] = str(root / "qualification-vllm-cache")
    recipe = {
        "target": {
            "id": target_id, "platform": platform_name, "architecture": architecture,
            "runtime": engine, "runtime_version": target_version, "backend": backend,
            "accelerator": accelerator_identity,
        },
        "coverage": {
            "workloads": workload_tests,
            "models": [model, SMALLEST_EMBEDDING_MODEL]
                      + ([SMALLEST_IMAGE_MODEL] if engine == "llamacpp" else []),
            "notes": "Smallest-model functional coverage for every compatible workload; not performance qualification.",
        },
        "environment": environment,
        "steps": {
            "install": step(install),
            "discovery": step(lifecycle + [
                "discover", "--root", root, "--engine", engine, "--version", baseline_version,
            ]),
            "first_valid_run": step(qualification_run, timeout=7200),
            "cancellation": step(
                lifecycle_smoke + ["--out", interrupted],
                exit_codes=(-2, 130, 149, -1073741510),
                interrupt=INTERRUPT_MARKER,
            ),
            "resume": step([py, "-m", "scripts.results.recovery_executor", interrupted]),
            "report_generation": step([
                py, "-m", "scripts.results.decision_report_cli", result,
                "--html", output / "baseline-report.html", "--reviewed-metadata",
                "--system-alias", target_id, "--hardware-alias", target_id,
            ]),
            "bundle_export": step(lifecycle + [
                "bundle", "--root", root, "--engine", engine, "--result", result,
                "--bundle", output / "baseline-result.lab.zip", "--alias", target_id,
                "--artifact-dir", output / "artifacts" / "baseline" / "images",
            ]),
            "upgrade": step(lifecycle + [
                "upgrade", "--root", root, "--engine", engine,
                "--model", model, "--version", target_version,
                "--smoke-output", output / "upgraded-smoke-result.json",
                "--inventory", output / "target-installation.json",
                "--report", output / "target-report.html",
                "--bundle", output / "target-result.lab.zip", "--alias", target_id,
                "--artifact-dir", output / "artifacts" / "target" / "images",
            ]),
            "rollback": step(lifecycle + [
                "rollback", "--root", root, "--engine", engine, "--version", baseline_version,
            ]),
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
