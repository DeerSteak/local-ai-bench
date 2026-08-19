"""Automatic target and version selection for unattended qualification."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

from scripts.release.qualification_automation import (
    next_qualification_step, qualification_preview, run_qualification,
)
from scripts.release.qualification_recipe import build_recipe, write_recipe
from scripts.runtime.shared import Shared


PINNED_VERSIONS = {
    "llamacpp": ("b10486", "b10488"),
    "vllm-cuda": ("0.27.0", "0.27.1"),
    "vllm-rocm": ("0.27.1+rocm723", "0.27.1+rocm723"),
    "vllm-cu130": (
        "0.26.1rc1.dev925+gf1178f3a0", "0.26.1rc1.dev925+gf1178f3a0",
    ),
}
AUTOMATION_REVISION = "v3"


def detected_targets(system: str, machine: str, hostname: str, *, wsl: bool) -> list[str]:
    identity = hostname.casefold()
    if system == "Darwin" and machine in {"arm64", "aarch64"} and "m5 pro" in identity:
        return ["macos-m5-pro-llamacpp-metal"]
    if system == "Windows":
        if "nvidia" in identity or "geforce" in identity:
            return ["geforce-windows-llamacpp-cuda"]
        if "intel" in identity and "arc" in identity:
            return ["intel-arc-windows-llamacpp-vulkan"]
        if "amd" in identity or "radeon" in identity:
            return ["radeon-windows-llamacpp-vulkan"]
    if system == "Linux" and wsl and ("nvidia" in identity or "geforce" in identity):
        return ["geforce-wsl2-llamacpp-cuda", "geforce-wsl2-vllm-cuda"]
    if system == "Linux" and machine in {"arm64", "aarch64"} and "gb10" in identity:
        return ["dgx-spark-llamacpp-cuda", "dgx-spark-vllm-cuda"]
    if system == "Linux" and any(value in identity for value in ("8060s", "ryzen ai max")):
        return ["ryzen-ai-halo-llamacpp-rocm", "ryzen-ai-halo-vllm-rocm"]
    if system == "Linux" and ("nvidia" in identity or "geforce" in identity):
        return ["nvidia-linux-llamacpp-cuda", "nvidia-linux-vllm-cuda"]
    raise ValueError(f"no automatic qualification target matches {system} {machine}: {hostname}")


def target_versions(target_id: str) -> tuple[str, str]:
    if "llamacpp" in target_id:
        return PINNED_VERSIONS["llamacpp"]
    if "rocm" in target_id:
        return PINNED_VERSIONS["vllm-rocm"]
    if "dgx-spark-vllm" in target_id:
        return PINNED_VERSIONS["vllm-cu130"]
    return PINNED_VERSIONS["vllm-cuda"]


def safe_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def automatic_recipes(root: Path, output_root: Path, *, system: str | None = None,
                      machine: str | None = None, hostname: str | None = None,
                      wsl: bool | None = None) -> list[tuple[Path, dict]]:
    system = system or platform.system()
    machine = machine or platform.machine()
    hostname = hostname or Shared.get_hostname()
    if wsl is None:
        wsl = Shared.detect_wsl(system, platform.release())
    recipes = []
    for target_id in detected_targets(system, machine, hostname, wsl=wsl):
        baseline, target = target_versions(target_id)
        output = Path(output_root) / f"{target_id}-{safe_version(target)}-{AUTOMATION_REVISION}"
        recipe = build_recipe(
            target_id=target_id, root=root, output=output,
            baseline_version=baseline, target_version=target,
            accelerator_identity=hostname,
        )
        recipes.append((output, recipe))
    return recipes


def execution_summary(target: dict, state: dict, output: Path) -> dict:
    failed_step = next_qualification_step(state)
    record = state["steps"].get(failed_step, {}) if failed_step else {}
    return {
        "target": target["id"], "status": "passed" if failed_step is None else "failed",
        "failed_step": failed_step, "detail": record.get("detail"),
        "log": str(Path(output) / record["log"]) if record.get("log") else None,
        "evidence_dir": str(output),
    }


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Automatically qualify this machine")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("qualification-evidence"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        recipes = automatic_recipes(args.root.resolve(), args.output.resolve())
        summaries = []
        failed = False
        for output, recipe in recipes:
            recipe_path = output / "qualification-recipe.json"
            write_recipe(recipe_path, recipe)
            if args.execute:
                state = run_qualification(recipe, output)
                summary = execution_summary(recipe["target"], state, output)
                summaries.append(summary)
                failed = failed or summary["status"] == "failed"
            else:
                summaries.append(qualification_preview(recipe, output))
        print(json.dumps(summaries, indent=2))
        return 1 if failed else 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
