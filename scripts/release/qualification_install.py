"""Install an isolated engine runtime and smoke model for qualification."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from scripts.setup import llamacpp_install
from scripts.setup.model_download import provision_catalog_models
from scripts.setup.setup_discovery import discover_nvidia, discover_rocm, rocm_version
from scripts.setup.vllm_install import (
    install_vllm, vllm_platform_support,
)
from scripts.workloads.models import LLM_MODELS


def qualification_model(tag: str, catalog=None) -> dict:
    catalog = LLM_MODELS if catalog is None else catalog
    matches = [model for model in catalog if model["tag"] == tag]
    if len(matches) != 1:
        raise ValueError(f"qualification model must match one catalog entry: {tag}")
    return matches[0]


def qualification_install_plan(*, root: Path, engine: str, model_tag: str,
                               system: str, machine: str, nvidia: bool,
                               rocm: bool, vllm_support=None) -> dict:
    root = Path(root).resolve()
    model = qualification_model(model_tag)
    if engine not in {"llamacpp", "vllm"}:
        raise ValueError(f"unsupported qualification engine: {engine}")
    if engine == "vllm" and "vllm_repo" not in model:
        raise ValueError(f"model has no vLLM artifact: {model_tag}")
    if engine == "vllm" and (vllm_support is None or not vllm_support.installable):
        detail = vllm_support.detail if vllm_support is not None else "support was not inspected"
        raise ValueError(f"vLLM cannot be installed on this target: {detail}")
    runtime_dir = root / ("llama.cpp" if engine == "llamacpp" else "vllm-env")
    cache_dir = root / "qualification-cache"
    models_dir = root / "models"
    return {
        "mode": "preview", "root": str(root), "engine": engine,
        "model": {"tag": model_tag, "label": model["label"]},
        "platform": {"system": system, "machine": machine,
                     "nvidia": nvidia, "rocm": rocm},
        "runtime_dir": str(runtime_dir), "models_dir": str(models_dir),
        "cache_dir": str(cache_dir),
        "actions": [
            f"install isolated {engine} runtime", f"download {model_tag} for {engine}",
        ],
    }


def inspect_install_plan(root: Path, engine: str, model_tag: str) -> tuple[dict, object, object]:
    nvidia = discover_nvidia()
    rocm = discover_rocm()
    rocm_available = rocm.available and not nvidia.available
    support = None
    if engine == "vllm":
        support = vllm_platform_support(
            os_name=platform.system(), machine=platform.machine(),
            python_version=sys.version_info[:2], nvidia_ok=nvidia.available,
            rocm_ok=rocm_available, gpu_names=[gpu["name"] for gpu in nvidia.gpus],
            compute_cap=nvidia.compute_capability,
            rocm_version=rocm_version() if rocm_available else None,
            rocm_gfx_targets=rocm.gfx_targets if rocm_available else [],
        )
    plan = qualification_install_plan(
        root=root, engine=engine, model_tag=model_tag, system=platform.system(),
        machine=platform.machine(), nvidia=nvidia.available, rocm=rocm_available,
        vllm_support=support,
    )
    return plan, nvidia, rocm


def install_qualification_stack(plan: dict, nvidia, rocm) -> bool:  # pragma: no cover
    root = Path(plan["root"])
    runtime_dir = Path(plan["runtime_dir"])
    cache_dir = Path(plan["cache_dir"])
    model = qualification_model(plan["model"]["tag"])
    issues = []
    log = lambda message: print(message, flush=True)
    if plan["engine"] == "llamacpp":
        installed = llamacpp_install.install(
            runtime_dir, cache_dir, plan["platform"]["system"],
            nvidia=plan["platform"]["nvidia"], rocm=plan["platform"]["rocm"],
            compute_capability=nvidia.compute_capability,
            max_cuda_version=nvidia.max_cuda_version,
            info=log, warn=log, fail=log, ok=log,
        )
        model_cache = root / "qualification-vllm-cache"
    else:
        rocm_available = rocm.available and not nvidia.available
        support = vllm_platform_support(
            os_name=plan["platform"]["system"], machine=plan["platform"]["machine"],
            python_version=sys.version_info[:2], nvidia_ok=plan["platform"]["nvidia"],
            rocm_ok=rocm_available, gpu_names=[gpu["name"] for gpu in nvidia.gpus],
            compute_cap=nvidia.compute_capability,
            rocm_version=rocm_version() if rocm_available else None,
            rocm_gfx_targets=rocm.gfx_targets if rocm_available else [],
        )
        installed = install_vllm(support, log=log, venv_dir=runtime_dir)
        model_cache = Path(os.environ.get("HF_HOME", root / "qualification-vllm-cache"))
    if not installed:
        return False
    provision_catalog_models(
        [model], [plan["engine"]], models_dir=Path(plan["models_dir"]),
        vllm_cache=model_cache, load_token=lambda: os.environ.get("HF_TOKEN"),
        issues=issues, info=log, warn=log, fail=log, ok=log,
    )
    return not issues


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Install an isolated qualification stack")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--engine", required=True, choices=("llamacpp", "vllm"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-isolated-root", action="store_true")
    args = parser.parse_args(argv)
    plan, nvidia, rocm = inspect_install_plan(args.root, args.engine, args.model)
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    if not args.confirm_isolated_root:
        parser.error("--execute requires --confirm-isolated-root")
    return 0 if install_qualification_stack(plan, nvidia, rocm) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
