"""Install isolated runtimes and minimum workload models for qualification."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from scripts.setup import llamacpp_install
from scripts.runtime.comfyui_installation import find_image_asset
from scripts.setup.comfyui_assets import provision as provision_comfyui_assets
from scripts.setup.comfyui_install import ensure as ensure_comfyui
from scripts.setup.comfyui_runtime import prepare as prepare_comfyui_runtime
from scripts.setup.model_download import download_hf_files, provision_catalog_models
from scripts.setup.setup_discovery import (
    discover_nvidia, discover_rocm, discover_windows_gpu, rocm_version,
)
from scripts.setup.setup_config import vllm_setup_config, write_setup_config
from scripts.setup.vllm_install import (
    DGX_CU130_INDEX, find_vllm_binary, install_vllm, vllm_platform_support,
)
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


SMALLEST_EMBEDDING_MODEL = EMBED_MODELS[0]["tag"]
SMALLEST_IMAGE_MODEL = IMAGE_MODELS[0]["short"]
def qualification_vllm_index(method: str | None) -> str | None:
    return DGX_CU130_INDEX if method == "cu130_wheel" else None


def validate_vllm_runtime(runtime_dir: Path, run_fn=subprocess.run) -> tuple[bool, str]:
    python = Path(runtime_dir) / ("Scripts" if os.name == "nt" else "bin") / \
        ("python.exe" if os.name == "nt" else "python")
    try:
        result = run_fn(
            [str(python), "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0 and bool(output), output or f"exit code {result.returncode}"


def qualification_vllm_handoff(runtime_dir: Path, model_cache: Path,
                               *, system: str) -> dict:
    executable = find_vllm_binary(
        platform_name=system, venv_dir=runtime_dir, which_fn=lambda _name: None,
    )
    if executable is None:
        raise ValueError(f"vLLM executable was not found in {runtime_dir}")
    return vllm_setup_config(
        executable=executable, launcher=None, server_url=None,
        launcher_extra_args=[], hf_home=model_cache,
    )


def qualification_model(tag: str, catalog=None) -> dict:
    catalog = LLM_MODELS if catalog is None else catalog
    matches = [model for model in catalog if model["tag"] == tag]
    if len(matches) != 1:
        raise ValueError(f"qualification model must match one catalog entry: {tag}")
    return matches[0]


def qualification_install_plan(*, root: Path, engine: str, model_tag: str,
                               system: str, machine: str, nvidia: bool,
                               rocm: bool, runtime_version: str | None = None,
                               vllm_support=None) -> dict:
    root = Path(root).resolve()
    model = qualification_model(model_tag)
    if engine not in {"llamacpp", "vllm"}:
        raise ValueError(f"unsupported qualification engine: {engine}")
    if engine == "vllm" and "vllm_repo" not in model:
        raise ValueError(f"model has no vLLM artifact: {model_tag}")
    if engine == "vllm" and (vllm_support is None or not vllm_support.installable):
        detail = vllm_support.detail if vllm_support is not None else "support was not inspected"
        raise ValueError(f"vLLM cannot be installed on this target: {detail}")
    if not runtime_version:
        raise ValueError("qualification requires an exact runtime version")
    runtime_dir = root / ("llama.cpp" if engine == "llamacpp" else "vllm-env")
    cache_dir = root / "qualification-cache"
    models_dir = root / "models"
    return {
        "mode": "preview", "root": str(root), "engine": engine,
        "runtime_version": runtime_version,
        "model": {"tag": model_tag, "label": model["label"]},
        "coverage_models": {
            "llm": model_tag, "embeddings": SMALLEST_EMBEDDING_MODEL,
            "images": SMALLEST_IMAGE_MODEL if engine == "llamacpp" else None,
        },
        "platform": {"system": system, "machine": machine,
                     "nvidia": nvidia, "rocm": rocm},
        "runtime_dir": str(runtime_dir), "models_dir": str(models_dir),
        "cache_dir": str(cache_dir),
        "actions": [
            f"install isolated {engine} runtime", f"download {model_tag} for {engine}",
        ],
    }


def inspect_install_plan(root: Path, engine: str, model_tag: str,
                         runtime_version: str | None = None) -> tuple[dict, object, object]:
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
        runtime_version=runtime_version, vllm_support=support,
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
            info=log, warn=log, fail=log, ok=log, version=plan["runtime_version"],
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
        installed = install_vllm(
            support, log=log, venv_dir=runtime_dir, version=plan["runtime_version"],
            index_url=qualification_vllm_index(support.method),
        )
        model_cache = Path(os.environ.get("HF_HOME", root / "qualification-vllm-cache"))
    if not installed:
        return False
    if plan["engine"] == "vllm":
        valid, detail = validate_vllm_runtime(runtime_dir)
        if not valid:
            log(f"Installed vLLM could not be imported: {detail}")
            return False
        write_setup_config(
            root / "local_ai_bench_config.json", comfyui_dir=None,
            llamacpp_tools={}, gpu_devices=[
                {**gpu, "vendor": "nvidia", "backend": "cuda"} for gpu in nvidia.gpus
            ], vllm=qualification_vllm_handoff(
                runtime_dir, model_cache, system=plan["platform"]["system"],
            ),
        )
    provision_catalog_models(
        [model, EMBED_MODELS[0]], [plan["engine"]], models_dir=Path(plan["models_dir"]),
        vllm_cache=model_cache, load_token=lambda: os.environ.get("HF_TOKEN"),
        issues=issues, info=log, warn=log, fail=log, ok=log,
    )
    if plan["coverage_models"]["images"] is None:
        return not issues
    comfyui_root = root / "qualification-comfyui-runtime"
    comfyui_dir = comfyui_root / "ComfyUI"
    portable_python = comfyui_root / "python_embeded" / "python.exe"
    display = discover_windows_gpu()
    windows_gpu = "nvidia" if nvidia.available else display.vendor
    ensure_comfyui(
        comfyui_dir, comfyui_root, plan["platform"]["system"], windows_gpu,
        compute_capability=nvidia.compute_capability, issues=issues,
        info=log, warn=log, fail=log, ok=log,
    )
    image_models_dir = Path(plan["models_dir"]) / "comfyui"
    extra_paths = image_models_dir / "extra_model_paths.yaml"
    prepared = prepare_comfyui_runtime(
        comfyui_dir, image_models_dir, extra_paths, portable_python=portable_python,
        intel_xpu=display.vendor == "intel", rocm=rocm.available and not nvidia.available,
        issues=issues, info=log, warn=log, fail=log, ok=log,
    )
    if prepared:
        def asset(name, subdir):
            return find_image_asset(name, image_models_dir, subdir, comfyui_dir)

        def download(repo, filename, token=None, dest_dir: Path | None = None, save_as=None):
            if dest_dir is None:
                raise ValueError("qualification image download requires a destination")
            return download_hf_files(
                repo, filename, dest_dir, token=token, save_as=save_as, warn=log,
            )

        found = provision_comfyui_assets(
            [IMAGE_MODELS[0]], image_models_dir, find_asset=asset, download=download,
            load_token=lambda: os.environ.get("HF_TOKEN"), info=log, warn=log,
            fail=log, ok=log,
        )
        if IMAGE_MODELS[0]["checkpoint"] not in found:
            issues.append("smallest image checkpoint was not provisioned")
    return not issues


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Install an isolated qualification stack")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--engine", required=True, choices=("llamacpp", "vllm"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-version")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-isolated-root", action="store_true")
    args = parser.parse_args(argv)
    plan, nvidia, rocm = inspect_install_plan(
        args.root, args.engine, args.model, args.runtime_version,
    )
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    if not args.confirm_isolated_root:
        parser.error("--execute requires --confirm-isolated-root")
    return 0 if install_qualification_stack(plan, nvidia, rocm) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
