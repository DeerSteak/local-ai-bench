#!/usr/bin/env python3
"""Pre-flight setup/install assistant — see docs/setup.md."""

import argparse
import atexit
import sys
import os
import platform
import re
import shlex
import signal
import subprocess
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.runtime import config
from scripts.runtime import hardware
from scripts.runtime.comfyui_installation import (
    checkpoint_names_from_object_info,
    find_comfyui_installation,
    find_image_asset,
    managed_checkpoints_visible,
    normalize_comfyui_dir,
    resolve_comfyui_setup_choice,
)
from scripts.runtime.llamacpp_tools import find_nvcc
from scripts.setup.cuda_install import cuda_toolkit_plan, run_cuda_toolkit_install
from scripts.setup.model_inventory import (
    delete_non_catalog_model_dirs, delete_non_catalog_vllm_repos,
    engine_download_size, find_non_catalog_vllm_repos,
    hf_cache_repo_id,
    engine_model_complete, engine_model_dir, find_non_catalog_model_dirs,
    models_missing_engine_support,
)
from scripts.setup.model_download import download_hf_files, download_hf_snapshot
from scripts.setup import llamacpp_install
from scripts.setup.hf_credentials import HfTokenProvider
from scripts.setup.comfyui_assets import provision as provision_comfyui_assets
from scripts.setup.comfyui_runtime import prepare as prepare_comfyui_runtime
from scripts.setup.comfyui_install import ensure as ensure_comfyui
from scripts.workloads.models import LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, IMAGE_MODELS, EMBED_MODELS
from scripts.setup.setup_selection import additional_disk_space_needed, select_models
from scripts.setup.setup_config import configured_comfyui_dir, load_setup_config, write_setup_config
from scripts.setup.setup_progress import finish_setup_progress, start_setup_progress
from scripts.setup.setup_discovery import (
    discover_linux_intel_gpu, discover_metal, discover_nvidia, discover_rocm,
    discover_system, discover_windows_gpu, rocm_version,
)
from scripts.setup.setup_console import (
    BOLD, CYAN, GREEN, RESET, YELLOW, confirm, fail, info, link, ok, section, warn,
)
from scripts.setup.engine_selection import (
    LLAMACPP, VLLM, build_engine_entries, engines_needing_install,
    needs_python_headers, select_engines, selected_engine_names,
)
from scripts.setup.vllm_install import (
    find_vllm_binary, find_vllm_launcher, find_vllm_server, hf_cache_model_complete,
    build_tools_command, install_vllm, missing_build_tools, missing_python_headers,
    python_dev_package_command, python_include_dir, python_version_from_include_dir,
    read_launcher_extra_args, redact_launcher_extra_args, vllm_cache_home, vllm_platform_support,
    PINNED_PYTHON, python_bootstrap_plan, run_python_bootstrap,
)
from scripts.app.interface_mode import select_interface_mode


def main() -> None:  # pragma: no cover - real interactive installer
    # Repo root, one level up — sourced from config.py rather than redefined here.
    SCRIPT_DIR   = config.SCRIPT_DIR
    LLAMACPP_DIR = config.LLAMACPP_DIR

    _arg_parser = argparse.ArgumentParser(description="local-ai-bench setup")
    _arg_parser.add_argument("--comfyui", help="Path to an existing ComfyUI or portable root")
    _arg_parser.add_argument("--interface", choices=("auto", "gui", "terminal"), default="auto")
    args = _arg_parser.parse_args()
    _saved_setup = load_setup_config(config.SETUP_CONFIG_PATH)
    if args.comfyui and not normalize_comfyui_dir(Path(args.comfyui)):
        _arg_parser.error("--comfyui must contain main.py or a ComfyUI/main.py portable layout")
    _detected_comfyui = find_comfyui_installation(
        explicit=args.comfyui,
        saved_path=configured_comfyui_dir(_saved_setup),
        managed_dir=config.COMFYUI_DIR,
    )
    COMFYUI_DIR: Path = _detected_comfyui or config.COMFYUI_DIR

    INSTALL_STARTED = False  # flipped True once the unattended install phase begins
    GUI_CANCEL_EXIT = 10

    def cancel_setup(*_args):
        """SIGINT handler so Ctrl+C works mid-subprocess/download, not just at input()."""
        if INSTALL_STARTED:
            print("\n\n  Setup cancelled — some components may already be partially installed.\n")
        else:
            print("\n\n  Setup cancelled — nothing was installed.\n")
        sys.exit(130)

    signal.signal(signal.SIGINT, cancel_setup)


    def hf_download(repo, filenames, token=None, dest_dir=None, save_as=None):
        return download_hf_files(
            repo, filenames, dest_dir or CHECKPOINTS, token=token, save_as=save_as, warn=warn,
        )


    def hf_snapshot_download(repo, cache_home, token=None):
        return download_hf_snapshot(repo, cache_home, token=token, warn=warn)


    issues = []

    # Local aliases for hardware.py's sizes, shared with select_models()'s memory-fit check.
    CHECKPOINT_SIZES_GB = hardware.CHECKPOINT_SIZES_GB
    ENCODER_SIZES_GB = hardware.ENCODER_SIZES_GB
    GATED_IMAGE_SHORTS = {"sd35-large", "flux-dev", "flux2-dev"}

    # ── 1. Python version ──────────────────────────────────────────────────────────

    section("Python")
    major, minor = sys.version_info[:2]
    print(f"  Version: {sys.version.split()[0]}")
    if (major, minor) >= (3, 11):
        ok("Python 3.11+ detected")
    else:
        fail(f"Python 3.11+ required (found {major}.{minor})")
        issues.append("Upgrade Python to 3.11+")

    # ── 2. OS & hardware identity ──────────────────────────────────────────────────

    section("System")
    system = discover_system()
    os_name = system.os_name
    total_ram_gb = system.total_ram_gb
    print(f"  OS:       {system.os_name} {system.release}")
    print(f"  Machine:  {system.machine}")
    print(f"  Node:     {system.node}")
    if system.chip is not None:
        print(f"  Chip:     {system.chip}")
    if total_ram_gb is not None:
        print(f"  RAM:      {total_ram_gb:.0f} GB")

    # ── 3. GPU / acceleration backend ─────────────────────────────────────────────

    section("GPU / Acceleration Backend")

    nvidia = discover_nvidia()
    nvidia_gpus = nvidia.gpus
    nvidia_vram_gb = nvidia.total_vram_gb
    for device in nvidia_gpus:
        print(f"  GPU:     {device['name']}")
        vram = device["vram_gb"]
        print(f"  VRAM:    {f'{vram:.1f} GB' if vram is not None else 'unified with system RAM'}")
        print(f"  Driver:  {device['driver']}")

    rocm = discover_rocm() if not nvidia.available else None
    rocm_gfx = rocm.gfx_targets if rocm else []
    rocm_gpu_kind = rocm.kind if rocm else None
    rocm_vram_gb = rocm.total_vram_gb if rocm else None
    rocm_gpus = rocm.gpus if rocm else []
    for name in rocm.names[:3] if rocm else []:
        print(f"  ROCm GPU: {name}")

    # See docs/setup.md's Intel Arc platform notes.
    INTEL_GPU_RUNTIME_PACKAGES = ("intel-opencl-icd", "intel-level-zero-gpu", "level-zero")

    def check_linux_intel_gpu_runtime():
        """Detection-only check for Intel's GPU compute runtime via dpkg — see
        docs/setup.md's Intel Arc platform notes."""
        if platform.system() != "Linux" or not shutil.which("dpkg"):
            return False
        for pkg in INTEL_GPU_RUNTIME_PACKAGES:
            result = subprocess.run(["dpkg", "-s", pkg],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                return False
        return True

    nvidia_ok = nvidia.available
    nvidia_compute_cap = nvidia.compute_capability
    nvidia_max_cuda_version = nvidia.max_cuda_version
    rocm_ok             = False
    metal_ok            = False
    amd_windows         = False
    intel_windows       = False
    intel_linux         = False
    intel_linux_runtime = False

    if not nvidia_ok:
        rocm_ok = bool(rocm and rocm.available)
    metal_ok, metal_details = discover_metal() if not nvidia_ok and not rocm_ok else (False, [])
    for detail in metal_details:
        print(f"  {detail}")
    if not nvidia_ok and os_name == "Windows":
        windows_gpu = discover_windows_gpu()
        windows_gpu_kind = windows_gpu.kind
        amd_windows = windows_gpu.vendor == "amd"
        intel_windows = windows_gpu.vendor == "intel"
        if windows_gpu.name:
            print(f"  GPU:     {windows_gpu.name}")
    else:
        windows_gpu_kind = None
    if not nvidia_ok and not rocm_ok and os_name == "Linux":
        linux_gpu = discover_linux_intel_gpu()
        linux_intel_gpu_kind = linux_gpu.kind
        intel_linux = linux_gpu.vendor == "intel"
        if linux_gpu.name:
            print(f"  GPU:     {linux_gpu.name}")
        if intel_linux:
            intel_linux_runtime = check_linux_intel_gpu_runtime()
    else:
        linux_intel_gpu_kind = None

    if nvidia_ok:
        ok("CUDA / Nvidia GPU detected")
    elif rocm_ok:
        ok("ROCm / AMD GPU detected")
    elif amd_windows:
        ok("AMD/Radeon GPU detected on Windows")
    elif intel_windows:
        ok("Intel Arc GPU detected on Windows")
        info("Intel Arc support is experimental — this project's maintainers don't have "
             "Arc hardware to test against, so treat this as unverified")
        warn("LLM tests need llama.cpp's SYCL backend for Intel Arc acceleration, which "
             "this script doesn't build; they'll run on CPU unless you build it yourself "
             "with -DGGML_SYCL=ON")
    elif intel_linux:
        ok("Intel Arc GPU detected on Linux")
        info("Intel Arc support is experimental — this project's maintainers don't have "
             "Arc hardware to test against, so everything below (runtime check, XPU "
             "PyTorch install) is unverified. Please report back "
             "if you try it: https://github.com/DeerSteak/local-ai-bench/issues")
        warn("LLM tests need llama.cpp's SYCL backend for Intel Arc acceleration, which "
             "this script doesn't build; they'll run on CPU unless you build it yourself "
             "with -DGGML_SYCL=ON")
        if intel_linux_runtime:
            ok("Intel GPU compute runtime (Level Zero/OpenCL) detected — ready for XPU-accelerated PyTorch")
        else:
            warn("Intel GPU compute runtime not installed — image generation will run on "
                 "CPU until it is. This script won't add a third-party APT repo for you; "
                 "install it yourself:")
            warn("  https://dgpu-docs.intel.com/driver/installation.html")
            warn(f"  (adds Intel's graphics APT repo, then: {' '.join(INTEL_GPU_RUNTIME_PACKAGES)})")
    elif metal_ok:
        ok("Apple Metal detected")
    else:
        warn("No GPU acceleration detected — LLM and image tests may run slowly")

    # ── 3a. Memory ceiling ─────────────────────────────────────────────────────────
    # Defaults models that clearly won't fit to unchecked in the picker below — informational, not a hard block.

    section("Memory")

    if nvidia_ok:
        gpu_vendor = "nvidia"
        gpu_vram_gb = nvidia_vram_gb if nvidia_vram_gb > 0 else None
    elif rocm_ok:
        gpu_vendor = "amd" if rocm_gpu_kind == "discrete" else "integrated"
        gpu_vram_gb = rocm_vram_gb
    elif amd_windows:
        gpu_vendor = "amd" if windows_gpu_kind == "discrete" else "integrated"
        gpu_vram_gb = None  # no driver-agnostic VRAM query implemented on Windows
    elif intel_windows:
        gpu_vendor = "intel" if windows_gpu_kind == "discrete" else "integrated"
        gpu_vram_gb = None
    elif intel_linux:
        gpu_vendor = "intel" if linux_intel_gpu_kind == "discrete" else "integrated"
        gpu_vram_gb = None
    else:
        # Apple Silicon (metal_ok) and "no GPU detected" both land here — unified
        # memory and CPU-only both mean total system RAM is the only pool.
        gpu_vendor = "integrated" if metal_ok else "none"
        gpu_vram_gb = None

    memory_ceiling_gb, memory_ceiling_note = hardware.compute_memory_ceiling_gb(
        os_name=os_name, total_ram_gb=total_ram_gb,
        gpu_vendor=gpu_vendor, vram_gb=gpu_vram_gb,
        device_vram_gb=(
            [device["vram_gb"] for device in nvidia_gpus if device["vram_gb"] is not None]
            if nvidia_ok else
            [device["vram_gb"] for device in rocm_gpus] if rocm_ok else None
        ) or None,
    )
    if memory_ceiling_gb is not None:
        ok(f"Model memory ceiling: {memory_ceiling_note}")
    else:
        warn(memory_ceiling_note)

    # WSL2 caps the VM near half the host's RAM, and the host total isn't visible from in here.
    if hardware.detect_wsl(os_name, platform.release()):
        _reported = f"{total_ram_gb:.0f} GB" if total_ram_gb else "an unknown amount"
        warn(f"Running under WSL2, which reports {_reported} of RAM — if the Windows "
             "host has more, models that would fit are being filtered out silently")
        info("Raise it with memory=<N>GB under [wsl2] in %UserProfile%\\.wslconfig, "
             "then run 'wsl --shutdown' — see docs/setup.md")

    def find_llamacpp_binary():
        return llamacpp_install.find_tool("llama-server", LLAMACPP_DIR, os_name)

    def find_llamacpp_bench_binary():
        return llamacpp_install.find_tool("llama-bench", LLAMACPP_DIR, os_name)

    def find_llamacpp_batched_bench_binary():
        return llamacpp_install.find_tool("llama-batched-bench", LLAMACPP_DIR, os_name)

    def install_llamacpp():
        return llamacpp_install.install(
            LLAMACPP_DIR, SCRIPT_DIR, os_name, nvidia=nvidia_ok, rocm=rocm_ok,
            compute_capability=nvidia_compute_cap,
            max_cuda_version=nvidia_max_cuda_version,
            info=info, warn=warn, fail=fail, ok=ok,
        )

    # ── 4a. llama.cpp detection (read-only) ────────────────────────────────────────

    section("llama.cpp")

    LLAMACPP_BIN = find_llamacpp_binary()
    llamacpp_found = LLAMACPP_BIN is not None
    managed_mac_runtime = os_name == "Darwin" and LLAMACPP_DIR.is_dir() and any(
        path.is_file() for path in LLAMACPP_DIR.rglob("llama-server")
    )
    needs_llamacpp_install = not llamacpp_found or (os_name == "Darwin" and not managed_mac_runtime)
    if llamacpp_found:
        ok(f"llama-server found: {LLAMACPP_BIN}")
    else:
        warn("llama-server not found — will need to be installed")

    LLAMACPP_BENCH_BIN = find_llamacpp_bench_binary()
    llamacpp_bench_found = LLAMACPP_BENCH_BIN is not None
    if llamacpp_bench_found:
        ok(f"llama-bench found: {LLAMACPP_BENCH_BIN}")
    elif needs_llamacpp_install:
        info("llama-bench not found — will be installed alongside llama-server")
    else:
        warn("llama-bench not found (llama-server is installed, but without llama-bench) — "
             "rerun setup after a fresh llama.cpp install, or build it yourself, to use "
             "the llamabench test")

    LLAMACPP_BATCHED_BENCH_BIN = find_llamacpp_batched_bench_binary()
    llamacpp_batched_bench_found = LLAMACPP_BATCHED_BENCH_BIN is not None
    if llamacpp_batched_bench_found:
        ok(f"llama-batched-bench found: {LLAMACPP_BATCHED_BENCH_BIN}")
    elif needs_llamacpp_install:
        info("llama-batched-bench not found — will be installed alongside llama-server")
    else:
        warn("llama-batched-bench not found (llama-server is installed, but without llama-batched-bench) — "
             "rerun setup after a fresh llama.cpp install, or build it yourself, to use "
             "the llamabenchconc test")

    # ── 4b. vLLM detection (read-only) ─────────────────────────────────────────────

    section("vLLM")

    VLLM_BIN = find_vllm_binary(platform_name=os_name)
    # A reachable server counts as present even with no host-side binary — AMD's Strix Halo
    # image ships one preconfigured, and a container/remote server looks the same from here.
    VLLM_LAUNCHER = find_vllm_launcher()
    VLLM_SERVER_URL = find_vllm_server()
    VLLM_LAUNCHER_ARGS = (
        redact_launcher_extra_args(read_launcher_extra_args()) if VLLM_LAUNCHER else []
    )
    VLLM_CACHE_HOME = vllm_cache_home(VLLM_LAUNCHER)
    # Triton JIT-compiles a CUDA helper on import, so vLLM cannot start without these.
    _vllm_python = config.VLLM_VENV / "bin" / "python"
    _vllm_include_dir = python_include_dir(
        str(_vllm_python) if _vllm_python.is_file() else sys.executable)
    missing_python_header = missing_python_headers(_vllm_include_dir)
    missing_header_version = python_version_from_include_dir(_vllm_include_dir) or sys.version_info[:2]
    header_command = next(
        (command for manager in ("apt-get", "dnf", "zypper")
         for command in [python_dev_package_command(manager, missing_header_version)] if command),
        None,
    ) if missing_python_header else None
    header_package = header_command[-1] if header_command else None
    vllm_found = VLLM_BIN is not None or VLLM_LAUNCHER is not None or VLLM_SERVER_URL is not None
    _cuda_plan = cuda_toolkit_plan(
        is_wsl=hardware.detect_wsl(os_name, platform.release()),
        nvidia_ok=nvidia_ok, nvcc_found=find_nvcc() is not None,
    )
    if _cuda_plan:
        section("CUDA Toolkit")
        warn("An NVIDIA GPU is available under WSL2 but the CUDA toolkit (nvcc) is missing — "
             "llama.cpp would build CPU-only")
        info("Setup can install NVIDIA's WSL-Ubuntu CUDA toolkit. It contains no Linux GPU "
             "driver, so the Windows driver's passthrough is left intact. Needs sudo.")
        for _command in _cuda_plan:
            print(f"      {' '.join(_command)}")
        if input("\n  Install it? [y/N] ").strip().lower().startswith("y"):
            run_cuda_toolkit_install(_cuda_plan)
        else:
            info("Skipped — llama.cpp will build CPU-only")

    vllm_note = (f"server already running at {VLLM_SERVER_URL}" if VLLM_SERVER_URL
                 else f"platform launcher {VLLM_LAUNCHER}" if VLLM_LAUNCHER
                 else f"found at {VLLM_BIN}" if VLLM_BIN else None)
    vllm_support = vllm_platform_support(
        os_name=os_name, machine=platform.machine(), python_version=sys.version_info[:2],
        nvidia_ok=nvidia_ok, rocm_ok=rocm_ok, intel_gpu=intel_linux or intel_windows,
        gpu_names=[device["name"] for device in nvidia_gpus],
        compute_cap=nvidia_compute_cap, rocm_version=rocm_version() if rocm_ok else None,
        rocm_gfx_targets=rocm_gfx,
    )
    if not vllm_found and vllm_support.needs_python_bootstrap:
        bootstrap_plan = python_bootstrap_plan(python_version=sys.version_info[:2])
        if bootstrap_plan:
            warn(f"vLLM needs Python 3.10–3.13 and this system only has "
                 f"{sys.version_info.major}.{sys.version_info.minor}")
            info("Setup can install a private CPython "
                 f"{PINNED_PYTHON[0]}.{PINNED_PYTHON[1]} for vLLM to build its venv from. "
                 "This downloads uv from astral.sh and does not change your system Python.")
            for command in bootstrap_plan:
                print(f"      {shlex.join(command)}")
            if input("\n  Install it? [y/N] ").strip().lower().startswith("y"):
                if run_python_bootstrap(bootstrap_plan):
                    vllm_support = vllm_platform_support(
                        os_name=os_name, machine=platform.machine(),
                        python_version=sys.version_info[:2],
                        nvidia_ok=nvidia_ok, rocm_ok=rocm_ok,
                        intel_gpu=intel_linux or intel_windows,
                        gpu_names=[device["name"] for device in nvidia_gpus],
                        compute_cap=nvidia_compute_cap,
                        rocm_version=rocm_version() if rocm_ok else None,
                        rocm_gfx_targets=rocm_gfx,
                    )
            else:
                info("Skipped — continuing without vLLM")

    if VLLM_SERVER_URL:
        ok(f"vLLM server already running at {VLLM_SERVER_URL} — nothing to install")
    elif VLLM_BIN or VLLM_LAUNCHER:
        ok(f"vllm found: {VLLM_LAUNCHER or VLLM_BIN}")
        info(f"vLLM model cache: {VLLM_CACHE_HOME}")
    elif vllm_support.status == "unsupported":
        info(f"vLLM not available here — {vllm_support.reason}")
    else:
        warn(f"vllm not found — {vllm_support.reason}")
        if vllm_support.status == "experimental":
            info("This vLLM path is experimental and unverified by this project's maintainers")

    if missing_python_header and (vllm_found or vllm_support.installable):
        warn(f"Python development headers are missing ({missing_python_header}) — "
             "vLLM cannot start without them")

    if VLLM_LAUNCHER_ARGS:
        warn(f"{VLLM_LAUNCHER} injects extra vLLM arguments on every launch: "
             f"{' '.join(VLLM_LAUNCHER_ARGS)}")
        info("These are recorded in the setup configuration so a run reports the flags "
             "that actually ran")

    # ── 4c. Engine selection ───────────────────────────────────────────────────────
    # Whatever models are picked later are downloaded for every engine selected here.

    engine_entries = build_engine_entries(
        vllm_support=vllm_support, vllm_found=vllm_found, llamacpp_found=llamacpp_found,
        vllm_note=vllm_note,
    )

    # ── 5. Welcome / prerequisites approval ────────────────────────────────────────

    try:
        import tkinter  # noqa: F401
        _tkinter_available = True
    except ImportError:
        _tkinter_available = False
    try:
        _interface = select_interface_mode(
            args.interface, platform_name=os_name, env=dict(os.environ),
            stdin_is_tty=sys.stdin.isatty(), gui_available=_tkinter_available,
        )
    except ValueError as exc:
        _arg_parser.error(str(exc))

    if _interface != "gui":
        select_engines(engine_entries)

    section("Setup Plan")
    print(f"  {BOLD}local-ai-bench{RESET} needs a few things before it can run benchmarks.\n")
    print("  This will:")
    print("    • Install Python dependencies from requirements.txt")
    if needs_llamacpp_install and LLAMACPP in selected_engine_names(engine_entries):
        build_note = " (source build — can take several minutes)" if os_name == "Linux" else ""
        print(f"    • Install llama.cpp{build_note}, including llama-bench and llama-batched-bench")
    if _detected_comfyui:
        print(f"    • Reuse ComfyUI at {COMFYUI_DIR}")
    if _interface != "gui" and VLLM in engines_needing_install(engine_entries):
        print("    • Install vLLM (several GB, into its own vllm-env/ environment)")
    if _interface != "gui" and needs_python_headers(engine_entries, missing_python_header):
        print("    • Install the Python development headers vLLM needs (requires sudo)")
    print()
    print("  You'll then pick which models to install — everything after that")
    print("  runs on its own, with no further prompts.")
    print()

    _gui_plan = None
    if _interface == "gui":
        from scripts.setup.setup_gui import run_setup_wizard_process
        _cleanup_candidates = find_non_catalog_model_dirs(config.MODELS_DIR / "llamacpp")
        _gui_plan = run_setup_wizard_process(
            memory_ceiling_gb=memory_ceiling_gb,
            detected_comfyui=_detected_comfyui,
            cleanup_names=[path.name for path in _cleanup_candidates],
            vllm_cleanup=find_non_catalog_vllm_repos(VLLM_CACHE_HOME),
            existing_hf_token=bool(os.environ.get("HF_TOKEN", "").strip() or (
                (SCRIPT_DIR / "hf.txt").is_file() and (SCRIPT_DIR / "hf.txt").read_text().strip()
            )),
            engine_entries=engine_entries,
            sudo_package=header_package,
        )
        if _gui_plan is None:
            print("\n  Setup cancelled — nothing was installed.\n")
            sys.exit(GUI_CANCEL_EXIT)
        _chosen = set(_gui_plan.get("engines", [LLAMACPP]))
        for entry in engine_entries:
            entry["checked"] = entry["enabled"] and entry["name"] in _chosen
    elif not confirm("Continue?", default=True):
        print(f"\n  Setup cancelled — nothing was installed.\n")
        sys.exit(0)

    selected_engines = selected_engine_names(engine_entries)
    pending_engines = engines_needing_install(engine_entries)
    _engine_labels = {entry["name"]: entry["label"] for entry in engine_entries}
    ok(f"Engines selected: {', '.join(_engine_labels[name] for name in selected_engines)}")
    for _name in _engine_labels:
        if _name not in selected_engines:
            info(f"{_engine_labels[_name]} not selected — its models and tools are skipped")

    # ── 6. Model selection ──────────────────────────────────────────────────────────

    section("Model Selection")

    if _gui_plan is None:
        selected_llm, selected_images, selected_embed, cleanup_names, vllm_cleanup_names = select_models(
            memory_ceiling_gb, engines=selected_engines,
            vllm_cache_home=VLLM_CACHE_HOME, cancel=cancel_setup,
        )
    else:
        _llm_tags = set(_gui_plan["llm_tags"])
        _image_shorts = set(_gui_plan["image_shorts"])
        _embed_tags = set(_gui_plan["embedding_tags"])
        selected_llm = [
            model for tier in (LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE)
            for model in tier if model["tag"] in _llm_tags
        ]
        selected_images = [model for model in IMAGE_MODELS if model["short"] in _image_shorts]
        selected_embed = [model for model in EMBED_MODELS if model["tag"] in _embed_tags]
        cleanup_names = list(_gui_plan["cleanup_names"])
        vllm_cleanup_names = list(_gui_plan.get("vllm_cleanup_names", []))
    selected_llm_tags     = {m["tag"] for m in selected_llm}
    selected_image_shorts = {m["short"] for m in selected_images}

    print()
    info(f"LLM models selected: {len(selected_llm)}/{len(LLM_MODELS_XSMALL) + len(LLM_MODELS_SMALL) + len(LLM_MODELS_MEDIUM) + len(LLM_MODELS_LARGE)}")
    info(f"Image models selected: {len(selected_images)}/{len(IMAGE_MODELS)}")
    info(f"Embeddings models selected: {len(selected_embed)}/{len(EMBED_MODELS)}")
    if cleanup_names:
        warn(f"Non-catalog model cleanup selected: {len(cleanup_names)} folder(s)")
    if vllm_cleanup_names:
        warn(f"Cached vLLM weight cleanup selected: {len(vllm_cleanup_names)} repo(s)")

    # ── 7. HuggingFace token (only if a selected image model needs one) ───────────

    token_provider = HfTokenProvider(
        SCRIPT_DIR, bool(GATED_IMAGE_SHORTS & selected_image_shorts),
    )


    def load_token():
        return token_provider.load()

    if _gui_plan is not None:
        _gui_token = _gui_plan["hf_token"]
        if not _gui_token and _gui_plan["use_existing_hf_token"]:
            _gui_token = token_provider.load_existing()
        token_provider.set(_gui_token)
        if _gui_token and _gui_plan["save_hf_token"]:
            try:
                token_provider.save(_gui_token)
                ok("Hugging Face token saved to hf.txt")
            except OSError as exc:
                warn(f"Could not save hf.txt: {exc}")
    elif selected_llm or selected_embed or selected_images:
        section("HuggingFace Token")
        load_token()

    if _gui_plan is not None and selected_images:
        _gui_comfy_mode = _gui_plan["comfyui_mode"]
        if _gui_comfy_mode == "existing":
            _normalized_comfyui = normalize_comfyui_dir(Path(_gui_plan["comfyui_path"]))
            COMFYUI_DIR = _normalized_comfyui or config.COMFYUI_DIR
            _detected_comfyui = _normalized_comfyui
        elif _gui_comfy_mode == "detected" and _detected_comfyui:
            COMFYUI_DIR = _detected_comfyui
        else:
            COMFYUI_DIR = config.COMFYUI_DIR
            _detected_comfyui = None
    elif selected_images and not _detected_comfyui:
        section("ComfyUI Installation")
        print("  No usable ComfyUI installation was detected.")
        print("  1. Download and manage a compatible ComfyUI copy here (default)")
        print("  2. Enter the path to an existing ComfyUI installation")
        try:
            comfyui_choice = input(f"  {CYAN}Choose [1/2]:{RESET} ")
        except EOFError:
            comfyui_choice = ""
        entered_path = ""
        if comfyui_choice.strip().lower() in {"2", "p", "path"}:
            try:
                entered_path = input(
                    f"  {CYAN}ComfyUI directory, main.py, or portable launcher path:{RESET} "
                ).strip()
            except EOFError:
                entered_path = ""
        choice_status, chosen_comfyui = resolve_comfyui_setup_choice(comfyui_choice, entered_path)
        if choice_status == "existing" and chosen_comfyui:
            COMFYUI_DIR = chosen_comfyui
            _detected_comfyui = chosen_comfyui
            ok(f"Using existing ComfyUI at {COMFYUI_DIR}")
        elif choice_status == "invalid":
            warn(f"No usable ComfyUI installation was found at {entered_path!r}")
            info(f"Downloading a managed copy to {config.COMFYUI_DIR}")
        else:
            info(f"Downloading a managed copy to {config.COMFYUI_DIR}")

    # ── 8. Installing — everything below runs unattended, no more prompts ─────────

    INSTALL_STARTED = True

    _gui_progress_path: Path | None = None
    _gui_progress_status = ["stopped"]
    if _gui_plan is not None:
        try:
            _gui_progress_process, _gui_progress_path = start_setup_progress()
            _started_progress_path = _gui_progress_path
            atexit.register(
                lambda: finish_setup_progress(_started_progress_path, _gui_progress_status[0]),
            )
        except OSError as exc:
            warn(f"Could not open the graphical progress window: {exc}")

    section("Installing")

    req_file = SCRIPT_DIR / "requirements.txt"
    info("Installing Python dependencies ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        ok("Packages installed from requirements.txt")
    else:
        fail("pip install -r requirements.txt failed")
        info(result.stderr.strip().splitlines()[-1] if result.stderr else "")
        sys.exit(1)

    if LLAMACPP in pending_engines:
        llamacpp_installed = install_llamacpp()
        if llamacpp_installed:
            ok("llama.cpp installed successfully")
            llamacpp_found = True
            LLAMACPP_BIN = find_llamacpp_binary()
            LLAMACPP_BENCH_BIN = find_llamacpp_bench_binary()
            if LLAMACPP_BENCH_BIN:
                ok(f"llama-bench found: {LLAMACPP_BENCH_BIN}")
            else:
                warn("llama-bench still not found after install — llama-bench-based tests won't be available")
            LLAMACPP_BATCHED_BENCH_BIN = find_llamacpp_batched_bench_binary()
            if LLAMACPP_BATCHED_BENCH_BIN:
                ok(f"llama-batched-bench found: {LLAMACPP_BATCHED_BENCH_BIN}")
            else:
                warn("llama-batched-bench still not found after install — the llamabenchconc test won't be available")
        else:
            fail("llama.cpp installation failed")
            issues.append("Install llama.cpp manually: https://github.com/ggml-org/llama.cpp "
                           "(needs a 'llama-server' binary on PATH, or built under "
                          f"{LLAMACPP_DIR})")

    if needs_python_headers(engine_entries, missing_python_header):
        if header_command is None:
            fail(f"Python development headers are missing ({missing_python_header}) and no known "
                 "package manager was found — install your distribution's python3 dev package")
            issues.append(f"Install the Python development headers providing {missing_python_header}")
        else:
            info(f"Installing Python development headers: {' '.join(header_command)} ...")
            if subprocess.run(header_command).returncode == 0:
                ok("Python development headers installed")
            else:
                fail("Python development header install failed — vLLM will not start")
                issues.append(f"Run: {' '.join(header_command)}")

    if VLLM in pending_engines:
        if install_vllm(vllm_support, log=info):
            VLLM_BIN = find_vllm_binary(platform_name=os_name)
            if VLLM_BIN:
                ok(f"vLLM installed: {VLLM_BIN}")
                vllm_found = True
            else:
                warn("vLLM install reported success but no 'vllm' executable was found")
        else:
            fail("vLLM installation failed")
            issues.append("Install vLLM manually: https://docs.vllm.ai/en/stable/getting_started/installation/")

    if VLLM in selected_engines:
        _vllm_venv_python = config.VLLM_VENV / "bin" / "python"
        _missing_tools = missing_build_tools(config.VLLM_VENV) if _vllm_venv_python.is_file() else []
        if not _vllm_venv_python.is_file():
            info(f"No project vLLM venv at {config.VLLM_VENV} — build tools are that "
                 "installation's own responsibility")
        elif not _missing_tools:
            ok("vLLM build tools already present")
        if _missing_tools:
            info(f"Installing vLLM build tools ({', '.join(_missing_tools)}) — "
                 "FlashInfer compiles kernels on first use ...")
            _tools_command = build_tools_command(str(_vllm_venv_python), _missing_tools)
            if _tools_command and subprocess.run(_tools_command).returncode == 0:
                ok("vLLM build tools installed")
            else:
                fail("vLLM build tool install failed — kernel compilation will fail at run time")
                issues.append(f"Run: {config.VLLM_VENV}/bin/pip install {' '.join(_missing_tools)}")

    # ── 8a. Disk space ──────────────────────────────────────────────────────────────

    section("Disk Space")

    CHECKPOINTS = config.COMFYUI_MODELS_DIR / "checkpoints"

    def image_asset(name, subdir):
        """Existing path of an image asset, including the pre-4.1 <ComfyUI>/models location."""
        return find_image_asset(name, config.COMFYUI_MODELS_DIR, subdir, COMFYUI_DIR)
    CLIP_DIR    = config.COMFYUI_MODELS_DIR / "clip"
    VAE_DIR     = config.COMFYUI_MODELS_DIR / "vae"

    remaining_gb = 0.0

    def model_downloaded(m, engine=LLAMACPP):
        """True if `engine`'s weights for `m` are already present."""
        if engine == VLLM:
            return hf_cache_model_complete(VLLM_CACHE_HOME, m["vllm_repo"])
        filenames = m["hf_file"] if isinstance(m["hf_file"], list) else [m["hf_file"]]
        model_dir = engine_model_dir(config.MODELS_DIR, engine, m["tag"])
        return engine_model_complete(model_dir, engine, filenames)

    all_llm = selected_embed + selected_llm
    for engine in selected_engines:
        for m in all_llm:
            if not model_downloaded(m, engine):
                remaining_gb += hardware.parse_size_gb(engine_download_size(m, engine) or "")

    sd35_selected  = "sd35-large" in selected_image_shorts
    flux1_selected = "flux-dev" in selected_image_shorts
    flux2_selected = "flux2-dev" in selected_image_shorts

    for m in selected_images:
        if not image_asset(m["checkpoint"], "checkpoints"):
            remaining_gb += CHECKPOINT_SIZES_GB.get(m["checkpoint"], 0.0)

    # Shared T5-XXL + CLIP-L text encoders: used by SD3.5 Large and Flux.1-dev,
    # NOT Flux.2-dev (which has its own Mistral-based encoder below).
    if (sd35_selected or flux1_selected):
        for fname in ("t5xxl_fp16.safetensors", "clip_l.safetensors"):
            if not image_asset(fname, "clip"):
                remaining_gb += ENCODER_SIZES_GB[fname]
    if sd35_selected and not image_asset("clip_g.safetensors", "clip"):
        remaining_gb += ENCODER_SIZES_GB["clip_g.safetensors"]
    if flux1_selected and not image_asset("ae.safetensors", "vae"):
        remaining_gb += ENCODER_SIZES_GB["ae.safetensors"]
    if flux2_selected:
        if not image_asset("mistral_3_small_flux2_fp8.safetensors", "text_encoders"):
            remaining_gb += ENCODER_SIZES_GB["mistral_3_small_flux2_fp8.safetensors"]
        if not image_asset("flux2-vae.safetensors", "vae"):
            remaining_gb += ENCODER_SIZES_GB["flux2-vae.safetensors"]

    try:
        total, used, free = shutil.disk_usage(SCRIPT_DIR)
        free_gb  = free  // (1024**3)
        total_gb = total // (1024**3)
        print(f"  Free:              {free_gb} GB / {total_gb} GB total")
        if remaining_gb > 0:
            print(f"  Still to download: ~{remaining_gb:.0f} GB")
        def _warn_if_drive_fills_up():
            # Informational only (doesn't block or add to `issues`) — warns even when the downloads themselves fit.
            projected_free_gb = free_gb - remaining_gb
            if projected_free_gb < total_gb * 0.10:
                warn(f"After these downloads, free space would be ~{projected_free_gb:.0f} GB — "
                     f"less than 10% of your {total_gb:.0f} GB drive. Continuing in 5s ...")
                time.sleep(5)

        if remaining_gb == 0:
            ok("All selected models already downloaded — no additional space needed")
        elif free_gb >= remaining_gb + 10:
            ok(f"Sufficient free space for remaining ~{remaining_gb:.0f} GB of downloads")
            if total_gb > 0:
                _warn_if_drive_fills_up()
        elif free_gb >= remaining_gb:
            warn(f"Space is tight — ~{remaining_gb:.0f} GB needed, {free_gb} GB free (less than 10 GB buffer)")
            if total_gb > 0:
                _warn_if_drive_fills_up()
        else:
            needed_more = additional_disk_space_needed(free_gb, remaining_gb)
            fail(f"Insufficient space — ~{remaining_gb:.0f} GB needed, only {free_gb} GB free")
            fail(f"Setup stopped to avoid a partial installation or filling this volume. "
                 f"Free at least ~{needed_more:.0f} GB and run setup again.")
            sys.exit(1)
    except Exception as e:
        warn(f"Could not check disk space: {e}")

    if cleanup_names:
        section("Non-catalog Model Cleanup")
        cleanup_root = config.MODELS_DIR / "llamacpp"
        removed, cleanup_failures = delete_non_catalog_model_dirs(cleanup_root, cleanup_names)
        for name in removed:
            ok(f"{name!r} — deleted")
        for name, reason in cleanup_failures.items():
            fail(f"{name!r} — could not delete: {reason}")
            issues.append(f"Delete non-catalog model folder {str(cleanup_root / name)!r}")

    if vllm_cleanup_names:
        section("Cached vLLM Weight Cleanup")
        _vllm_cache = VLLM_CACHE_HOME
        removed, cleanup_failures = delete_non_catalog_vllm_repos(_vllm_cache, vllm_cleanup_names)
        for name in removed:
            ok(f"{hf_cache_repo_id(name)} — deleted")
        for name, reason in cleanup_failures.items():
            fail(f"{name!r} — could not delete: {reason}")
            issues.append(f"Delete cached vLLM weights {str(_vllm_cache / 'hub' / name)!r}")

    # ── 8b. LLM/embedding models — download selected GGUFs, skip the rest ─────────

    section("LLM/Embedding Models")

    deselected_llm = [
        m for tier in (LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE)
        for m in tier if m["tag"] not in selected_llm_tags
    ]
    for m in deselected_llm:
        info(f"{m['label']} — skipped (not selected)")

    for engine in selected_engines:
        if len(selected_engines) > 1:
            info(f"Models for {engine} ...")
        unsupported = models_missing_engine_support(selected_embed + selected_llm, engine)
        for tag in unsupported:
            warn(f"{tag} — no {engine} weights defined in the catalog, skipping for this engine")
            issues.append(f"No {engine} weights for {tag} — it will only be benchmarked on other engines")
        for m in selected_embed + selected_llm:
            tag, label = m["tag"], m["label"]
            if tag in unsupported:
                continue
            size = engine_download_size(m, engine)
            if model_downloaded(m, engine):
                ok(f"{label} [{engine}] — already downloaded")
                continue
            warn(f"{label} [{engine}] ({size}) — not found, downloading now ...")
            if engine == VLLM:
                repo, dest = m["vllm_repo"], VLLM_CACHE_HOME
                success = hf_snapshot_download(repo, VLLM_CACHE_HOME, token=load_token())
            else:
                repo = m["hf_repo"]
                dest = engine_model_dir(config.MODELS_DIR, engine, tag)
                success = hf_download(repo, m["hf_file"], token=load_token(), dest_dir=dest)
            if success:
                ok(f"{label} [{engine}] — downloaded successfully")
            else:
                fail(f"{label} [{engine}] — download failed")
                issues.append(f"Download {repo} manually into {dest}")

    # ── 8c. ComfyUI — only if at least one image model was selected ───────────────

    if selected_images:
        section("ComfyUI")

        PORTABLE_PYTHON = COMFYUI_DIR.parent / "python_embeded" / "python.exe"
        nvidia_windows  = nvidia_ok and os_name == "Windows"

        windows_gpu = (
            "amd" if amd_windows else "nvidia" if nvidia_windows
            else "intel" if intel_windows else None
        )
        ensure_comfyui(
            COMFYUI_DIR, SCRIPT_DIR, os_name, windows_gpu,
            compute_capability=nvidia_compute_cap, issues=issues,
            info=info, warn=warn, fail=fail, ok=ok,
        )
        if prepare_comfyui_runtime(
            COMFYUI_DIR, config.COMFYUI_MODELS_DIR, config.COMFYUI_EXTRA_MODEL_PATHS,
            portable_python=PORTABLE_PYTHON, intel_xpu=intel_linux, rocm=rocm_ok,
            issues=issues, info=info, warn=warn, fail=fail, ok=ok,
        ):
            found_ckpts = provision_comfyui_assets(
                selected_images, config.COMFYUI_MODELS_DIR,
                find_asset=image_asset, download=hf_download, load_token=load_token,
                info=info, warn=warn, fail=fail, ok=ok,
            )
            if found_ckpts:
                ok(f"{len(found_ckpts)}/{len(selected_images)} image checkpoints ready: "
                   f"{', '.join(found_ckpts)}")
                try:
                    with urllib.request.urlopen(
                        f"{config.COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=3,
                    ) as response:
                        available = checkpoint_names_from_object_info(json.load(response))
                    if managed_checkpoints_visible(available, set(found_ckpts)):
                        ok("Running ComfyUI already sees Local AI Bench's managed models")
                    else:
                        warn("Restart running ComfyUI once to load the managed model path")
                except (OSError, urllib.error.URLError, json.JSONDecodeError):
                    pass
            else:
                fail("No image checkpoints available — image benchmarks will be skipped")
                issues.append("Download at least one image checkpoint into models/comfyui/checkpoints/")

    # ── 9. Summary ────────────────────────────────────────────────────────────────

    write_setup_config(
        config.SETUP_CONFIG_PATH,
        comfyui_dir=COMFYUI_DIR if (COMFYUI_DIR / "main.py").is_file() else None,
        llamacpp_tools={
            "llama-server": LLAMACPP_BIN,
            "llama-bench": LLAMACPP_BENCH_BIN,
            "llama-batched-bench": LLAMACPP_BATCHED_BENCH_BIN,
        },
        gpu_devices=(
            [{**device, "vendor": "nvidia", "backend": "cuda"} for device in nvidia_gpus]
            if nvidia_ok else rocm_gpus
        ),
        vllm={
            "executable": VLLM_BIN,
            "launcher": VLLM_LAUNCHER,
            "server_url": VLLM_SERVER_URL,
            "launcher_extra_args": VLLM_LAUNCHER_ARGS,
            "hf_home": str(VLLM_CACHE_HOME),
        } if vllm_found else {},
    )

    section("Summary")

    if not issues:
        print(f"\n  {GREEN}{BOLD}All checks passed — ready to benchmark!{RESET}")
        run_hint = "run_bench.bat" if os_name == "Windows" else "bash run_bench.sh"
        print(f"  Run: {run_hint}\n")
    else:
        print(f"\n  {YELLOW}{BOLD}Action items before benchmarking:{RESET}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()

    if _gui_progress_path is not None:
        _gui_progress_status[0] = "complete" if not issues else "action_items"
        finish_setup_progress(_gui_progress_path, _gui_progress_status[0])


if __name__ == "__main__":  # pragma: no cover
    main()
