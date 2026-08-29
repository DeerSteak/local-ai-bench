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
from collections.abc import Collection
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
from scripts.runtime.llamacpp_tools import (
    find_nvcc, llamacpp_backend_error, llamacpp_backend_mismatch, probe_llamacpp_backend,
)
from scripts.setup.cuda_install import (
    NATIVE_NVIDIA_REBOOT_EXIT_CODE, cuda_toolkit_plan, native_cuda_toolkit_plan,
    native_nvidia_driver_plan, nouveau_loaded, qualification_needs_native_nvidia_driver,
    run_cuda_toolkit_install, run_native_nvidia_driver_install,
)
from scripts.setup.intel_xpu_install import (
    intel_xpu_install_plan, oneapi_environment, run_intel_xpu_install, sycl_gpu_available,
)
from scripts.setup.rocm_install import (
    NATIVE_ROCM_VERSION, WINDOWS_DRIVER, native_rocm_install_plan,
    qualification_needs_native_rocm, qualification_needs_wsl_rocm, run_rocm_install,
    ryzen_ai_halo_dkms_packages, ryzen_ai_halo_oem_kernel_ready, setup_needs_wsl_rocm,
    wsl_rocm_install_plan,
)
from scripts.setup.model_inventory import (
    delete_non_catalog_model_dirs, delete_non_catalog_vllm_repos,
    engine_download_size, find_non_catalog_vllm_repos,
    hf_cache_repo_id,
    find_non_catalog_model_dirs,
)
from scripts.setup.model_download import (
    catalog_model_downloaded, catalog_mtp_artifact_download_size,
    catalog_mtp_artifact_downloaded, download_hf_files, download_hf_snapshot,
    provision_catalog_models,
)
from scripts.setup import llamacpp_install
from scripts.setup.hf_credentials import HfTokenProvider
from scripts.setup.comfyui_assets import (
    missing_download_size_gb, provision as provision_comfyui_assets,
)
from scripts.setup.comfyui_runtime import prepare as prepare_comfyui_runtime
from scripts.setup.comfyui_install import ensure as ensure_comfyui
from scripts.workloads.models import (
    EMBED_MODELS, IMAGE_MODELS, LLM_MODELS_LARGE, LLM_MODELS_MEDIUM,
    LLM_MODELS_SMALL, LLM_MODELS_XSMALL, image_checkpoint_groups,
)
from scripts.workloads.model_variants import expanded_variant_catalog
from scripts.setup.setup_selection import (
    additional_disk_space_needed, qualification_model_selection, select_models,
)
from scripts.setup.setup_config import (
    configured_comfyui_dir, load_setup_config, vllm_setup_config, write_setup_config,
)
from scripts.setup.setup_progress import finish_setup_progress, start_setup_progress
from scripts.setup.setup_discovery import (
    discover_intel_vram_gb, discover_linux_amd_gpu, discover_linux_intel_gpu,
    discover_linux_nvidia_gpu,
    discover_metal, discover_nvidia, discover_rocm, discover_system,
    discover_windows_amd_gpus, discover_windows_gpu, discover_wsl_windows_amd_gpus,
    rocm_version,
)
from scripts.setup.setup_console import (
    BOLD, CYAN, GREEN, RESET, YELLOW, confirm, fail, info, link, ok, section, warn,
)
from scripts.setup.engine_selection import (
    LLAMACPP, LLAMACPP_VULKAN, VLLM, apply_engine_preset, build_engine_entries,
    engines_needing_install, llamacpp_vulkan_setup_state, model_engine_names,
    needs_python_headers, qualification_engines_needing_install, qualification_setup_failed,
    select_engines,
    selected_engine_names,
)
from scripts.setup.vllm_install import (
    find_vllm_binary, find_vllm_launcher, find_vllm_server,
    install_vllm, install_vllm_build_tools, missing_python_headers,
    python_dev_package_command, python_include_dir, python_version_from_include_dir,
    read_launcher_extra_args, redact_launcher_extra_args, vllm_cache_home, vllm_platform_support,
    PINNED_PYTHON, python_bootstrap_plan, resolve_python, run_python_bootstrap,
    vllm_runtime_expectations, vllm_runtime_import_error,
)
from scripts.setup.vulkan_install import (
    missing_vulkan_build_requirements, run_vulkan_build_install,
    vulkan_build_install_plan,
)
from scripts.app.interface_mode import select_interface_mode
from scripts.release.qualification_targets import qualification_host_error, qualification_target


def llamacpp_backend_rebuild_warning(installed_backend: str | None, required_backend: str | None,
                                     *, qualification: bool) -> str:
    context = "qualification" if qualification else "setup"
    installed = installed_backend or "no detectable backend"
    required = required_backend or "an accelerator backend"
    return (f"llama-server exposes {installed}, but {context} requires {required} "
            "— it will be rebuilt")


def llamacpp_install_action(required_backend: str | None, failure: str | None,
                            runtime_dir: Path) -> str:
    backend = {
        "cuda": "CUDA", "rocm": "ROCm/HIP", "xpu": "Intel oneAPI/SYCL",
    }.get(required_backend or "", "CPU")
    detail = f" Last failure: {failure}." if failure else ""
    return (f"Rerun setup to retry the managed llama.cpp {backend} build at "
            f"{runtime_dir}.{detail} No manual llama.cpp installation is required.")


def accessible_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def running_comfyui_checkpoints_visible(selected_images: list[dict],
                                         found_checkpoints: Collection[str], *,
                                         urlopen=urllib.request.urlopen) -> bool | None:
    ready_models = [
        model for model in selected_images if model["checkpoint"] in found_checkpoints
    ]
    try:
        visible = True
        for loader, expected in image_checkpoint_groups(ready_models).items():
            with urlopen(f"{config.COMFYUI_URL}/object_info/{loader}", timeout=3) as response:
                available = checkpoint_names_from_object_info(json.load(response), loader)
            visible = visible and managed_checkpoints_visible(available, expected)
        return visible
    except (OSError, ValueError):
        return None


def main() -> None:  # pragma: no cover - real interactive installer
    # Repo root, one level up — sourced from config.py rather than redefined here.
    SCRIPT_DIR   = config.SCRIPT_DIR
    LLAMACPP_DIR = config.LLAMACPP_DIR
    LLAMACPP_VULKAN_DIR = config.LLAMACPP_VULKAN_DIR

    _arg_parser = argparse.ArgumentParser(description="local-ai-bench setup")
    _arg_parser.add_argument("--comfyui", help="Path to an existing ComfyUI or portable root")
    _arg_parser.add_argument("--interface", choices=("auto", "gui", "terminal"), default="auto")
    _arg_parser.add_argument("--qualification", choices=(LLAMACPP, VLLM))
    _arg_parser.add_argument("--qualification-target")
    args = _arg_parser.parse_args()
    if args.qualification_target and not args.qualification:
        _arg_parser.error("--qualification-target requires --qualification")
    try:
        _qualification_target = (
            qualification_target(args.qualification_target) if args.qualification_target else None
        )
    except ValueError as exc:
        _arg_parser.error(str(exc))
    if _qualification_target and _qualification_target["runtime"] != args.qualification:
        _arg_parser.error(
            f"target {args.qualification_target} requires "
            f"--qualification {_qualification_target['runtime']}"
        )
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

    GATED_IMAGE_SHORTS = {"flux-dev", "flux2-dev"}

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

    _wsl_windows_amd_gpus = discover_wsl_windows_amd_gpus()
    _setup_wsl_rocm_plan = None

    if _qualification_target:
        if _host_error := qualification_host_error(_qualification_target):
            fail(_host_error)
            sys.exit(1)
        _initial_nvidia = discover_nvidia()
        try:
            _install_native_nvidia = qualification_needs_native_nvidia_driver(
                _qualification_target, os_name=os_name, release=platform.release(),
                nvidia_available=_initial_nvidia.available,
            )
        except ValueError as exc:
            _arg_parser.error(str(exc))
        if _install_native_nvidia:
            section("NVIDIA driver for native Linux")
            _nvidia_gpu = discover_linux_nvidia_gpu()
            if _nvidia_gpu.vendor != "nvidia":
                fail("native NVIDIA qualification found no NVIDIA display adapter")
                sys.exit(1)
            info(f"GPU: {_nvidia_gpu.name}")
            try:
                _disable_nouveau = nouveau_loaded()
                _driver_plan = native_nvidia_driver_plan(
                    platform.freedesktop_os_release(), platform.release(),
                    disable_nouveau=_disable_nouveau,
                )
                if _disable_nouveau:
                    info("nouveau is loaded; replacing it with the NVIDIA CUDA driver")
                info("Installing Ubuntu's recommended signed NVIDIA driver (requires sudo) ...")
                run_native_nvidia_driver_install(_driver_plan)
            except (OSError, RuntimeError, ValueError) as exc:
                fail(f"NVIDIA driver installation failed: {exc}")
                sys.exit(1)
            warn("NVIDIA driver installed; reboot to load it, then rerun qualification")
            sys.exit(NATIVE_NVIDIA_REBOOT_EXIT_CODE)

        _initial_rocm = discover_rocm()
        try:
            _install_wsl_rocm = qualification_needs_wsl_rocm(
                _qualification_target, os_name=os_name, release=platform.release(),
                rocm_available=_initial_rocm.available,
            )
            _install_native_rocm = qualification_needs_native_rocm(
                _qualification_target, os_name=os_name, release=platform.release(),
                rocm_available=_initial_rocm.available,
            )
            _install_native_rocm = _install_native_rocm or not ryzen_ai_halo_oem_kernel_ready(
                _qualification_target["id"], platform.release(),
            )
            _halo_dkms_packages = ryzen_ai_halo_dkms_packages(_qualification_target["id"])
            if _halo_dkms_packages:
                info(
                    "Ryzen AI Halo requires inbox drivers; removing "
                    f"{', '.join(_halo_dkms_packages)}"
                )
                _install_native_rocm = True
        except ValueError as exc:
            _arg_parser.error(str(exc))
        if _install_wsl_rocm or _install_native_rocm:
            section("ROCm for WSL2" if _install_wsl_rocm else "ROCm for native Linux")
            try:
                _os_release = Path("/etc/os-release").read_text(encoding="utf-8")
                if _install_wsl_rocm:
                    _rocm_plan = wsl_rocm_install_plan(_os_release)
                else:
                    _amd_gpu = discover_linux_amd_gpu()
                    if _amd_gpu.vendor != "amd":
                        raise ValueError("native ROCm qualification found no AMD display adapter")
                    info(f"GPU: {_amd_gpu.name}")
                    _rocm_plan = native_rocm_install_plan(
                        _os_release, platform.release(), target_id=_qualification_target["id"],
                        user=os.environ.get("SUDO_USER") or os.environ.get("USER"),
                    )
            except (OSError, ValueError) as exc:
                fail(str(exc))
                sys.exit(1)
            if _install_wsl_rocm:
                info("Installing AMD ROCm 7.2 for WSL2 (requires sudo) ...")
                info(f"Compatible Windows host driver required: {WINDOWS_DRIVER}")
            else:
                info(f"Installing AMD ROCm {NATIVE_ROCM_VERSION} for native Linux (requires sudo) ...")
            try:
                run_rocm_install(_rocm_plan)
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                fail(f"ROCm installation failed: {exc}")
                sys.exit(1)
            if _rocm_plan.reboot_required:
                fail(
                    "Ryzen AI Halo inbox driver prepared; reboot to load its kernel and "
                    "firmware, then rerun qualification"
                )
                sys.exit(1)
            if not discover_rocm().available:
                fail("ROCm installed but rocminfo cannot see an AMD GPU")
                if _install_wsl_rocm:
                    fail(f"Install {WINDOWS_DRIVER}, reboot Windows, then run qualification again")
                else:
                    fail("Reboot so the AMD driver and render/video group access take effect, then rerun qualification")
                sys.exit(1)
            ok("ROCm installed and GPU access verified")

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
    if not _qualification_target and setup_needs_wsl_rocm(
        os_name=os_name, release=platform.release(), amd_gpus=_wsl_windows_amd_gpus,
        rocm_available=bool(rocm and rocm.available),
    ):
        try:
            _setup_wsl_rocm_plan = wsl_rocm_install_plan(
                Path("/etc/os-release").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            fail(str(exc))
            sys.exit(1)
    rocm_gfx = rocm.gfx_targets if rocm else []
    rocm_gpu_kind = rocm.kind if rocm else None
    rocm_vram_gb = rocm.total_vram_gb if rocm else None
    rocm_gpus = rocm.gpus if rocm else []
    for name in rocm.names[:3] if rocm else []:
        print(f"  ROCm GPU: {name}")

    def check_linux_intel_gpu_runtime():
        """Detection-only check for a usable Intel GPU through oneAPI SYCL."""
        environment = oneapi_environment()
        return bool(environment and sycl_gpu_available(env=environment))

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
    if _setup_wsl_rocm_plan:
        rocm_ok = True
        rocm_gpus = [{**device, "backend": "rocm"} for device in _wsl_windows_amd_gpus]
        rocm_vram_gb = sum(
            device["vram_gb"] for device in rocm_gpus if device["vram_gb"] is not None
        ) or None
        rocm_gpu_kind = (
            "discrete" if any(device["kind"] == "discrete" for device in rocm_gpus)
            else "integrated"
        )
        for device in rocm_gpus:
            print(f"  WSL host GPU: {device['name']}")
            vram = device["vram_gb"]
            print(f"  VRAM:    {f'{vram:.1f} GB' if vram is not None else 'unknown'}")
    metal_ok, metal_details = discover_metal() if not nvidia_ok and not rocm_ok else (False, [])
    for detail in metal_details:
        print(f"  {detail}")
    if not nvidia_ok and os_name == "Windows":
        windows_amd_gpus = discover_windows_amd_gpus()
        windows_gpu = discover_windows_gpu() if not windows_amd_gpus else None
        windows_gpu_kind = "discrete" if windows_amd_gpus else windows_gpu.kind
        amd_windows = bool(windows_amd_gpus) or windows_gpu.vendor == "amd"
        intel_windows = not windows_amd_gpus and windows_gpu.vendor == "intel"
        for device in windows_amd_gpus:
            print(f"  GPU:     {device['name']}")
            vram = device["vram_gb"]
            print(f"  VRAM:    {f'{vram:.1f} GB' if vram is not None else 'unknown'}")
            if device["driver"]:
                print(f"  Driver:  {device['driver']}")
        if windows_gpu and windows_gpu.name:
            print(f"  GPU:     {windows_gpu.name}")
    else:
        windows_amd_gpus = []
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
    elif _setup_wsl_rocm_plan:
        ok("AMD GPU detected through the Windows host")
        info("Setup will install ROCm 7.2 for WSL2 before building llama.cpp")
    elif rocm_ok:
        ok("ROCm / AMD GPU detected")
    elif amd_windows:
        ok("AMD/Radeon GPU detected on Windows")
    elif intel_windows:
        ok("Intel Arc GPU detected on Windows")
        info("Intel Arc support remains unverified until qualification completes")
        info("Setup will use llama.cpp's official self-contained Windows SYCL package")
    elif intel_linux:
        ok("Intel Arc GPU detected on Linux")
        if intel_linux_runtime:
            ok("Intel oneAPI/SYCL GPU runtime detected")
        else:
            warn("Intel oneAPI/SYCL GPU runtime not detected — setup will install it")
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
        known_windows_vram = [
            device["vram_gb"] for device in windows_amd_gpus
            if device["vram_gb"] is not None
        ]
        gpu_vram_gb = sum(known_windows_vram) if known_windows_vram else None
    elif intel_windows:
        gpu_vendor = "intel" if windows_gpu_kind == "discrete" else "integrated"
        gpu_vram_gb = discover_intel_vram_gb() if gpu_vendor == "intel" else None
    elif intel_linux:
        gpu_vendor = "intel" if linux_intel_gpu_kind == "discrete" else "integrated"
        gpu_vram_gb = discover_intel_vram_gb() if gpu_vendor == "intel" else None
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
            [device["vram_gb"] for device in rocm_gpus] if rocm_ok else
            [device["vram_gb"] for device in windows_amd_gpus
             if device["vram_gb"] is not None] if amd_windows else None
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

    _llamacpp_install_failures: list[str] = []

    def install_llamacpp():
        def record_failure(message: str) -> None:
            _llamacpp_install_failures.append(message)
            fail(message)

        return llamacpp_install.install(
            LLAMACPP_DIR, SCRIPT_DIR, os_name, nvidia=nvidia_ok, rocm=rocm_ok,
            intel_xpu=intel_linux or intel_windows,
            compute_capability=nvidia_compute_cap,
            max_cuda_version=nvidia_max_cuda_version,
            info=info, warn=warn, fail=record_failure, ok=ok,
        )

    def install_llamacpp_vulkan():
        def record_failure(message: str) -> None:
            _llamacpp_install_failures.append(message)
            fail(message)

        return llamacpp_install.install(
            LLAMACPP_VULKAN_DIR, SCRIPT_DIR, os_name,
            nvidia=nvidia_ok, rocm=rocm_ok, intel_xpu=intel_linux or intel_windows,
            compute_capability=nvidia_compute_cap,
            max_cuda_version=nvidia_max_cuda_version,
            info=info, warn=warn, fail=record_failure, ok=ok, vulkan=True,
        )

    # ── 4a. llama.cpp detection (read-only) ────────────────────────────────────────

    section("llama.cpp")

    llamacpp_tools = llamacpp_install.find_tools(LLAMACPP_DIR, os_name)
    LLAMACPP_BIN = llamacpp_tools["llama-server"]
    _required_llamacpp_backend = None
    if _qualification_target and _qualification_target["runtime"] == LLAMACPP:
        _required_llamacpp_backend = _qualification_target["backend"]
    elif os_name == "Linux":
        if nvidia_ok:
            _required_llamacpp_backend = "cuda"
        elif rocm_ok:
            _required_llamacpp_backend = "rocm"
        elif intel_linux:
            _required_llamacpp_backend = "xpu"
    _llamacpp_probe_env = oneapi_environment() if _required_llamacpp_backend == "xpu" else None
    _installed_llamacpp_backend = (
        probe_llamacpp_backend(LLAMACPP_BIN, env=_llamacpp_probe_env) if LLAMACPP_BIN else None
    )
    _llamacpp_backend_mismatch = llamacpp_backend_mismatch(
        LLAMACPP_BIN, _installed_llamacpp_backend, _required_llamacpp_backend,
    )
    llamacpp_found = LLAMACPP_BIN is not None and not _llamacpp_backend_mismatch
    _managed_llamacpp_ready = llamacpp_install.managed_toolset_ready(LLAMACPP_DIR, os_name)
    managed_mac_runtime = os_name == "Darwin" and LLAMACPP_DIR.is_dir() and any(
        path.is_file() for path in LLAMACPP_DIR.rglob("llama-server")
    )
    needs_llamacpp_install = (
        not llamacpp_found or (os_name == "Darwin" and not managed_mac_runtime)
        or (args.qualification == LLAMACPP and not _managed_llamacpp_ready)
    )
    if llamacpp_found:
        ok(f"llama-server found: {LLAMACPP_BIN}")
    elif _llamacpp_backend_mismatch:
        warn(llamacpp_backend_rebuild_warning(
            _installed_llamacpp_backend, _required_llamacpp_backend,
            qualification=bool(_qualification_target),
        ))
    else:
        warn("llama-server not found — will need to be installed")

    LLAMACPP_BENCH_BIN = llamacpp_tools["llama-bench"]
    llamacpp_bench_found = LLAMACPP_BENCH_BIN is not None
    if llamacpp_bench_found:
        ok(f"llama-bench found: {LLAMACPP_BENCH_BIN}")
    elif needs_llamacpp_install:
        info("llama-bench not found — will be installed alongside llama-server")
    else:
        warn("llama-bench not found (llama-server is installed, but without llama-bench) — "
             "rerun setup after a fresh llama.cpp install, or build it yourself, to use "
             "the llamabench test")

    LLAMACPP_BATCHED_BENCH_BIN = llamacpp_tools["llama-batched-bench"]
    llamacpp_batched_bench_found = LLAMACPP_BATCHED_BENCH_BIN is not None
    if llamacpp_batched_bench_found:
        ok(f"llama-batched-bench found: {LLAMACPP_BATCHED_BENCH_BIN}")
    elif needs_llamacpp_install:
        info("llama-batched-bench not found — will be installed alongside llama-server")
    else:
        warn("llama-batched-bench not found (llama-server is installed, but without llama-batched-bench) — "
             "rerun setup after a fresh llama.cpp install, or build it yourself, to use "
             "the llamabenchconc test")

    vulkan_tools = llamacpp_install.find_tools(LLAMACPP_VULKAN_DIR, os_name)
    LLAMACPP_VULKAN_BIN = vulkan_tools["llama-server"]
    _installed_vulkan_backend = (
        probe_llamacpp_backend(LLAMACPP_VULKAN_BIN) if LLAMACPP_VULKAN_BIN else None
    )
    _managed_vulkan_ready = llamacpp_install.managed_toolset_ready(
        LLAMACPP_VULKAN_DIR, os_name,
    )
    _vulkan_state = llamacpp_vulkan_setup_state(
        os_name, platform.machine(), runtime_present=LLAMACPP_VULKAN_BIN is not None,
        backend=_installed_vulkan_backend, toolset_ready=_managed_vulkan_ready,
    )
    llamacpp_vulkan_found = _vulkan_state["found"]
    LLAMACPP_VULKAN_BENCH_BIN = vulkan_tools["llama-bench"]
    LLAMACPP_VULKAN_BATCHED_BENCH_BIN = vulkan_tools["llama-batched-bench"]
    llamacpp_vulkan_supported = _vulkan_state["supported"]
    _vulkan_note = _vulkan_state["note"]
    if _vulkan_state["problem"]:
        if _vulkan_state["problem"] == "wrong_backend":
            warn(llamacpp_backend_rebuild_warning(
                _installed_vulkan_backend, "vulkan", qualification=False,
            ))
        else:
            warn("llama.cpp Vulkan is missing llama-bench or llama-batched-bench — "
                 "select it to repair the managed toolset")

    # ── 4b. vLLM detection (read-only) ─────────────────────────────────────────────

    section("vLLM")

    VLLM_BIN = find_vllm_binary(
        platform_name=os_name, managed_only=args.qualification == VLLM,
    )
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
        str(_vllm_python) if accessible_file(_vllm_python) else sys.executable)
    missing_python_header = missing_python_headers(_vllm_include_dir)
    missing_header_version = python_version_from_include_dir(_vllm_include_dir) or sys.version_info[:2]
    header_command = next(
        (command for manager in ("apt-get", "dnf", "zypper")
         for command in [python_dev_package_command(manager, missing_header_version)] if command),
        None,
    ) if missing_python_header else None
    header_package = header_command[-1] if header_command else None
    vllm_found = VLLM_BIN is not None or VLLM_LAUNCHER is not None or VLLM_SERVER_URL is not None
    _is_wsl = hardware.detect_wsl(os_name, platform.release())
    _cuda_plan = cuda_toolkit_plan(
        is_wsl=_is_wsl,
        nvidia_ok=nvidia_ok, nvcc_found=find_nvcc() is not None,
    )
    if not _cuda_plan and os_name == "Linux" and not _is_wsl:
        _cuda_plan = native_cuda_toolkit_plan(
            platform.freedesktop_os_release(), platform.machine(),
            nvidia_ok=nvidia_ok, nvcc_found=find_nvcc() is not None,
        )
    if _cuda_plan:
        section("CUDA Toolkit")
        location = "under WSL2" if _is_wsl else "on native Linux"
        warn(f"An NVIDIA GPU is available {location} but nvcc is missing — "
             "llama.cpp would build CPU-only")
        info("Setup can install NVIDIA's driver-free CUDA toolkit package. Needs sudo.")
        for _command in _cuda_plan:
            print(f"      {' '.join(_command)}")
        if args.qualification or input("\n  Install it? [y/N] ").strip().lower().startswith("y"):
            if not run_cuda_toolkit_install(_cuda_plan) or find_nvcc() is None:
                fail("CUDA toolkit installation did not provide nvcc")
                if args.qualification:
                    sys.exit(1)
        else:
            info("Skipped — llama.cpp will build CPU-only")

    vllm_note = (f"server already running at {VLLM_SERVER_URL}" if VLLM_SERVER_URL
                 else f"platform launcher {VLLM_LAUNCHER}" if VLLM_LAUNCHER
                 else f"found at {VLLM_BIN}" if VLLM_BIN else None)
    vllm_support = vllm_platform_support(
        os_name=os_name, machine=platform.machine(), python_version=sys.version_info[:2],
        is_wsl=hardware.detect_wsl(os_name, platform.release()),
        nvidia_ok=nvidia_ok, rocm_ok=rocm_ok, intel_gpu=intel_linux or intel_windows,
        gpu_names=[device["name"] for device in nvidia_gpus],
        compute_cap=nvidia_compute_cap, rocm_version=rocm_version() if rocm_ok else None,
        rocm_gfx_targets=rocm_gfx,
    )
    qualification_needs_native_vllm = args.qualification == VLLM and VLLM_BIN is None
    missing_required_python = resolve_python(
        vllm_support.requires_python, sys.version_info[:2],
    ) is None
    if ((not vllm_found and vllm_support.needs_python_bootstrap)
            or (qualification_needs_native_vllm and missing_required_python)):
        bootstrap_plan = python_bootstrap_plan(
            python_version=sys.version_info[:2],
            requires_python=vllm_support.requires_python,
        )
        if bootstrap_plan:
            python_requirement = (
                f"Python {vllm_support.requires_python[0]}.{vllm_support.requires_python[1]}"
                if vllm_support.requires_python else "Python 3.10–3.13"
            )
            warn(f"vLLM needs {python_requirement} and no matching interpreter was found")
            info("Setup can install a private CPython "
                 f"{PINNED_PYTHON[0]}.{PINNED_PYTHON[1]} for vLLM to build its venv from. "
                 "This downloads uv from astral.sh and does not change your system Python.")
            for command in bootstrap_plan:
                print(f"      {shlex.join(command)}")
            if args.qualification or input("\n  Install it? [y/N] ").strip().lower().startswith("y"):
                if run_python_bootstrap(bootstrap_plan):
                    vllm_support = vllm_platform_support(
                        os_name=os_name, machine=platform.machine(),
                        python_version=sys.version_info[:2],
                        is_wsl=hardware.detect_wsl(os_name, platform.release()),
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
        llamacpp_vulkan_supported=llamacpp_vulkan_supported,
        llamacpp_vulkan_found=llamacpp_vulkan_found,
        llamacpp_vulkan_note=_vulkan_note,
        vllm_note=vllm_note,
    )
    if args.qualification:
        try:
            apply_engine_preset(engine_entries, args.qualification)
        except ValueError as exc:
            _arg_parser.error(str(exc))

    # ── 5. Welcome / prerequisites approval ────────────────────────────────────────

    try:
        import tkinter  # noqa: F401
        _tkinter_available = True
    except ImportError:
        _tkinter_available = False
    try:
        _interface = "terminal" if args.qualification else select_interface_mode(
            args.interface, platform_name=os_name, env=dict(os.environ),
            stdin_is_tty=sys.stdin.isatty(), gui_available=_tkinter_available,
        )
    except ValueError as exc:
        _arg_parser.error(str(exc))

    if _interface != "gui" and not args.qualification:
        select_engines(engine_entries)

    _preview_vulkan_missing = (
        missing_vulkan_build_requirements()
        if os_name == "Linux" and LLAMACPP_VULKAN in selected_engine_names(engine_entries)
        else ()
    )
    _preview_vulkan_plan = vulkan_build_install_plan(_preview_vulkan_missing)

    section("Setup Plan")
    print(f"  {BOLD}local-ai-bench{RESET} needs a few things before it can run benchmarks.\n")
    print("  This will:")
    print("    • Install Python dependencies from requirements.txt")
    if _setup_wsl_rocm_plan:
        print("    • Install AMD ROCm 7.2 for WSL2 (requires sudo)")
    if intel_linux and not intel_linux_runtime:
        print("    • Install Intel GPU compute and oneAPI/SYCL prerequisites (requires sudo)")
    if needs_llamacpp_install and LLAMACPP in selected_engine_names(engine_entries):
        build_note = " (source build — can take several minutes)" if os_name == "Linux" else ""
        print(f"    • Install llama.cpp{build_note}, including llama-bench and llama-batched-bench")
    if LLAMACPP_VULKAN in engines_needing_install(engine_entries):
        build_note = " (source build — can take several minutes)" if os_name == "Linux" else ""
        print(f"    • Install llama.cpp Vulkan{build_note}, including native benchmark tools")
        if _preview_vulkan_missing and _preview_vulkan_plan:
            print("    • Install Vulkan build prerequisites:")
            for _command in _preview_vulkan_plan:
                print(f"      {shlex.join(_command)}")
        elif _preview_vulkan_missing:
            print(f"    • Stop before the Vulkan build: missing {', '.join(_preview_vulkan_missing)}")
    if _detected_comfyui:
        print(f"    • Reuse ComfyUI at {COMFYUI_DIR}")
    if _interface != "gui" and VLLM in engines_needing_install(engine_entries):
        print("    • Install vLLM (several GB, into its own vllm-env/ environment)")
    if _interface != "gui" and needs_python_headers(engine_entries, missing_python_header):
        print("    • Install the Python development headers vLLM needs (requires sudo)")
    print()
    print("  The selected models will install next — everything after that"
          if args.qualification else
          "  You'll then pick which models to install — everything after that")
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
    elif not args.qualification and not confirm("Continue?", default=True):
        print(f"\n  Setup cancelled — nothing was installed.\n")
        sys.exit(0)

    selected_engines = selected_engine_names(engine_entries)
    selected_model_engines = model_engine_names(selected_engines)
    _qualification_vllm_runtime_error = None
    _expected_vllm_device, _expected_vllm_runtime = vllm_runtime_expectations(
        vllm_support.method,
    )
    if args.qualification == VLLM and VLLM_BIN is not None \
            and (config.VLLM_VENV / "bin" / "python").is_file():
        _qualification_vllm_runtime_error = vllm_runtime_import_error(
            config.VLLM_VENV,
            expected_device_type=_expected_vllm_device,
            expected_runtime=_expected_vllm_runtime,
        )
        if _qualification_vllm_runtime_error:
            warn("Managed vLLM runtime failed preflight — setup will rebuild it")
    pending_engines = (
        qualification_engines_needing_install(
            engine_entries, args.qualification, vllm_bench_found=VLLM_BIN is not None,
            vllm_runtime_ready=_qualification_vllm_runtime_error is None,
            llamacpp_runtime_ready=_managed_llamacpp_ready,
        ) if args.qualification else engines_needing_install(engine_entries)
    )
    _vulkan_missing = (
        missing_vulkan_build_requirements()
        if os_name == "Linux" and LLAMACPP_VULKAN in pending_engines else ()
    )
    _vulkan_install_plan = vulkan_build_install_plan(_vulkan_missing)
    _engine_labels = {entry["name"]: entry["label"] for entry in engine_entries}
    ok(f"Engines selected: {', '.join(_engine_labels[name] for name in selected_engines)}")
    for _name in _engine_labels:
        if _name not in selected_engines:
            info(f"{_engine_labels[_name]} not selected — its models and tools are skipped")

    # ── 6. Model selection ──────────────────────────────────────────────────────────

    section("Model Selection")

    if args.qualification:
        selected_llm, selected_images, selected_embed = qualification_model_selection(
            args.qualification,
        )
        cleanup_names, vllm_cleanup_names = [], []
    elif _gui_plan is None:
        selected_llm, selected_images, selected_embed, cleanup_names, vllm_cleanup_names = select_models(
            memory_ceiling_gb, engines=selected_model_engines,
            vllm_cache_home=VLLM_CACHE_HOME, cancel=cancel_setup,
        )
    else:
        _llm_tags = set(_gui_plan["llm_tags"])
        _image_shorts = set(_gui_plan["image_shorts"])
        _embed_tags = set(_gui_plan["embedding_tags"])
        selected_llm = [
            model for tier in (LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE)
            for model in expanded_variant_catalog(tier) if model["tag"] in _llm_tags
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
            comfyui_choice = "" if args.qualification else input(f"  {CYAN}Choose [1/2]:{RESET} ")
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

    if _setup_wsl_rocm_plan:
        info("Installing AMD ROCm 7.2 for WSL2 (requires sudo) ...")
        info(f"Compatible Windows host driver required: {WINDOWS_DRIVER}")
        try:
            run_rocm_install(_setup_wsl_rocm_plan)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            fail(f"ROCm installation failed: {exc}")
            sys.exit(1)
        rocm = discover_rocm()
        if not rocm.available:
            fail("ROCm installed but rocminfo cannot see an AMD GPU")
            fail(f"Install {WINDOWS_DRIVER}, reboot Windows, then rerun setup")
            sys.exit(1)
        rocm_gfx = rocm.gfx_targets
        rocm_gpu_kind = rocm.kind
        rocm_vram_gb = rocm.total_vram_gb
        if rocm.gpus:
            rocm_gpus = rocm.gpus
        ok("ROCm installed and GPU access verified")

    if intel_linux and not intel_linux_runtime:
        try:
            _os_release = Path("/etc/os-release").read_text(encoding="utf-8")
            _intel_plan = intel_xpu_install_plan(
                _os_release, user=os.environ.get("SUDO_USER") or os.environ.get("USER"),
            )
        except (OSError, ValueError) as exc:
            fail(str(exc))
            sys.exit(1)
        else:
            info("Installing Intel GPU compute and oneAPI/SYCL prerequisites ...")
            if run_intel_xpu_install(_intel_plan, log=info):
                ok("Intel XPU prerequisites installed")
                intel_linux_runtime = check_linux_intel_gpu_runtime()
                if not intel_linux_runtime:
                    fail("Intel packages installed, but this kernel/login cannot access the GPU yet")
                    info("Reboot so the HWE kernel and render-group access take effect, then rerun setup")
                    sys.exit(1)
            else:
                fail("Intel XPU prerequisite installation failed")
                sys.exit(1)

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

    _vulkan_prerequisites_ready = True
    if _vulkan_missing:
        if _vulkan_install_plan is None:
            _vulkan_prerequisites_ready = False
            detail = ", ".join(_vulkan_missing)
            fail(f"llama.cpp Vulkan build prerequisites are missing: {detail}")
            issues.append("Install the Vulkan SDK build prerequisites and rerun setup")
        else:
            info(f"Installing llama.cpp Vulkan build prerequisites: {', '.join(_vulkan_missing)}")
            for _command in _vulkan_install_plan:
                info(shlex.join(_command))
            _vulkan_prerequisites_ready = run_vulkan_build_install(_vulkan_install_plan)
            if not _vulkan_prerequisites_ready:
                fail("llama.cpp Vulkan build prerequisite installation failed")
                issues.append("Install the Vulkan SDK build prerequisites and rerun setup")

    if LLAMACPP in pending_engines:
        llamacpp_installed = install_llamacpp()
        if llamacpp_installed:
            ok("llama.cpp installed successfully")
            llamacpp_found = True
            llamacpp_tools = llamacpp_install.find_tools(LLAMACPP_DIR, os_name)
            LLAMACPP_BIN = llamacpp_tools["llama-server"]
            _post_install_probe_env = (
                oneapi_environment() if _required_llamacpp_backend == "xpu" else None
            )
            _post_install_backend_error = llamacpp_backend_error(
                LLAMACPP_BIN, _required_llamacpp_backend, env=_post_install_probe_env,
                context="setup",
            )
            if _post_install_backend_error:
                fail(_post_install_backend_error)
                issues.append(_post_install_backend_error)
                llamacpp_found = False
            LLAMACPP_BENCH_BIN = llamacpp_tools["llama-bench"]
            if LLAMACPP_BENCH_BIN:
                ok(f"llama-bench found: {LLAMACPP_BENCH_BIN}")
            else:
                warn("llama-bench still not found after install — llama-bench-based tests won't be available")
            LLAMACPP_BATCHED_BENCH_BIN = llamacpp_tools["llama-batched-bench"]
            if LLAMACPP_BATCHED_BENCH_BIN:
                ok(f"llama-batched-bench found: {LLAMACPP_BATCHED_BENCH_BIN}")
            else:
                warn("llama-batched-bench still not found after install — the llamabenchconc test won't be available")
        else:
            fail("llama.cpp installation failed")
            issues.append(llamacpp_install_action(
                _required_llamacpp_backend,
                _llamacpp_install_failures[-1] if _llamacpp_install_failures else None,
                LLAMACPP_DIR,
            ))

    if LLAMACPP_VULKAN in pending_engines and _vulkan_prerequisites_ready:
        if install_llamacpp_vulkan():
            vulkan_tools = llamacpp_install.find_tools(LLAMACPP_VULKAN_DIR, os_name)
            LLAMACPP_VULKAN_BIN = vulkan_tools["llama-server"]
            LLAMACPP_VULKAN_BENCH_BIN = vulkan_tools["llama-bench"]
            LLAMACPP_VULKAN_BATCHED_BENCH_BIN = vulkan_tools["llama-batched-bench"]
            _vulkan_backend_error = llamacpp_backend_error(
                LLAMACPP_VULKAN_BIN, "vulkan", context="setup",
            )
            if _vulkan_backend_error:
                fail(_vulkan_backend_error)
                issues.append(_vulkan_backend_error)
            elif not LLAMACPP_VULKAN_BENCH_BIN or not LLAMACPP_VULKAN_BATCHED_BENCH_BIN:
                fail("llama.cpp Vulkan installed without its required benchmark tools")
                issues.append("Repair the managed llama.cpp Vulkan toolset")
            else:
                ok("llama.cpp Vulkan installed successfully")
        else:
            fail("llama.cpp Vulkan installation failed")
            issues.append(llamacpp_install_action(
                "vulkan",
                _llamacpp_install_failures[-1] if _llamacpp_install_failures else None,
                LLAMACPP_VULKAN_DIR,
            ))

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
        if install_vllm(
            vllm_support, log=info,
            recreate=_qualification_vllm_runtime_error is not None,
        ):
            VLLM_BIN = find_vllm_binary(
                platform_name=os_name, managed_only=args.qualification == VLLM,
            )
            if VLLM_BIN:
                ok(f"vLLM installed: {VLLM_BIN}")
                vllm_found = True
            else:
                warn("vLLM install reported success but no 'vllm' executable was found")
        else:
            fail("vLLM installation failed")
            issues.append("Install vLLM manually: https://docs.vllm.ai/en/stable/getting_started/installation/")

    if VLLM in selected_engines and (config.VLLM_VENV / "bin" / "python").is_file():
        if not install_vllm_build_tools(config.VLLM_VENV, log=info):
            fail("vLLM build tool install failed — kernel compilation will fail at run time")
            issues.append(f"Install the vLLM build tools in {config.VLLM_VENV}")
        runtime_error = vllm_runtime_import_error(
            config.VLLM_VENV, expected_device_type=_expected_vllm_device,
            expected_runtime=_expected_vllm_runtime,
        )
        if runtime_error:
            fail("vLLM runtime preflight failed")
            info(runtime_error.splitlines()[-1] if runtime_error else "")
            issues.append("Repair the managed vLLM environment before benchmarking")

    # ── 8a. Disk space ──────────────────────────────────────────────────────────────

    section("Disk Space")

    CHECKPOINTS = config.COMFYUI_MODELS_DIR / "checkpoints"

    def image_asset(name, subdir):
        """Existing path of an image asset, including the pre-4.1 <ComfyUI>/models location."""
        return find_image_asset(name, config.COMFYUI_MODELS_DIR, subdir, COMFYUI_DIR)
    CLIP_DIR    = config.COMFYUI_MODELS_DIR / "clip"
    VAE_DIR     = config.COMFYUI_MODELS_DIR / "vae"

    remaining_gb = 0.0

    all_llm = selected_embed + selected_llm
    for engine in selected_model_engines:
        for m in all_llm:
            if not catalog_model_downloaded(
                m, engine, models_dir=config.MODELS_DIR, vllm_cache=VLLM_CACHE_HOME,
            ):
                remaining_gb += hardware.parse_size_gb(engine_download_size(m, engine) or "")
            if not catalog_mtp_artifact_downloaded(
                m, engine, models_dir=config.MODELS_DIR,
            ):
                remaining_gb += hardware.parse_size_gb(
                    catalog_mtp_artifact_download_size(m, engine) or ""
                )

    remaining_gb += missing_download_size_gb(selected_images, image_asset)

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

    provision_catalog_models(
        selected_embed + selected_llm, selected_model_engines,
        models_dir=config.MODELS_DIR, vllm_cache=VLLM_CACHE_HOME,
        load_token=load_token, issues=issues, info=info, warn=warn, fail=fail, ok=ok,
    )

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
            rocm_version=rocm_version() if rocm_ok else None,
            wsl=hardware.detect_wsl(os_name, platform.release()),
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
                visible = running_comfyui_checkpoints_visible(selected_images, found_ckpts)
                if visible is True:
                    ok("Running ComfyUI already sees Local AI Bench's managed models")
                elif visible is False:
                    warn("Restart running ComfyUI once to load the managed model path")
            else:
                fail("No image checkpoints available — image benchmarks will be skipped")
                issues.append("Download at least one image model through setup")

    # ── 9. Summary ────────────────────────────────────────────────────────────────

    write_setup_config(
        config.SETUP_CONFIG_PATH,
        comfyui_dir=COMFYUI_DIR if (COMFYUI_DIR / "main.py").is_file() else None,
        llamacpp_tools={
            "llama-server": LLAMACPP_BIN,
            "llama-bench": LLAMACPP_BENCH_BIN,
            "llama-batched-bench": LLAMACPP_BATCHED_BENCH_BIN,
        },
        llamacpp_vulkan_tools={
            "llama-server": LLAMACPP_VULKAN_BIN,
            "llama-bench": LLAMACPP_VULKAN_BENCH_BIN,
            "llama-batched-bench": LLAMACPP_VULKAN_BATCHED_BENCH_BIN,
        },
        gpu_devices=(
            [{**device, "vendor": "nvidia", "backend": "cuda"} for device in nvidia_gpus]
            if nvidia_ok else rocm_gpus if rocm_ok else windows_amd_gpus
        ),
        vllm=vllm_setup_config(
            executable=VLLM_BIN, launcher=VLLM_LAUNCHER if VLLM_BIN is None else None,
            server_url=(
                VLLM_SERVER_URL if VLLM_BIN is None and VLLM_LAUNCHER is None else None
            ),
            launcher_extra_args=VLLM_LAUNCHER_ARGS if VLLM_BIN is None else [],
            hf_home=VLLM_CACHE_HOME,
        ) if vllm_found else {},
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

    if qualification_setup_failed(args.qualification, issues):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
