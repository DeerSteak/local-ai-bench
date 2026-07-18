#!/usr/bin/env python3
"""
setup_check.py — Pre-flight verification for LLM benchmark suite.
Run this on each machine before running benchmark.py.

Flow: detect the machine -> show what prerequisites need installing and ask
once -> let the user pick which models to install (numbered list, defaults
to all) -> gather any HuggingFace token needed for the picks -> install
everything with no further prompts.
"""

import argparse
import sys
import os
import platform
import re
import signal
import subprocess
import json
import shutil
import time
from pathlib import Path

import config
import hardware
from models import LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, IMAGE_MODELS, EMBED_MODELS

# Every asset this script manages (requirements.txt, ComfyUI/, hf.txt, ...)
# lives at the repo root, one level up. Sourced from config.py rather than
# redefined here.
SCRIPT_DIR   = config.SCRIPT_DIR
COMFYUI_DIR  = config.COMFYUI_DIR
LLAMACPP_DIR = config.LLAMACPP_DIR

_arg_parser = argparse.ArgumentParser(description="local-ai-bench setup")
_arg_parser.add_argument(
    "--engine", choices=["ollama", "llamacpp", "both"], default="both",
    help="Which inference engine(s) to set up (default: both — llama.cpp is "
         "a light install on top of Ollama, so it's set up unless you opt "
         "out). Ollama is always needed regardless of choice — even "
         "'llamacpp' pulls models through it, see engines/llamacpp.py — this "
         "only controls whether the llama-server binary is also "
         "detected/installed, for benchmark.py --engine llamacpp. Pass "
         "--engine ollama to skip it.",
)
args = _arg_parser.parse_args()
setup_llamacpp = args.engine in ("llamacpp", "both")

# ── Formatting helpers ─────────────────────────────────────────────────────────

GREEN, YELLOW, RED, CYAN, RESET, BOLD = (
    config.GREEN, config.YELLOW, config.RED, config.CYAN, config.RESET, config.BOLD
)

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}→{RESET}  {msg}")
def section(title): print(f"\n{BOLD}{title}{RESET}\n" + "─" * 50)

def link(url, text=None):
    """OSC 8 terminal hyperlink. Terminals without support swallow the escape
    codes as an unrecognized control sequence, leaving just the visible text."""
    return f"\033]8;;{url}\033\\{text or url}\033]8;;\033\\"

INSTALL_STARTED = False  # flipped True once the unattended install phase begins

def cancel_setup(*_args):
    """
    Ctrl+C always means 'get me out' — installed as the SIGINT handler so it
    fires everywhere (mid-subprocess, mid-download), not just at an input()
    prompt. Nothing rolls back partial work, so the message only claims
    "nothing installed" if the install phase hadn't started.
    """
    if INSTALL_STARTED:
        print("\n\n  Setup cancelled — some components may already be partially installed.\n")
    else:
        print("\n\n  Setup cancelled — nothing was installed.\n")
    sys.exit(130)

signal.signal(signal.SIGINT, cancel_setup)

def confirm(prompt, default=True):
    """
    Plain (non-raw) y/n prompt — reads a full line via input(), so it's
    immune to stray keypresses or escape sequences from earlier prompts.
    Defaults to `default` on a bare Enter or a non-interactive/EOF stdin.
    """
    hint = "[Y/n]" if default else "[y/N]"
    try:
        reply = input(f"  {CYAN}{prompt} {hint}{RESET} ").strip().lower()
    except EOFError:
        print()
        return default
    if reply == "":
        return default
    return reply in ("y", "yes")

issues = []

# Approximate download sizes keyed by filename — shown on the model-selection
# screen and used to estimate remaining disk space. Each is the on-disk size
# rounded UP to the next 0.1 GB so the section-8a disk check errs toward
# requiring more free space, not less.
CHECKPOINT_SIZES_GB = {
    "v1-5-pruned-emaonly.safetensors": 4.3,
    "sd_xl_base_1.0.safetensors": 7.0,
    "sd3.5_large.safetensors":    16.5,
    "flux1-dev.safetensors":      23.9,
    "flux2-dev.safetensors":      64.5,
}
ENCODER_SIZES_GB = {
    "t5xxl_fp16.safetensors":               9.8,
    "clip_l.safetensors":                   0.3,
    "clip_g.safetensors":                   1.4,
    "ae.safetensors":                       0.4,
    "flux2-vae.safetensors":                0.4,
    "mistral_3_small_flux2_fp8.safetensors": 18.1,
}
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
os_name = platform.system()
print(f"  OS:       {platform.system()} {platform.release()}")
print(f"  Machine:  {platform.machine()}")
print(f"  Node:     {platform.node()}")

# Populated below per-OS, used later (section 3a) as the memory ceiling for
# Darwin/integrated-GPU/CPU-only machines — None if it couldn't be read.
total_ram_gb = None

if os_name == "Darwin":
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = "unknown"
    print(f"  Chip:     {chip}")
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
        total_ram_gb = mem_bytes / (1024**3)
        print(f"  RAM:      {mem_bytes // (1024**3)} GB")
    except Exception:
        pass

elif os_name == "Linux":
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    total_ram_gb = kb / (1024**2)
                    print(f"  RAM:      {kb // (1024**2)} GB")
                    break
    except Exception:
        pass

elif os_name == "Windows":
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        mem_bytes = int(out.splitlines()[-1].strip())
        total_ram_gb = mem_bytes / (1024**3)
        print(f"  RAM:      {mem_bytes // (1024**3)} GB")
    except Exception:
        pass

# ── 3. GPU / acceleration backend ─────────────────────────────────────────────

section("GPU / Acceleration Backend")

nvidia_vram_gb = 0.0  # sum across GPUs — Ollama can span a model across multiple

def check_nvidia():
    global nvidia_vram_gb
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            print(f"  GPU:     {parts[0]}")
            print(f"  VRAM:    {parts[1]}")
            print(f"  Driver:  {parts[2]}")
            m = re.match(r"([\d.]+)\s*MiB", parts[1])
            if m:
                nvidia_vram_gb += float(m.group(1)) / 1024
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def get_nvidia_compute_cap():
    """Return the GPU's CUDA compute capability (e.g. '12.0'), or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        )
        return out.strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None

rocm_gpu_kind = None  # "discrete" or "integrated", set by check_rocm()
rocm_vram_gb  = None  # only queried for a discrete GPU — see compute_memory_ceiling_gb

def check_rocm():
    global rocm_gpu_kind, rocm_vram_gb
    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.DEVNULL
        )
        agents = [l for l in out.splitlines() if "Marketing Name" in l]
        for a in agents[:3]:
            print(f"  ROCm GPU: {a.split(':', 1)[-1].strip()}")
        if agents:
            name = agents[0].split(":", 1)[-1].strip()
            rocm_gpu_kind = hardware.classify_gpu(name)
            if rocm_gpu_kind == "discrete":
                # An APU's VRAM figure here is often just the small
                # BIOS-fixed carve-out, not the real usable pool via
                # GTT/system-RAM addressing — only trust this for a
                # confirmed-discrete card. Fails open (leaves rocm_vram_gb
                # None) on any missing binary or unexpected output.
                try:
                    mem_out = subprocess.check_output(
                        ["rocm-smi", "--showmeminfo", "vram", "--json"],
                        text=True, stderr=subprocess.DEVNULL,
                    )
                    mem_data = json.loads(mem_out)
                    total_bytes = sum(
                        int(card.get("VRAM Total Memory (B)", 0))
                        for card in mem_data.values()
                    )
                    if total_bytes > 0:
                        rocm_vram_gb = total_bytes / (1024**3)
                except (FileNotFoundError, subprocess.CalledProcessError,
                        json.JSONDecodeError, ValueError):
                    pass
        return bool(agents)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def check_metal():
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True
        )
        if "Metal" in result or "Apple" in result:
            for line in result.splitlines():
                if "Chipset Model" in line or "Metal" in line:
                    print(f"  {line.strip()}")
            return True
    except Exception:
        pass
    return False

windows_gpu_kind = None  # "discrete" or "integrated", set by check_windows_gpu()

def check_windows_gpu():
    """Detect GPU vendor on Windows via PowerShell. Returns 'amd', 'intel', or None."""
    global windows_gpu_kind
    if platform.system() != "Windows":
        return None

    names = []
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            text=True, stderr=subprocess.DEVNULL,
        )
        names = [n.strip() for n in out.splitlines() if n.strip()]
    except Exception:
        pass

    for name in names:
        if "AMD" in name or "Radeon" in name:
            print(f"  GPU:     {name}")
            windows_gpu_kind = hardware.classify_gpu(name)
            return "amd"
        if "Intel" in name and "Arc" in name:
            print(f"  GPU:     {name}")
            windows_gpu_kind = hardware.classify_gpu(name)
            return "intel"

    return None

linux_intel_gpu_kind = None  # "discrete" or "integrated", set by check_linux_intel_gpu()

def check_linux_intel_gpu():
    """Detect an Intel Arc GPU on Linux via lspci. Detection/labeling only —
    unlike the AMD/NVIDIA paths it unlocks no GPU-accelerated install path
    here: whether LLM tests use the GPU depends on the user's own Ollama
    version (native Intel SYCL support landed in v0.17, Feb 2026 —
    https://github.com/ollama/ollama/pull/11160), not anything this script
    installs. Requires 'Arc' in the name (not just 'Intel') so integrated
    graphics with no discrete acceleration aren't misreported."""
    global linux_intel_gpu_kind
    if platform.system() != "Linux":
        return False
    try:
        out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if (any(k in line for k in ("VGA", "3D controller", "Display"))
                    and "Intel" in line and "Arc" in line):
                name = line.split(":", 2)[-1].strip()
                print(f"  GPU:     {name}")
                linux_intel_gpu_kind = hardware.classify_gpu(name)
                return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return False

# Intel's Level Zero / OpenCL GPU compute runtime — the actual missing piece
# for XPU-accelerated PyTorch/ComfyUI on Linux, distinct from the GPU merely
# appearing in lspci. Packages from Intel's install guide
# (https://dgpu-docs.intel.com/driver/installation.html), via a third-party
# APT repo this script deliberately does NOT add itself (see
# check_linux_intel_gpu_runtime()).
INTEL_GPU_RUNTIME_PACKAGES = ("intel-opencl-icd", "intel-level-zero-gpu", "level-zero")

def check_linux_intel_gpu_runtime():
    """Check whether Intel's GPU compute runtime is installed on Linux, via
    dpkg (Debian/Ubuntu). Detection-only: installing it means adding a
    third-party APT repo + GPG key, a more invasive, harder-to-reverse change
    than the plain-package installs this script automates — so it tells the
    user the commands to run themselves rather than modifying apt sources
    unattended."""
    if platform.system() != "Linux" or not shutil.which("dpkg"):
        return False
    for pkg in INTEL_GPU_RUNTIME_PACKAGES:
        result = subprocess.run(["dpkg", "-s", pkg],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return False
    return True

nvidia_ok           = check_nvidia()
nvidia_compute_cap  = get_nvidia_compute_cap() if nvidia_ok else None
rocm_ok             = False
metal_ok            = False
amd_windows         = False
intel_windows       = False
intel_linux         = False
intel_linux_runtime = False

if not nvidia_ok:
    rocm_ok = check_rocm()
if not nvidia_ok and not rocm_ok:
    metal_ok = check_metal()
if not nvidia_ok and os_name == "Windows":
    _win_vendor   = check_windows_gpu()
    amd_windows   = _win_vendor == "amd"
    intel_windows = _win_vendor == "intel"
if not nvidia_ok and not rocm_ok and os_name == "Linux":
    intel_linux = check_linux_intel_gpu()
    if intel_linux:
        intel_linux_runtime = check_linux_intel_gpu_runtime()

if nvidia_ok:
    ok("CUDA / Nvidia GPU detected")
elif rocm_ok:
    ok("ROCm / AMD GPU detected")
elif amd_windows:
    ok("AMD/Radeon GPU detected on Windows")
elif intel_windows:
    ok("Intel Arc GPU detected on Windows")
    info("Intel Arc support is experimental — this project's maintainers don't have "
         "Arc hardware to test against, so the Ollama version check below is unverified")
    # Whether Ollama actually accelerates on this GPU depends on its version —
    # reported precisely against the installed Ollama below, once it's found.
elif intel_linux:
    ok("Intel Arc GPU detected on Linux")
    info("Intel Arc support is experimental — this project's maintainers don't have "
         "Arc hardware to test against, so everything below (runtime check, XPU "
         "PyTorch install, Ollama version check) is unverified. Please report back "
         "if you try it: https://github.com/DeerSteak/local-ai-bench/issues")
    # Same as above for Ollama — see report_gpu_acceleration_status() below.
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
# How much memory a model can realistically use on this machine — VRAM for a
# confirmed-discrete GPU, total system RAM for everything else (Apple Silicon
# unified memory, integrated GPUs, or no GPU at all). Used below to default
# models that clearly won't fit to unchecked in the picker — informational,
# not a hard block (see hardware.py).

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
)
if memory_ceiling_gb is not None:
    ok(f"Model memory ceiling: {memory_ceiling_note}")
else:
    warn(memory_ceiling_note)

# ── 4. Ollama detection (read-only — nothing is installed or started here) ────

section("Ollama")

def parse_ollama_version(ver_string):
    """Extract a (major, minor, patch) tuple from an 'ollama version X.Y.Z'
    string (or any string containing a dotted version number). Returns None
    if no version number can be found, e.g. malformed --version output."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", ver_string)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())

# Ollama's native Intel GPU (SYCL) backend landed in this release — an Intel
# Arc GPU needs at least this version for LLM tests to run on the GPU instead
# of silently falling back to CPU. Source:
# https://github.com/ollama/ollama/pull/11160
OLLAMA_INTEL_SYCL_MIN_VERSION = (0, 17, 0)

def report_gpu_acceleration_status(ollama_version):
    """Print whether the already-detected GPU backend is actually accelerated
    by the installed Ollama version. NVIDIA/AMD/Metal acceleration isn't
    version-gated for any Ollama version users are expected to have, so those
    report unconditionally; Intel Arc is the one case that depends on
    OLLAMA_INTEL_SYCL_MIN_VERSION. Prints nothing if no GPU was detected."""
    if nvidia_ok:
        ok("GPU acceleration: CUDA/NVIDIA — supported")
    elif rocm_ok or amd_windows:
        ok("GPU acceleration: ROCm/AMD — supported")
    elif metal_ok:
        ok("GPU acceleration: Apple Metal — supported")
    elif intel_windows or intel_linux:
        min_str = ".".join(map(str, OLLAMA_INTEL_SYCL_MIN_VERSION))
        if ollama_version is None:
            warn(f"GPU acceleration: Intel Arc — could not determine Ollama version; "
                 f"native SYCL support requires >= {min_str} "
                 "(https://github.com/ollama/ollama/pull/11160)")
        elif ollama_version >= OLLAMA_INTEL_SYCL_MIN_VERSION:
            ver_str = ".".join(map(str, ollama_version))
            ok(f"GPU acceleration: Intel Arc — supported (Ollama {ver_str} has native SYCL support)")
        else:
            ver_str = ".".join(map(str, ollama_version))
            warn(f"GPU acceleration: Intel Arc — NOT supported by Ollama {ver_str} "
                 f"(native SYCL support requires >= {min_str}, added Feb 2026: "
                 "https://github.com/ollama/ollama/pull/11160). LLM tests will run on "
                 "CPU until you update Ollama — don't install IPEX-LLM instead, Intel "
                 "archived that repo in Jan 2026 citing security issues.")

def ollama_running():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}

def find_ollama_binary():
    """
    Return the path to the ollama binary, or None if not found.
    On Windows, Ollama installs to %LOCALAPPDATA%\\Programs\\Ollama which is
    not always on the subprocess PATH even when it works in PowerShell.
    """
    found = shutil.which("ollama")
    if found:
        return found
    if os_name == "Windows":
        candidates = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
    return None

def ollama_pull(tag, ollama_bin="ollama"):
    """Pull a model via ollama CLI, streaming progress to stdout."""
    print(f"  Pulling {tag} ...")
    result = subprocess.run([ollama_bin, "pull", tag])
    return result.returncode == 0

def install_ollama():
    """Install Ollama using the appropriate method for this OS."""
    if os_name == "Darwin":
        if shutil.which("brew"):
            info("Installing Ollama via Homebrew ...")
            # --no-ask / HOMEBREW_NO_ASK skips brew's "proceed? [y/n]"
            # confirmation (NOT what NONINTERACTIVE controls — that only covers
            # brew's installer script and sudo). We already got consent in the
            # prerequisites screen.
            result = subprocess.run(
                ["brew", "install", "--no-ask", "ollama"],
                env={**os.environ, "HOMEBREW_NO_ASK": "1", "NONINTERACTIVE": "1"},
            )
            return result.returncode == 0
        else:
            fail("Homebrew not found — install Ollama manually from https://ollama.com/download")
            return False

    elif os_name == "Linux":
        if shutil.which("snap"):
            info("Installing Ollama via snap ...")
            result = subprocess.run(["sudo", "snap", "install", "ollama"])
            if result.returncode == 0:
                time.sleep(3)
                return True
        info("Installing Ollama via official install script ...")
        result = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True
        )
        return result.returncode == 0

    elif os_name == "Windows":
        if shutil.which("winget"):
            info("Installing Ollama via winget ...")
            result = subprocess.run([
                "winget", "install", "Ollama.Ollama",
                "--silent", "--accept-package-agreements", "--accept-source-agreements"
            ])
            if result.returncode == 0:
                time.sleep(5)
                return True
        else:
            fail("winget not found — install Ollama manually from https://ollama.com/download")
        return False

    return False

def find_llamacpp_binary():
    """Same resolution LlamaCppEngine uses at runtime (see its _binary_path):
    vendored under LLAMACPP_DIR first (source build on Linux, prebuilt zip on
    Windows), then PATH (the Homebrew install path on macOS, or a manual
    install anywhere), then — if PATH doesn't have it — the two well-known
    Homebrew prefixes directly. A brew install only updates PATH in the shell
    that ran it (and only for shells started after its rc-file was touched),
    so a symlink that brew just created may not be on PATH yet in whatever
    shell re-runs this script later; checking the fixed prefixes means this
    never depends on the terminal being restarted or an rc file re-sourced."""
    exe_name = "llama-server.exe" if os_name == "Windows" else "llama-server"
    if LLAMACPP_DIR.exists():
        match = next(iter(LLAMACPP_DIR.rglob(exe_name)), None)
        if match is not None:
            return str(match)
    found = shutil.which("llama-server")
    if found:
        return found
    if os_name == "Darwin":
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            candidate = Path(prefix) / exe_name
            if candidate.exists():
                return str(candidate)
    return None

def download_llamacpp_windows():
    """Download the latest llama.cpp Windows release and extract it into
    LLAMACPP_DIR. Picks the Vulkan-backend build specifically: it runs on
    NVIDIA/AMD/Intel GPUs alike without having to match a specific CUDA
    toolkit version to whatever's installed, which is the safer bet for an
    unattended install (a CUDA-specific asset that doesn't match the user's
    toolkit would simply fail to load its driver at runtime)."""
    import urllib.request
    import zipfile

    info("Fetching latest llama.cpp release info ...")
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            release = json.load(r)
        asset = next(
            (a for a in release["assets"]
             if "win-vulkan-x64" in a["name"].lower() and a["name"].endswith(".zip")),
            None,
        )
        if not asset:
            fail("No Windows Vulkan build found in the latest llama.cpp release")
            return False
        url  = asset["browser_download_url"]
        size = asset["size"] // (1024 ** 2)
        tag  = release["tag_name"]
    except Exception as e:
        fail(f"Could not fetch llama.cpp release info: {e}")
        return False

    info(f"Downloading llama.cpp {tag} (Vulkan, {size} MB) ...")
    tmp = SCRIPT_DIR / asset["name"]
    try:
        urllib.request.urlretrieve(url, str(tmp))
    except Exception as e:
        fail(f"Download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False

    info(f"Extracting {asset['name']} ...")
    try:
        LLAMACPP_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(LLAMACPP_DIR)
        tmp.unlink()
    except Exception as e:
        fail(f"Extraction failed: {e}")
        tmp.unlink(missing_ok=True)
        return False

    if not any(LLAMACPP_DIR.rglob("llama-server.exe")):
        fail(f"Extracted {asset['name']} but llama-server.exe wasn't found inside it")
        return False
    ok(f"llama.cpp {tag} (Vulkan) extracted to {LLAMACPP_DIR}")
    return True

def install_llamacpp():
    """Install llama-server using the appropriate method for this OS. Picks a
    GPU backend using the same detection this script already ran for Ollama
    (nvidia_ok/rocm_ok), so it's accelerated the same way — covers DGX Spark
    the same as any other Linux+NVIDIA box, since a source build has no
    prebuilt-binary architecture to match (Spark is ARM64)."""
    if os_name == "Darwin":
        if shutil.which("brew"):
            info("Installing llama.cpp via Homebrew (includes Metal support) ...")
            result = subprocess.run(
                ["brew", "install", "--no-ask", "llama.cpp"],
                env={**os.environ, "HOMEBREW_NO_ASK": "1", "NONINTERACTIVE": "1"},
            )
            return result.returncode == 0
        fail("Homebrew not found — install llama.cpp manually: https://github.com/ggml-org/llama.cpp")
        return False

    elif os_name == "Linux":
        if not shutil.which("git") or not shutil.which("cmake"):
            fail("git and cmake are required to build llama.cpp from source — "
                 "install them (e.g. sudo apt install git cmake build-essential) and re-run")
            return False

        cmake_flags = []
        if nvidia_ok:
            if not shutil.which("nvcc"):
                warn("NVIDIA GPU detected but the CUDA toolkit (nvcc) isn't installed — "
                     "building CPU-only. Install the CUDA toolkit and re-run for GPU support.")
            else:
                info("Building with CUDA support ...")
                cmake_flags.append("-DGGML_CUDA=ON")
        elif rocm_ok:
            info("Building with ROCm/HIP support ...")
            cmake_flags += ["-DGGML_HIP=ON"]
        else:
            info("No GPU backend detected — building CPU-only ...")

        if LLAMACPP_DIR.exists():
            info("Updating existing llama.cpp checkout ...")
            pull = subprocess.run(["git", "pull"], cwd=str(LLAMACPP_DIR))
            if pull.returncode != 0:
                warn("git pull failed — building from the existing checkout as-is")
        else:
            info("Cloning llama.cpp ...")
            clone = subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/ggml-org/llama.cpp", str(LLAMACPP_DIR),
            ])
            if clone.returncode != 0:
                fail("git clone failed")
                return False

        build_dir = LLAMACPP_DIR / "build"
        info(f"Configuring build ({' '.join(cmake_flags) or 'CPU-only'}) ...")
        configure = subprocess.run(
            ["cmake", "-B", str(build_dir), "-S", str(LLAMACPP_DIR)] + cmake_flags
        )
        if configure.returncode != 0:
            fail("cmake configure failed")
            return False

        info("Building llama-server — this can take several minutes ...")
        build = subprocess.run([
            "cmake", "--build", str(build_dir), "--target", "llama-server",
            "--config", "Release", "-j",
        ])
        if build.returncode != 0:
            fail("Build failed")
            return False

        if not any(build_dir.rglob("llama-server")):
            fail(f"Build finished but llama-server wasn't found under {build_dir}")
            return False
        return True

    elif os_name == "Windows":
        return download_llamacpp_windows()

    return False

ollama_up, tag_data = ollama_running()
OLLAMA_BIN = find_ollama_binary()
ollama_found = OLLAMA_BIN is not None

if ollama_found:
    ollama_version = None
    try:
        ver_out = subprocess.check_output(
            [OLLAMA_BIN, "--version"], text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        ver_line = next(
            (l for l in ver_out.splitlines() if "ollama version" in l.lower()),
            ver_out.splitlines()[0] if ver_out else "unknown"
        )
        print(f"  Binary:  {ver_line.strip()}")
        ollama_version = parse_ollama_version(ver_line)
        ok("Ollama binary found")
    except Exception:
        ok("Ollama binary found")
    report_gpu_acceleration_status(ollama_version)
else:
    warn("Ollama not found in PATH — will need to be installed")

if ollama_up:
    ok("Ollama server is running (port 11434)")
else:
    warn("Ollama server not running — will need to be started")

needs_ollama_install = not ollama_found
needs_ollama_start   = not ollama_up

# ── 4a. llama.cpp detection (opt-in via --engine llamacpp/both; read-only) ────

LLAMACPP_BIN = None
llamacpp_found = False
needs_llamacpp_install = False

if setup_llamacpp:
    section("llama.cpp")
    LLAMACPP_BIN = find_llamacpp_binary()
    llamacpp_found = LLAMACPP_BIN is not None
    needs_llamacpp_install = not llamacpp_found
    if llamacpp_found:
        ok(f"llama-server found: {LLAMACPP_BIN}")
    else:
        warn("llama-server not found — will need to be installed")
    info("llama.cpp reuses models pulled via Ollama (see engines/llamacpp.py) "
         "— Ollama itself is still required regardless of --engine")

# ── 5. Welcome / prerequisites approval ────────────────────────────────────────

section("Setup Plan")
print(f"  {BOLD}local-ai-bench{RESET} needs a few things before it can run benchmarks.\n")
print("  This will:")
print("    • Install Python dependencies from requirements.txt")
if needs_ollama_install:
    print("    • Install Ollama")
if needs_ollama_start:
    print("    • Start the Ollama server")
if needs_llamacpp_install:
    build_note = " (source build — can take several minutes)" if os_name == "Linux" else ""
    print(f"    • Install llama.cpp{build_note}")
print()
print("  You'll then pick which models to install — everything after that")
print("  runs on its own, with no further prompts.")
print()

if not confirm("Continue?", default=True):
    print(f"\n  Setup cancelled — nothing was installed.\n")
    sys.exit(0)

# ── 6. Model selection ──────────────────────────────────────────────────────────

section("Model Selection")

def select_models(memory_ceiling_gb=None):
    """
    Flat numbered list spanning every LLM tier, embeddings, and image models,
    checked by default; returns (selected_llm, selected_images,
    selected_embed). The toggle syntax is printed to the user below. Plain
    input() only — no raw terminal mode — so stray keys from earlier prompts
    can't leak in and there's nothing to restore/flush.

    An LLM model that hardware.model_fits() says won't fit in
    memory_ceiling_gb starts unchecked instead, with a note on its line
    explaining why — informational, not a hard block, since a model that's
    merely tight (MoE expert offload, a little KV-cache spillover into
    system RAM) can still be worth trying. memory_ceiling_gb=None (couldn't
    be determined) means no filtering — every LLM model still starts checked.
    """
    TIER_KEYS = {"xs": "xsmall", "s": "small", "m": "medium", "l": "large"}
    groups = [
        ("LLM — Extra-small tier (<6B params)", LLM_MODELS_XSMALL, "llm",   "xs"),
        ("LLM — Small tier (≤20B params)",   LLM_MODELS_SMALL,  "llm",   "s"),
        ("LLM — Medium tier (26–35B params)", LLM_MODELS_MEDIUM, "llm",   "m"),
        ("LLM — Large tier (70B+ params)",   LLM_MODELS_LARGE,  "llm",   "l"),
        ("Embeddings models",                 EMBED_MODELS,      "embed", "emb"),
        ("Image generation models",           IMAGE_MODELS,      "image", "img"),
    ]
    group_keys = {g[3] for g in groups}
    entries = []
    for _, items, kind, group_key in groups:
        # LLM groups are already one-per-tier; image models carry their own
        # "tier" field (see models.py) so xs/s/m/l can reach them too.
        tier = TIER_KEYS.get(group_key) if kind == "llm" else None
        for m in items:
            entry_tier = tier if kind == "llm" else m.get("tier")
            fits = (hardware.model_fits(m["download_size"], memory_ceiling_gb)
                    if kind == "llm" else True)
            entries.append({"item": m, "kind": kind, "group": group_key,
                            "tier": entry_tier, "checked": fits is not False,
                            "fits": fits})

    def size_label(e, m, kind):
        if kind == "embed":
            return f"  ({m['download_size']})"
        if kind == "llm":
            label = f"  ({m['download_size']})"
            if e["fits"] is False:
                needed = hardware.model_memory_requirement_gb(m["download_size"])
                label += f"  {YELLOW}⚠ needs ~{needed:.1f} GB, ~{memory_ceiling_gb:.1f} GB available{RESET}"
            return label
        gb = CHECKPOINT_SIZES_GB.get(m["checkpoint"])
        return f"  (~{gb:.1f} GB)" if gb else ""

    def render():
        header_note = ("all selected by default" if memory_ceiling_gb is None
                        else "selected by default, except LLMs that likely won't fit in memory")
        print(f"  {BOLD}Choose which models to install ({header_note}){RESET}")
        n = 1
        for header, items, kind, group_key in groups:
            if not items:
                continue
            print(f"  {CYAN}{header} [{group_key}]{RESET}")
            for m in items:
                e = entries[n - 1]
                box = "[x]" if e["checked"] else "[ ]"
                print(f"    {box} {n:>2}  {m['label']}{size_label(e, m, kind)}")
                n += 1
            print()

    render()
    print("  Type numbers to toggle (e.g. '2 4 7-9'), a size tier (xs/s/m/l — LLM")
    print("  and image checkpoints together) or 'emb'/'img' to toggle a whole")
    print("  section, 'a' to select/deselect all, 'q' to cancel,")
    while True:
        try:
            raw = input("  or press Enter to install everything checked above: ").strip().lower()
        except EOFError:
            print()
            break
        if raw == "":
            break
        if raw in ("q", "quit", "cancel"):
            cancel_setup()
        if raw in ("a", "all"):
            all_checked = all(e["checked"] for e in entries)
            for e in entries:
                e["checked"] = not all_checked
            print()
            render()
            continue
        if raw in TIER_KEYS:
            matching = [e for e in entries if e["tier"] == TIER_KEYS[raw]]
            all_checked = all(e["checked"] for e in matching)
            for e in matching:
                e["checked"] = not all_checked
            print()
            render()
            continue
        if raw in group_keys:
            matching = [e for e in entries if e["group"] == raw]
            all_checked = all(e["checked"] for e in matching)
            for e in matching:
                e["checked"] = not all_checked
            print()
            render()
            continue

        nums = set()
        valid = True
        for tok in raw.replace(",", " ").split():
            if "-" in tok:
                a, b = tok.split("-", 1)
                if a.isdigit() and b.isdigit():
                    nums.update(range(int(a), int(b) + 1))
                else:
                    valid = False
                    break
            elif tok.isdigit():
                nums.add(int(tok))
            else:
                valid = False
                break
        if not valid or not nums or any(x < 1 or x > len(entries) for x in nums):
            warn("Couldn't parse that — use numbers/ranges like '2 4 7-9', 'a', or Enter to continue")
            continue

        for x in nums:
            entries[x - 1]["checked"] = not entries[x - 1]["checked"]
        print()
        render()

    selected_llm    = [e["item"] for e in entries if e["checked"] and e["kind"] == "llm"]
    selected_images = [e["item"] for e in entries if e["checked"] and e["kind"] == "image"]
    selected_embed  = [e["item"] for e in entries if e["checked"] and e["kind"] == "embed"]
    return selected_llm, selected_images, selected_embed

selected_llm, selected_images, selected_embed = select_models(memory_ceiling_gb)
selected_llm_tags     = {m["tag"] for m in selected_llm}
selected_image_shorts = {m["short"] for m in selected_images}

print()
info(f"LLM models selected: {len(selected_llm)}/{len(LLM_MODELS_XSMALL) + len(LLM_MODELS_SMALL) + len(LLM_MODELS_MEDIUM) + len(LLM_MODELS_LARGE)}")
info(f"Image models selected: {len(selected_images)}/{len(IMAGE_MODELS)}")
info(f"Embeddings models selected: {len(selected_embed)}/{len(EMBED_MODELS)}")

# ── 7. HuggingFace token (only if a selected image model needs one) ───────────

_hf_token_cache = [None]

def load_token():
    """Load HF token from env var, hf.txt, or prompt — cached after first load."""
    if _hf_token_cache[0] is not None:
        return _hf_token_cache[0]
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        ok("HuggingFace token loaded from HF_TOKEN env var")
        _hf_token_cache[0] = token
        return token
    hf_txt = SCRIPT_DIR / "hf.txt"
    if hf_txt.exists():
        token = hf_txt.read_text().strip()
        if token:
            ok("HuggingFace token loaded from hf.txt")
            _hf_token_cache[0] = token
            return token
    needs_gated = bool(GATED_IMAGE_SHORTS & selected_image_shorts)
    print()
    if needs_gated:
        print(f"  {YELLOW}SD3.5 Large, Flux.1-dev, and Flux.2-dev require a free HuggingFace account.{RESET}")
        print(f"  1. Create an account at {link('https://huggingface.co')}")
        print(f"  2. Accept the licenses at:")
        print(f"       {link('https://huggingface.co/stabilityai/stable-diffusion-3.5-large')}")
        print(f"       {link('https://huggingface.co/black-forest-labs/FLUX.1-dev')}")
        print(f"       {link('https://huggingface.co/black-forest-labs/FLUX.2-dev')}")
        print(f"  3. Generate a token at {link('https://huggingface.co/settings/tokens')}")
    else:
        print(f"  {CYAN}A free HuggingFace token isn't required for the models you selected,{RESET}")
        print(f"  {CYAN}but HuggingFace gives token holders faster downloads.{RESET}")
        print(f"  Generate one (optional) at {link('https://huggingface.co/settings/tokens')}")
    print()
    try:
        skip_hint = "skip gated models" if needs_gated else "skip and download without one"
        token = input(
            f"  {CYAN}Paste your HuggingFace token and press Enter{RESET}\n  (or press Enter to {skip_hint}): "
        ).strip()
    except EOFError:
        token = ""
    if token:
        try:
            save = input("  Save token to hf.txt for future runs? [y/N]: ").strip().lower()
            if save == "y":
                (SCRIPT_DIR / "hf.txt").write_text(token)
                ok("Token saved to hf.txt")
        except EOFError:
            pass
    _hf_token_cache[0] = token or ""
    return token

if selected_image_shorts:
    section("HuggingFace Token")
    load_token()

# ── 8. Installing — everything below runs unattended, no more prompts ─────────

INSTALL_STARTED = True

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
    issues.append("pip install -r requirements.txt")

if needs_ollama_install:
    installed = install_ollama()
    if installed:
        ok("Ollama installed successfully")
        ollama_found = True
        OLLAMA_BIN = find_ollama_binary()
    else:
        fail("Ollama installation failed")
        issues.append("Install Ollama manually from https://ollama.com/download")
        info("On Linux (DGX Spark / Ubuntu): sudo snap install ollama")

if needs_llamacpp_install:
    llamacpp_installed = install_llamacpp()
    if llamacpp_installed:
        ok("llama.cpp installed successfully")
        llamacpp_found = True
        LLAMACPP_BIN = find_llamacpp_binary()
    else:
        fail("llama.cpp installation failed")
        issues.append("Install llama.cpp manually: https://github.com/ggml-org/llama.cpp "
                       "(needs a 'llama-server' binary on PATH, or built under "
                       f"{LLAMACPP_DIR})")

if needs_ollama_start and ollama_found:
    info("Starting Ollama server ...")
    try:
        if os_name == "Windows":
            subprocess.Popen(
                [OLLAMA_BIN or "ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            subprocess.Popen(
                [OLLAMA_BIN or "ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for _ in range(15):
            time.sleep(1)
            ollama_up, tag_data = ollama_running()
            if ollama_up:
                ok("Ollama started successfully")
                break
        else:
            fail("Ollama did not respond within 15 seconds")
            issues.append("Start Ollama manually: ollama serve")
    except FileNotFoundError:
        fail("Ollama binary not found — cannot start server")
        issues.append("Install Ollama from https://ollama.com/download")
elif needs_ollama_start and not ollama_found:
    issues.append("Start Ollama manually: ollama serve")

# ── 8a. Disk space ──────────────────────────────────────────────────────────────

section("Disk Space")

CHECKPOINTS = COMFYUI_DIR / "models" / "checkpoints"
CLIP_DIR    = COMFYUI_DIR / "models" / "clip"
VAE_DIR     = COMFYUI_DIR / "models" / "vae"

remaining_gb = 0.0

all_llm = selected_embed + selected_llm
if ollama_up:
    already_pulled = {m["name"] for m in tag_data.get("models", [])}
    for m in all_llm:
        tag = m["tag"]
        if not (tag in already_pulled or any(tag in a for a in already_pulled)):
            remaining_gb += hardware.parse_size_gb(m["download_size"])
else:
    for m in all_llm:
        remaining_gb += hardware.parse_size_gb(m["download_size"])

sd35_selected  = "sd35-large" in selected_image_shorts
flux1_selected = "flux-dev" in selected_image_shorts
flux2_selected = "flux2-dev" in selected_image_shorts

for m in selected_images:
    ckpt_path = CHECKPOINTS / m["checkpoint"]
    if not ckpt_path.exists():
        remaining_gb += CHECKPOINT_SIZES_GB.get(m["checkpoint"], 0.0)

# Shared T5-XXL + CLIP-L text encoders: used by SD3.5 Large and Flux.1-dev,
# NOT Flux.2-dev (which has its own Mistral-based encoder below).
if (sd35_selected or flux1_selected):
    for fname in ("t5xxl_fp16.safetensors", "clip_l.safetensors"):
        if not (CLIP_DIR / fname).exists():
            remaining_gb += ENCODER_SIZES_GB[fname]
if sd35_selected and not (CLIP_DIR / "clip_g.safetensors").exists():
    remaining_gb += ENCODER_SIZES_GB["clip_g.safetensors"]
if flux1_selected and not (VAE_DIR / "ae.safetensors").exists():
    remaining_gb += ENCODER_SIZES_GB["ae.safetensors"]
if flux2_selected:
    text_encoder_dir = COMFYUI_DIR / "models" / "text_encoders"
    if not (text_encoder_dir / "mistral_3_small_flux2_fp8.safetensors").exists():
        remaining_gb += ENCODER_SIZES_GB["mistral_3_small_flux2_fp8.safetensors"]
    if not (VAE_DIR / "flux2-vae.safetensors").exists():
        remaining_gb += ENCODER_SIZES_GB["flux2-vae.safetensors"]

try:
    check_path = "C:\\" if os_name == "Windows" else "/"
    total, used, free = shutil.disk_usage(check_path)
    free_gb  = free  // (1024**3)
    total_gb = total // (1024**3)
    print(f"  Free:              {free_gb} GB / {total_gb} GB total")
    if remaining_gb > 0:
        print(f"  Still to download: ~{remaining_gb:.0f} GB")
    def _warn_if_drive_fills_up():
        # Separate from the absolute-GB checks above: even with room for the
        # downloads, warn if so little would be left afterward that the drive
        # gets uncomfortably full. Informational only — doesn't block or add
        # to `issues`.
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
        needed_more = remaining_gb - free_gb
        fail(f"Insufficient space — ~{remaining_gb:.0f} GB needed, only {free_gb} GB free")
        issues.append(f"Free up ~{needed_more:.0f} GB more disk space before downloading models")
except Exception as e:
    warn(f"Could not check disk space: {e}")

# ── 8b. Ollama models — pull selected, skip the rest ──────────────────────────

section("Ollama Models")

deselected_llm = [
    m for tier in (LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE)
    for m in tier if m["tag"] not in selected_llm_tags
]
for m in deselected_llm:
    info(f"{m['label']} — skipped (not selected)")

if ollama_up:
    available = {m["name"] for m in tag_data.get("models", [])}
    all_models = selected_embed + selected_llm
    for m in all_models:
        tag, label, size = m["tag"], m["label"], m["download_size"]
        already = tag in available or any(tag in a for a in available)
        if already:
            ok(f"{label} — already pulled")
        else:
            warn(f"{label} ({size}) — not found, pulling now ...")
            success = ollama_pull(tag, ollama_bin=OLLAMA_BIN or "ollama")
            if success:
                ok(f"{label} — pulled successfully")
            else:
                fail(f"{label} — pull failed")
                issues.append(f"ollama pull {tag}")
else:
    for m in selected_embed + selected_llm:
        warn(f"Cannot check {m['tag']} — Ollama server not running")
        issues.append(f"ollama pull {m['tag']}  (once Ollama is running)")

# ── 8c. ComfyUI — only if at least one image model was selected ───────────────

if not selected_images:
    section("ComfyUI")
    info("No image models selected — skipping ComfyUI/image setup")
else:
    section("ComfyUI")

    PORTABLE_PYTHON = SCRIPT_DIR / "python_embeded" / "python.exe"
    nvidia_windows  = nvidia_ok and os_name == "Windows"

    def check_and_fix_torch_cuda_arch(python_exe, compute_cap):
        """
        ComfyUI's Windows portable build bundles a pinned torch wheel. New GPU
        architectures (e.g. Blackwell / RTX 50-series, compute capability 12.0)
        aren't recognized by older cu12x wheels, so every CUDA kernel launch
        fails with "no kernel image is available for execution on the device"
        (first seen during CLIP text encoding in warmup). Compare the GPU's
        compute capability against the bundled torch build and reinstall from
        the cu128 wheel index if the architecture isn't listed.
        """
        if not compute_cap:
            return
        major, minor = compute_cap.split(".")
        sm = f"sm_{major}{minor}"
        check_script = "import torch; print(','.join(torch.cuda.get_arch_list()))"
        try:
            out = subprocess.check_output(
                [str(python_exe), "-c", check_script],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception as e:
            warn(f"Could not check torch CUDA architecture support: {e}")
            return
        arch_list = out.split(",") if out else []
        if sm in arch_list:
            ok(f"torch build supports {sm} (GPU compute capability {compute_cap})")
            return

        warn(f"torch build does not support {sm} (GPU compute capability {compute_cap}) "
             f"— reinstalling torch with Blackwell-compatible (cu128) wheels ...")
        # --force-reinstall is required, not just --upgrade: pip treats an
        # already-installed torch+cu126 as satisfying a bare "torch" requirement
        # and skips it, while torchvision/torchaudio get swapped to +cu128 —
        # leaving a mismatched trio torchaudio's _check_cuda_version() refuses to
        # import. Forcing reinstall keeps all three in lockstep. The wheels are
        # ~800MB-2GB each, so stream pip's progress live instead of capturing
        # silently, which would look hung for minutes.
        proc = subprocess.Popen(
            [str(python_exe), "-s", "-m", "pip", "install",
             "--force-reinstall", "--no-deps", "--progress-bar", "raw",
             "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu128"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        tail = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            print(f"      {line}")
            tail.append(line)
            tail = tail[-5:]
        proc.wait()
        if proc.returncode != 0:
            fail("torch reinstall failed")
            if tail:
                info(tail[-1])
            issues.append(
                f"Reinstall torch manually: {python_exe} -s -m pip install --force-reinstall "
                "--no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
            )
            return

        try:
            out2 = subprocess.check_output(
                [str(python_exe), "-c", check_script],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            arch_list2 = out2.split(",") if out2 else []
        except Exception:
            arch_list2 = []

        if sm in arch_list2:
            ok(f"torch reinstalled — {sm} now supported")
        else:
            warn(f"torch reinstalled but {sm} still not listed — may need a newer/nightly build")
            issues.append(
                f"GPU compute capability {compute_cap} may need a PyTorch nightly build: "
                f"{python_exe} -s -m pip install --pre --upgrade torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/nightly/cu128"
            )

    def download_comfyui_portable(asset_filter, label):
        """Download and extract an official ComfyUI Windows portable build."""
        import urllib.request
        import json as _json
        info("Fetching latest ComfyUI release info ...")
        try:
            req = urllib.request.Request(
                "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest",
                headers={"Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                release = _json.load(r)
            asset = next(
                (a for a in release["assets"]
                 if asset_filter in a["name"].lower() and a["name"].endswith(".7z")),
                None,
            )
            if not asset:
                fail(f"No {label} portable build found in latest ComfyUI release")
                return False
            url  = asset["browser_download_url"]
            size = asset["size"] // (1024 ** 2)
            tag  = release["tag_name"]
        except Exception as e:
            fail(f"Could not fetch ComfyUI release info: {e}")
            return False

        info(f"Downloading ComfyUI {tag} {label} portable ({size} MB) — this may take a while ...")
        tmp = SCRIPT_DIR / asset["name"]
        try:
            urllib.request.urlretrieve(url, str(tmp))
        except Exception as e:
            fail(f"Download failed: {e}")
            tmp.unlink(missing_ok=True)
            return False

        info(f"Extracting {asset['name']} ...")
        # py7zr doesn't support BCJ2 (used in ComfyUI portables); use a real 7-zip binary.
        seven_zip = (shutil.which("7z") or shutil.which("7za") or shutil.which("7zr"))
        if not seven_zip and os_name == "Windows":
            szr = SCRIPT_DIR / "7zr.exe"
            if not szr.exists():
                info("Downloading 7zr.exe for extraction ...")
                try:
                    urllib.request.urlretrieve(
                        "https://github.com/ip7z/7zip/releases/download/26.02/7zr.exe",
                        str(szr),
                    )
                    ok("7zr.exe downloaded")
                except Exception as e:
                    fail(f"Could not download 7zr.exe: {e}")
                    tmp.unlink(missing_ok=True)
                    return False
            seven_zip = str(szr)

        def _flatten_portable():
            """Move ComfyUI_windows_portable/* up to SCRIPT_DIR if the wrapper folder exists."""
            wrapper = SCRIPT_DIR / "ComfyUI_windows_portable"
            if not wrapper.is_dir():
                return
            for child in wrapper.iterdir():
                dest = SCRIPT_DIR / child.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(child), str(dest))
            wrapper.rmdir()

        if seven_zip:
            try:
                result = subprocess.run(
                    [seven_zip, "x", str(tmp), f"-o{SCRIPT_DIR}", "-y"],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    fail(f"Extraction failed:\n{result.stderr.strip()}")
                    tmp.unlink(missing_ok=True)
                    return False
                tmp.unlink()
                _flatten_portable()
                ok(f"ComfyUI {tag} {label} portable extracted")
                return True
            except Exception as e:
                fail(f"Extraction failed: {e}")
                tmp.unlink(missing_ok=True)
                return False
        else:
            # Last resort: py7zr (may fail on BCJ2-compressed archives)
            try:
                import py7zr
                with py7zr.SevenZipFile(str(tmp), mode="r") as z:
                    z.extractall(path=str(SCRIPT_DIR))
                tmp.unlink()
                _flatten_portable()
                ok(f"ComfyUI {tag} {label} portable extracted")
                return True
            except Exception as e:
                fail(f"Extraction failed (install 7-zip for best results): {e}")
                tmp.unlink(missing_ok=True)
                return False

    if not COMFYUI_DIR.exists():
        if amd_windows:
            info("AMD GPU detected on Windows — downloading official ComfyUI AMD portable build ...")
            if not download_comfyui_portable("amd", "AMD"):
                issues.append("Download ComfyUI AMD portable from https://github.com/Comfy-Org/ComfyUI/releases")
        elif nvidia_windows:
            info("NVIDIA GPU detected on Windows — downloading official ComfyUI NVIDIA portable build ...")
            if not download_comfyui_portable("nvidia_cu", "NVIDIA"):
                issues.append("Download ComfyUI NVIDIA portable from https://github.com/Comfy-Org/ComfyUI/releases")
            elif PORTABLE_PYTHON.exists():
                check_and_fix_torch_cuda_arch(PORTABLE_PYTHON, nvidia_compute_cap)
        elif intel_windows:
            info("Intel Arc GPU detected on Windows — downloading official ComfyUI Intel portable build ...")
            if not download_comfyui_portable("intel", "Intel"):
                issues.append("Download ComfyUI Intel portable from https://github.com/Comfy-Org/ComfyUI/releases")
        else:
            comfyui_repo = "https://github.com/comfyanonymous/ComfyUI"
            info(f"Cloning ComfyUI from {comfyui_repo} ...")
            result = subprocess.run(["git", "clone", comfyui_repo, str(COMFYUI_DIR)])
            if result.returncode == 0:
                ok(f"ComfyUI cloned to {COMFYUI_DIR}")
            else:
                fail("ComfyUI clone failed — check your internet connection and git install")
                issues.append(f"git clone {comfyui_repo}")
    else:
        if amd_windows or nvidia_windows or intel_windows:
            gpu_label = "AMD" if amd_windows else ("Intel" if intel_windows else "NVIDIA")
            if not PORTABLE_PYTHON.exists():
                warn(f"ComfyUI found at {COMFYUI_DIR} but python_embeded is missing")
                warn(f"Delete {COMFYUI_DIR} and re-run setup to download the {gpu_label} portable build")
                issues.append(f"Delete {COMFYUI_DIR} and re-run setup ({gpu_label} portable build required)")
            else:
                ok(f"ComfyUI found at {COMFYUI_DIR} ({gpu_label} portable)")
                if nvidia_windows:
                    check_and_fix_torch_cuda_arch(PORTABLE_PYTHON, nvidia_compute_cap)
        else:
            ok(f"ComfyUI found at {COMFYUI_DIR}")

    if COMFYUI_DIR.exists():
        comfy_req_file = COMFYUI_DIR / "requirements.txt"
        if PORTABLE_PYTHON.exists():
            ok("Windows portable build detected — skipping requirements install (uses bundled python_embeded)")
        elif comfy_req_file.exists():
            already_installed = subprocess.run(
                [sys.executable, "-m", "pip", "show", "aiohttp"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode == 0

            if already_installed:
                ok("ComfyUI requirements already installed")
            else:
                info("Installing ComfyUI requirements ...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(comfy_req_file)]
                )
                if result.returncode == 0:
                    ok("ComfyUI requirements installed")
                else:
                    fail("ComfyUI requirements install failed")
                    issues.append(f"pip install -r {comfy_req_file}")
        else:
            warn("ComfyUI requirements.txt not found — clone may be incomplete")

        # Intel Arc on Linux: ComfyUI's requirements.txt pulls in a plain
        # (non-XPU) torch, so overwrite it with Intel's XPU wheels afterward —
        # a plain pip install from a public index, no sudo or new APT repo
        # needed. Requires PyTorch >= 2.5; no IPEX involved.
        if intel_linux and not PORTABLE_PYTHON.exists():
            torch_show = subprocess.run(
                [sys.executable, "-m", "pip", "show", "torch"],
                capture_output=True, text=True
            )
            torch_is_xpu = torch_show.returncode == 0 and "+xpu" in torch_show.stdout.lower()

            if torch_is_xpu:
                ok("XPU-enabled PyTorch already installed")
            else:
                info("Intel Arc detected — installing XPU-enabled PyTorch "
                     "(https://download.pytorch.org/whl/xpu) so ComfyUI uses the GPU ...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "--index-url", "https://download.pytorch.org/whl/xpu",
                    "torch", "torchvision", "torchaudio",
                ])
                if result.returncode == 0:
                    ok("XPU-enabled PyTorch installed")
                else:
                    fail("XPU-enabled PyTorch install failed — image tests will run on CPU")
                    issues.append(
                        "pip install --upgrade --index-url https://download.pytorch.org/whl/xpu "
                        "torch torchvision torchaudio"
                    )

        found_ckpts = []
        if CHECKPOINTS.exists():
            for m in selected_images:
                p = CHECKPOINTS / m["checkpoint"]
                if p.exists():
                    size_gb = p.stat().st_size / (1024**3)
                    ok(f"Checkpoint found: {m['checkpoint']} ({size_gb:.1f} GB)")
                    found_ckpts.append(m["checkpoint"])

        def hf_download(repo, filename, token=None, dest_dir=None, save_as=None):
            if dest_dir is None:
                dest_dir = CHECKPOINTS
            dest_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if token:
                env["HF_TOKEN"] = token
            # Try `hf` first, fall back to `huggingface-cli`, then Python API
            success = False
            for cli in ["hf", "huggingface-cli"]:
                if shutil.which(cli):
                    result = subprocess.run(
                        [cli, "download", repo, filename, "--local-dir", str(dest_dir)],
                        env=env, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        success = True
                    else:
                        stderr = (result.stderr or result.stdout or "").strip()
                        if stderr:
                            warn(f"{cli} error: {stderr}")
                    break
            if not success:
                try:
                    from huggingface_hub import hf_hub_download  # type: ignore
                    hf_hub_download(repo_id=repo, filename=filename,
                                    local_dir=str(dest_dir), token=token)
                    success = True
                except Exception as e:
                    warn(f"Python API download failed: {e}")
            # If the remote file lives in a subdirectory, move it flat into dest_dir
            if success and save_as:
                src = dest_dir / filename
                dst = dest_dir / save_as
                if src.exists() and src != dst:
                    shutil.move(str(src), str(dst))
                    try:
                        src.parent.rmdir()
                    except OSError:
                        pass
            return success

        # ── Download missing checkpoints for the selected image models ────────
        missing = [m for m in selected_images if m["checkpoint"] not in found_ckpts]
        if missing:
            info(f"Downloading {len(missing)} missing checkpoint(s): "
                 f"{', '.join(m['checkpoint'] for m in missing)}")
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)

            for m in missing:
                short, ckpt = m["short"], m["checkpoint"]

                if short == "sd15":
                    info("Downloading Stable Diffusion 1.5 (no login required) ...")
                    if hf_download("Comfy-Org/stable-diffusion-v1-5-archive", ckpt, token=load_token()):
                        ok(f"{ckpt} downloaded")
                        found_ckpts.append(ckpt)
                    else:
                        warn("SD1.5 download failed — image benchmarks will run without it")

                elif short == "sdxl":
                    info("Downloading SDXL base model (no login required) ...")
                    if hf_download("stabilityai/stable-diffusion-xl-base-1.0", ckpt, token=load_token()):
                        ok(f"{ckpt} downloaded")
                        found_ckpts.append(ckpt)
                    else:
                        warn("SDXL download failed — image benchmarks will run without it")

                elif short == "sd35-large":
                    info("Downloading SD3.5 Large (requires HuggingFace token) ...")
                    token = load_token()
                    if token:
                        if hf_download("stabilityai/stable-diffusion-3.5-large", ckpt, token=token):
                            ok(f"{ckpt} downloaded")
                            found_ckpts.append(ckpt)
                        else:
                            fail("SD3.5 Large download failed — check token and license acceptance")
                            info(f"Accept license at: {link('https://huggingface.co/stabilityai/stable-diffusion-3.5-large')}")
                    else:
                        info("Skipping SD3.5 Large — no token provided")

                elif short == "flux-dev":
                    info("Downloading Flux.1-dev (requires HuggingFace token) ...")
                    token = load_token()
                    if token:
                        if hf_download("black-forest-labs/FLUX.1-dev", ckpt, token=token):
                            ok(f"{ckpt} downloaded")
                            found_ckpts.append(ckpt)
                        else:
                            fail("Flux.1-dev download failed — check token and license acceptance")
                            info(f"Accept license at: {link('https://huggingface.co/black-forest-labs/FLUX.1-dev')}")
                    else:
                        info("Skipping Flux.1-dev — no token provided")

                elif short == "flux2-dev":
                    info("Downloading Flux.2-dev (requires HuggingFace token) ...")
                    token = load_token()
                    if token:
                        if hf_download("black-forest-labs/FLUX.2-dev", ckpt, token=token):
                            ok(f"{ckpt} downloaded")
                            found_ckpts.append(ckpt)
                        else:
                            fail("Flux.2-dev download failed — check token and license acceptance")
                            info(f"Accept license at: {link('https://huggingface.co/black-forest-labs/FLUX.2-dev')}")
                    else:
                        info("Skipping Flux.2-dev — no token provided")

        # Text encoders shared by Flux.1 and SD3.5 Large: T5-XXL + CLIP-L (public).
        # Flux.2-dev uses a different (Mistral-3-24B) text encoder — handled below.
        sd35_present  = any("sd3.5" in c for c in found_ckpts)
        flux1_present = "flux1-dev.safetensors" in found_ckpts
        flux2_present = "flux2-dev.safetensors" in found_ckpts

        if flux1_present or sd35_present:
            shared_clip_files = [
                ("t5xxl_fp16.safetensors", CLIP_DIR),
                ("clip_l.safetensors",     CLIP_DIR),
            ]
            for fname, dest in shared_clip_files:
                if not (dest / fname).exists():
                    info(f"Downloading {fname} (public, no token required) ...")
                    if hf_download("comfyanonymous/flux_text_encoders", fname, token=load_token(), dest_dir=dest):
                        ok(f"{fname} downloaded")
                    else:
                        warn(f"{fname} download failed — image generation will error")
                else:
                    ok(f"{fname} already present")

        # SD3.5 Large also needs CLIP-G (gated, same license as checkpoint)
        if sd35_present:
            clip_g = CLIP_DIR / "clip_g.safetensors"
            if not clip_g.exists():
                info("Downloading clip_g.safetensors for SD3.5 Large (requires HuggingFace token) ...")
                token = load_token()
                if token:
                    if hf_download("stabilityai/stable-diffusion-3.5-large",
                                   "text_encoders/clip_g.safetensors", token=token,
                                   dest_dir=CLIP_DIR, save_as="clip_g.safetensors"):
                        ok("clip_g.safetensors downloaded")
                    else:
                        warn("clip_g.safetensors download failed — SD3.5 image generation will error")
                        info(f"Accept license at: {link('https://huggingface.co/stabilityai/stable-diffusion-3.5-large')}")
                else:
                    info("Skipping clip_g.safetensors — no token provided")
            else:
                ok("clip_g.safetensors already present")

        if flux1_present:
            vae_file = VAE_DIR / "ae.safetensors"
            if not vae_file.exists():
                info("Downloading ae.safetensors (Flux VAE, requires HuggingFace token) ...")
                token = load_token()
                if token:
                    if hf_download("black-forest-labs/FLUX.1-schnell", "ae.safetensors",
                                   token=token, dest_dir=VAE_DIR):
                        ok("ae.safetensors downloaded")
                    else:
                        warn("ae.safetensors download failed — Flux image generation will error")
                else:
                    info("Skipping ae.safetensors — no token provided")
            else:
                ok("ae.safetensors already present")

        # Flux.2-dev needs its own (public, no token) text encoder + VAE —
        # a different architecture from Flux.1/SD3.5, not interchangeable.
        if flux2_present:
            text_encoder_dir = COMFYUI_DIR / "models" / "text_encoders"
            mistral_file = "mistral_3_small_flux2_fp8.safetensors"
            if not (text_encoder_dir / mistral_file).exists():
                info(f"Downloading {mistral_file} for Flux.2-dev (public, no token required) ...")
                if hf_download("Comfy-Org/flux2-dev",
                               f"split_files/text_encoders/{mistral_file}",
                               token=load_token(), dest_dir=text_encoder_dir, save_as=mistral_file):
                    ok(f"{mistral_file} downloaded")
                else:
                    warn(f"{mistral_file} download failed — Flux.2-dev image generation will error")
            else:
                ok(f"{mistral_file} already present")

            flux2_vae = VAE_DIR / "flux2-vae.safetensors"
            if not flux2_vae.exists():
                info("Downloading flux2-vae.safetensors (Flux.2 VAE, public, no token required) ...")
                if hf_download("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors",
                               token=load_token(), dest_dir=VAE_DIR, save_as="flux2-vae.safetensors"):
                    ok("flux2-vae.safetensors downloaded")
                else:
                    warn("flux2-vae.safetensors download failed — Flux.2-dev image generation will error")
            else:
                ok("flux2-vae.safetensors already present")

        n_expected = len(selected_images)
        if found_ckpts:
            ok(f"{len(found_ckpts)}/{n_expected} image checkpoints ready: "
               f"{', '.join(found_ckpts)}")
        else:
            fail("No image checkpoints available — image benchmarks will be skipped")
            issues.append("Download at least one image checkpoint into ComfyUI/models/checkpoints/")

# ── 9. Summary ────────────────────────────────────────────────────────────────

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
