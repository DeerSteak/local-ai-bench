#!/usr/bin/env python3
"""
setup_check.py — Pre-flight verification for LLM benchmark suite.
Run this on each machine before running benchmark.py.

Checks: Python version, Ollama, required packages, GPU/backend, model availability.
Pulls any missing Ollama models automatically.
Prompts for a HuggingFace token if the Flux model download fails.
"""

import sys
import os
import platform
import subprocess
import importlib
import json
import shutil
from pathlib import Path

# ── Formatting helpers ─────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}→{RESET}  {msg}")
def section(title): print(f"\n{BOLD}{title}{RESET}\n" + "─" * 50)

issues = []

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
        print(f"  RAM:      {mem_bytes // (1024**3)} GB")
    except Exception:
        pass

elif os_name == "Linux":
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    print(f"  RAM:      {kb // (1024**2)} GB")
                    break
    except Exception:
        pass

elif os_name == "Windows":
    try:
        result = subprocess.check_output(
            ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
            text=True
        ).strip().splitlines()
        mem_bytes = int([r for r in result if r.strip().isdigit()][0])
        print(f"  RAM:      {mem_bytes // (1024**3)} GB")
    except Exception:
        pass

# ── 3. GPU / acceleration backend ─────────────────────────────────────────────

section("GPU / Acceleration Backend")

def check_nvidia():
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
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def check_rocm():
    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.DEVNULL
        )
        agents = [l for l in out.splitlines() if "Marketing Name" in l]
        for a in agents[:3]:
            print(f"  ROCm GPU: {a.split(':', 1)[-1].strip()}")
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

nvidia_ok = check_nvidia()
rocm_ok   = False
metal_ok  = False

if not nvidia_ok:
    rocm_ok = check_rocm()
if not nvidia_ok and not rocm_ok:
    metal_ok = check_metal()

if nvidia_ok:
    ok("CUDA / Nvidia GPU detected")
elif rocm_ok:
    ok("ROCm / AMD GPU detected")
elif metal_ok:
    ok("Apple Metal detected")
else:
    warn("No GPU acceleration detected — embeddings will run on CPU")

# ── 4. PyTorch & backend ───────────────────────────────────────────────────────

section("PyTorch")
try:
    import torch  # type: ignore
    print(f"  Version: {torch.__version__}")

    if torch.cuda.is_available():
        ok(f"CUDA available — {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        ok("MPS (Metal) available")
    else:
        warn("No GPU backend detected for PyTorch — embeddings will run on CPU")
        info("This is expected on Windows with AMD GPU; Ollama uses the GPU independently")

except ImportError:
    fail("PyTorch not installed")
    info("Install: pip install torch torchvision torchaudio")
    issues.append("pip install torch torchvision torchaudio")

# ── 5. Required Python packages ────────────────────────────────────────────────

section("Python Packages")

REQUIRED = {
    "requests":              "requests",
    "psutil":                "psutil",
    "sentence_transformers": "sentence-transformers",
    "numpy":                 "numpy",
    "tqdm":                  "tqdm",
}

for import_name, install_name in REQUIRED.items():
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        ok(f"{install_name} ({ver})")
    except ImportError:
        warn(f"{install_name} not found — installing ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", install_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            ok(f"{install_name} installed ({ver})")
        else:
            fail(f"{install_name} install failed")
            info(result.stderr.strip().splitlines()[-1] if result.stderr else "")
            issues.append(f"pip install {install_name}")

# ── 6. Ollama ──────────────────────────────────────────────────────────────────

section("Ollama")

def ollama_running():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}

def ollama_pull(tag):
    """Pull a model via ollama CLI, streaming progress to stdout."""
    print(f"  Pulling {tag} ...")
    result = subprocess.run(["ollama", "pull", tag])
    return result.returncode == 0

def install_ollama():
    """Install Ollama using the appropriate method for this OS."""
    if os_name == "Darwin":
        # macOS — use Homebrew if available, otherwise direct download
        if shutil.which("brew"):
            info("Installing Ollama via Homebrew ...")
            result = subprocess.run(["brew", "install", "ollama"])
            return result.returncode == 0
        else:
            fail("Homebrew not found — install Ollama manually from https://ollama.com/download")
            return False

    elif os_name == "Linux":
        # Check if snap is available (DGX Spark / Ubuntu)
        if shutil.which("snap"):
            info("Installing Ollama via snap ...")
            result = subprocess.run(["sudo", "snap", "install", "ollama"])
            if result.returncode == 0:
                # snap installs may need a moment before the binary is on PATH
                import time
                time.sleep(3)
                return True
        # Fall back to the official install script
        info("Installing Ollama via official install script ...")
        result = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True
        )
        return result.returncode == 0

    elif os_name == "Windows":
        # Use winget if available
        if shutil.which("winget"):
            info("Installing Ollama via winget ...")
            result = subprocess.run(["winget", "install", "Ollama.Ollama", "--silent"])
            return result.returncode == 0
        else:
            fail("winget not found — install Ollama manually from https://ollama.com/download")
            return False

    return False

ollama_up, tag_data = ollama_running()

ollama_found = shutil.which("ollama") is not None

if ollama_found:
    try:
        ver_out = subprocess.check_output(
            ["ollama", "--version"], text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        ver_line = next(
            (l for l in ver_out.splitlines() if "ollama version" in l.lower()),
            ver_out.splitlines()[0] if ver_out else "unknown"
        )
        print(f"  Binary:  {ver_line.strip()}")
        ok("Ollama binary found")
    except Exception:
        ok("Ollama binary found")
else:
    warn("Ollama not found in PATH — attempting to install ...")
    installed = install_ollama()
    if installed:
        ok("Ollama installed successfully")
        ollama_found = True
    else:
        fail("Ollama installation failed")
        issues.append("Install Ollama manually from https://ollama.com/download")
        info("On Linux (DGX Spark / Ubuntu): sudo snap install ollama")

if ollama_up:
    ok("Ollama server is running (port 11434)")
else:
    warn("Ollama server not running — attempting to start it ...")
    try:
        if os_name == "Windows":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # Wait up to 15s for it to come up
        import time
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

# ── 7. Ollama models — pull if missing ────────────────────────────────────────

section("Ollama Models")

OLLAMA_MODELS = [
    # Small tier — fit in 16GB VRAM or less (tags verified June 2026)
    # Note: llama3.2 tops out at 3B; 8B slot is llama3.1
    # Note: qwen3:14b has no q3_K_M — q4_K_M and q8_0 are the available quantizations
    # Note: gpt-oss:20b ships in MXFP4 only — no separate Q3/Q4 variants
    ("llama3.1:8b-instruct-q3_K_M", "Llama 3.1 8B Q3_K_M",    "~4.3 GB"),
    ("llama3.1:8b-instruct-q4_K_M", "Llama 3.1 8B Q4_K_M",    "~4.9 GB"),
    ("qwen3:14b-q4_K_M",            "Qwen3 14B Q4_K_M",        "~9.3 GB"),
    ("qwen3:14b-q8_0",              "Qwen3 14B Q8_0",           "~16 GB"),
    ("gpt-oss:20b",                  "GPT-OSS 20B (MXFP4)",    "~14 GB"),
    # Large tier — 32GB+ memory required
    # Note: gpt-oss:120b ships in MXFP4 only — no Q3/Q4 variants exist
    ("llama3.1:70b-instruct-q3_K_M", "Llama 3.1 70B Q3_K_M",  "~32 GB"),
    ("llama3.1:70b-instruct-q4_K_M", "Llama 3.1 70B Q4_K_M",  "~42 GB"),
    ("gpt-oss:120b",                  "GPT-OSS 120B (MXFP4)",  "~65 GB"),
]

if ollama_up:
    available = {m["name"] for m in tag_data.get("models", [])}

    for tag, label, size in OLLAMA_MODELS:
        already = tag in available or any(tag in a for a in available)
        if already:
            ok(f"{label} — already pulled")
        else:
            warn(f"{label} ({size}) — not found, pulling now ...")
            success = ollama_pull(tag)
            if success:
                ok(f"{label} — pulled successfully")
            else:
                fail(f"{label} — pull failed")
                issues.append(f"ollama pull {tag}")
else:
    for tag, label, size in OLLAMA_MODELS:
        warn(f"Cannot check {label} — Ollama server not running")
        issues.append(f"ollama pull {tag}  (once Ollama is running)")

# ── 8. Disk space ─────────────────────────────────────────────────────────────

section("Disk Space")
try:
    check_path = "C:\\" if os_name == "Windows" else "/"
    total, used, free = shutil.disk_usage(check_path)
    free_gb  = free  // (1024**3)
    total_gb = total // (1024**3)
    print(f"  Free:  {free_gb} GB / {total_gb} GB total")
    if free_gb >= 280:
        ok("Sufficient free space for all ten models")
    elif free_gb >= 60:
        warn(f"Only {free_gb} GB free — enough for small-tier models only")
        warn("Large-tier models need ~250 GB total; run with --small-only if disk is limited")
    else:
        fail(f"Only {free_gb} GB free — may not fit even small models (~57 GB total)")
        issues.append("Free up at least 60 GB of disk space for small-tier models")
except Exception as e:
    warn(f"Could not check disk space: {e}")

# ── 9. ComfyUI — clone if missing, pip install requirements ───────────────────

section("ComfyUI")

SCRIPT_DIR   = Path(__file__).resolve().parent
COMFYUI_DIR  = SCRIPT_DIR / "ComfyUI"
IMAGE_CHECKPOINTS = [
    "sd_xl_base_1.0.safetensors",
    "flux1-schnell.safetensors",
    "flux1-dev.safetensors",
]
CHECKPOINTS = COMFYUI_DIR / "models" / "checkpoints"

if not COMFYUI_DIR.exists():
    info("ComfyUI not found — cloning ...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/comfyanonymous/ComfyUI",
         str(COMFYUI_DIR)]
    )
    if result.returncode == 0:
        ok("ComfyUI cloned successfully")
    else:
        fail("ComfyUI clone failed — check your internet connection and git install")
        issues.append("git clone https://github.com/comfyanonymous/ComfyUI")
else:
    ok(f"ComfyUI directory found at {COMFYUI_DIR}")

if COMFYUI_DIR.exists():
    req_file = COMFYUI_DIR / "requirements.txt"
    if req_file.exists():
        # Check if aiohttp (a ComfyUI dep) is already installed
        already_installed = subprocess.run(
            [sys.executable, "-m", "pip", "show", "aiohttp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0

        if already_installed:
            ok("ComfyUI requirements already installed")
        else:
            info("Installing ComfyUI requirements ...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            )
            if result.returncode == 0:
                ok("ComfyUI requirements installed")
            else:
                fail("ComfyUI requirements install failed")
                issues.append(f"pip install -r {req_file}")
    else:
        warn("ComfyUI requirements.txt not found — clone may be incomplete")

    # Check for image checkpoints
    found_ckpts = []
    if CHECKPOINTS.exists():
        for name in IMAGE_CHECKPOINTS:
            p = CHECKPOINTS / name
            if p.exists():
                size_gb = p.stat().st_size / (1024**3)
                ok(f"Checkpoint found: {name} ({size_gb:.1f} GB)")
                found_ckpts.append(name)

    # ── Token resolution ──────────────────────────────────────────────────────
    # Used only for gated models (Flux.1-dev). hf.txt takes priority over prompt.
    def load_token():
        hf_txt = SCRIPT_DIR / "hf.txt"
        if hf_txt.exists():
            token = hf_txt.read_text().strip()
            if token:
                ok(f"HuggingFace token loaded from {hf_txt}")
                return token
        return None

    def prompt_token():
        print()
        print(f"  {YELLOW}Flux.1-dev is a gated model. To download it you need to:{RESET}")
        print(f"  1. Visit https://huggingface.co/black-forest-labs/FLUX.1-dev")
        print(f"  2. Accept the license agreement")
        print(f"  3. Generate a token at https://huggingface.co/settings/tokens")
        print(f"  4. Save it to hf.txt in this directory, or paste it below")
        print()
        try:
            token = input(
                f"  {CYAN}Paste your HuggingFace token and press Enter{RESET}\n"
                f"  (or press Enter to skip Flux.1-dev): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            token = ""
        if token:
            # Offer to save it for future runs
            try:
                save = input(f"  Save token to hf.txt for future runs? [y/N]: ").strip().lower()
                if save == "y":
                    (SCRIPT_DIR / "hf.txt").write_text(token)
                    ok("Token saved to hf.txt")
            except (EOFError, KeyboardInterrupt):
                pass
        return token

    def hf_download(repo, filename, token=None):
        env = os.environ.copy()
        if token:
            env["HF_TOKEN"] = token
        cmd = ["huggingface-cli", "download", repo,
               filename, "--local-dir", str(CHECKPOINTS)]
        return subprocess.run(cmd, env=env).returncode == 0

    # ── Download missing checkpoints ───────────────────────────────────────────
    missing = [n for n in IMAGE_CHECKPOINTS if n not in found_ckpts]
    if missing:
        info(f"Downloading {len(missing)} missing checkpoint(s): {', '.join(missing)}")
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)

        # 1. SDXL — public, no token needed
        if "sd_xl_base_1.0.safetensors" in missing:
            info("Downloading SDXL base model (no login required) ...")
            if hf_download("stabilityai/stable-diffusion-xl-base-1.0",
                           "sd_xl_base_1.0.safetensors"):
                ok("sd_xl_base_1.0.safetensors downloaded successfully")
                found_ckpts.append("sd_xl_base_1.0.safetensors")
            else:
                warn("SDXL download failed — image benchmarks will run without it")

        # 2. Flux.1-schnell — public, Apache 2.0, no token needed
        if "flux1-schnell.safetensors" in missing:
            info("Downloading Flux.1-schnell (no login required) ...")
            if hf_download("black-forest-labs/FLUX.1-schnell",
                           "flux1-schnell.safetensors"):
                ok("flux1-schnell.safetensors downloaded successfully")
                found_ckpts.append("flux1-schnell.safetensors")
            else:
                warn("Flux.1-schnell download failed — image benchmarks will run without it")

        # 3. Flux.1-dev — gated; try hf.txt first, then cached login, then prompt
        if "flux1-dev.safetensors" in missing:
            info("Downloading Flux.1-dev (gated model) ...")
            token = load_token()
            if hf_download("black-forest-labs/FLUX.1-dev",
                           "flux1-dev.safetensors", token=token):
                ok("flux1-dev.safetensors downloaded successfully")
                found_ckpts.append("flux1-dev.safetensors")
            else:
                if not token:
                    token = prompt_token()
                    if token and hf_download("black-forest-labs/FLUX.1-dev",
                                             "flux1-dev.safetensors", token=token):
                        ok("flux1-dev.safetensors downloaded successfully")
                        found_ckpts.append("flux1-dev.safetensors")
                    elif not token:
                        info("Skipping Flux.1-dev")
                    else:
                        fail("Flux.1-dev download failed — check token and license acceptance")
                else:
                    fail("Flux.1-dev download failed — check token and license acceptance")

    if found_ckpts:
        ok(f"{len(found_ckpts)}/{len(IMAGE_CHECKPOINTS)} image checkpoints ready: "
           f"{', '.join(found_ckpts)}")
    else:
        fail("No image checkpoints available — image benchmarks will be skipped")
        issues.append("Download at least one image checkpoint into ComfyUI/models/checkpoints/")


# ── 10. Container check (DGX Spark / Linux) ───────────────────────────────────



# ── 11. Summary ────────────────────────────────────────────────────────────────

section("Summary")

if not issues:
    print(f"\n  {GREEN}{BOLD}All checks passed — ready to benchmark!{RESET}")
    print(f"  Run: python benchmark.py\n")
else:
    print(f"\n  {YELLOW}{BOLD}Action items before benchmarking:{RESET}")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print()

# Write machine profile JSON for use by benchmark.py
profile = {
    "hostname":  platform.node(),
    "os":        f"{platform.system()} {platform.release()}",
    "arch":      platform.machine(),
    "python":    sys.version.split()[0],
    "backend":   "cuda" if nvidia_ok else ("rocm" if rocm_ok else ("metal" if metal_ok else "cpu")),
    "ollama_up": ollama_up,
    "issues":    issues,
}

profile_path = Path("machine_profile.json")
profile_path.write_text(json.dumps(profile, indent=2))
info(f"Machine profile saved to {profile_path}")
