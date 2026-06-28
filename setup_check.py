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
import json
import shutil
from pathlib import Path

from models import LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, EMBED_MODEL

SCRIPT_DIR = Path(__file__).resolve().parent

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
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        mem_bytes = int(out.splitlines()[-1].strip())
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

def check_windows_gpu():
    """Detect GPU vendor on Windows via PowerShell. Returns 'amd', 'intel', or None."""
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
            return "amd"
        if "Intel" in name and "Arc" in name:
            print(f"  GPU:     {name}")
            return "intel"

    return None

nvidia_ok     = check_nvidia()
rocm_ok       = False
metal_ok      = False
amd_windows   = False
intel_windows = False

if not nvidia_ok:
    rocm_ok = check_rocm()
if not nvidia_ok and not rocm_ok:
    metal_ok = check_metal()
if not nvidia_ok and os_name == "Windows":
    _win_vendor   = check_windows_gpu()
    amd_windows   = _win_vendor == "amd"
    intel_windows = _win_vendor == "intel"

if nvidia_ok:
    ok("CUDA / Nvidia GPU detected")
elif rocm_ok:
    ok("ROCm / AMD GPU detected")
elif amd_windows:
    ok("AMD/Radeon GPU detected on Windows")
elif intel_windows:
    ok("Intel Arc GPU detected on Windows")
elif metal_ok:
    ok("Apple Metal detected")
else:
    warn("No GPU acceleration detected — LLM and image tests may run slowly")

# ── 4. Required Python packages ────────────────────────────────────────────────

section("Python Packages")

req_file = SCRIPT_DIR / "requirements.txt"
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
    capture_output=True, text=True,
)
if result.returncode == 0:
    ok(f"Packages installed from requirements.txt")
else:
    fail("pip install -r requirements.txt failed")
    info(result.stderr.strip().splitlines()[-1] if result.stderr else "")
    issues.append("pip install -r requirements.txt")

# ── 5. Ollama ──────────────────────────────────────────────────────────────────

section("Ollama")

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
    On Windows, Ollama installs to %LOCALAPPDATA%\Programs\Ollama which is
    not always on the subprocess PATH even when it works in PowerShell.
    """
    # Standard PATH lookup first
    found = shutil.which("ollama")
    if found:
        return found
    # Windows fallback — check known install locations
    if os_name == "Windows":
        import os as _os
        candidates = [
            _os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
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
            result = subprocess.run(["brew", "install", "ollama"])
            return result.returncode == 0
        else:
            fail("Homebrew not found — install Ollama manually from https://ollama.com/download")
            return False

    elif os_name == "Linux":
        if shutil.which("snap"):
            info("Installing Ollama via snap ...")
            result = subprocess.run(["sudo", "snap", "install", "ollama"])
            if result.returncode == 0:
                import time
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
                # Give Windows a moment to finish writing the binary
                import time
                time.sleep(5)
                return True
        else:
            fail("winget not found — install Ollama manually from https://ollama.com/download")
        return False

    return False

ollama_up, tag_data = ollama_running()

OLLAMA_BIN = find_ollama_binary()
ollama_found = OLLAMA_BIN is not None

if ollama_found:
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

if ollama_up:
    available = {m["name"] for m in tag_data.get("models", [])}

    all_models = (
        [{"tag": EMBED_MODEL, "label": f"Embed: {EMBED_MODEL}", "vram": "~274 MB"}]
        + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE
    )
    for m in all_models:
        tag, label, size = m["tag"], m["label"], m["vram"]
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
    for m in [{"tag": EMBED_MODEL}] + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE:
        warn(f"Cannot check {m['tag']} — Ollama server not running")
        issues.append(f"ollama pull {m['tag']}  (once Ollama is running)")

# ── 8. Disk space ─────────────────────────────────────────────────────────────

section("Disk Space")
try:
    check_path = "C:\\" if os_name == "Windows" else "/"
    total, used, free = shutil.disk_usage(check_path)
    free_gb  = free  // (1024**3)
    total_gb = total // (1024**3)
    print(f"  Free:  {free_gb} GB / {total_gb} GB total")
    if free_gb >= 280:
        ok("Sufficient free space for all models")
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

COMFYUI_DIR  = SCRIPT_DIR / "ComfyUI"
IMAGE_CHECKPOINTS = [
    "sd_xl_base_1.0.safetensors",
    "sd3.5_large.safetensors",
    "flux1-dev.safetensors",
]
CHECKPOINTS = COMFYUI_DIR / "models" / "checkpoints"

PORTABLE_PYTHON = SCRIPT_DIR / "python_embeded" / "python.exe"
nvidia_windows  = nvidia_ok and os_name == "Windows"

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
        return False

    info(f"Extracting {asset['name']} ...")
    try:
        import py7zr
        with py7zr.SevenZipFile(str(tmp), mode="r") as z:
            z.extractall(path=str(SCRIPT_DIR))
        tmp.unlink()
        ok(f"ComfyUI {tag} {label} portable extracted")
        return True
    except Exception as e:
        fail(f"Extraction failed: {e}")
        return False

if not COMFYUI_DIR.exists():
    if amd_windows:
        info("AMD GPU detected on Windows — downloading official ComfyUI AMD portable build ...")
        if not download_comfyui_portable("amd", "AMD"):
            issues.append("Download ComfyUI AMD portable from https://github.com/Comfy-Org/ComfyUI/releases")
    elif nvidia_windows:
        info("NVIDIA GPU detected on Windows — downloading official ComfyUI NVIDIA portable build ...")
        if not download_comfyui_portable("nvidia_cu126", "NVIDIA"):
            issues.append("Download ComfyUI NVIDIA portable from https://github.com/Comfy-Org/ComfyUI/releases")
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
    else:
        ok(f"ComfyUI found at {COMFYUI_DIR}")

if COMFYUI_DIR.exists():
    req_file = COMFYUI_DIR / "requirements.txt"
    if PORTABLE_PYTHON.exists():
        ok("Windows portable build detected — skipping requirements install (uses bundled python_embeded)")
    elif req_file.exists():
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
        print()
        print(f"  {YELLOW}SD3.5 Large and Flux.1-dev require a free HuggingFace account.{RESET}")
        print(f"  1. Create an account at https://huggingface.co")
        print(f"  2. Accept the licenses at:")
        print(f"       https://huggingface.co/stabilityai/stable-diffusion-3.5-large")
        print(f"       https://huggingface.co/black-forest-labs/FLUX.1-dev")
        print(f"  3. Generate a token at https://huggingface.co/settings/tokens")
        print()
        try:
            token = input(
                f"  {CYAN}Paste your HuggingFace token and press Enter{RESET}\n  (or press Enter to skip gated models): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            token = ""
        if token:
            try:
                save = input("  Save token to hf.txt for future runs? [y/N]: ").strip().lower()
                if save == "y":
                    (SCRIPT_DIR / "hf.txt").write_text(token)
                    ok("Token saved to hf.txt")
            except (EOFError, KeyboardInterrupt):
                pass
        _hf_token_cache[0] = token or ""
        return token

    CLIP_DIR = COMFYUI_DIR / "models" / "clip"
    VAE_DIR  = COMFYUI_DIR / "models" / "vae"

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

    # ── Download missing checkpoints ───────────────────────────────────────────
    missing = [n for n in IMAGE_CHECKPOINTS if n not in found_ckpts]
    if missing:
        info(f"Downloading {len(missing)} missing checkpoint(s): {', '.join(missing)}")
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)

        # SDXL — public, no token needed
        if "sd_xl_base_1.0.safetensors" in missing:
            info("Downloading SDXL base model (no login required) ...")
            if hf_download("stabilityai/stable-diffusion-xl-base-1.0",
                           "sd_xl_base_1.0.safetensors"):
                ok("sd_xl_base_1.0.safetensors downloaded")
                found_ckpts.append("sd_xl_base_1.0.safetensors")
            else:
                warn("SDXL download failed — image benchmarks will run without it")

        # SD3.5 Large — gated (free account + license acceptance required)
        if "sd3.5_large.safetensors" in missing:
            info("Downloading SD3.5 Large (requires HuggingFace token) ...")
            token = load_token()
            if token:
                if hf_download("stabilityai/stable-diffusion-3.5-large",
                               "sd3.5_large.safetensors", token=token):
                    ok("sd3.5_large.safetensors downloaded")
                    found_ckpts.append("sd3.5_large.safetensors")
                else:
                    fail("SD3.5 Large download failed — check token and license acceptance")
                    info("Accept license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-large")
            else:
                info("Skipping SD3.5 Large — no token provided")

        # Flux.1-dev — gated (free account + license acceptance required)
        if "flux1-dev.safetensors" in missing:
            info("Downloading Flux.1-dev (requires HuggingFace token) ...")
            token = load_token()
            if token:
                if hf_download("black-forest-labs/FLUX.1-dev",
                               "flux1-dev.safetensors", token=token):
                    ok("flux1-dev.safetensors downloaded")
                    found_ckpts.append("flux1-dev.safetensors")
                else:
                    fail("Flux.1-dev download failed — check token and license acceptance")
                    info("Accept license at: https://huggingface.co/black-forest-labs/FLUX.1-dev")
            else:
                info("Skipping Flux.1-dev — no token provided")

    # Text encoders shared by Flux and SD3.5 Large: T5-XXL + CLIP-L (public)
    sd35_present  = any("sd3.5" in c for c in found_ckpts)
    flux_present  = any("flux"  in c for c in found_ckpts)

    if flux_present or sd35_present:
        shared_clip_files = [
            ("t5xxl_fp16.safetensors", CLIP_DIR),
            ("clip_l.safetensors",     CLIP_DIR),
        ]
        for fname, dest in shared_clip_files:
            if not (dest / fname).exists():
                info(f"Downloading {fname} (public, no token required) ...")
                if hf_download("comfyanonymous/flux_text_encoders", fname, dest_dir=dest):
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
                    info("Accept license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-large")
            else:
                info("Skipping clip_g.safetensors — no token provided")
        else:
            ok("clip_g.safetensors already present")

    if flux_present:
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

    if found_ckpts:
        ok(f"{len(found_ckpts)}/{len(IMAGE_CHECKPOINTS)} image checkpoints ready: "
           f"{', '.join(found_ckpts)}")
    else:
        fail("No image checkpoints available — image benchmarks will be skipped")
        issues.append("Download at least one image checkpoint into ComfyUI/models/checkpoints/")


# ── 10. Summary ────────────────────────────────────────────────────────────────

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
    "backend":   "cuda" if nvidia_ok else ("rocm" if (rocm_ok or amd_windows) else ("xpu" if intel_windows else ("metal" if metal_ok else "cpu"))),
    "ollama_up": ollama_up,
    "issues":    issues,
}

profile_path = Path("machine_profile.json")
profile_path.write_text(json.dumps(profile, indent=2))
info(f"Machine profile saved to {profile_path}")
