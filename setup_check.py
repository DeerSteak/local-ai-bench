#!/usr/bin/env python3
"""
setup_check.py — Pre-flight verification for LLM benchmark suite.
Run this on each machine before running benchmark.py.

Flow: detect the machine -> show what prerequisites need installing and ask
once -> let the user pick which models to install (numbered list, defaults
to all) -> gather any HuggingFace token needed for the picks -> install
everything with no further prompts.
"""

import sys
import os
import platform
import signal
import subprocess
import json
import shutil
from pathlib import Path

from models import LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, IMAGE_MODELS, EMBED_MODELS

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

def link(url, text=None):
    """OSC 8 terminal hyperlink. Terminals without support swallow the escape
    codes as an unrecognized control sequence, leaving just the visible text."""
    return f"\033]8;;{url}\033\\{text or url}\033]8;;\033\\"

INSTALL_STARTED = False  # flipped True once the unattended install phase begins

def cancel_setup(*_args):
    """
    Ctrl+C always means 'get me out' — installed as the SIGINT handler below
    so it fires everywhere (mid-subprocess, mid-download), not just at an
    input() prompt, and never silently falls back to a default. Nothing here
    rolls back partial work, so the message only claims "nothing installed"
    if we hadn't started installing yet.
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

# Approximate download sizes, keyed by filename — used both to show sizes on
# the model-selection screen and to estimate remaining disk space needed.
CHECKPOINT_SIZES_GB = {
    "v1-5-pruned-emaonly.safetensors": 2.1,
    "sd_xl_base_1.0.safetensors": 6.9,
    "sd3.5_large.safetensors":    10.1,
    "flux1-dev.safetensors":      23.8,
    "flux2-dev.safetensors":      23.8,
}
ENCODER_SIZES_GB = {
    "t5xxl_fp16.safetensors": 9.8,
    "clip_l.safetensors":     0.25,
    "clip_g.safetensors":     1.4,
    "ae.safetensors":         0.33,
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

# ── 4. Ollama detection (read-only — nothing is installed or started here) ────

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
            # --no-ask / HOMEBREW_NO_ASK skips brew's own "Do you want to
            # proceed with the installation? [y/n]" confirmation (this is
            # NOT what NONINTERACTIVE controls — that only covers brew's
            # installer script and sudo prompting). We already got explicit
            # consent in the prerequisites screen, so there's no need for a
            # second prompt here.
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
    warn("Ollama not found in PATH — will need to be installed")

if ollama_up:
    ok("Ollama server is running (port 11434)")
else:
    warn("Ollama server not running — will need to be started")

needs_ollama_install = not ollama_found
needs_ollama_start   = not ollama_up

# ── 5. Welcome / prerequisites approval ────────────────────────────────────────

section("Setup Plan")
print(f"  {BOLD}local-ai-bench{RESET} needs a few things before it can run benchmarks.\n")
print("  This will:")
print("    • Install Python dependencies from requirements.txt")
if needs_ollama_install:
    print("    • Install Ollama")
if needs_ollama_start:
    print("    • Start the Ollama server")
print()
print("  You'll then pick which models to install — everything after that")
print("  runs on its own, with no further prompts.")
print()

if not confirm("Continue?", default=True):
    print(f"\n  Setup cancelled — nothing was installed.\n")
    sys.exit(0)

# ── 6. Model selection ──────────────────────────────────────────────────────────

section("Model Selection")

def select_models():
    """
    Flat numbered list spanning every LLM tier, the embeddings model, and
    image models, all checked by default. Type numbers/ranges to toggle,
    a size-tier keyword (xs/s/m/l) to toggle every model at that tier —
    LLM and image checkpoints alike — 'emb'/'img' to toggle a whole
    model-type section, 'a' to select/deselect all, or press Enter to
    accept the current selection.
    Plain input() only — no raw terminal mode — so stray keys from earlier
    prompts can't leak in and there's nothing to restore/flush.
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
            entries.append({"item": m, "kind": kind, "group": group_key,
                            "tier": entry_tier, "checked": True})

    def size_label(m, kind):
        if kind in ("llm", "embed"):
            return f"  ({m['vram']})"
        gb = CHECKPOINT_SIZES_GB.get(m["checkpoint"])
        return f"  (~{gb:.1f} GB)" if gb else ""

    def render():
        print(f"  {BOLD}Choose which models to install (all selected by default){RESET}")
        n = 1
        for header, items, kind, group_key in groups:
            if not items:
                continue
            print(f"  {CYAN}{header} [{group_key}]{RESET}")
            for m in items:
                e = entries[n - 1]
                box = "[x]" if e["checked"] else "[ ]"
                print(f"    {box} {n:>2}  {m['label']}{size_label(m, kind)}")
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

selected_llm, selected_images, selected_embed = select_models()
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
elif needs_ollama_start and not ollama_found:
    issues.append("Start Ollama manually: ollama serve")

# ── 8a. Disk space ──────────────────────────────────────────────────────────────

section("Disk Space")

def _parse_size_gb(s):
    """Parse a size string like '~4.9 GB' or '~274 MB' to float GB."""
    s = s.strip().lstrip("~≈")
    try:
        if "MB" in s:
            return float(s.replace("MB", "").strip()) / 1024
        if "GB" in s:
            return float(s.replace("GB", "").strip())
    except ValueError:
        pass
    return 0.0

COMFYUI_DIR = SCRIPT_DIR / "ComfyUI"
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
            remaining_gb += _parse_size_gb(m["vram"])
else:
    for m in all_llm:
        remaining_gb += _parse_size_gb(m["vram"])

sd35_selected  = "sd35-large" in selected_image_shorts
flux_selected  = bool({"flux-dev", "flux2-dev"} & selected_image_shorts)

for m in selected_images:
    ckpt_path = CHECKPOINTS / m["checkpoint"]
    if not ckpt_path.exists():
        remaining_gb += CHECKPOINT_SIZES_GB.get(m["checkpoint"], 0.0)

if (sd35_selected or flux_selected):
    for fname in ("t5xxl_fp16.safetensors", "clip_l.safetensors"):
        if not (CLIP_DIR / fname).exists():
            remaining_gb += ENCODER_SIZES_GB[fname]
if sd35_selected and not (CLIP_DIR / "clip_g.safetensors").exists():
    remaining_gb += ENCODER_SIZES_GB["clip_g.safetensors"]
if flux_selected and not (VAE_DIR / "ae.safetensors").exists():
    remaining_gb += ENCODER_SIZES_GB["ae.safetensors"]

try:
    check_path = "C:\\" if os_name == "Windows" else "/"
    total, used, free = shutil.disk_usage(check_path)
    free_gb  = free  // (1024**3)
    total_gb = total // (1024**3)
    print(f"  Free:              {free_gb} GB / {total_gb} GB total")
    if remaining_gb > 0:
        print(f"  Still to download: ~{remaining_gb:.0f} GB")
    if remaining_gb == 0:
        ok("All selected models already downloaded — no additional space needed")
    elif free_gb >= remaining_gb + 10:
        ok(f"Sufficient free space for remaining ~{remaining_gb:.0f} GB of downloads")
    elif free_gb >= remaining_gb:
        warn(f"Space is tight — ~{remaining_gb:.0f} GB needed, {free_gb} GB free (less than 10 GB buffer)")
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
