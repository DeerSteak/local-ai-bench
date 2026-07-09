#!/usr/bin/env python3
"""
benchmark.py — Cross-platform LLM benchmark suite.

Tests:
  1. LLM generation — 10 models across small/medium/large tiers via Ollama
     Metrics: time-to-first-token (TTFT), tokens/sec
     Context lengths: 2K, 8K, 32K, 64K
     Models that exceed the warmup timeout are skipped automatically

  1b. LLM conversation — same models/context depths, but via a real multi-turn
      chat (/api/chat) instead of one padded single-shot prompt: the model
      explains Plato's Allegory of the Cave in sections, then each turn asks
      for more detail on a section. TTFT/tokens-per-sec at each depth reflect
      processing a new turn against an already-filled context (relying on
      llama.cpp's slot prefix cache), not a cold fill from empty.

  2. Image generation — SDXL, SD3.5 Large, Flux.1-dev via ComfyUI HTTP API
     Metrics: seconds/image at 1024×1024 and 1536×1536
     (models skipped automatically if checkpoint not found)

  3. Embeddings — nomic-embed-text via Ollama
     Metrics: sentences/sec at batch sizes 32, 128, 512

Servers are managed automatically:
  - Ollama: started if not already running, shut down on exit if we started it
  - ComfyUI: started before image tests, shut down cleanly when done

Usage:
  python benchmark.py                  # run all tests
  python benchmark.py --tests llm      # run only LLM single-shot tests
  python benchmark.py --tests llm emb  # run LLM + embeddings
  python benchmark.py --tests conv     # run only LLM conversation tests
  python benchmark.py --runs 3         # override number of measured runs
  python benchmark.py --comfyui /path/to/ComfyUI  # override ComfyUI path
"""

import argparse
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434"
COMFYUI_URL  = "http://localhost:8188"

# Default ComfyUI path — relative to this script's directory
SCRIPT_DIR   = Path(__file__).resolve().parent
COMFYUI_DIR  = SCRIPT_DIR / "ComfyUI"

from models import IMAGE_MODELS, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS, EMBED_MODEL  # noqa: E402

CONTEXT_LENGTHS = [2048, 8192, 32768, 65536]   # tokens (approximate, via prompt padding)
EMBED_BATCH_SIZES = [32, 128, 512]
IMAGE_RESOLUTIONS = [(1024, 1024), (1536, 1536)]
# Steps are now per-model in IMAGE_MODELS
IMAGE_SEED  = 42
IMAGE_PROMPT = (
    "A photorealistic high-end gaming PC build with RGB lighting, "
    "multiple GPUs, custom water cooling, shot in a dark room, "
    "highly detailed, 8k resolution"
)

VERSION        = "1.0"
WARMUP_RUNS    = 2
DEFAULT_RUNS   = 5
RUN_TIMEOUT = 300   # seconds per run (warmup and measured) before aborting

# ── Helpers ────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def log(msg):   print(f"  {CYAN}→{RESET}  {msg}")
def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET}  {msg}")
def err(msg):   print(f"  {RED}✗{RESET}  {msg}")
def section(t): print(f"\n{BOLD}{'─'*50}\n  {t}\n{'─'*50}{RESET}")

def mean(vals):   return statistics.mean(vals) if vals else 0
def stdev(vals):  return statistics.stdev(vals) if len(vals) >= 2 else 0

def system_ram_gb():
    return psutil.virtual_memory().total / (1024 ** 3)

# ── Server management ─────────────────────────────────────────────────────────

# Tracks processes we started so we can shut them down cleanly
_managed_procs: list[subprocess.Popen] = []

def _shutdown_managed():
    """Terminate any servers we started."""
    for proc in _managed_procs:
        if proc.poll() is None:
            log(f"Stopping managed process (pid {proc.pid}) ...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    _managed_procs.clear()

def ensure_ollama():
    """Start Ollama if not already running. Returns True if available."""
    if ollama_available():
        ok("Ollama already running")
        return True

    log("Ollama not running — attempting to start ...")
    os_name = platform.system()

    try:
        if os_name == "Windows":
            # On Windows Ollama is a system tray app; 'ollama serve' starts the server
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        _managed_procs.append(proc)
    except FileNotFoundError:
        err("'ollama' not found in PATH — install from https://ollama.com/download")
        return False

    # Wait up to 15s for it to come up
    for i in range(15):
        time.sleep(1)
        if ollama_available():
            ok(f"Ollama started (pid {proc.pid})")
            return True
        if proc.poll() is not None:
            err(f"Ollama exited unexpectedly (code {proc.returncode})")
            return False

    err("Ollama did not respond within 15 seconds")
    return False

def find_comfyui_python(comfyui_dir: Path) -> str:
    """
    Return the Python executable to use for ComfyUI.
    Prefers a venv inside comfyui_dir, then the venv running this script,
    then whatever 'python' resolves to.
    """
    # Official AMD portable build: python_embeded sits next to ComfyUI/, not inside it
    for candidate in [
        comfyui_dir.parent / "python_embeded" / "python.exe",
        comfyui_dir / "python_env" / "python.exe",
        comfyui_dir / "venv" / "bin" / "python",
        comfyui_dir / ".venv" / "bin" / "python",
        comfyui_dir / "venv" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)

    # The venv currently running this script (most likely on Mac/Linux)
    current_venv = os.environ.get("VIRTUAL_ENV")
    if current_venv:
        for rel in ["bin/python", "Scripts/python.exe"]:
            p = Path(current_venv) / rel
            if p.exists():
                return str(p)

    return sys.executable

def ensure_comfyui(comfyui_dir: Path) -> bool:
    """
    Start ComfyUI if not already running.
    Returns True if ComfyUI is available (either was already running or we started it).
    """
    if comfyui_available():
        ok("ComfyUI already running")
        return True

    if not comfyui_dir.exists():
        warn(f"ComfyUI directory not found at {comfyui_dir}")
        warn("Clone it with: git clone https://github.com/comfyanonymous/ComfyUI")
        return False

    main_py = comfyui_dir / "main.py"
    if not main_py.exists():
        warn(f"main.py not found in {comfyui_dir}")
        return False

    # Check at least one image model checkpoint is present
    checkpoints_dir = comfyui_dir / "models" / "checkpoints"
    known = [m["checkpoint"] for m in IMAGE_MODELS]
    found = [c for c in known if (checkpoints_dir / c).exists()]
    if not found:
        warn("No image model checkpoints found in " + str(checkpoints_dir))
        warn("Expected one of: " + ", ".join(known))
        warn("Run setup_check.py to download Flux models automatically")
        return False
    log(f"Found {len(found)}/{len(known)} image checkpoints: {found}")

    python_exe = find_comfyui_python(comfyui_dir)

    # Windows portable builds: python_embeded is a sibling of ComfyUI/, cwd must be the parent
    portable_windows = (comfyui_dir.parent / "python_embeded" / "python.exe").exists()
    if portable_windows:
        cmd = [python_exe, "-s", str(main_py), "--windows-standalone-build", "--listen"]
        launch_cwd = str(comfyui_dir.parent)
    else:
        cmd = [python_exe, str(main_py), "--listen"]
        launch_cwd = str(comfyui_dir)

    log(f"Starting ComfyUI from {comfyui_dir} using {python_exe} ...")

    env = os.environ.copy()
    # AMD on Windows: Triton JIT compilation fails; interpreter mode works around it
    if portable_windows and detect_backend() == "rocm":
        env["TRITON_INTERPRET"] = "1"

    # Capture stderr to a temp file so we can show it if ComfyUI exits unexpectedly
    try:
        stderr_fh = tempfile.NamedTemporaryFile(
            mode="w", suffix="-comfyui-stderr.log", delete=False
        )
        stderr_log = Path(stderr_fh.name)
        proc = subprocess.Popen(
            cmd,
            cwd=launch_cwd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
            env=env,
        )
        stderr_fh.close()
        _managed_procs.append(proc)
    except Exception as e:
        err(f"Failed to start ComfyUI: {e}")
        return False

    # Wait up to 60s — model loading takes time
    log("Waiting for ComfyUI to be ready (up to 60s) ...")
    for i in range(60):
        time.sleep(1)
        if comfyui_available():
            ok(f"ComfyUI started (pid {proc.pid})")
            try:
                stderr_log.unlink()
            except Exception:
                pass
            return True
        if proc.poll() is not None:
            err(f"ComfyUI exited unexpectedly (code {proc.returncode})")
            try:
                lines = stderr_log.read_text(errors="replace").strip().splitlines()
                if lines:
                    err("Last output from ComfyUI:")
                    for line in lines[-8:]:
                        err(f"  {line}")
            except Exception:
                pass
            try:
                stderr_log.unlink()
            except Exception:
                pass
            err(f"Try starting manually: cd {comfyui_dir} && python main.py {' '.join(cmd[2:])}")
            return False
        if (i + 1) % 10 == 0:
            log(f"Still waiting ... ({i+1}s)")

    err("ComfyUI did not respond within 60 seconds")
    try:
        stderr_log.unlink()
    except Exception:
        pass
    return False


# ── Machine profile ────────────────────────────────────────────────────────────

def _get_hostname():
    system = platform.system()
    ram_gb = round(system_ram_gb())

    if system == "Darwin":
        try:
            sp = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True, timeout=10,
            )
            model = chip = ram = None
            for line in sp.stdout.splitlines():
                if "Model Name:" in line:
                    model = line.split(":", 1)[1].strip()
                elif "Chip:" in line:
                    chip = line.split(":", 1)[1].strip().removeprefix("Apple ").strip()
                elif "Memory:" in line:
                    ram = line.split(":", 1)[1].strip()
            if model and chip and ram:
                return f"{model}\n{chip} {ram}"
        except Exception:
            pass

    elif system == "Windows":
        cpu = gpu = None

        def _ps_names(cim_class):
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"(Get-CimInstance {cim_class}).Name"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                return [n.strip() for n in out.splitlines() if n.strip()]
            except Exception:
                return []

        cpu_names = _ps_names("Win32_Processor")
        if cpu_names:
            cpu = cpu_names[0]

        _skip = {"microsoft basic display adapter", "microsoft remote display adapter"}
        gpus = [n for n in _ps_names("Win32_VideoController") if n and n.lower() not in _skip]
        if gpus:
            gpu = gpus[0]
        if cpu and gpu:
            return f"{cpu}\n{gpu} {ram_gb} GB"
        elif cpu:
            return f"{cpu}\n{ram_gb} GB"
        elif gpu:
            return f"{gpu} {ram_gb} GB"

    elif system == "Linux":
        cpu = gpu = None
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        # NVIDIA first, then AMD via rocminfo, then lspci fallback
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                gpu = out.splitlines()[0].strip()
        except Exception:
            pass
        if not gpu:
            try:
                out = subprocess.run(
                    ["rocminfo"], capture_output=True, text=True, timeout=10,
                ).stdout
                for line in out.splitlines():
                    if "Marketing Name:" in line:
                        gpu = line.split(":", 1)[1].strip()
                        break
            except Exception:
                pass
        if not gpu:
            try:
                out = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=10,
                ).stdout
                for line in out.splitlines():
                    if any(k in line for k in ("VGA", "3D controller", "Display")):
                        gpu = line.split(":", 2)[-1].strip()
                        break
            except Exception:
                pass
        if cpu and gpu:
            return f"{cpu}\n{gpu} {ram_gb} GB"
        elif cpu:
            return f"{cpu}\n{ram_gb} GB"
        elif gpu:
            return f"{gpu} {ram_gb} GB"

    return platform.node()


def build_profile():
    os_name = platform.system()
    profile = {
        "hostname":   _get_hostname(),
        "os":         f"{os_name} {platform.release()}",
        "arch":       platform.machine(),
        "python":     sys.version.split()[0],
        "ram_gb":     round(system_ram_gb(), 1),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "backend":    detect_backend(),
    }
    return profile

def detect_backend():
    # Nvidia
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # ROCm (Linux)
    try:
        out = subprocess.check_output(["rocminfo"], text=True,
                                       stderr=subprocess.DEVNULL)
        if "Marketing Name" in out:
            return "rocm"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # AMD/Intel on Windows — rocminfo/xpu-smi don't exist; detect via PowerShell
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                text=True, stderr=subprocess.DEVNULL,
            )
            names = [n.strip() for n in out.splitlines() if n.strip()]
            if any("AMD" in n or "Radeon" in n for n in names):
                return "rocm"
            if any("Intel" in n and "Arc" in n for n in names):
                return "xpu"
        except Exception:
            pass
    # Metal
    if platform.system() == "Darwin":
        return "metal"
    return "cpu"

# ── Prompt builders ────────────────────────────────────────────────────────────

SHORT_PROMPT = (
    "You are a hardware reviewer for HotHardware.com. Compare the performance, "
    "power efficiency, and value proposition of the latest GPU architectures "
    "for gaming and content creation workloads. Discuss thermal design, memory "
    "bandwidth, ray tracing capabilities, and how driver maturity affects "
    "real-world performance across AAA titles and professional applications."
)

_PADDING_UNIT = (
    " Additionally, analyze how CPU and platform choices — including chiplet "
    "designs, memory controllers, and PCIe bandwidth — interact with GPU "
    "performance, and what this means for system builders choosing between "
    "competing platforms at different price points."
)

def build_prompt_for_context(target_tokens: int) -> str:
    """
    Pad a prompt to approximate a target context length (1 token ≈ 4 chars).

    Prepends a unique per-call nonce so repeated calls at the same context
    length don't share an identical prefix — without it, Ollama's slot cache
    recognizes the exact same prompt on every subsequent run and serves a
    cache hit instead of genuinely reprocessing it, making every run after
    the first measure cache-hit latency rather than real prompt-processing
    time (the one real cold-prefill measurement then gets discarded as the
    "slow outlier").
    """
    nonce = uuid.uuid4().hex
    prefix = f"[run {nonce}] "
    chars_needed = target_tokens * 4
    parts = [prefix, SHORT_PROMPT]
    total = len(prefix) + len(SHORT_PROMPT)
    while total < chars_needed:
        parts.append(_PADDING_UNIT)
        total += len(_PADDING_UNIT)
    return "".join(parts)[:chars_needed]

# ── Ollama ─────────────────────────────────────────────────────────────────────

def ollama_available():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def model_pulled(tag):
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        names = [m["name"] for m in models]
        return tag in names or any(tag in n for n in names)
    except Exception:
        return False

def ollama_generate(model_tag: str, prompt: str, timeout: int = 600,
                    num_ctx: int | None = None):
    """
    Generate via Ollama and return timing metrics.
    Returns: (ttft_sec, tokens_generated, tokens_per_sec)

    Uses urllib instead of requests for streaming to avoid TCP buffering
    that causes iter_lines() to batch all chunks and inflate TTFT.

    Ollama's final chunk includes server-side timing fields:
      prompt_eval_duration  — time to process the prompt (nanoseconds)
      eval_count            — tokens generated
      eval_duration         — time spent generating (nanoseconds)
    These are authoritative and used in preference to wall-clock where available.

    num_ctx must match the context length being tested. Without it, Ollama uses
    the model's default, and sending a prompt longer than that default triggers a
    full model reload — inflating TTFT by minutes rather than seconds.
    """
    options: dict = {"num_predict": 512, "temperature": 0.0}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload = json.dumps({
        "model":  model_tag,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    ttft    = None
    tokens  = 0
    tps     = 0
    eval_count = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            # First token received — record TTFT from wall clock
            if ttft is None and chunk.get("response"):
                ttft = time.perf_counter() - t_start

            if chunk.get("response"):
                tokens += 1

            if chunk.get("done"):
                eval_count    = chunk.get("eval_count", tokens)
                eval_duration = chunk.get("eval_duration", 0)      # nanoseconds
                prompt_dur    = chunk.get("prompt_eval_duration", 0) # nanoseconds

                # Prefer server-side TTFT (prompt processing time) if available
                if prompt_dur and prompt_dur > 0:
                    ttft = prompt_dur / 1e9

                if eval_duration and eval_duration > 0:
                    tps = eval_count / (eval_duration / 1e9)
                break

    total = time.perf_counter() - t_start
    if ttft is None:
        ttft = total
    return ttft, eval_count, tps

def ollama_chat(model_tag: str, messages: list, timeout: int = 600,
                 num_ctx: int | None = None, num_predict: int = 1024):
    """
    Generate via Ollama's /api/chat and return timing metrics plus the reply text.
    Returns: (ttft_sec, tokens_generated, tokens_per_sec, prompt_eval_count, response_text)

    prompt_eval_count reflects the number of *new* prompt tokens the backend had
    to process this turn — when `messages` shares a prefix with a prior call on
    the same loaded model, llama.cpp's slot cache skips re-processing it, so
    this is the incremental prefill cost against an already-filled context
    rather than a cold fill from empty.
    """
    options: dict = {"num_predict": num_predict, "temperature": 0.0}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload = json.dumps({
        "model":    model_tag,
        "messages": messages,
        "stream":   True,
        "options":  options,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    ttft   = None
    tokens = 0
    tps    = 0
    eval_count        = 0
    prompt_eval_count = 0
    response_parts    = []

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            content = chunk.get("message", {}).get("content")

            if ttft is None and content:
                ttft = time.perf_counter() - t_start

            if content:
                tokens += 1
                response_parts.append(content)

            if chunk.get("done"):
                eval_count        = chunk.get("eval_count", tokens)
                eval_duration      = chunk.get("eval_duration", 0)      # nanoseconds
                prompt_dur         = chunk.get("prompt_eval_duration", 0)
                prompt_eval_count  = chunk.get("prompt_eval_count", 0)

                if prompt_dur and prompt_dur > 0:
                    ttft = prompt_dur / 1e9

                if eval_duration and eval_duration > 0:
                    tps = eval_count / (eval_duration / 1e9)
                break

    total = time.perf_counter() - t_start
    if ttft is None:
        ttft = total
    return ttft, eval_count, tps, prompt_eval_count, "".join(response_parts)

# ── Ollama model loading/unloading ────────────────────────────────────────────

def unload_model(model_tag: str):
    """Force Ollama to evict a model from memory immediately."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_tag, "keep_alive": 0},
            timeout=30,
        )
        ok(f"Unloaded {model_tag}")
    except Exception as e:
        warn(f"Could not unload {model_tag}: {e}")

def unload_all_models():
    """Unload every model currently loaded in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10)
        loaded = r.json().get("models", [])
        if not loaded:
            ok("No models currently loaded in Ollama")
            return
        for m in loaded:
            unload_model(m["name"])
    except Exception as e:
        warn(f"Could not query loaded models: {e}")

def wait_until_unloaded(model_tag: str, timeout: int = 30):
    """Poll /api/ps until the model no longer appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
            loaded = [m["name"] for m in r.json().get("models", [])]
            if not any(model_tag in name for name in loaded):
                return True
        except Exception:
            pass
        time.sleep(1)
    warn(f"{model_tag} still appears loaded after {timeout}s")
    return False


# ── LLM benchmark ──────────────────────────────────────────────────────────────

def run_llm_benchmarks(models, context_lengths, n_runs, warmup_runs):
    results = {}

    if not ollama_available():
        err("Ollama server not reachable — skipping LLM benchmarks")
        err("Start with: ollama serve")
        return results

    for model in models:
        tag   = model["tag"]
        label = model["label"]
        short = model["short"]

        section(f"LLM: {label}")

        if not model_pulled(tag):
            warn(f"{tag} not pulled — skipping")
            warn(f"Pull with: ollama pull {tag}")
            continue

        # Warm up model (load into memory), with a timeout so we don't get stuck.
        # Use the largest context this model will actually run so Ollama
        # pre-allocates the full KV cache once — avoiding a reload at max context.
        max_ctx = min(model.get("max_ctx", max(context_lengths)), max(context_lengths))
        log(f"Warming up {label} at num_ctx={max_ctx} (timeout: {RUN_TIMEOUT}s per run) ...")
        warmup_ok = True
        for warmup_i in range(warmup_runs):
            result_box = [None]   # mutable container so thread can write back
            exc_box    = [None]

            def _warmup():
                try:
                    result_box[0] = ollama_generate(
                        tag, "Hello.", timeout=RUN_TIMEOUT, num_ctx=max_ctx)
                except Exception as e:
                    exc_box[0] = e

            t = threading.Thread(target=_warmup, daemon=True)
            t_start = time.perf_counter()
            t.start()
            t.join(timeout=RUN_TIMEOUT)

            if t.is_alive():
                elapsed = time.perf_counter() - t_start
                warn(f"{label}: warmup run {warmup_i+1} did not complete within {elapsed:.0f}s")
                warn(f"{label}: model is likely too large for available memory — skipping")
                warmup_ok = False
                break
            elif exc_box[0] is not None:
                warn(f"Warmup run {warmup_i+1} failed: {exc_box[0]}")
                warmup_ok = False
                break
            else:
                log(f"Warmup run {warmup_i+1}/{warmup_runs} done")

        if not warmup_ok:
            unload_model(tag)
            continue

        results[short] = {}

        model_ctx_lengths = [c for c in context_lengths
                             if c <= model.get("max_ctx", max(context_lengths))]

        model_timed_out = False
        for ctx_len in model_ctx_lengths:
            label_ctx = f"{ctx_len // 1024}K"
            log(f"Context {label_ctx} — {n_runs} runs ...")

            ttfts, tps_list = [], []
            ctx_timed_out = False

            for run_i in range(n_runs):
                try:
                    prompt = build_prompt_for_context(ctx_len)
                    ttft, tokens, tps = ollama_generate(
                        tag, prompt, timeout=RUN_TIMEOUT, num_ctx=ctx_len
                    )
                    ttfts.append(ttft)
                    tps_list.append(tps)
                    print(
                        f"    run {run_i+1}/{n_runs}: "
                        f"TTFT={ttft:.2f}s  "
                        f"TPS={tps:.1f}"
                    )
                except Exception as e:
                    is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                    if is_timeout:
                        err(f"Run {run_i+1} timed out — skipping remaining runs and context lengths for {label}")
                        ctx_timed_out = True
                        break
                    err(f"Run {run_i+1} failed: {e}")

            if ttfts:
                raw_ttfts = ttfts[:]
                raw_tps = tps_list[:]
                # Drop the single slowest run (by TTFT) from both lists before
                # averaging — a run flagged as an outlier is dropped entirely,
                # not just its TTFT value.
                if len(ttfts) > 1:
                    worst_idx = ttfts.index(max(ttfts))
                    ttfts    = ttfts[:worst_idx]    + ttfts[worst_idx+1:]
                    tps_list = tps_list[:worst_idx] + tps_list[worst_idx+1:]

                results[short][label_ctx] = {
                    "ttft_mean_sec":  round(mean(ttfts),    3),
                    "ttft_stdev_sec": round(stdev(ttfts),   3),
                    "tps_mean":       round(mean(tps_list), 2),
                    "tps_stdev":      round(stdev(tps_list),2),
                    "n_runs":         len(tps_list),
                    "ttft_runs":      [round(t, 3) for t in raw_ttfts],
                    "tps_runs":       [round(t, 2) for t in raw_tps],
                }
                ok(
                    f"Context {label_ctx} done: "
                    f"TTFT={results[short][label_ctx]['ttft_mean_sec']:.2f}s  "
                    f"TPS={results[short][label_ctx]['tps_mean']:.1f}"
                )

            if ctx_timed_out:
                model_timed_out = True
                results[short]["timed_out"] = label_ctx
                break

        # Unload model and confirm it's gone before moving on
        if model_timed_out:
            warn(f"{label}: timed out — moving to next model")
        log(f"Unloading {label} ...")
        unload_model(tag)
        wait_until_unloaded(tag)

    return results

# ── LLM conversation benchmark ──────────────────────────────────────────────────
# Simulates a real multi-turn chat (rather than one huge padded single-shot
# prompt): the model explains Plato's Allegory of the Cave in numbered
# sections, then each subsequent turn asks for more detail on one section.
# Every turn is sent with the full message history via /api/chat, so
# llama.cpp's slot cache carries the prior turns forward — TTFT/TPS at each
# context depth reflect processing the new turn against an already-filled
# context, not a cold fill from empty.

CONV_NUM_SECTIONS = 6
CONV_MIN_PREDICT  = 64    # turn size floor, used at the smallest depths
CONV_MAX_PREDICT  = 1024  # turn size cap, used at the largest depths — longer,
                          # steadier responses reduce turn-to-turn variance

def _conv_turn_budget(ctx_len: int) -> int:
    """
    Per-turn generation length, scaled to the depth currently being targeted.

    Small checkpoints (e.g. 2K) use short turns so the crossing overshoot stays
    a small fraction of the target — otherwise a single 1024-token turn could
    blow straight past a 2K checkpoint. Large checkpoints (e.g. 64K) use long
    turns so we're not spending hundreds of tiny turns to get there.
    """
    return max(CONV_MIN_PREDICT, min(CONV_MAX_PREDICT, ctx_len // 32))

CONV_OPENING_PROMPT = (
    "Explain Plato's Allegory of the Cave in detail. Structure your answer into "
    f"{CONV_NUM_SECTIONS} numbered sections (Section 1 through Section {CONV_NUM_SECTIONS}): "
    "the setup and the prisoners, the escape and the ascent, the sun and the Form "
    "of the Good, the return to the cave, philosophical interpretation, and modern "
    "relevance. Write several detailed paragraphs for each section."
)

def _conv_followup_prompt(section_n: int) -> str:
    section = ((section_n - 1) % CONV_NUM_SECTIONS) + 1
    return (
        f"Give much more detail about Section {section}, including additional "
        "examples, counterarguments, and analysis."
    )

def run_conversation_benchmarks(models, context_lengths, n_runs, warmup_runs):
    results = {}

    if not ollama_available():
        err("Ollama server not reachable — skipping LLM conversation benchmarks")
        err("Start with: ollama serve")
        return results

    for model in models:
        tag   = model["tag"]
        label = model["label"]
        short = model["short"]

        section(f"LLM Conversation: {label}")

        if not model_pulled(tag):
            warn(f"{tag} not pulled — skipping")
            warn(f"Pull with: ollama pull {tag}")
            continue

        # Unlike the single-shot test, this session keeps growing past each
        # checkpoint within the *same* num_ctx for the whole conversation. If
        # num_ctx == the top checkpoint exactly, growth lands right on the
        # ceiling with zero headroom — Ollama has to truncate/context-shift and
        # fully reprocess, which looks like a cache miss (a 100x+ TTFT spike)
        # rather than the incremental cost we're trying to measure. Pad the
        # ceiling so the top checkpoint's overshoot + all its measured turns
        # still fit comfortably inside num_ctx.
        headroom = (n_runs + 2) * CONV_MAX_PREDICT
        session_ctx_ceiling = max(context_lengths) + headroom
        max_ctx = min(model.get("max_ctx", session_ctx_ceiling), session_ctx_ceiling)
        log(f"Warming up {label} at num_ctx={max_ctx} (timeout: {RUN_TIMEOUT}s per run) ...")
        warmup_ok = True
        for warmup_i in range(warmup_runs):
            result_box = [None]
            exc_box    = [None]

            def _warmup():
                try:
                    result_box[0] = ollama_generate(
                        tag, "Hello.", timeout=RUN_TIMEOUT, num_ctx=max_ctx)
                except Exception as e:
                    exc_box[0] = e

            t = threading.Thread(target=_warmup, daemon=True)
            t_start = time.perf_counter()
            t.start()
            t.join(timeout=RUN_TIMEOUT)

            if t.is_alive():
                elapsed = time.perf_counter() - t_start
                warn(f"{label}: warmup run {warmup_i+1} did not complete within {elapsed:.0f}s")
                warn(f"{label}: model is likely too large for available memory — skipping")
                warmup_ok = False
                break
            elif exc_box[0] is not None:
                warn(f"Warmup run {warmup_i+1} failed: {exc_box[0]}")
                warmup_ok = False
                break
            else:
                log(f"Warmup run {warmup_i+1}/{warmup_runs} done")

        if not warmup_ok:
            unload_model(tag)
            continue

        results[short] = {}

        model_ctx_lengths = [c for c in context_lengths
                             if c <= model.get("max_ctx", session_ctx_ceiling)]

        messages          = []
        cumulative_tokens = 0
        section_n         = 1
        first_turn_done   = False
        model_timed_out   = False
        model_failed      = False

        def _turn(prompt_text, num_predict):
            nonlocal cumulative_tokens
            messages.append({"role": "user", "content": prompt_text})
            ttft, eval_count, tps, prompt_eval_count, response_text = ollama_chat(
                tag, messages, timeout=RUN_TIMEOUT, num_ctx=max_ctx,
                num_predict=num_predict,
            )
            messages.append({"role": "assistant", "content": response_text})
            # prompt_eval_count is the *total* prompt length for this call, not just
            # the new suffix — even when the slot cache means only the suffix was
            # actually recomputed (that's why ttft stays flat as this grows). So the
            # true context size after this turn is a plain overwrite, not a sum.
            cumulative_tokens = prompt_eval_count + eval_count
            return ttft, tps

        def _next_prompt():
            nonlocal section_n, first_turn_done
            if not first_turn_done:
                first_turn_done = True
                return CONV_OPENING_PROMPT
            prompt_text = _conv_followup_prompt(section_n)
            section_n += 1
            return prompt_text

        for ctx_len in model_ctx_lengths:
            label_ctx = f"{ctx_len // 1024}K"
            turn_budget = _conv_turn_budget(ctx_len)

            # Grow the conversation (untimed) until we've crossed this depth.
            log(f"Conversation depth {label_ctx} — growing context "
                f"(currently ~{cumulative_tokens} tokens, turn budget {turn_budget}) ...")
            try:
                while cumulative_tokens < ctx_len:
                    _turn(_next_prompt(), turn_budget)
            except Exception as e:
                is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                if is_timeout:
                    err(f"Timed out growing context for {label} — skipping remaining depths")
                    model_timed_out = True
                else:
                    err(f"Failed growing context for {label}: {e}")
                    model_failed = True
                break

            log(f"Context {label_ctx} (~{cumulative_tokens} tokens actual) — "
                f"{n_runs} runs ...")

            ttfts, tps_list = [], []
            ctx_timed_out = False

            for run_i in range(n_runs):
                try:
                    ttft, tps = _turn(_next_prompt(), turn_budget)
                    ttfts.append(ttft)
                    tps_list.append(tps)
                    print(
                        f"    run {run_i+1}/{n_runs}: "
                        f"TTFT={ttft:.2f}s  "
                        f"TPS={tps:.1f}  "
                        f"(depth~{cumulative_tokens})"
                    )
                except Exception as e:
                    is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                    if is_timeout:
                        err(f"Run {run_i+1} timed out — skipping remaining runs and depths for {label}")
                        ctx_timed_out = True
                        break
                    err(f"Run {run_i+1} failed: {e}")

            if ttfts:
                raw_ttfts = ttfts[:]
                raw_tps = tps_list[:]
                # Drop the single slowest run (by TTFT) from both lists before
                # averaging — a run flagged as an outlier is dropped entirely,
                # not just its TTFT value.
                if len(ttfts) > 1:
                    worst_idx = ttfts.index(max(ttfts))
                    ttfts    = ttfts[:worst_idx]    + ttfts[worst_idx+1:]
                    tps_list = tps_list[:worst_idx] + tps_list[worst_idx+1:]

                results[short][label_ctx] = {
                    "ttft_mean_sec":  round(mean(ttfts),    3),
                    "ttft_stdev_sec": round(stdev(ttfts),   3),
                    "tps_mean":       round(mean(tps_list), 2),
                    "tps_stdev":      round(stdev(tps_list),2),
                    "n_runs":         len(tps_list),
                    "ttft_runs":      [round(t, 3) for t in raw_ttfts],
                    "tps_runs":       [round(t, 2) for t in raw_tps],
                    "depth_tokens":   cumulative_tokens,
                }
                ok(
                    f"Context {label_ctx} done: "
                    f"TTFT={results[short][label_ctx]['ttft_mean_sec']:.2f}s  "
                    f"TPS={results[short][label_ctx]['tps_mean']:.1f}"
                )

            if ctx_timed_out:
                model_timed_out = True
                results[short]["timed_out"] = label_ctx
                break

        if model_timed_out or model_failed:
            warn(f"{label}: stopped early — moving to next model")
        log(f"Unloading {label} ...")
        unload_model(tag)
        wait_until_unloaded(tag)

    return results

# ── Embeddings benchmark ───────────────────────────────────────────────────────

CORPUS_SENTENCES = [
    "The transformer architecture revolutionized natural language processing.",
    "Attention mechanisms allow models to weigh the importance of each token.",
    "Retrieval-augmented generation combines search with language model generation.",
    "Vector databases store embeddings for efficient similarity search.",
    "Semantic search finds documents based on meaning rather than keywords.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task.",
    "Quantization reduces model size by using lower precision arithmetic.",
    "The context window determines how much text a model can process at once.",
    "Flash attention reduces memory usage during transformer forward passes.",
    "Mixture-of-experts models activate only a subset of parameters per token.",
] * 500  # 5,000 sentences total

def run_embedding_benchmarks(batch_sizes, n_runs):
    results = {}
    section(f"Embeddings: {EMBED_MODEL}")

    if not ollama_available():
        err("Ollama not running — skipping embedding benchmarks")
        return results

    if not model_pulled(EMBED_MODEL):
        warn(f"{EMBED_MODEL} not pulled — skipping")
        warn(f"Pull with: ollama pull {EMBED_MODEL}")
        return results

    ok(f"Using Ollama model: {EMBED_MODEL}")

    corpus = CORPUS_SENTENCES
    log(f"Corpus: {len(corpus)} sentences")

    for bs in batch_sizes:
        log(f"Batch size {bs} — {n_runs} runs ...")
        rates = []

        for run_i in range(n_runs):
            t0 = time.perf_counter()
            try:
                for i in range(0, len(corpus), bs):
                    batch = corpus[i:i + bs]
                    resp = requests.post(
                        f"{OLLAMA_URL}/api/embed",
                        json={"model": EMBED_MODEL, "input": batch},
                        timeout=120,
                    )
                    resp.raise_for_status()
                elapsed = time.perf_counter() - t0
                rate = len(corpus) / elapsed
                rates.append(rate)
                print(f"    run {run_i+1}/{n_runs}: {rate:.0f} sent/sec")
            except Exception as e:
                err(f"Run {run_i+1} failed: {e}")

        if rates:
            raw_rates = rates[:]
            # Drop the single slowest run before averaging
            if len(rates) > 1:
                worst_idx = rates.index(min(rates))
                rates = rates[:worst_idx] + rates[worst_idx+1:]
            key = f"batch_{bs}"
            results[key] = {
                "sentences_per_sec_mean":  round(mean(rates), 1),
                "sentences_per_sec_stdev": round(stdev(rates), 1),
                "device":                  "gpu",
                "n_runs":                  len(rates),
                "runs":                   [round(r, 1) for r in raw_rates],
            }
            ok(f"Batch {bs}: {results[key]['sentences_per_sec_mean']:.0f} sent/sec")

    return results

# ── Image generation benchmark ─────────────────────────────────────────────────

def comfyui_available():
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def build_flux_workflow(checkpoint, width, height, steps, cfg,
                        sampler, scheduler, seed, prompt, filename_prefix="bench_flux"):
    """
    Flux.1 txt2img workflow.

    The BFL flux1-schnell/dev .safetensors files are transformer-only (no CLIP,
    no VAE), so we load CLIP and VAE via separate nodes rather than relying on
    CheckpointLoaderSimple output slots 1 and 2 (which would be None).
    """
    return {
        # UNet from checkpoint (output 0 = model; slots 1/2 are None for BFL files)
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        # Dual CLIP for Flux: T5-XXL + CLIP-L
        "12": {"class_type": "DualCLIPLoader",
               "inputs": {
                   "clip_name1": "t5xxl_fp16.safetensors",
                   "clip_name2": "clip_l.safetensors",
                   "type": "flux",
               }},
        # VAE loaded separately
        "13": {"class_type": "VAELoader",
               "inputs": {"vae_name": "ae.safetensors"}},
        # Encode prompt using dual CLIP — no negative for Flux
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["12", 0]}},
        # Flux guidance node (replaces CFGGuider)
        "3": {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["2", 0], "guidance": cfg}},
        # Empty latent image
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        # Noise source
        "5": {"class_type": "RandomNoise",
              "inputs": {"noise_seed": seed}},
        # Basic guider wrapping FluxGuidance conditioning
        "6": {"class_type": "BasicGuider",
              "inputs": {"model": ["1", 0], "conditioning": ["3", 0]}},
        # Sampler selection
        "7": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": sampler}},
        # Scheduler
        "8": {"class_type": "BasicScheduler",
              "inputs": {
                  "model": ["1", 0],
                  "scheduler": scheduler,
                  "steps": steps,
                  "denoise": 1.0,
              }},
        # Run the sampler
        "9": {"class_type": "SamplerCustomAdvanced",
              "inputs": {
                  "noise": ["5", 0],
                  "guider": ["6", 0],
                  "sampler": ["7", 0],
                  "sigmas": ["8", 0],
                  "latent_image": ["4", 0],
              }},
        # Decode latent to image using separate VAE
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["13", 0]}},
        # Save
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix}},
    }

def build_sd3_workflow(checkpoint, width, height, steps, cfg,
                       sampler, scheduler, seed, prompt, filename_prefix="bench_sd3"):
    """
    SD3.5 Large txt2img workflow for ComfyUI.

    sd3.5_large.safetensors contains the UNet and VAE but NOT the text encoders.
    clip_l.safetensors, clip_g.safetensors, and t5xxl_fp16.safetensors must be
    present in ComfyUI/models/clip/ (downloaded by setup_check.py).
    SD3 uses 16-channel latents — EmptySD3LatentImage is required.
    """
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "TripleCLIPLoader",
              "inputs": {
                  "clip_name1": "clip_l.safetensors",
                  "clip_name2": "clip_g.safetensors",
                  "clip_name3": "t5xxl_fp16.safetensors",
                  "type": "sd3",
              }},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "", "clip": ["2", 0]}},
        "5": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {
                  "model":          ["1", 0],
                  "positive":       ["3", 0],
                  "negative":       ["4", 0],
                  "latent_image":   ["5", 0],
                  "seed":           seed,
                  "steps":          steps,
                  "cfg":            cfg,
                  "sampler_name":   sampler,
                  "scheduler":      scheduler,
                  "denoise":        1.0,
              }},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
    }

def build_sdxl_workflow(checkpoint, width, height, steps, cfg,
                        sampler, scheduler, seed, prompt, filename_prefix="bench"):
    """Minimal SDXL txt2img workflow for ComfyUI API."""
    return {
        "4":  {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": checkpoint}},
        "6":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": "", "clip": ["4", 1]}},
        "8":  {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["4", 2]}},
        "9":  {"class_type": "SaveImage",
               "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}},
        "5":  {"class_type": "EmptyLatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {
                   "model": ["4", 0], "positive": ["6", 0],
                   "negative": ["7", 0], "latent_image": ["5", 0],
                   "seed": seed, "steps": steps, "cfg": cfg,
                   "sampler_name": sampler, "scheduler": scheduler,
                   "denoise": 1.0,
               }},
    }

def comfyui_submit(workflow: dict, timeout: int = 300) -> tuple[float, list[dict]]:
    """Submit a workflow to ComfyUI, poll until done.

    Returns (elapsed_sec, images) where images is a list of
    {"filename": str, "subfolder": str, "type": str} dicts from all output nodes.
    """
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        raise RuntimeError(f"ComfyUI rejected workflow (HTTP {resp.status_code}): {detail}")
    prompt_id = resp.json()["prompt_id"]

    # Start timing AFTER submission so we measure generation time only,
    # and stale history entries from previous runs won't match this prompt_id.
    t0 = time.perf_counter()
    seen = False  # True once we see this prompt_id appear in history

    while True:
        time.sleep(1)
        try:
            status = requests.get(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            ).json()
        except Exception:
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"ComfyUI job timed out after {timeout}s")
            continue

        if prompt_id in status:
            seen = True
            job = status[prompt_id]
            job_status = job.get("status", {})

            # Check for errors first
            if job_status.get("status_str") == "error" or job.get("error"):
                msgs = job.get("error") or job_status.get("messages", [])
                raise RuntimeError(f"ComfyUI job failed: {msgs}")

            if job_status.get("completed"):
                elapsed = time.perf_counter() - t0
                images = []
                for node_out in job.get("outputs", {}).values():
                    images.extend(node_out.get("images", []))
                return elapsed, images

        if time.perf_counter() - t0 > timeout:
            if not seen:
                raise TimeoutError(
                    f"ComfyUI job never appeared in history after {timeout}s "
                    f"— workflow may have errored before queuing"
                )
            raise TimeoutError(f"ComfyUI job timed out after {timeout}s")

def save_comfyui_image(img: dict, dest: Path) -> None:
    """Fetch a generated image from ComfyUI and save it locally."""
    resp = requests.get(
        f"{COMFYUI_URL}/view",
        params={
            "filename": img["filename"],
            "subfolder": img.get("subfolder", ""),
            "type":     img.get("type", "output"),
        },
        timeout=30,
    )
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)

def run_image_benchmarks(image_models, resolutions, seed, prompt, n_runs,
                         comfyui_dir, timeout=None, save_fn=None):
    if timeout is None:
        timeout = RUN_TIMEOUT
    results = {}
    section("Image Generation via ComfyUI")

    checkpoints_dir = comfyui_dir / "models" / "checkpoints"

    for model in image_models:
        label      = model["label"]
        checkpoint = model["checkpoint"]
        workflow_t = model["workflow"]
        steps      = model["steps"]
        cfg        = model["cfg"]
        sampler    = model["sampler"]
        scheduler  = model["scheduler"]
        short      = model["short"]

        try:
            # Skip if checkpoint not present
            ckpt_path = checkpoints_dir / checkpoint
            if not ckpt_path.exists():
                warn(f"{label}: checkpoint not found at {ckpt_path} — skipping")
                log(f"Download and place at: {ckpt_path}")
                continue

            ok(f"{label}: checkpoint found ({ckpt_path.stat().st_size / (1024**3):.1f} GB)")
            results[short] = {"label": label, "checkpoint": checkpoint,
                              "steps": steps, "resolutions": {}}

            # Warmup: one generation at the smallest resolution to trigger Metal/CUDA
            # shader compilation before timing starts.
            w0, h0 = resolutions[0]
            log(f"{label}: warmup run ({w0}x{h0}, timeout: {timeout}s) ...")
            warmup_ok = True
            try:
                if workflow_t == "flux":
                    wf = build_flux_workflow(checkpoint, w0, h0, steps, cfg,
                                             sampler, scheduler, seed, prompt,
                                             filename_prefix=f"{short}_warmup")
                elif workflow_t == "sd3":
                    wf = build_sd3_workflow(checkpoint, w0, h0, steps, cfg,
                                            sampler, scheduler, seed, prompt,
                                            filename_prefix=f"{short}_warmup")
                else:
                    wf = build_sdxl_workflow(checkpoint, w0, h0, steps, cfg,
                                             sampler, scheduler, seed, prompt,
                                             filename_prefix=f"{short}_warmup")
                comfyui_submit(wf, timeout=timeout)
                ok(f"{label}: warmup done")
            except Exception as e:
                warn(f"{label}: warmup failed ({e}) — skipping")
                warmup_ok = False

            if not warmup_ok:
                continue

            img_dir = SCRIPT_DIR / "benchmark_images"

            model_timed_out = False
            for (w, h) in resolutions:
                res_label = f"{w}x{h}"
                log(f"{label} @ {res_label} — {n_runs} runs ...")
                times = []
                last_images: list[dict] = []

                for run_i in range(n_runs):
                    try:
                        prefix = f"{short}_{res_label}_run{run_i + 1}"
                        if workflow_t == "flux":
                            wf = build_flux_workflow(
                                checkpoint, w, h, steps, cfg,
                                sampler, scheduler, seed, prompt,
                                filename_prefix=prefix)
                        elif workflow_t == "sd3":
                            wf = build_sd3_workflow(
                                checkpoint, w, h, steps, cfg,
                                sampler, scheduler, seed, prompt,
                                filename_prefix=prefix)
                        else:
                            wf = build_sdxl_workflow(
                                checkpoint, w, h, steps, cfg,
                                sampler, scheduler, seed, prompt,
                                filename_prefix=prefix)

                        elapsed, images = comfyui_submit(wf, timeout=timeout)
                        times.append(elapsed)
                        last_images = images
                        print(f"    run {run_i+1}/{n_runs}: {elapsed:.1f}s")
                    except TimeoutError:
                        err(f"Run {run_i+1} timed out — skipping {label}")
                        model_timed_out = True
                        break
                    except Exception as e:
                        err(f"Run {run_i+1} failed: {e}")

                if times:
                    raw_times = times[:]
                    # Drop the single slowest run before averaging
                    if len(times) > 1:
                        worst_idx = times.index(max(times))
                        times = times[:worst_idx] + times[worst_idx+1:]
                    results[short]["resolutions"][res_label] = {
                        "sec_per_image_mean":  round(mean(times),  2),
                        "sec_per_image_stdev": round(stdev(times) if len(times) > 1 else 0.0, 2),
                        "n_runs":              len(times),
                        "runs":               [round(t, 2) for t in raw_times],
                    }
                    ok(f"{label} @ {res_label}: "
                       f"{results[short]['resolutions'][res_label]['sec_per_image_mean']:.1f}s/image")

                if not last_images:
                    warn(f"{label} @ {res_label}: no images in ComfyUI history response — skipping save")
                else:
                    img  = last_images[0]
                    dest = img_dir / f"{short}_{res_label}.png"
                    saved = False
                    try:
                        save_comfyui_image(img, dest)
                        ok(f"Saved image → benchmark_images/{dest.name}")
                        saved = True
                    except Exception as e:
                        warn(f"HTTP image fetch failed ({e}) — trying direct file copy")
                    if not saved:
                        # Fallback: copy directly from ComfyUI's output directory
                        subfolder = img.get("subfolder", "")
                        src = (comfyui_dir / "output" / subfolder / img["filename"]
                               if subfolder else comfyui_dir / "output" / img["filename"])
                        try:
                            import shutil
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dest)
                            ok(f"Saved image (file copy) → benchmark_images/{dest.name}")
                        except Exception as e:
                            warn(f"Could not save image: {e}")

                if model_timed_out:
                    warn(f"{label}: timed out — moving to next model")
                    break

        finally:
            if save_fn:
                save_fn(results)

    log("Unloading ComfyUI models from VRAM ...")
    try:
        requests.post(f"{COMFYUI_URL}/free",
                      json={"unload_models": True, "free_memory": True},
                      timeout=10)
        ok("ComfyUI models unloaded")
    except Exception as e:
        warn(f"Could not unload ComfyUI models: {e}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM benchmark suite")
    parser.add_argument(
        "--tests", nargs="+",
        choices=["llm", "conv", "emb", "img"],
        default=["llm", "conv", "emb", "img"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Number of measured runs per test (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help=f"Warmup runs before measuring (default: {WARMUP_RUNS})",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Seconds per run (warmup and measured) before aborting this model (default: 300)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON file (default: results_<hostname>_<timestamp>.json)",
    )
    parser.add_argument(
        "--comfyui", type=str, default=None,
        help=f"Path to ComfyUI directory (default: {COMFYUI_DIR})",
    )
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--small-only", action="store_true",
        help="Run only small-tier models (≤20B params): Llama 3.1 8B Q4_K_M, DeepSeek-R1 8B, Gemma 4 E4B, GPT-OSS 20B (MXFP4)",
    )
    size_group.add_argument(
        "--medium-only", action="store_true",
        help="Run only medium-tier models (26–35B params): Gemma 4 26B, DeepSeek-R1 32B, Qwen3.6 35B-A3B",
    )
    size_group.add_argument(
        "--large-only", action="store_true",
        help="Run only large-tier models (70B+ params): Llama 3.3 70B Q4_K_M, DeepSeek-R1 70B, GPT-OSS 120B (MXFP4)",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level config
    global RUN_TIMEOUT
    if args.timeout is not None:
        RUN_TIMEOUT = args.timeout

    # Select model tier
    if args.small_only:
        llm_models = LLM_MODELS_SMALL
        tier_label = "small only (≤16GB)"
    elif args.medium_only:
        llm_models = LLM_MODELS_MEDIUM
        tier_label = "medium only (16–32GB)"
    elif args.large_only:
        llm_models = LLM_MODELS_LARGE
        tier_label = "large only (32GB+)"
    else:
        llm_models = LLM_MODELS
        tier_label = "all (small + medium + large)"

    comfyui_dir = Path(args.comfyui) if args.comfyui else COMFYUI_DIR

    profile  = build_profile()
    _safe = re.sub(r'[\\/:*?"<>|\s]+', '_', profile['hostname']).strip('_')
    _start_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = args.out or f"results_{_safe}_{_start_stamp}.json"

    print(f"\n{BOLD}LLM Benchmark Suite{RESET}")
    print(f"  Host:      {profile['hostname']}")
    print(f"  OS:        {profile['os']}")
    print(f"  Backend:   {profile['backend']}")
    print(f"  RAM:       {profile['ram_gb']} GB")
    print(f"  Runs:      {args.runs} measured + {args.warmup} warmup")
    print(f"  Timeout:   {RUN_TIMEOUT}s per run")
    print(f"  Models:    {tier_label}")
    print(f"  Tests:     {', '.join(args.tests)}")
    print(f"  ComfyUI:   {comfyui_dir}")

    # Register cleanup for Ctrl-C and normal exit
    def _cleanup(sig=None, frame=None):
        if _managed_procs:
            print(f"\n{YELLOW}Cleaning up managed servers ...{RESET}")
            _shutdown_managed()
        if sig is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    results = {
        "version":         VERSION,
        "profile":         profile,
        "llm":             {},
        "llm_conversation": {},
        "embeddings":      {},
        "images":          {},
    }

    def _checkpoint(label=""):
        Path(out_path).write_text(json.dumps(results, indent=2))
        if label:
            log(f"Partial results saved to {out_path} ({label})")

    try:
        # ── LLM ───────────────────────────────────────────────────────────────
        if "llm" in args.tests or "conv" in args.tests:
            section("Starting Servers")
            ensure_ollama()

        if "llm" in args.tests:
            results["llm"] = run_llm_benchmarks(
                models=llm_models,
                context_lengths=CONTEXT_LENGTHS,
                n_runs=args.runs,
                warmup_runs=args.warmup,
            )
            _checkpoint("LLM done")

        if "conv" in args.tests:
            results["llm_conversation"] = run_conversation_benchmarks(
                models=llm_models,
                context_lengths=CONTEXT_LENGTHS,
                n_runs=args.runs,
                warmup_runs=args.warmup,
            )
            _checkpoint("LLM conversation done")

        # ── Embeddings ─────────────────────────────────────────────────────────
        if "emb" in args.tests:
            results["embeddings"] = run_embedding_benchmarks(
                batch_sizes=EMBED_BATCH_SIZES,
                n_runs=args.runs,
            )
            _checkpoint("embeddings done")

        # ── Image generation ───────────────────────────────────────────────────
        if "img" in args.tests:
            section("Starting Servers")
            # Hard guarantee: nothing from Ollama in memory before ComfyUI loads
            if ollama_available():
                log("Ensuring all Ollama models are unloaded ...")
                unload_all_models()
            comfyui_started = ensure_comfyui(comfyui_dir)
            if not comfyui_started:
                warn("Image benchmarks will be skipped")
            else:
                def _img_save(img_partial):
                    results["images"] = img_partial
                    _checkpoint()

                results["images"] = run_image_benchmarks(
                    image_models=IMAGE_MODELS,
                    resolutions=IMAGE_RESOLUTIONS,
                    seed=IMAGE_SEED,
                    prompt=IMAGE_PROMPT,
                    n_runs=args.runs,
                    comfyui_dir=comfyui_dir,
                    save_fn=_img_save,
                )
                # Shut down ComfyUI as soon as image tests are done
                # to free GPU memory before saving results
                _shutdown_managed()

    finally:
        # Always shut down anything still running, even on error
        _shutdown_managed()

    # ── Save results ───────────────────────────────────────────────────────────
    section("Saving Results")
    Path(out_path).write_text(json.dumps(results, indent=2))
    ok(f"Results saved to: {out_path}")
    print(f"\n  Copy this file to your comparison machine and run:")
    print(f"  python compare.py results_*.json\n")
    section("Done")
    ok("All servers shut down. Benchmark complete.")

if __name__ == "__main__":
    main()
