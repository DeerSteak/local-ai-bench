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

  3. Embeddings — nomic-embed-text and mxbai-embed-large via Ollama
     Metrics: sentences/sec at batch sizes 32, 128, 512

Servers are managed automatically:
  - Ollama: started if not already running, shut down on exit if we started it
  - ComfyUI: started before image tests, shut down cleanly when done

Usage:
  python benchmark.py                  # run all tests
  python benchmark.py --tests llm      # run only LLM single-shot tests
  python benchmark.py --tests llm emb  # run LLM + embeddings
  python benchmark.py --tests conv     # run only LLM conversation tests
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
import urllib.error
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

from models import IMAGE_MODELS, LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS, EMBED_MODELS  # noqa: E402

CONTEXT_LENGTHS = [2048, 8192, 32768, 65536]   # tokens (approximate, via prompt padding)
IMAGE_RESOLUTIONS = [(1024, 1024), (1536, 1536)]
# Steps are now per-model in IMAGE_MODELS
IMAGE_SEED  = 42
IMAGE_PROMPT = (
    "A photorealistic high-end gaming PC build with RGB lighting, "
    "multiple GPUs, custom water cooling, shot in a dark room, "
    "highly detailed, 8k resolution"
)

VERSION        = "1.2"
WARMUP_RUNS    = 2
N_RUNS         = 3   # measured runs per test — every test averages exactly this many
RUN_TIMEOUT = 300   # seconds per run (warmup and measured) before aborting

# Tokens/sec below which a model is considered unusable for real conversation
# use and skipped from the (expensive) conversation test — decode speed this
# low means every turn of a real back-and-forth chat is a slog, regardless of
# how the single-shot test's TTFT looked. Checked against every context depth
# the single-shot LLM test reported, not just one.
SLOW_MODEL_MIN_TPS = 15.0   # tokens/sec

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

# True while Ollama is running with GPU devices hidden (--emb-cpu-only). If the
# script dies before it restores normal mode, _shutdown_managed() must not
# leave that GPU-hidden process running silently in the background.
_cpu_only_active = False

def _shutdown_managed():
    """Terminate any servers we started."""
    if _cpu_only_active:
        warn("Exiting while Ollama is in forced CPU-only mode — killing it "
             "rather than leaving GPU devices hidden in the background")
        stop_all_ollama()
    for proc in _managed_procs:
        if proc.poll() is None:
            log(f"Stopping managed process (pid {proc.pid}) ...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    _managed_procs.clear()

# Path to the log file capturing the current/most recent Ollama server's
# stdout+stderr — set by start_ollama() so a crash's actual message can be
# surfaced later instead of silently discarded.
_ollama_log_path: Path | None = None

def tail_ollama_log(n_lines: int = 40) -> str:
    """Return the last n_lines of the current Ollama server's captured
    output, for surfacing the real crash reason instead of guessing."""
    if _ollama_log_path is None:
        return "(no Ollama log captured this session)"
    try:
        lines = _ollama_log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n_lines:]) or "(log file is empty)"
    except Exception as e:
        return f"(failed to read Ollama log: {e})"

def start_ollama(extra_env: dict | None = None, timeout: int = 15) -> bool:
    """Start 'ollama serve', optionally with extra/overridden environment
    variables (e.g. HIP_VISIBLE_DEVICES="" to force CPU-only). Tracked in
    _managed_procs for cleanup on exit. Returns True once reachable."""
    global _ollama_log_path

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    os_name = platform.system()
    try:
        log_fh = tempfile.NamedTemporaryFile(
            mode="w", suffix="-ollama-server.log", delete=False
        )
        _ollama_log_path = Path(log_fh.name)
        kwargs = dict(stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        if os_name == "Windows":
            # On Windows Ollama is a system tray app; 'ollama serve' starts the server
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(["ollama", "serve"], **kwargs)
        log_fh.close()
        _managed_procs.append(proc)
    except FileNotFoundError:
        err("'ollama' not found in PATH — install from https://ollama.com/download")
        return False

    for i in range(timeout):
        time.sleep(1)
        if ollama_available():
            ok(f"Ollama started (pid {proc.pid}) — log: {_ollama_log_path}")
            return True
        if proc.poll() is not None:
            err(f"Ollama exited unexpectedly (code {proc.returncode})")
            err(f"Last output:\n{tail_ollama_log()}")
            return False

    err(f"Ollama did not respond within {timeout} seconds")
    return False

def stop_all_ollama(timeout: int = 15) -> None:
    """Kill any running Ollama server, including one this script didn't
    start itself, so a fresh instance can be launched with different
    environment variables (e.g. to force CPU-only for embeddings)."""
    os_name = platform.system()
    try:
        if os_name == "Windows":
            subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "ollama serve"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if not ollama_available():
            return
        time.sleep(1)
    warn(f"Ollama still reachable {timeout}s after attempting to stop it")

def ensure_ollama():
    """Start Ollama if not already running. Returns True if available."""
    if ollama_available():
        ok("Ollama already running")
        return True

    log("Ollama not running — attempting to start ...")
    return start_ollama()
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

def _ollama_urlopen(req, timeout):
    """urlopen wrapper that surfaces the response body on HTTP error status.

    Ollama puts the actual failure reason (OOM, backend/driver crash, model
    load failure, etc.) in a JSON body even on a 500 — the bare HTTPError
    only says "HTTP Error 500: Internal Server Error" and hides it.
    """
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"Ollama returned HTTP {e.code}: {detail[:500]}") from None

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

    with _ollama_urlopen(req, timeout) as resp:
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

    prompt_eval_count is the *total* number of tokens in this call's prompt
    (ground truth for how deep the context is), not just the new suffix — even
    when `messages` shares a prefix with a prior call and llama.cpp's slot
    cache skips re-processing it. The cache reuse shows up as a low
    prompt_eval_duration/ttft instead, not a smaller prompt_eval_count.
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
    thinking_parts    = []

    with _ollama_urlopen(req, timeout) as resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            message  = chunk.get("message", {})
            content  = message.get("content")
            thinking = message.get("thinking")

            if ttft is None and (content or thinking):
                ttft = time.perf_counter() - t_start

            if content:
                tokens += 1
                response_parts.append(content)
            if thinking:
                thinking_parts.append(thinking)

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
    # Reasoning models (Qwen3.x, DeepSeek-R1, Gemma-thinking, ...) can stream
    # their entire turn through message.thinking with message.content left
    # empty. Fall back to the thinking text so the conversation history we
    # feed back on the next turn isn't an empty assistant message — otherwise
    # the growth loop below wildly overcounts how much context actually
    # persists (eval_count reflects thinking tokens that never make it into
    # the next turn's prompt at all).
    response_text = "".join(response_parts) or "".join(thinking_parts)
    return ttft, eval_count, tps, prompt_eval_count, response_text

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

def _slow_tps_early_exit(results, short, label, label_ctx, is_first_ctx, tps_list, force_all):
    """Shared by the LLM prefill and conversation tests: if the first context
    depth's decode speed is below SLOW_MODEL_MIN_TPS, mark the model slow and
    tell the caller to stop testing deeper contexts (unless force_all)."""
    if not (is_first_ctx and tps_list and mean(tps_list) < SLOW_MODEL_MIN_TPS):
        return False
    if force_all:
        warn(f"{label}: {mean(tps_list):.1f} tok/s at {label_ctx} context is below "
             f"{SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff — --force-all set, continuing anyway")
        return False
    warn(f"{label}: {mean(tps_list):.1f} tok/s at {label_ctx} context is below "
         f"{SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff — marking slow, skipping deeper contexts")
    results[short]["slow_tps"] = label_ctx
    return True

def run_llm_benchmarks(models, context_lengths, warmup_runs, force_all=False):
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
            log(f"Context {label_ctx} — {N_RUNS} runs ...")

            ttfts, tps_list = [], []
            ctx_timed_out = False

            for run_i in range(N_RUNS):
                try:
                    prompt = build_prompt_for_context(ctx_len)
                    ttft, tokens, tps = ollama_generate(
                        tag, prompt, timeout=RUN_TIMEOUT, num_ctx=ctx_len
                    )
                    ttfts.append(ttft)
                    tps_list.append(tps)
                    print(
                        f"    run {run_i+1}/{N_RUNS}: "
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
                results[short][label_ctx] = {
                    "ttft_mean_sec":  round(mean(ttfts),    3),
                    "ttft_stdev_sec": round(stdev(ttfts),   3),
                    "tps_mean":       round(mean(tps_list), 2),
                    "tps_stdev":      round(stdev(tps_list),2),
                    "n_runs":         len(tps_list),
                    "ttft_runs":      [round(t, 3) for t in ttfts],
                    "tps_runs":       [round(t, 2) for t in tps_list],
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

            is_first_ctx = ctx_len == model_ctx_lengths[0]
            if _slow_tps_early_exit(results, short, label, label_ctx, is_first_ctx, tps_list, force_all):
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

CONV_NUM_SECTIONS    = 6
CONV_MIN_PREDICT     = 64    # measured-phase turn size floor, smallest depths
CONV_MAX_PREDICT     = 1024  # measured-phase turn size cap, largest depths —
                              # longer, steadier responses reduce turn-to-turn
                              # variance. Governs only the N_RUNS measured runs
                              # reported per checkpoint (via _conv_turn_budget
                              # below) — growth turns are separate, see
                              # CONV_GROWTH_PREDICT.
CONV_GROWTH_PREDICT  = 2000  # turn size for every growth turn (opening answer
                              # and all "give more detail" follow-ups used to
                              # build up to each checkpoint) — large so growth
                              # is a handful of substantial turns rather than
                              # dozens of small ones, more like a real
                              # conversation. Doesn't affect what's measured;
                              # some overshoot past each checkpoint is expected
                              # and fine.

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

def run_conversation_benchmarks(models, context_lengths, warmup_runs, force_all=False):
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
        # ceiling so the top checkpoint's crossing overshoot (one growth turn,
        # up to CONV_GROWTH_PREDICT) plus all its measured turns (up to
        # CONV_MAX_PREDICT each, plus 2 extra turns of buffer) still fit
        # comfortably inside num_ctx.
        headroom = CONV_GROWTH_PREDICT + (N_RUNS + 2) * CONV_MAX_PREDICT
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
            # prompt_eval_count is the *total* prompt length Ollama reports for this
            # call (ground truth of what's actually in context going into it) — even
            # when the slot cache means only the suffix was actually recomputed
            # (that's why ttft stays flat as this grows). We deliberately don't add
            # eval_count on top: for reasoning models a turn's generated tokens can
            # include large amounts of thinking content that a template silently
            # drops from history on the next turn, so eval_count doesn't reliably
            # predict what will actually persist. The next turn's prompt_eval_count
            # (i.e. this same assignment, one call later) is what tells us the truth.
            cumulative_tokens = prompt_eval_count
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

            # Grow the conversation (untimed) until we've crossed this depth,
            # using large CONV_GROWTH_PREDICT turns regardless of this
            # checkpoint's (much smaller) measured-phase turn_budget — see
            # CONV_GROWTH_PREDICT above for why. Once within CONV_GROWTH_PREDICT
            # of the ceiling, shrink num_predict to the remaining gap instead of
            # blindly using the full budget, so the crossing turn lands close to
            # the checkpoint rather than potentially overshooting it by up to
            # CONV_GROWTH_PREDICT tokens. num_predict is a cap, not a target —
            # a turn can still come in short (early stop) — so this can still
            # take more than one closing turn; the loop just keeps recomputing
            # the remaining gap each time.
            log(f"Conversation depth {label_ctx} — growing context "
                f"(currently ~{cumulative_tokens} tokens) ...")
            try:
                while cumulative_tokens < ctx_len:
                    remaining = ctx_len - cumulative_tokens
                    num_predict = (CONV_GROWTH_PREDICT if remaining > CONV_GROWTH_PREDICT
                                   else max(CONV_MIN_PREDICT, remaining))
                    _turn(_next_prompt(), num_predict)
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
                f"{N_RUNS} runs ...")

            ttfts, tps_list = [], []
            ctx_timed_out = False

            for run_i in range(N_RUNS):
                try:
                    ttft, tps = _turn(_next_prompt(), turn_budget)
                    ttfts.append(ttft)
                    tps_list.append(tps)
                    print(
                        f"    run {run_i+1}/{N_RUNS}: "
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
                results[short][label_ctx] = {
                    "ttft_mean_sec":  round(mean(ttfts),    3),
                    "ttft_stdev_sec": round(stdev(ttfts),   3),
                    "tps_mean":       round(mean(tps_list), 2),
                    "tps_stdev":      round(stdev(tps_list),2),
                    "n_runs":         len(tps_list),
                    "ttft_runs":      [round(t, 3) for t in ttfts],
                    "tps_runs":       [round(t, 2) for t in tps_list],
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

            is_first_ctx = ctx_len == model_ctx_lengths[0]
            if _slow_tps_early_exit(results, short, label, label_ctx, is_first_ctx, tps_list, force_all):
                break

        if model_timed_out or model_failed:
            warn(f"{label}: stopped early — moving to next model")
        log(f"Unloading {label} ...")
        unload_model(tag)
        wait_until_unloaded(tag)

    return results

# ── Embeddings benchmark ───────────────────────────────────────────────────────

# Real document-ingestion workload: chunk one real document the way a RAG
# pipeline would (paragraph-sized pieces), then embed every chunk from it in
# a single call, the way an app actually ingests one document — rather than
# sweeping arbitrary "batch sizes" that don't correspond to any real client
# behavior. max_words caps every chunk well under any embedding model's
# context length (mxbai-embed-large's is 512 tokens) regardless of how the
# source document is formatted — this is also what fixes "content length
# exceeds context length" errors that an unbounded chunker can produce when
# it runs into a document's own markdown tables/code blocks.
EMBED_DOCUMENT_PATH = SCRIPT_DIR / "sample_document.txt"
EMBED_CHUNK_MAX_WORDS = 150
EMBED_CHUNK_MIN_WORDS = 6

def chunk_document(path: Path = EMBED_DOCUMENT_PATH,
                    max_words: int = EMBED_CHUNK_MAX_WORDS,
                    min_words: int = EMBED_CHUNK_MIN_WORDS) -> list[str]:
    """Split a document into paragraph-sized chunks, each capped at
    max_words. Paragraphs longer than max_words are packed sentence-by-
    sentence up to the cap; anything that's still too long after that
    (e.g. a code block or table with no sentence punctuation) is hard-split
    by word count so no chunk can ever exceed the cap."""
    paragraphs = [p.strip() for p in path.read_text().split("\n\n") if p.strip()]

    def split_oversized(words: list[str]) -> list[str]:
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    chunks = []
    for para in paragraphs:
        words = " ".join(para.split()).split()
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            chunks.append(" ".join(words))
            continue

        current, current_len = [], 0
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(words)):
            sentence_words = sentence.split()
            if len(sentence_words) > max_words:
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                chunks.extend(split_oversized(sentence_words))
                continue
            if current_len + len(sentence_words) > max_words and current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.extend(sentence_words)
            current_len += len(sentence_words)
        if current:
            chunks.append(" ".join(current))

    return chunks

# Records model/document combos that crashed Ollama's runner repeatedly
# (deterministically, not a transient blip) so future runs don't waste time
# rediscovering the same crash. Delete this file to retry a skipped model.
EMBED_CRASH_CACHE = Path(".embed_crash_cache.json")

def _load_crash_cache() -> dict:
    try:
        return json.loads(EMBED_CRASH_CACHE.read_text())
    except Exception:
        return {}

def _save_crash_cache(cache: dict) -> None:
    try:
        EMBED_CRASH_CACHE.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        warn(f"Failed to save embedding crash cache: {e}")

def run_embedding_benchmarks(models, warmup_runs=WARMUP_RUNS):
    results = {}

    if not ollama_available():
        err("Ollama not running — skipping embedding benchmarks")
        return results

    crash_cache = _load_crash_cache()
    chunks = chunk_document()
    log(f"Corpus: {len(chunks)} chunks from {EMBED_DOCUMENT_PATH.name} "
        f"(max {EMBED_CHUNK_MAX_WORDS} words/chunk)")

    for model in models:
        tag   = model["tag"]
        label = model["label"]
        short = model["short"]

        section(f"Embeddings: {label}")

        if not model_pulled(tag):
            warn(f"{tag} not pulled — skipping")
            warn(f"Pull with: ollama pull {tag}")
            continue

        ok(f"Using Ollama model: {tag}")

        if tag in crash_cache:
            detail = crash_cache[tag]
            warn(f"{tag} previously crashed Ollama's runner repeatedly on "
                 f"{detail.get('crashed_at', 'an earlier run')} — skipping "
                 f"(delete {EMBED_CRASH_CACHE} to retry)")
            results[short] = {
                "label": label,
                "skipped": True,
                "skip_reason": "known_crash",
                "skip_detail": f"Crashed Ollama's runner repeatedly on {detail.get('crashed_at', 'an earlier run')}",
            }
            continue

        # Warm up model (load into memory) before measuring. The first embed
        # call against a freshly-unloaded model pays a one-time cost, model
        # weights loading into memory, first-call kernel/graph setup, that
        # has nothing to do with steady-state throughput — folding it into a
        # measured run would understate this model's real performance.
        log(f"Warming up {label} ...")
        for warmup_i in range(warmup_runs):
            try:
                resp = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": tag, "input": chunks},
                    timeout=120,
                )
                if not resp.ok:
                    warn(f"Warmup run {warmup_i+1} failed: HTTP {resp.status_code}")
                else:
                    log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
            except Exception as e:
                warn(f"Warmup run {warmup_i+1} failed: {e}")
                if isinstance(e, requests.exceptions.ConnectionError) or "actively refused" in str(e):
                    wait_t0 = time.perf_counter()
                    while time.perf_counter() - wait_t0 < 30:
                        if ollama_available():
                            break
                        time.sleep(2)

        log(f"Embedding {len(chunks)} chunks in one call — {N_RUNS} runs ...")
        rates = []

        MAX_CRASH_RETRIES = 2
        run_i = 0
        crash_retries = 0
        gave_up_from_crashes = False
        while run_i < N_RUNS:
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": tag, "input": chunks},
                    timeout=120,
                )
                if not resp.ok:
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text[:500]
                    raise RuntimeError(
                        f"Ollama rejected embed request (HTTP {resp.status_code}, "
                        f"n_chunks={len(chunks)}): {detail}"
                    )
                elapsed = time.perf_counter() - t0
                rate = len(chunks) / elapsed
                rates.append(rate)
                print(f"    run {run_i+1}/{N_RUNS}: {rate:.0f} chunks/sec")
                run_i += 1
            except Exception as e:
                err(f"Run {run_i+1} failed: {e}")
                # A connection-refused error means Ollama's model runner
                # subprocess had already died (commonly OOM) before this
                # request. Wait for the main Ollama server to notice and
                # respawn it, then retry this same run — up to a capped
                # number of attempts, since a deterministic crash on this
                # document would just recur identically forever.
                if isinstance(e, requests.exceptions.ConnectionError) or "actively refused" in str(e):
                    crash_retries += 1
                    err(f"Ollama's model runner appears to have crashed embedding {tag} "
                        f"— last server output:\n{tail_ollama_log()}")
                    if crash_retries > MAX_CRASH_RETRIES:
                        err(f"Ollama's model runner crashed {crash_retries} times — giving up on {tag}")
                        gave_up_from_crashes = True
                        break
                    warn(f"Waiting for recovery, retry {crash_retries}/{MAX_CRASH_RETRIES} ...")
                    recovered = False
                    wait_t0 = time.perf_counter()
                    while time.perf_counter() - wait_t0 < 30:
                        if ollama_available():
                            recovered = True
                            break
                        time.sleep(2)
                    if not recovered:
                        warn("Ollama did not become reachable again within 30s — giving up on this model")
                        break
                    # don't advance run_i — retry the same run now that Ollama is back
                else:
                    run_i += 1

        if rates:
            results[short] = {
                "label": label,
                "chunks_per_sec_mean":  round(mean(rates), 1),
                "chunks_per_sec_stdev": round(stdev(rates), 1),
                "device":               "gpu",
                "n_chunks":             len(chunks),
                "n_runs":               len(rates),
                "runs":                [round(r, 1) for r in rates],
            }
            ok(f"{label}: {results[short]['chunks_per_sec_mean']:.0f} chunks/sec")
        elif gave_up_from_crashes:
            crashed_at = datetime.now().isoformat(timespec="seconds")
            crash_cache[tag] = {"crashed_at": crashed_at}
            _save_crash_cache(crash_cache)
            results[short] = {
                "label": label,
                "skipped": True,
                "skip_reason": "known_crash",
                "skip_detail": f"Ollama's runner crashed repeatedly embedding this document ({crashed_at})",
            }

    return results

# ── Image generation benchmark ─────────────────────────────────────────────────

def comfyui_available():
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def comfyui_free_models(timeout: int = 10) -> None:
    """Unload whatever checkpoint(s) ComfyUI currently has resident in memory.

    ComfyUI's own automatic model-swap-on-load is the only thing that would
    otherwise free a previous checkpoint, and on the MPS backend its free-VRAM
    detection is unreliable — models can stay resident far longer than on
    CUDA. Call this between models so each one starts from a clean memory
    state instead of stacking on top of whatever the last one left behind.
    """
    try:
        requests.post(f"{COMFYUI_URL}/free",
                      json={"unload_models": True, "free_memory": True},
                      timeout=timeout)
    except Exception as e:
        warn(f"Could not unload ComfyUI models: {e}")

def comfyui_interrupt_and_clear(timeout: int = 10, confirm_timeout: int = 15) -> None:
    """Stop ComfyUI's currently running job and drop anything still queued.

    ComfyUI executes one job at a time. If we give up on a job client-side
    after a timeout without telling the server, it (or whatever we submit
    next) keeps occupying that single execution slot — every subsequent
    submission queues silently behind it and can time out in turn without
    ever actually starting. Call this right after a timeout so the next
    submission starts from a clean queue.

    /interrupt and /queue clear only signal ComfyUI — they return before the
    running job has actually unwound, so we poll /queue afterward until both
    queue_running and queue_pending are actually empty (or we give up and warn).
    """
    try:
        requests.post(f"{COMFYUI_URL}/interrupt", timeout=timeout)
    except Exception as e:
        warn(f"Failed to interrupt ComfyUI job: {e}")
    try:
        requests.post(f"{COMFYUI_URL}/queue", json={"clear": True}, timeout=timeout)
    except Exception as e:
        warn(f"Failed to clear ComfyUI queue: {e}")

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < confirm_timeout:
        try:
            status = requests.get(f"{COMFYUI_URL}/queue", timeout=10).json()
        except Exception as e:
            warn(f"Failed to confirm ComfyUI queue is clear: {e}")
            return
        if not status.get("queue_running") and not status.get("queue_pending"):
            return
        time.sleep(1)
    warn(f"ComfyUI queue still not empty {confirm_timeout}s after interrupt/clear — "
         f"a stuck job may still be occupying the execution slot")

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

def build_flux2_workflow(checkpoint, width, height, steps, cfg,
                         sampler, scheduler, seed, prompt, filename_prefix="bench_flux2"):
    """
    Flux.2-dev txt2img workflow.

    Flux.2 uses a Mistral-3-24B text encoder (loaded via a single CLIPLoader,
    type "flux2") instead of the T5-XXL + CLIP-L pair used by Flux.1/SD3, and
    a dedicated flux2-vae.safetensors — reusing Flux.1's DualCLIPLoader/VAE
    here silently produces a text-embedding-dimension mismatch deep in the
    transformer (txt_in linear layer) rather than a clear error.
    """
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        "12": {"class_type": "CLIPLoader",
               "inputs": {
                   "clip_name": "mistral_3_small_flux2_fp8.safetensors",
                   "type": "flux2",
               }},
        "13": {"class_type": "VAELoader",
               "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["12", 0]}},
        "3": {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["2", 0], "guidance": cfg}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "RandomNoise",
              "inputs": {"noise_seed": seed}},
        "6": {"class_type": "BasicGuider",
              "inputs": {"model": ["1", 0], "conditioning": ["3", 0]}},
        "7": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": sampler}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {
                  "model": ["1", 0],
                  "scheduler": scheduler,
                  "steps": steps,
                  "denoise": 1.0,
              }},
        "9": {"class_type": "SamplerCustomAdvanced",
              "inputs": {
                  "noise": ["5", 0],
                  "guider": ["6", 0],
                  "sampler": ["7", 0],
                  "sigmas": ["8", 0],
                  "latent_image": ["4", 0],
              }},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["13", 0]}},
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
    # A prior job (from an earlier model/test) can be left running or queued
    # if its own timeout handling didn't fully clear it — e.g. the interrupt
    # or queue-clear request itself failed. Check first so a stuck job never
    # silently eats our execution slot and causes a fresh, unrelated timeout.
    try:
        queue_status = requests.get(f"{COMFYUI_URL}/queue", timeout=10).json()
        if queue_status.get("queue_running") or queue_status.get("queue_pending"):
            warn("ComfyUI queue has leftover job(s) from a prior submission — clearing before continuing")
            comfyui_interrupt_and_clear()
    except Exception as e:
        warn(f"Failed to check ComfyUI queue before submission: {e}")

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
                comfyui_interrupt_and_clear()
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
            comfyui_interrupt_and_clear()
            if not seen:
                raise TimeoutError(
                    f"ComfyUI job never appeared in history after {timeout}s "
                    f"— may be queued behind a still-running prior job, or the "
                    f"workflow errored before queuing"
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

def run_image_benchmarks(image_models, resolutions, seed, prompt,
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
        model_resolutions = model.get("resolutions", resolutions)

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
            w0, h0 = model_resolutions[0]
            log(f"{label}: warmup run ({w0}x{h0}, timeout: {timeout}s) ...")
            warmup_ok = True
            # Use a seed outside the measured runs' range (seed .. seed+N_RUNS-1) so
            # this warmup can't hit the same ComfyUI node cache as measured run 1,
            # which would otherwise return near-instantly instead of regenerating.
            warmup_seed = seed - 1
            try:
                if workflow_t == "flux":
                    wf = build_flux_workflow(checkpoint, w0, h0, steps, cfg,
                                             sampler, scheduler, warmup_seed, prompt,
                                             filename_prefix=f"{short}_warmup")
                elif workflow_t == "flux2":
                    wf = build_flux2_workflow(checkpoint, w0, h0, steps, cfg,
                                              sampler, scheduler, warmup_seed, prompt,
                                              filename_prefix=f"{short}_warmup")
                elif workflow_t == "sd3":
                    wf = build_sd3_workflow(checkpoint, w0, h0, steps, cfg,
                                            sampler, scheduler, warmup_seed, prompt,
                                            filename_prefix=f"{short}_warmup")
                else:
                    wf = build_sdxl_workflow(checkpoint, w0, h0, steps, cfg,
                                             sampler, scheduler, warmup_seed, prompt,
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
            for (w, h) in model_resolutions:
                res_label = f"{w}x{h}"
                log(f"{label} @ {res_label} — {N_RUNS} runs ...")
                times = []
                last_images: list[dict] = []

                for run_i in range(N_RUNS):
                    try:
                        prefix = f"{short}_{res_label}_run{run_i + 1}"
                        # Vary the seed per run — an identical seed/workflow lets
                        # ComfyUI cache every node, so repeat runs return near-
                        # instantly instead of re-running generation.
                        run_seed = seed + run_i
                        if workflow_t == "flux":
                            wf = build_flux_workflow(
                                checkpoint, w, h, steps, cfg,
                                sampler, scheduler, run_seed, prompt,
                                filename_prefix=prefix)
                        elif workflow_t == "flux2":
                            wf = build_flux2_workflow(
                                checkpoint, w, h, steps, cfg,
                                sampler, scheduler, run_seed, prompt,
                                filename_prefix=prefix)
                        elif workflow_t == "sd3":
                            wf = build_sd3_workflow(
                                checkpoint, w, h, steps, cfg,
                                sampler, scheduler, run_seed, prompt,
                                filename_prefix=prefix)
                        else:
                            wf = build_sdxl_workflow(
                                checkpoint, w, h, steps, cfg,
                                sampler, scheduler, run_seed, prompt,
                                filename_prefix=prefix)

                        elapsed, images = comfyui_submit(wf, timeout=timeout)
                        times.append(elapsed)
                        last_images = images
                        print(f"    run {run_i+1}/{N_RUNS}: {elapsed:.1f}s")
                    except TimeoutError:
                        err(f"Run {run_i+1} timed out — skipping {label}")
                        model_timed_out = True
                        results[short]["timed_out"] = res_label
                        break
                    except Exception as e:
                        err(f"Run {run_i+1} failed: {e}")

                if times:
                    results[short]["resolutions"][res_label] = {
                        "sec_per_image_mean":  round(mean(times),  2),
                        "sec_per_image_stdev": round(stdev(times) if len(times) > 1 else 0.0, 2),
                        "n_runs":              len(times),
                        "runs":               [round(t, 2) for t in times],
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
            log(f"Unloading {label} from VRAM ...")
            comfyui_free_models()

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
    parser.add_argument(
        "--emb-cpu-only", action="store_true",
        help="Force CPU-only inference for the embedding benchmarks by restarting "
             "Ollama with GPU devices hidden (HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES "
             "/ ROCR_VISIBLE_DEVICES set empty). Stops any running Ollama server "
             "(even one this script didn't start) and restores normal GPU mode "
             "afterward. Useful on GPU backends unstable under embedding batching.",
    )
    parser.add_argument(
        "--maxtier", type=str, default=None,
        choices=["xsmall", "small", "medium", "large"],
        help="Cap LLM models (single-shot and conversation tests) at this size tier "
             "and below (default: all tiers). xsmall: <6B params. small: adds ≤20B. "
             "medium: adds 26-35B. large: adds 70B+ (i.e. no cap).",
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help=f"Ignore the {SLOW_MODEL_MIN_TPS:.0f} tok/s slow-model cutoff: run every "
             "context length in the LLM prefill test and always run the conversation "
             "test, even for models that would otherwise be marked slow and skipped. "
             "Does not override real failures (timeouts, missing data). (default: false)",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level config
    global RUN_TIMEOUT
    if args.timeout is not None:
        RUN_TIMEOUT = args.timeout

    # Select model tier — cumulative: --maxtier caps at that tier and includes everything below it
    TIER_MODELS = {
        "xsmall": LLM_MODELS_XSMALL,
        "small":  LLM_MODELS_XSMALL + LLM_MODELS_SMALL,
        "medium": LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM,
        "large":  LLM_MODELS,
    }
    TIER_LABELS = {
        "xsmall": "extra-small only (≤4GB)",
        "small":  "small and below (≤16GB)",
        "medium": "medium and below (≤32GB)",
        "large":  "large and below — all tiers (32GB+)",
    }
    TIER_ORDER = ["xsmall", "small", "medium", "large"]
    if args.maxtier:
        llm_models = TIER_MODELS[args.maxtier]
        tier_label = TIER_LABELS[args.maxtier]
        max_idx = TIER_ORDER.index(args.maxtier)
        image_models = [m for m in IMAGE_MODELS if TIER_ORDER.index(m["tier"]) <= max_idx]
    else:
        llm_models = LLM_MODELS
        tier_label = "all (extra-small + small + medium + large)"
        image_models = IMAGE_MODELS

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
    print(f"  Runs:      {N_RUNS} measured + {args.warmup} warmup")
    print(f"  Timeout:   {RUN_TIMEOUT}s per run")
    print(f"  Models:    {tier_label}")
    if args.maxtier:
        print(f"  Images:    {', '.join(m['label'] for m in image_models) or '(none — tier too small)'}")
    print(f"  Tests:     {', '.join(args.tests)}")
    print(f"  ComfyUI:   {comfyui_dir}")

    # Register cleanup for Ctrl-C and normal exit
    def _cleanup(sig=None, frame=None):
        if sig is not None:
            print(f"\n{YELLOW}Interrupted — unloading models before exit ...{RESET}")
        if ollama_available():
            unload_all_models()
        if comfyui_available():
            comfyui_free_models()
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
                warmup_runs=args.warmup,
                force_all=args.force_all,
            )
            _checkpoint("LLM done")

        if "conv" in args.tests:
            conv_models = llm_models
            llm_conv_skips = {}
            if "llm" in args.tests:
                conv_models = []
                for model in llm_models:
                    short = model["short"]
                    llm_data = results["llm"].get(short)
                    if not llm_data:
                        detail = "no LLM benchmark data (checkpoint skipped or model failed)"
                        warn(f"{model['label']}: skipping conversation test — {detail}")
                        llm_conv_skips[short] = {
                            "label": model["label"], "skipped": True,
                            "skip_reason": "no_llm_data", "skip_detail": detail,
                        }
                        continue
                    first_ctx_label = f"{CONTEXT_LENGTHS[0] // 1024}K"
                    if llm_data.get("timed_out") == first_ctx_label:
                        detail = f"LLM test timed out at {llm_data['timed_out']} context"
                        warn(f"{model['label']}: skipping conversation test — {detail}")
                        llm_conv_skips[short] = {
                            "label": model["label"], "skipped": True,
                            "skip_reason": "timed_out", "skip_detail": detail,
                        }
                        continue
                    # A timeout at a deeper context (8K/32K/64K) doesn't disqualify
                    # the model — it passed the 2K prefill test, so fall through to
                    # the tok/s check below just like a model that wasn't timed out.
                    slow_ctx = None if args.force_all else llm_data.get("slow_tps") or (
                        first_ctx_label if isinstance(llm_data.get(first_ctx_label), dict)
                        and llm_data[first_ctx_label].get("tps_mean") is not None
                        and llm_data[first_ctx_label]["tps_mean"] < SLOW_MODEL_MIN_TPS
                        else None
                    )
                    if slow_ctx is not None:
                        ctx_data = llm_data.get(slow_ctx)
                        detail = (f"{ctx_data['tps_mean']:.1f} tok/s at {slow_ctx} "
                                  f"context (below {SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff)"
                                  if isinstance(ctx_data, dict) and ctx_data.get("tps_mean") is not None
                                  else f"below {SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff at {slow_ctx} context")
                        warn(f"{model['label']}: skipping conversation test — {detail}")
                        llm_conv_skips[short] = {
                            "label": model["label"], "skipped": True,
                            "skip_reason": "slow_tps", "skip_detail": detail,
                        }
                        continue
                    conv_models.append(model)

            results["llm_conversation"] = run_conversation_benchmarks(
                models=conv_models,
                context_lengths=CONTEXT_LENGTHS,
                warmup_runs=args.warmup,
                force_all=args.force_all,
            )
            results["llm_conversation"].update(llm_conv_skips)
            _checkpoint("LLM conversation done")

        # ── Embeddings ─────────────────────────────────────────────────────────
        if "emb" in args.tests:
            if args.emb_cpu_only:
                global _cpu_only_active
                section("Embeddings: forcing CPU-only")
                warn("Stopping Ollama to relaunch in CPU-only mode for embeddings ...")
                stop_all_ollama()
                cpu_env = {
                    "HIP_VISIBLE_DEVICES": "",
                    "CUDA_VISIBLE_DEVICES": "",
                    "ROCR_VISIBLE_DEVICES": "",
                }
                if not start_ollama(extra_env=cpu_env):
                    err("Failed to start Ollama in CPU-only mode — skipping embeddings")
                    results["embeddings"] = {}
                else:
                    _cpu_only_active = True
                    results["embeddings"] = run_embedding_benchmarks(
                        models=EMBED_MODELS,
                        warmup_runs=args.warmup,
                    )
                    _checkpoint("embeddings done")
                    log("Restoring normal (GPU-enabled) Ollama ...")
                    stop_all_ollama()
                    _cpu_only_active = False
                    start_ollama()
            else:
                if not ollama_available():
                    section("Starting Servers")
                    ensure_ollama()
                results["embeddings"] = run_embedding_benchmarks(
                    models=EMBED_MODELS,
                    warmup_runs=args.warmup,
                )
                _checkpoint("embeddings done")

        # ── Image generation ───────────────────────────────────────────────────
        if "img" in args.tests:
            section("Starting Servers")
            # Hard guarantee: nothing from Ollama in memory before ComfyUI loads.
            # Image generation is always the last phase (see phase order above),
            # so there's nothing left in this run that needs Ollama afterward —
            # kill the whole server rather than just unloading its models, to
            # free up whatever memory the idle process itself still holds.
            if ollama_available():
                log("Stopping Ollama entirely to free memory for ComfyUI ...")
                stop_all_ollama()
            comfyui_started = ensure_comfyui(comfyui_dir)
            if not comfyui_started:
                warn("Image benchmarks will be skipped")
            else:
                def _img_save(img_partial):
                    results["images"] = img_partial
                    _checkpoint()

                results["images"] = run_image_benchmarks(
                    image_models=image_models,
                    resolutions=IMAGE_RESOLUTIONS,
                    seed=IMAGE_SEED,
                    prompt=IMAGE_PROMPT,
                    comfyui_dir=comfyui_dir,
                    timeout=RUN_TIMEOUT * 2,
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
