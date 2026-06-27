#!/usr/bin/env python3
"""
benchmark.py — Cross-platform LLM benchmark suite.

Tests:
  1. LLM generation — Llama 3.1 70B and GPT-OSS 120B via Ollama
     Metrics: time-to-first-token (TTFT), tokens/sec
     Context lengths: 2K and 8K tokens

  2. Image generation — Flux.1-dev via ComfyUI HTTP API
     Metrics: seconds/image at 1024×1024 and 1536×1536
     (skipped automatically if Flux model not found)

  3. Embeddings — bge-large-en-v1.5 via sentence-transformers
     Metrics: sentences/sec at batch sizes 32, 128, 512
     Memory tracked throughout

Servers are managed automatically:
  - Ollama: started if not already running, left running after (it's a service)
  - ComfyUI: started before image tests, shut down cleanly when done

Usage:
  python benchmark.py                  # run all tests
  python benchmark.py --tests llm      # run only LLM tests
  python benchmark.py --tests llm emb  # run LLM + embeddings
  python benchmark.py --runs 3         # override number of measured runs
  python benchmark.py --comfyui /path/to/ComfyUI  # override ComfyUI path
"""

import argparse
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import threading
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434"
COMFYUI_URL  = "http://localhost:8188"

# Default ComfyUI path — relative to this script's directory (~/llamabench/ComfyUI)
SCRIPT_DIR   = Path(__file__).resolve().parent
COMFYUI_DIR  = SCRIPT_DIR / "ComfyUI"

from models import IMAGE_MODELS, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS  # noqa: E402

EMBED_MODEL = "BAAI/bge-large-en-v1.5"

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

WARMUP_RUNS    = 2
DEFAULT_RUNS   = 5
WARMUP_TIMEOUT = 300   # seconds per warmup run before aborting this model

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

_this_proc = psutil.Process(os.getpid())

def peak_ram_mb():
    return _this_proc.memory_info().rss / (1024 ** 2)

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
    # venv inside ComfyUI dir
    for candidate in [
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
        warn("Run setup_check.py to download Flux.1-schnell automatically")
        return False
    log(f"Found {len(found)}/{len(known)} image checkpoints: {found}")

    python_exe = find_comfyui_python(comfyui_dir)

    cmd = [python_exe, str(main_py), "--listen"]
    if platform.system() == "Windows":
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # AMD/Intel GPU on Windows — ComfyUI needs DirectML instead of CUDA
            cmd.append("--directml")
            log("Windows non-NVIDIA GPU detected — adding --directml (AMD/Intel)")

    log(f"Starting ComfyUI from {comfyui_dir} using {python_exe} ...")

    # Capture stderr to a temp file so we can show it if ComfyUI exits unexpectedly
    stderr_log = Path(tempfile.mktemp(suffix="-comfyui-stderr.log"))
    try:
        stderr_fh = open(stderr_log, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=str(comfyui_dir),
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
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
    if platform.system() == "Darwin":
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
                    # "Apple M4 Max" → "M4 Max"
                    chip = line.split(":", 1)[1].strip().removeprefix("Apple ").strip()
                elif "Memory:" in line:
                    ram = line.split(":", 1)[1].strip()
            if model and chip and ram:
                return f"{model} {chip} {ram}"
        except Exception:
            pass
    return platform.node()


def build_profile():
    os_name = platform.system()
    profile = {
        "hostname":   _get_hostname(),
        "os":         f"{os_name} {platform.release()}",
        "arch":       platform.machine(),
        "python":     sys.version.split()[0],
        "ram_gb":     round(system_ram_gb(), 1),
        "timestamp":  datetime.utcnow().isoformat() + "Z",
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
    # ROCm
    try:
        out = subprocess.check_output(["rocminfo"], text=True,
                                       stderr=subprocess.DEVNULL)
        if "Marketing Name" in out:
            return "rocm"
    except (FileNotFoundError, subprocess.CalledProcessError):
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
    """Pad a prompt to approximate a target context length (1 token ≈ 4 chars)."""
    chars_needed = target_tokens * 4
    parts = [SHORT_PROMPT]
    total = len(SHORT_PROMPT)
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
        # Use the largest context length so Ollama pre-allocates the full KV cache
        # once — avoiding a reload on the first measured run at max context.
        max_ctx = max(context_lengths)
        log(f"Warming up {label} at num_ctx={max_ctx} (timeout: {WARMUP_TIMEOUT}s per run) ...")
        warmup_ok = True
        for warmup_i in range(warmup_runs):
            result_box = [None]   # mutable container so thread can write back
            exc_box    = [None]

            def _warmup():
                try:
                    result_box[0] = ollama_generate(
                        tag, "Hello.", timeout=WARMUP_TIMEOUT, num_ctx=max_ctx)
                except Exception as e:
                    exc_box[0] = e

            t = threading.Thread(target=_warmup, daemon=True)
            t_start = time.perf_counter()
            t.start()
            t.join(timeout=WARMUP_TIMEOUT)

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

        for ctx_len in context_lengths:
            prompt = build_prompt_for_context(ctx_len)
            label_ctx = f"{ctx_len // 1024}K"
            log(f"Context {label_ctx} — {n_runs} runs ...")

            ttfts, tps_list = [], []

            for run_i in range(n_runs):
                try:
                    ttft, tokens, tps = ollama_generate(
                        tag, prompt, timeout=600, num_ctx=ctx_len
                    )
                    ttfts.append(ttft)
                    tps_list.append(tps)
                    print(
                        f"    run {run_i+1}/{n_runs}: "
                        f"TTFT={ttft:.2f}s  "
                        f"TPS={tps:.1f}"
                    )
                except Exception as e:
                    err(f"Run {run_i+1} failed: {e}")

            if ttfts:
                # Drop the single slowest TTFT run before averaging
                if len(ttfts) > 1:
                    worst_idx = ttfts.index(max(ttfts))
                    ttfts = ttfts[:worst_idx] + ttfts[worst_idx+1:]

                results[short][label_ctx] = {
                    "ttft_mean_sec":  round(mean(ttfts),    3),
                    "ttft_stdev_sec": round(stdev(ttfts),   3),
                    "tps_mean":       round(mean(tps_list), 2),
                    "tps_stdev":      round(stdev(tps_list),2),
                    "n_runs":         len(tps_list),
                }
                ok(
                    f"Context {label_ctx} done: "
                    f"TTFT={results[short][label_ctx]['ttft_mean_sec']:.2f}s  "
                    f"TPS={results[short][label_ctx]['tps_mean']:.1f}"
                )

        # Unload model and confirm it's gone before moving on
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

def detect_embed_device():
    """Return the best available device for sentence-transformers."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

def run_embedding_benchmarks(batch_sizes, n_runs):
    results = {}

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        err("sentence-transformers not installed — skipping embedding benchmarks")
        err("Install: pip install sentence-transformers")
        return results

    device = detect_embed_device()
    section(f"Embeddings: {EMBED_MODEL}")

    if device == "cpu":
        warn("No GPU backend detected — embeddings will run on CPU")
        warn("Results are tagged 'cpu' and will be flagged in compare.py")
    else:
        ok(f"Using device: {device}")

    log(f"Loading model {EMBED_MODEL} on {device} ...")

    try:
        model = SentenceTransformer(EMBED_MODEL, device=device)
        ok("Model loaded")
    except Exception as e:
        err(f"Failed to load embedding model: {e}")
        return results

    corpus = CORPUS_SENTENCES
    log(f"Corpus: {len(corpus)} sentences")

    for bs in batch_sizes:
        log(f"Batch size {bs} — {n_runs} runs ...")
        rates = []
        ram_peaks = []

        for run_i in range(n_runs):
            t0 = time.perf_counter()
            try:
                model.encode(corpus, batch_size=bs, show_progress_bar=False)
                elapsed = time.perf_counter() - t0
                rate = len(corpus) / elapsed
                ram_mb = peak_ram_mb()
                rates.append(rate)
                ram_peaks.append(ram_mb)
                print(
                    f"    run {run_i+1}/{n_runs}: "
                    f"{rate:.0f} sent/sec  "
                    f"RAM={ram_mb:.0f} MB"
                )
            except Exception as e:
                err(f"Run {run_i+1} failed: {e}")

        if rates:
            key = f"batch_{bs}"
            results[key] = {
                "sentences_per_sec_mean":  round(mean(rates),      1),
                "sentences_per_sec_stdev": round(stdev(rates),     1),
                "peak_ram_mb_mean":        round(mean(ram_peaks),  1),
                "device":                  device,
                "n_runs":                  len(rates),
            }
            ok(
                f"Batch {bs}: {results[key]['sentences_per_sec_mean']:.0f} sent/sec"
                f"  [{device}]"
            )

    return results

# ── Image generation benchmark ─────────────────────────────────────────────────

def comfyui_available():
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def build_flux_workflow(checkpoint, width, height, steps, cfg,
                        sampler, scheduler, seed, prompt):
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
               "inputs": {"images": ["10", 0], "filename_prefix": "bench_flux"}},
    }

def build_sdxl_workflow(checkpoint, width, height, steps, cfg,
                        sampler, scheduler, seed, prompt):
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
               "inputs": {"images": ["8", 0], "filename_prefix": "bench"}},
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
                         comfyui_dir):
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
        log(f"{label}: warmup run ({w0}x{h0}) ...")
        try:
            if workflow_t == "flux":
                wf = build_flux_workflow(checkpoint, w0, h0, steps, cfg,
                                         sampler, scheduler, seed, prompt)
            else:
                wf = build_sdxl_workflow(checkpoint, w0, h0, steps, cfg,
                                         sampler, scheduler, seed, prompt)
            comfyui_submit(wf)
            ok(f"{label}: warmup done")
        except Exception as e:
            warn(f"{label}: warmup failed: {e}")

        img_dir = SCRIPT_DIR / "benchmark_images"

        for (w, h) in resolutions:
            res_label = f"{w}x{h}"
            log(f"{label} @ {res_label} — {n_runs} runs ...")
            times = []
            last_images: list[dict] = []

            for run_i in range(n_runs):
                try:
                    if workflow_t == "flux":
                        wf = build_flux_workflow(
                            checkpoint, w, h, steps, cfg,
                            sampler, scheduler, seed, prompt)
                    else:
                        wf = build_sdxl_workflow(
                            checkpoint, w, h, steps, cfg,
                            sampler, scheduler, seed, prompt)

                    elapsed, images = comfyui_submit(wf)
                    times.append(elapsed)
                    last_images = images
                    print(f"    run {run_i+1}/{n_runs}: {elapsed:.1f}s")
                except Exception as e:
                    err(f"Run {run_i+1} failed: {e}")

            if times:
                results[short]["resolutions"][res_label] = {
                    "sec_per_image_mean":  round(mean(times),  2),
                    "sec_per_image_stdev": round(stdev(times), 2),
                    "n_runs":              len(times),
                }
                ok(f"{label} @ {res_label}: "
                   f"{results[short]['resolutions'][res_label]['sec_per_image_mean']:.1f}s/image")

            if last_images:
                dest = img_dir / f"{short}_{res_label}.png"
                try:
                    save_comfyui_image(last_images[0], dest)
                    ok(f"Saved image → benchmark_images/{dest.name}")
                except Exception as e:
                    warn(f"Could not save image: {e}")

    return results

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM benchmark suite")
    parser.add_argument(
        "--tests", nargs="+",
        choices=["llm", "emb", "img"],
        default=["llm", "emb", "img"],
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
        "--warmup-timeout", type=int, default=None,
        help="Seconds to wait per warmup run before aborting this model (default: 300)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON file (default: results_<hostname>.json)",
    )
    parser.add_argument(
        "--comfyui", type=str, default=None,
        help=f"Path to ComfyUI directory (default: {COMFYUI_DIR})",
    )
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--small-only", action="store_true",
        help="Run only small-tier models (≤16GB VRAM): Llama 3.1 8B, Qwen3 14B Q4, GPT-OSS 20B",
    )
    size_group.add_argument(
        "--medium-only", action="store_true",
        help="Run only medium-tier models (16–32GB VRAM): Qwen3 14B Q8, Qwen3.6 35B-A3B",
    )
    size_group.add_argument(
        "--large-only", action="store_true",
        help="Run only large-tier models (32GB+ VRAM): Llama 3.1 70B, GPT-OSS 120B",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level config
    global WARMUP_TIMEOUT
    if args.warmup_timeout is not None:
        WARMUP_TIMEOUT = args.warmup_timeout

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
    out_path = args.out or f"results_{profile['hostname'].replace(' ', '_')}.json"

    print(f"\n{BOLD}LLM Benchmark Suite{RESET}")
    print(f"  Host:      {profile['hostname']}")
    print(f"  OS:        {profile['os']}")
    print(f"  Backend:   {profile['backend']}")
    print(f"  RAM:       {profile['ram_gb']} GB")
    print(f"  Runs:      {args.runs} measured + {args.warmup} warmup")
    print(f"  Warmup TO: {WARMUP_TIMEOUT}s per run")
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
        "profile":    profile,
        "llm":        {},
        "embeddings": {},
        "images":     {},
    }

    try:
        # ── LLM ───────────────────────────────────────────────────────────────
        if "llm" in args.tests:
            section("Starting Servers")
            ensure_ollama()
            results["llm"] = run_llm_benchmarks(
                models=llm_models,
                context_lengths=CONTEXT_LENGTHS,
                n_runs=args.runs,
                warmup_runs=args.warmup,
            )

        # ── Embeddings ─────────────────────────────────────────────────────────
        if "emb" in args.tests:
            results["embeddings"] = run_embedding_benchmarks(
                batch_sizes=EMBED_BATCH_SIZES,
                n_runs=args.runs,
            )

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
                results["images"] = run_image_benchmarks(
                    image_models=IMAGE_MODELS,
                    resolutions=IMAGE_RESOLUTIONS,
                    seed=IMAGE_SEED,
                    prompt=IMAGE_PROMPT,
                    n_runs=args.runs,
                    comfyui_dir=comfyui_dir,
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
