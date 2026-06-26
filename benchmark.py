#!/usr/bin/env python3
"""
benchmark.py — Cross-platform LLM benchmark suite.

Tests:
  1. LLM generation — Llama 3.1 70B and GPT-OSS 120B via Ollama
     Metrics: time-to-first-token (TTFT), tokens/sec, total time
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
import subprocess
import sys
import time
import threading
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

# Image generation models — each entry defines the checkpoint file and workflow type.
# Models whose checkpoint file is not found in ComfyUI/models/checkpoints/ are
# skipped automatically with a clear message.
IMAGE_MODELS = [
    {
        "label":      "SDXL",
        "checkpoint": "sd_xl_base_1.0.safetensors",
        "workflow":   "sdxl",
        "steps":      20,
        "cfg":        7.0,
        "sampler":    "euler_ancestral",
        "scheduler":  "normal",
        "short":      "sdxl",
    },
    {
        "label":      "Flux.1-schnell",
        "checkpoint": "flux1-schnell.safetensors",
        "workflow":   "flux",
        "steps":      4,
        "cfg":        1.0,
        "sampler":    "euler",
        "scheduler":  "simple",
        "short":      "flux-schnell",
    },
    {
        "label":      "Flux.1-dev",
        "checkpoint": "flux1-dev.safetensors",
        "workflow":   "flux",
        "steps":      20,
        "cfg":        1.0,
        "sampler":    "euler",
        "scheduler":  "simple",
        "short":      "flux-dev",
    },
]


# Models
# Small-tier models (≤16GB VRAM) — run on all hardware including 8GB GPUs
# Tags verified against ollama.com/library June 2026
LLM_MODELS_SMALL = [
    {
        "tag":   "llama3.1:8b-instruct-q3_K_M",
        "label": "Llama 3.1 8B Q3_K_M",
        "short": "llama3.1-8b-q3",
        "vram":  "~4.3 GB",
    },
    {
        "tag":   "llama3.1:8b-instruct-q4_K_M",
        "label": "Llama 3.1 8B Q4_K_M",
        "short": "llama3.1-8b-q4",
        "vram":  "~4.9 GB",
    },
    {
        "tag":   "qwen3:14b-q4_K_M",
        "label": "Qwen3 14B Q4_K_M",
        "short": "qwen3-14b-q4",
        "vram":  "~9.3 GB",
    },
    {
        "tag":   "qwen3:14b-q8_0",
        "label": "Qwen3 14B Q8_0",
        "short": "qwen3-14b-q8",
        "vram":  "~16 GB",
    },
    {
        "tag":   "gpt-oss:20b",
        "label": "GPT-OSS 20B (MXFP4)",
        "short": "gpt-oss-20b",
        "vram":  "~14 GB",
    },
]

# Large-tier models (≥32GB) — for high-memory machines
# Note: gpt-oss:120b ships in MXFP4 only — no Q3/Q4 variants exist
LLM_MODELS_LARGE = [
    {
        "tag":   "llama3.1:70b-instruct-q3_K_M",
        "label": "Llama 3.1 70B Q3_K_M",
        "short": "llama3.1-70b-q3",
        "vram":  "~32 GB",
    },
    {
        "tag":   "llama3.1:70b-instruct-q4_K_M",
        "label": "Llama 3.1 70B Q4_K_M",
        "short": "llama3.1-70b-q4",
        "vram":  "~42 GB",
    },
    {
        "tag":   "gpt-oss:120b",
        "label": "GPT-OSS 120B (MXFP4)",
        "short": "gpt-oss-120b",
        "vram":  "~65 GB",
    },
]

# Default: run both tiers. Use --small-only or --large-only to restrict.
LLM_MODELS = LLM_MODELS_SMALL + LLM_MODELS_LARGE

EMBED_MODEL = "BAAI/bge-large-en-v1.5"

CONTEXT_LENGTHS = [2048, 8192]   # tokens (approximate, via prompt padding)
EMBED_BATCH_SIZES = [32, 128, 512]
IMAGE_RESOLUTIONS = [(1024, 1024), (1536, 1536)]
# Steps are now per-model in IMAGE_MODELS
IMAGE_SEED  = 42
IMAGE_PROMPT = (
    "A photorealistic mountain landscape at golden hour, "
    "dramatic clouds, highly detailed, 8k resolution"
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

def mean(vals):   return sum(vals) / len(vals) if vals else 0
def stdev(vals):
    if len(vals) < 2:
        return 0
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5

def peak_ram_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)

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
    log(f"Starting ComfyUI from {comfyui_dir} using {python_exe} ...")

    try:
        proc = subprocess.Popen(
            [python_exe, str(main_py), "--listen"],
            cwd=str(comfyui_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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
            return True
        if proc.poll() is not None:
            err(f"ComfyUI exited unexpectedly (code {proc.returncode})")
            err(f"Try starting it manually: cd {comfyui_dir} && python main.py --listen")
            return False
        if (i + 1) % 10 == 0:
            log(f"Still waiting ... ({i+1}s)")

    err("ComfyUI did not respond within 60 seconds")
    return False


# ── Machine profile ────────────────────────────────────────────────────────────

def build_profile():
    os_name = platform.system()
    profile = {
        "hostname":   platform.node(),
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
    "Explain the concept of neural scaling laws in machine learning. "
    "Discuss how model size, dataset size, and compute interact, "
    "and what the empirical findings suggest about future AI development."
)

def build_prompt_for_context(target_tokens: int) -> str:
    """
    Pad a prompt to approximate a target context length.
    Roughly 1 token ≈ 4 chars in English.
    """
    base = SHORT_PROMPT
    chars_needed = target_tokens * 4
    padding_unit = (
        " Furthermore, consider the implications for hardware design, "
        "energy consumption, and the economics of training large models."
    )
    while len(base) < chars_needed:
        base += padding_unit
    return base[:chars_needed]

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

def ollama_generate(model_tag: str, prompt: str, timeout: int = 600):
    """
    Generate via Ollama and return timing metrics.
    Returns: (ttft_sec, total_sec, tokens_generated, tokens_per_sec)

    Uses urllib instead of requests for streaming to avoid TCP buffering
    that causes iter_lines() to batch all chunks and inflate TTFT.

    Ollama's final chunk includes server-side timing fields:
      prompt_eval_duration  — time to process the prompt (nanoseconds)
      eval_count            — tokens generated
      eval_duration         — time spent generating (nanoseconds)
    These are authoritative and used in preference to wall-clock where available.
    """
    import urllib.request

    payload = json.dumps({
        "model":  model_tag,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": 512,
            "temperature": 0.0,
        },
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
    return ttft, total, eval_count, tps

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

        # Warm up model (load into memory), with a timeout so we don't get stuck
        log(f"Warming up {label} (timeout: {WARMUP_TIMEOUT}s per run) ...")
        warmup_ok = True
        for warmup_i in range(warmup_runs):
            result_box = [None]   # mutable container so thread can write back
            exc_box    = [None]

            def _warmup():
                try:
                    result_box[0] = ollama_generate(tag, "Hello.", timeout=WARMUP_TIMEOUT)
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
            label_ctx = f"{ctx_len // 1000}K"
            log(f"Context {label_ctx} — {n_runs} runs ...")

            ttfts, totals, tps_list = [], [], []

            for run_i in range(n_runs):
                try:
                    ttft, total, tokens, tps = ollama_generate(
                        tag, prompt, timeout=600
                    )
                    ttfts.append(ttft)
                    totals.append(total)
                    tps_list.append(tps)
                    print(
                        f"    run {run_i+1}/{n_runs}: "
                        f"TTFT={ttft:.2f}s  "
                        f"TPS={tps:.1f}  "
                        f"total={total:.1f}s"
                    )
                except Exception as e:
                    err(f"Run {run_i+1} failed: {e}")

            if ttfts:
                results[short][label_ctx] = {
                    "ttft_mean_sec":   round(mean(ttfts),    3),
                    "ttft_stdev_sec":  round(stdev(ttfts),   3),
                    "tps_mean":        round(mean(tps_list), 2),
                    "tps_stdev":       round(stdev(tps_list),2),
                    "total_mean_sec":  round(mean(totals),   2),
                    "n_runs":          len(ttfts),
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
    Flux.1 txt2img workflow using the native Flux node set.
    Flux requires: UNETLoader + DualCLIPLoader + FluxGuidance + ModelSamplingFlux
    + EmptyLatentImage (not EmptySD3LatentImage) + KSamplerSelect + SamplerCustomAdvanced.
    """
    return {
        # Load Flux UNet
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}},
        # Load dual CLIP (t5xxl + clip_l) — ComfyUI looks in models/clip/
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {
                  "clip_name1": "t5xxl_fp16.safetensors",
                  "clip_name2": "clip_l.safetensors",
                  "type": "flux",
              }},
        # Load VAE
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "ae.safetensors"}},
        # Encode prompt
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        # Flux guidance
        "5": {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["4", 0], "guidance": cfg}},
        # Empty latent
        "6": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        # Model sampling for Flux
        "7": {"class_type": "ModelSamplingFlux",
              "inputs": {
                  "model": ["1", 0],
                  "max_shift": 1.15,
                  "base_shift": 0.5,
                  "width": width,
                  "height": height,
              }},
        # Noise
        "8": {"class_type": "RandomNoise",
              "inputs": {"noise_seed": seed}},
        # Sampler
        "9": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": sampler}},
        # Scheduler
        "10": {"class_type": "BasicScheduler",
               "inputs": {
                   "model": ["7", 0],
                   "scheduler": scheduler,
                   "steps": steps,
                   "denoise": 1.0,
               }},
        # Run sampler
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {
                   "noise": ["8", 0],
                   "guider": ["12", 0],
                   "sampler": ["9", 0],
                   "sigmas": ["10", 0],
                   "latent_image": ["6", 0],
               }},
        # CFG guider
        "12": {"class_type": "BasicGuider",
               "inputs": {"model": ["7", 0], "conditioning": ["5", 0]}},
        # Decode
        "13": {"class_type": "VAEDecode",
               "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        # Save
        "14": {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0], "filename_prefix": "bench_flux"}},
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

def comfyui_submit(workflow: dict, timeout: int = 300) -> float:
    """Submit a workflow to ComfyUI, poll until done, return elapsed seconds."""
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    resp.raise_for_status()
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
                return time.perf_counter() - t0

        if time.perf_counter() - t0 > timeout:
            if not seen:
                raise TimeoutError(
                    f"ComfyUI job never appeared in history after {timeout}s "
                    f"— workflow may have errored before queuing"
                )
            raise TimeoutError(f"ComfyUI job timed out after {timeout}s")

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
            info(f"Download and place at: {ckpt_path}")
            continue

        ok(f"{label}: checkpoint found ({ckpt_path.stat().st_size / (1024**3):.1f} GB)")
        results[short] = {"label": label, "checkpoint": checkpoint,
                          "steps": steps, "resolutions": {}}

        for (w, h) in resolutions:
            res_label = f"{w}x{h}"
            log(f"{label} @ {res_label} — {n_runs} runs ...")
            times = []

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

                    elapsed = comfyui_submit(wf)
                    times.append(elapsed)
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
        help="Run only small-tier models (<=16GB VRAM): Llama 3.2 8B, Qwen3 14B, GPT-OSS 20B",
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
    elif args.large_only:
        llm_models = LLM_MODELS_LARGE
        tier_label = "large only (32GB+)"
    else:
        llm_models = LLM_MODELS
        tier_label = "all (small + large)"

    comfyui_dir = Path(args.comfyui) if args.comfyui else COMFYUI_DIR

    profile  = build_profile()
    out_path = args.out or f"results_{profile['hostname']}.json"

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
