# LLM Benchmark Suite

Cross-platform benchmarking for LLM generation, image generation, and embeddings.
Designed to run on any hardware from an 8GB GPU up to high-memory unified-memory
systems. Models that don't fit are skipped automatically — no configuration needed.

---

## Quick Start

```bash
git clone https://github.com/DeerSteak/local-ai-bench
cd local-ai-bench
```

Then run the setup script for your platform:

| Platform | Script | Notes |
|---|---|---|
| macOS | `bash setup.sh` | Installs Homebrew + Python if needed |
| Linux / DGX Spark | `bash setup.sh` | Installs Python + Ollama if needed |
| Windows | `setup.bat` | Double-click or run from terminal |

The setup script installs Python if missing, creates the venv, and runs
`setup_check.py` which handles everything else — dependencies, Ollama, models,
and image checkpoints. When it's done:

```bash
# macOS / Linux
source bench-env/bin/activate
python benchmark.py

# Windows
bench-env\Scripts\activate
python benchmark.py
```

---

## Files

| File | Purpose |
|---|---|
| `setup.sh` | One-shot setup for macOS and Linux |
| `setup.bat` | One-shot setup for Windows |
| `setup_check.py` | Called by setup scripts — installs deps, pulls models, downloads checkpoints |
| `benchmark.py` | Main benchmark — produces `results_<hostname>.json` |
| `compare.py` | Comparison — takes all result JSONs and prints a ranked summary table |

---

## Models

All eight models are attempted by default. If a model doesn't complete warmup
within 5 minutes it is skipped with a clear message and the benchmark moves on.
This means the same command works on any hardware — small GPUs naturally skip
the large models without any extra flags.

### Small tier (≤16GB)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.1 8B Q3_K_M | `llama3.1:8b-instruct-q3_K_M` | ~4.3 GB |
| Llama 3.1 8B Q4_K_M | `llama3.1:8b-instruct-q4_K_M` | ~4.9 GB |
| Qwen3 14B Q4_K_M | `qwen3:14b-q4_K_M` | ~9.3 GB |
| Qwen3 14B Q8_0 | `qwen3:14b-q8_0` | ~16 GB |
| GPT-OSS 20B (MXFP4) | `gpt-oss:20b` | ~14 GB |

### Large tier (32GB+)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.1 70B Q3_K_M | `llama3.1:70b-instruct-q3_K_M` | ~32 GB |
| Llama 3.1 70B Q4_K_M | `llama3.1:70b-instruct-q4_K_M` | ~42 GB |
| GPT-OSS 120B (MXFP4) | `gpt-oss:120b` | ~65 GB |

**Notes on GPT-OSS:** Both sizes ship in MXFP4 precision only — there are no
separate Q3/Q4 variants. Attempting to pull `gpt-oss:20b-q3_K_M` or
`gpt-oss:120b-q3_K_M` will fail; use the tags above.

**Notes on Qwen3 14B:** No Q3_K_M variant exists. Q4_K_M and Q8_0 are the
available quantizations.

**Notes on Llama 3.2:** Llama 3.2 tops out at 3B parameters. The 8B slot
belongs to Llama 3.1.

---

## How It Works

### Execution flow

```
1.  Start Ollama (if not already running)
--- LLM tests (all 8 models, small tier first) ---
2.  Llama 3.1 8B Q3_K_M  → warmup → measure → unload → confirm gone
3.  Llama 3.1 8B Q4_K_M  → warmup → measure → unload → confirm gone
4.  Qwen3 14B Q4_K_M     → warmup → measure → unload → confirm gone
5.  Qwen3 14B Q8_0       → warmup → measure → unload → confirm gone
6.  GPT-OSS 20B          → warmup → measure → unload → confirm gone
7.  Llama 3.1 70B Q3_K_M → warmup → measure → unload → confirm gone
8.  Llama 3.1 70B Q4_K_M → warmup → measure → unload → confirm gone
9.  GPT-OSS 120B         → warmup → measure → unload → confirm gone
    (any model whose warmup exceeds 5 minutes is skipped and the next runs)
--- After all LLM tests ---
10. Run embedding benchmarks (no server needed)
11. unload_all_models() — hard sweep to ensure GPU memory is clear
12. Start ComfyUI
13. Run image generation benchmarks
14. Shut down ComfyUI
15. Save results_<hostname>.json
```

### Model isolation

Only one model is ever in memory at a time. After each model's runs complete,
the script sends a `keep_alive: 0` request to Ollama to force eviction, then
polls `/api/ps` until the model is confirmed gone before loading the next one.
Before starting ComfyUI, `unload_all_models()` sweeps `/api/ps` one final time
as a hard guarantee.

### Servers

Servers are started and stopped automatically:
- **Ollama** — started if not running, left running after (it's a system service)
- **ComfyUI** — started just before image tests, shut down immediately after

Ctrl-C and crashes are handled — a signal handler and `finally` block ensure
ComfyUI is always shut down cleanly.

### Embedding device

| Machine | PyTorch backend | Notes |
|---|---|---|
| Apple Silicon Mac | MPS (Metal) | Auto-detected |
| Linux + NVIDIA | CUDA | Auto-detected |
| Windows + AMD | CPU | ROCm not available to PyTorch on Windows; Ollama uses the GPU independently for LLM tests |

CPU embedding results are tagged `(cpu)` in the JSON and dimmed in `compare.py`
output, excluded from rankings.

---

## Benchmark Parameters

| Parameter | Value |
|---|---|
| LLM context lengths | 2K and 8K tokens |
| LLM warmup runs | 2 (discarded) |
| LLM measured runs | 5 (averaged) |
| LLM warmup timeout | 300s per run — model skipped if exceeded |
| LLM metrics | TTFT, tokens/sec (TPS), total time |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| Embedding corpus | 5,000 sentences |
| Embedding batch sizes | 32, 128, 512 |
| Image models | SDXL (20 steps), Flux.1-schnell (4 steps), Flux.1-dev (20 steps) |
| Image resolutions | 1024×1024 and 1536×1536 |
| Image seed | 42 (fixed) |
| Image metrics | Seconds per image per model per resolution |
| Image skip | Model skipped automatically if checkpoint not found |

---

## Image Models

Three image models are tested. Each is skipped automatically if its checkpoint
file is not found in `ComfyUI/models/checkpoints/` — no errors, just a clear
skip message.

| Model | Checkpoint filename | Steps | Notes |
|---|---|---|---|
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~6.5GB, fits 8GB VRAM, no login required |
| Flux.1-schnell | `flux1-schnell.safetensors` | 4 | ~24GB, Apache 2.0, no login required |
| Flux.1-dev | `flux1-dev.safetensors` | 20 | ~24GB, highest quality, gated (HF login required) |

### Downloading checkpoints

**SDXL** (no login required):
```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0   sd_xl_base_1.0.safetensors   --local-dir ComfyUI/models/checkpoints
```

**Flux.1-schnell** (no login required, Apache 2.0):
```bash
huggingface-cli download black-forest-labs/FLUX.1-schnell   flux1-schnell.safetensors   --local-dir ComfyUI/models/checkpoints
```

**Flux.1-dev** (requires accepting license at huggingface.co/black-forest-labs/FLUX.1-dev):
```bash
huggingface-cli login   # paste your token
huggingface-cli download black-forest-labs/FLUX.1-dev   flux1-dev.safetensors   --local-dir ComfyUI/models/checkpoints
```

`setup_check.py` downloads all three automatically:
- **SDXL** and **Flux.1-schnell** need no login — downloaded silently
- **Flux.1-dev** is gated — the script checks for a file called `hf.txt` in
  your working directory first. If found, it uses the token inside it. If not,
  it tries your cached HuggingFace login, then prompts you to paste a token
  (offering to save it to `hf.txt` for future runs).

`hf.txt` format — just the token on a single line:
```
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

You can have any combination of checkpoints — the benchmark runs whatever it finds.

---

## Platform Notes

`setup.sh` / `setup.bat` handle everything — the notes below only cover
prerequisites the scripts can't install automatically, and platform-specific
quirks to be aware of.

### macOS
- If you don't have Homebrew, `setup.sh` installs it automatically.
- Before running benchmarks: plug in power, disable sleep (System Settings → Battery).

### Linux (NVIDIA GPU)
- Python 3.11 is installed via apt if missing. On non-Debian distros, install it manually first.

### DGX Spark
- Ollama is installed via snap if missing (`sudo snap install ollama`).
- After each model run, unused memory may not free immediately. The benchmark script flushes it automatically between models, but if RAM looks full outside of a run: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

### Windows (NVIDIA GPU)
- Install the CUDA Toolkit manually before running `setup.bat`: https://developer.nvidia.com/cuda-downloads
- If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Windows (AMD RDNA3+ / Ryzen AI Max+)
- PyTorch cannot access AMD GPUs on Windows, so embedding benchmarks fall back to CPU — this is expected. Ollama uses the GPU independently for LLM and image tests.
- The `No GPU backend detected` warning from setup_check is not an error.
- If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Running the Comparison

Copy result files from all machines to one machine, then:

```bash
python compare.py results_*.json
# or explicitly:
python compare.py results_mac.json results_dgx.json results_ryzen.json
```

Output is color-coded: green = best, red = slowest, dimmed `(cpu)` = ran on
CPU and excluded from rankings. A `compare_results.json` is also saved.

---

## CLI Reference

```
python benchmark.py [options]

--tests llm emb img     Tests to run (default: all three)
--runs N                Measured runs per test (default: 5)
--warmup N              Warmup runs before measuring (default: 2)
--warmup-timeout N      Seconds per warmup run before skipping model (default: 300)
--small-only            Run only small-tier LLM models (≤16GB)
--large-only            Run only large-tier LLM models (32GB+)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results_<hostname>.json)
```

Examples:
```bash
# Full run — all 8 models, large ones skipped automatically if they don't fit
python benchmark.py

# LLM tests only, quick check with 3 runs
python benchmark.py --tests llm --runs 3

# Skip image generation
python benchmark.py --tests llm emb

# Force only small models (useful if you know the large ones won't fit)
python benchmark.py --small-only

# Shorter warmup timeout
python benchmark.py --warmup-timeout 120
```

---

## Tips

- **All platforms:** Close other apps before running — GPU memory contention affects results.
- **Mac:** Watch Activity Monitor → Memory during 70B runs. If pressure turns red and TPS drops between runs, the system is swapping. The Q3 result is your reliable data point; Q4 may be skipped by the warmup timeout.
- **Linux:** Verify PyTorch sees your GPU before running: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- **Windows (NVIDIA):** If PyTorch doesn't detect your GPU, check that the CUDA version in your pip install URL matches `nvcc --version`.
- **Windows (AMD):** The `No GPU backend detected` warning from setup_check is expected — Ollama uses the GPU independently.
- **Expect 2–4 hours** for a full run on the Mac; faster on the Spark and Ryzen.
