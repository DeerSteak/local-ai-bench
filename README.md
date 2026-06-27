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

Nine LLM models across three tiers are attempted by default. If a model doesn't
complete warmup within 5 minutes it is skipped with a clear message and the
benchmark moves on. This means the same command works on any hardware — small
GPUs naturally skip the large models without any extra flags.

### Small tier (≤16GB VRAM)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.1 8B Q3_K_M | `llama3.1:8b-instruct-q3_K_M` | ~4.3 GB |
| Llama 3.1 8B Q4_K_M | `llama3.1:8b-instruct-q4_K_M` | ~4.9 GB |
| Qwen3 14B Q4_K_M | `qwen3:14b-q4_K_M` | ~9.3 GB |
| GPT-OSS 20B (MXFP4) | `gpt-oss:20b` | ~14 GB |

### Medium tier (16–32GB VRAM)

| Model | Ollama tag | Size |
|---|---|---|
| Qwen3 14B Q8_0 | `qwen3:14b-q8_0` | ~16 GB |
| Qwen3.6 35B-A3B | `qwen3.6:35b-a3b` | ~22 GB |

### Large tier (32GB+ VRAM)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.1 70B Q3_K_M | `llama3.1:70b-instruct-q3_K_M` | ~32 GB |
| Llama 3.1 70B Q4_K_M | `llama3.1:70b-instruct-q4_K_M` | ~42 GB |
| GPT-OSS 120B (MXFP4) | `gpt-oss:120b` | ~65 GB |

**Notes on GPT-OSS:** Both sizes ship in MXFP4 precision only — there are no
separate Q3/Q4 variants. Attempting to pull `gpt-oss:20b-q3_K_M` or
`gpt-oss:120b-q3_K_M` will fail; use the tags above.

**Notes on Qwen3 14B:** Both variants are capped at 32K context — 64K produces
multi-minute TTFT at this size and is not useful. No Q3_K_M variant exists.

**Notes on Llama 3.2:** Llama 3.2 tops out at 3B parameters. The 8B slot
belongs to Llama 3.1.

---

## How It Works

### Execution flow

```
1.  Start Ollama (if not already running)
--- LLM tests (all 9 models, small → medium → large) ---
2.  Llama 3.1 8B Q3_K_M   → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
3.  Llama 3.1 8B Q4_K_M   → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
4.  Qwen3 14B Q4_K_M      → warmup → measure (2K/8K/32K)      → unload → confirm gone
5.  GPT-OSS 20B           → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
6.  Qwen3 14B Q8_0        → warmup → measure (2K/8K/32K)      → unload → confirm gone
7.  Qwen3.6 35B-A3B       → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
8.  Llama 3.1 70B Q3_K_M  → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
9.  Llama 3.1 70B Q4_K_M  → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
10. GPT-OSS 120B          → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
    (any model whose warmup exceeds 5 minutes is skipped and the next runs)
--- After all LLM tests ---
11. Run embedding benchmarks via Ollama (mxbai-embed-large, batch sizes 32/128/512)
12. unload_all_models() — hard sweep to ensure GPU memory is clear
13. Start ComfyUI
14. Run image generation benchmarks
15. Shut down ComfyUI
16. Save results_<hostname>.json
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

### Embedding backend

Embeddings run via Ollama (`mxbai-embed-large`) on all platforms, so results are
directly comparable across machines. Ollama uses the GPU on every supported
platform — Metal, CUDA, and ROCm on Windows AMD.

---

## Benchmark Parameters

| Parameter | Value |
|---|---|
| LLM context lengths | 2K, 8K, 32K, 64K (Qwen3 14B capped at 32K) |
| LLM warmup runs | 2 (discarded) |
| LLM measured runs | 5 (averaged) |
| LLM warmup timeout | 300s per run — model skipped if exceeded |
| LLM metrics | TTFT, tokens/sec (TPS) |
| Embedding model | `mxbai-embed-large` (via Ollama) |
| Embedding corpus | 5,000 sentences |
| Embedding batch sizes | 32, 128, 512 |
| Image models | SDXL (20 steps), Flux.1-schnell (4 steps) |
| Image resolutions | 1024×1024 and 1536×1536 |
| Image seed | 42 (fixed) |
| Image metrics | Seconds per image per model per resolution |
| Image skip | Model skipped automatically if checkpoint not found |

---

## Image Models

Two image models are tested. Each is skipped automatically if its checkpoint
is not found in `ComfyUI/models/checkpoints/`. `setup_check.py` downloads
both automatically.

| Model | Checkpoint filename | Steps | Size | Login required |
|---|---|---|---|---|
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~6.5 GB | No |
| Flux.1-schnell | `flux1-schnell.safetensors` | 4 | ~24 GB | Yes (free) |

Flux.1-schnell requires a free HuggingFace account and license acceptance.
`setup_check.py` checks for a token in this order:

1. `HF_TOKEN` environment variable
2. `hf.txt` file in the repo root (just the token on a single line)
3. Interactive prompt (offers to save to `hf.txt` for future runs)

Accept the license at: https://huggingface.co/black-forest-labs/FLUX.1-schnell

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

### Windows (AMD GPU)
- `setup_check.py` detects AMD/Radeon GPUs via `wmic` and automatically clones [comfyui-rocm](https://github.com/patientx-cfz/comfyui-rocm) instead of standard ComfyUI, then runs its `install.bat` to set up a bundled ROCm Python environment. This can take several minutes on first run — it downloads PyTorch with ROCm support.
- Image generation runs on the AMD GPU via the ROCm ComfyUI fork.
- Embedding benchmarks run via Ollama (`mxbai-embed-large`) and use the GPU, same as every other platform.
- If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Dashboard

An interactive results explorer for visualising and exporting benchmark output.

### Setup

```bash
cd dashboard
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

### Loading results

Drag one or more `results_*.json` files onto the drop zone in the top-right corner, or click it to open a file picker. Up to six files can be loaded at once. Dropping a single file when fewer than six are loaded adds it to the current set; dropping multiple files at once replaces all.

Sample files for testing are in `samples/`.

### Sections

Use the **Section** buttons to switch between views:

| Section | Charts |
|---|---|
| LLM | Two charts per model — Tokens/sec and Time to First Token — each across context lengths (2K / 8K / 32K / 64K) |
| Embeddings | Sentences per second across batch sizes (32 / 128 / 512) |
| Images | One grouped bar chart per resolution — all image models side by side per host |

### Multi-file comparison

Each loaded file is assigned a colour (blue → orange → green → purple → red → teal). All charts use that colour to identify the host, so results from different machines are directly comparable at a glance. The **Models** filter in the controls lets you show or hide individual models.

### Exporting

Drop a logo image onto the **Logo** drop zone to embed it in the bottom-right corner of every chart.

Click **Save PNG** to export every visible chart as an individual file. Files are named by type:

```
# LLM
llama3.1-8b-q4_tps_hostname1_vs_hostname2.png
llama3.1-8b-q4_ttft_hostname1_vs_hostname2.png

# Embeddings
hostname1_vs_hostname2_embeddings.png

# Images
1024x1024_images_hostname1_vs_hostname2.png
```

The **Chart Width** field (default 708 px) controls the pixel width of the capture area — increase it for wider exports.

---

## Running the Comparison

Copy result files from all machines to one machine, then:

```bash
python compare.py results_*.json
# or explicitly:
python compare.py results_mac.json results_dgx.json results_ryzen.json
```

Output is color-coded: green = best, red = slowest. A `compare_results.json` is also saved.

---

## CLI Reference

```
python benchmark.py [options]

--tests llm emb img     Tests to run (default: all three)
--runs N                Measured runs per test (default: 5)
--warmup N              Warmup runs before measuring (default: 2)
--warmup-timeout N      Seconds per warmup run before skipping model (default: 300)
--small-only            Run only small-tier LLM models (≤16GB VRAM)
--medium-only           Run only medium-tier LLM models (16–32GB VRAM)
--large-only            Run only large-tier LLM models (32GB+ VRAM)
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
- **Linux:** Verify Ollama sees your GPU before running: `ollama run llama3.1:8b-instruct-q4_K_M "hello"` and check it loads on GPU in `nvidia-smi`.
- **Windows (AMD):** All three benchmarks use the GPU — LLM via Ollama, embeddings via Ollama (`mxbai-embed-large`), image generation via ROCm ComfyUI.
- **Expect 2–4 hours** for a full run on the Mac; faster on the Spark and Ryzen.
