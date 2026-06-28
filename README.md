# LLM Benchmark Suite

Cross-platform benchmarking for LLM generation, image generation, and embeddings.
Designed to run on any hardware from an 8GB GPU up to high-memory unified-memory
systems. Models that don't fit are skipped automatically — no configuration needed.

### License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Free to use, fork, modify, and redistribute for non-commercial purposes. For commercial licensing, contact [beatclikr@gmail.com](mailto:beatclikr@gmail.com).

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

## Workloads

### LLM

Nine models across three tiers are attempted by default. If a model doesn't
complete warmup within 5 minutes it is skipped with a clear message and the
benchmark moves on. This means the same command works on any hardware — small
GPUs naturally skip the large models without any extra flags.

#### Small tier (≤16GB VRAM)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.1 8B Q4_K_M | `llama3.1:8b-instruct-q4_K_M` | ~4.9 GB |
| DeepSeek-R1 8B | `deepseek-r1:8b` | ~5.2 GB |
| Gemma 4 E4B | `gemma4:e4b` | ~9.6 GB |
| GPT-OSS 20B (MXFP4) | `gpt-oss:20b` | ~14 GB |

#### Medium tier (16–32GB VRAM)

| Model | Ollama tag | Size |
|---|---|---|
| Gemma 4 27B (MoE, 4B active) | `gemma4:26b` | ~18 GB |
| DeepSeek-R1 32B | `deepseek-r1:32b` | ~20 GB |
| Qwen3.6 35B-A3B (MoE) | `qwen3.6:35b-a3b` | ~22 GB |

#### Large tier (42GB+ VRAM)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.3 70B Q4_K_M | `llama3.3:70b-instruct-q4_K_M` | ~43 GB |
| DeepSeek-R1 70B | `deepseek-r1:70b` | ~43 GB |
| GPT-OSS 120B (MXFP4) | `gpt-oss:120b` | ~65 GB |

**Notes on DeepSeek-R1:** R1 is a reasoning model — it generates internal thinking
tokens before producing its answer. TPS measurements include this thinking output,
and TTFT captures the first thinking token rather than the first answer token. This
is an accurate reflection of how the model runs in practice.

**Notes on GPT-OSS:** Both sizes ship in MXFP4 precision only — there are no
separate Q3/Q4 variants. Attempting to pull `gpt-oss:20b-q3_K_M` or
`gpt-oss:120b-q3_K_M` will fail; use the tags above.

**Notes on Llama versions:** Llama 3.2 tops out at 3B parameters. The 8B slot
uses Llama 3.1 (the largest 3.1 variant); the 70B slot uses Llama 3.3 (the
most recent 70B instruction model).

### Image Generation

Three models are tested at 1024×1024 and 1536×1536. Each is skipped automatically
if its checkpoint is not found in `ComfyUI/models/checkpoints/`. `setup_check.py`
downloads them automatically.

| Model | Checkpoint filename | Steps | Size | Login required |
|---|---|---|---|---|
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~6.5 GB | No |
| SD3.5 Large | `sd3.5_large.safetensors` | 28 | ~16.5 GB | Yes (free) |
| Flux.1-dev | `flux1-dev.safetensors` | 20 | ~24 GB | Yes (free) |

SD3.5 Large and Flux.1-dev require a free HuggingFace account and separate license acceptance.
`setup_check.py` checks for a token in this order:

1. `HF_TOKEN` environment variable
2. `hf.txt` file in the repo root (just the token on a single line)
3. Interactive prompt (offers to save to `hf.txt` for future runs)

Accept the licenses at:
- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/black-forest-labs/FLUX.1-dev

### Embeddings

Embeddings run via Ollama (`nomic-embed-text`) on all platforms — 5,000 sentences
across batch sizes of 32, 128, and 512. Ollama uses the GPU on every supported
platform (Metal, CUDA, and ROCm on Windows AMD), so results are directly comparable
across machines.

---

## How It Works

### Execution flow

```
1.  Start Ollama (if not already running)
--- LLM tests (all 10 models, small → medium → large) ---
2.  Llama 3.1 8B Q4_K_M  → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
3.  DeepSeek-R1 8B       → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
4.  Gemma 4 E4B          → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
5.  GPT-OSS 20B          → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
6.  Gemma 4 27B          → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
7.  DeepSeek-R1 32B      → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
8.  Qwen3.6 35B-A3B      → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
9.  Llama 3.3 70B Q4_K_M → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
10. DeepSeek-R1 70B      → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
11. GPT-OSS 120B         → warmup → measure (2K/8K/32K/64K) → unload → confirm gone
    (any run — warmup or measured — that exceeds the timeout is skipped; the model moves on)
--- After all LLM tests ---
12. Run embedding benchmarks via Ollama (nomic-embed-text, batch sizes 32/128/512)
13. unload_all_models() — hard sweep to ensure GPU memory is clear
14. Start ComfyUI
15. Run image generation benchmarks
16. Shut down ComfyUI
17. Save results_<hostname>.json
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

### Benchmark parameters

| Parameter | Value |
|---|---|
| LLM context lengths | 2K, 8K, 32K, 64K |
| LLM warmup runs | 2 (discarded) |
| LLM measured runs | 5 (averaged) |
| Run timeout | 300s per run (warmup and measured) — model skipped if exceeded |
| LLM metrics | TTFT, tokens/sec (TPS) |
| Embedding model | `nomic-embed-text` (via Ollama) |
| Embedding corpus | 5,000 sentences |
| Embedding batch sizes | 32, 128, 512 |
| Image models | SDXL (20 steps), SD3.5 Large (28 steps), Flux.1-dev (20 steps) |
| Image resolutions | 1024×1024 and 1536×1536 |
| Image seed | 42 (fixed) |
| Image metrics | Seconds per image per model per resolution |
| Image skip | Model skipped automatically if checkpoint not found |

---

## Dashboard

An interactive results explorer for visualising and exporting benchmark output.

### Quick launch

```bash
python launch_dashboard.py
```

Checks for Node.js/npm, installs dependencies if needed, builds the app on first run, starts a local server on port 3000, and opens the browser automatically.

```bash
python launch_dashboard.py --port 8080   # use a different port
python launch_dashboard.py --rebuild     # force a fresh npm build
```

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

### Manual setup (development)

```bash
cd dashboard
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

---

## Prerequisites & Setup

`setup.sh` / `setup.bat` handle most things automatically. The notes below cover
prerequisites the scripts can't install, and platform-specific quirks to be aware of.

### macOS
- If you don't have Homebrew, `setup.sh` installs it automatically.
- Before running benchmarks: plug in power, disable sleep (System Settings → Battery).

### Linux (NVIDIA GPU)
- Python 3.11 is installed via apt if missing. On non-Debian distros, install it manually first.

### DGX Spark
- Ollama is installed via snap if missing (`sudo snap install ollama`).
- After each model run, unused memory may not free immediately. The benchmark script flushes it automatically between models, but if RAM looks full outside of a run: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

### Windows (NVIDIA GPU)
- `setup.bat` detects the NVIDIA GPU and automatically downloads the official ComfyUI NVIDIA portable build (CUDA 12.6, bundled Python environment). No manual CUDA Toolkit install required.
- If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Windows (AMD GPU)
- `setup.bat` detects AMD/Radeon GPUs and automatically downloads the official ComfyUI AMD portable build (ROCm 7.1.1, bundled Python environment). No manual ROCm install required.
- If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Windows (Intel Arc GPU)
- `setup.bat` detects Intel Arc GPUs and automatically downloads the official ComfyUI Intel portable build (bundled Python environment with XPU support).
- If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## CLI Reference

```
python benchmark.py [options]

--tests llm emb img     Tests to run (default: all three)
--runs N                Measured runs per test (default: 5)
--warmup N              Warmup runs before measuring (default: 2)
--timeout N             Seconds per run (warmup and measured) before skipping model (default: 300)
--small-only            Run only small-tier LLM models (≤16GB VRAM)
--medium-only           Run only medium-tier LLM models (16–32GB VRAM)
--large-only            Run only large-tier LLM models (32GB+ VRAM)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results_<hostname>.json)
```

Examples:
```bash
# Full run — all 10 models, large ones skipped automatically if they don't fit
python benchmark.py

# LLM tests only, quick check with 3 runs
python benchmark.py --tests llm --runs 3

# Skip image generation
python benchmark.py --tests llm emb

# Force only small models (useful if you know the large ones won't fit)
python benchmark.py --small-only

# Longer timeout (gives slow hardware more time to complete)
python benchmark.py --timeout 600
```

### Running a comparison

Copy result files from all machines to one machine, then:

```bash
python compare.py results_*.json
# or explicitly:
python compare.py results_mac.json results_dgx.json results_ryzen.json
```

Output is color-coded: green = best, red = slowest. A `compare_results.json` is also saved.

---

## Files

| File | Purpose |
|---|---|
| `setup.sh` | One-shot setup for macOS and Linux |
| `setup.bat` | One-shot setup for Windows |
| `setup_check.py` | Called by setup scripts — installs deps, pulls models, downloads checkpoints |
| `benchmark.py` | Main benchmark — produces `results_<hostname>.json` |
| `compare.py` | Comparison — takes all result JSONs and prints a ranked summary table |
| `launch_dashboard.py` | Builds and serves the dashboard, opens browser automatically |

---

## Tips

- **All platforms:** Close other apps before running — GPU memory contention affects results.
- **Mac:** Watch Activity Monitor → Memory during 70B runs. Both large-tier models need ~42–43 GB — if memory pressure turns red and TPS drops between runs, the system is swapping. Use `--timeout 600` to give more time, or stick to `--medium-only` if memory is tight.
- **Linux:** Verify Ollama sees your GPU before running: `ollama run llama3.1:8b-instruct-q4_K_M "hello"` and check it loads on GPU in `nvidia-smi`.
- **Windows (AMD/Intel):** All three benchmarks use the GPU — LLM via Ollama, embeddings via Ollama (`nomic-embed-text`), image generation via the ComfyUI portable build.
- **Expect 2–4 hours** for a full run on the Mac; faster on the Spark and Ryzen.

