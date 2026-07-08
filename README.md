# LLM Benchmark Suite

Cross-platform benchmarking for LLM generation, image generation, and embeddings. Designed to run on any hardware from an 8GB GPU up to high-memory unified-memory systems — models that don't fit are skipped automatically with no configuration needed.

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) — free to use, fork, modify, and redistribute for non-commercial purposes. For commercial licensing, contact [beatclikr@gmail.com](mailto:beatclikr@gmail.com).

---

## Quick Start

```bash
git clone https://github.com/DeerSteak/local-ai-bench
cd local-ai-bench
```

| Platform | Script | What it installs automatically |
|---|---|---|
| macOS | `bash setup.sh` | Homebrew, Python |
| Linux / DGX Spark | `bash setup.sh` | Python, Ollama |
| Windows | `setup.bat` | Python, Ollama, ComfyUI portable |

Each script creates a virtual environment and runs `setup_check.py`, which installs Python dependencies, pulls Ollama models, and downloads image checkpoints. When setup is complete:

```bash
# macOS / Linux
source bench-env/bin/activate
python benchmark.py

# Windows
bench-env\Scripts\activate
python benchmark.py
```

### Platform notes

Close other apps before running — GPU memory contention affects results.

**macOS** — Plug in power and disable sleep (System Settings → Battery) before a long run. For 70B models, watch Activity Monitor → Memory: if pressure turns red and TPS drops between runs, the system is swapping — use `--timeout 600` or `--medium-only`.

**Linux (NVIDIA)** — Python 3.11 is installed via apt if missing; on non-Debian distros, install it manually. Verify Ollama sees your GPU before running: `ollama run llama3.1:8b-instruct-q4_K_M "hello"` and confirm it loads on GPU in `nvidia-smi`.

**DGX Spark** — Ollama is installed via snap if missing. If RAM looks full outside a benchmark run: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

**Windows (NVIDIA)** — The setup script detects the GPU and downloads the latest official ComfyUI NVIDIA portable build (bundled Python environment). No manual CUDA Toolkit install required.

**Windows (AMD)** — The setup script downloads the latest official ComfyUI AMD portable build. No manual ROCm install required.

**Windows (Intel Arc)** — The setup script downloads the latest official ComfyUI Intel portable build with XPU support.

**Windows (all)** — If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Workloads

### LLM

Ten models across three tiers are benchmarked by default. If any warmup or measured run exceeds the 300-second timeout, the model is skipped and the benchmark moves on — small GPUs naturally skip the large models without any flags.

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

**DeepSeek-R1** is a reasoning model that generates internal thinking tokens before its answer. TPS includes this thinking output, and TTFT captures the first thinking token rather than the first answer token — an accurate reflection of real-world inference.

**GPT-OSS** ships in MXFP4 precision only — no Q3/Q4 variants exist. Tags like `gpt-oss:20b-q3_K_M` will fail; use the tags above.

**Llama versions:** Llama 3.2 tops out at 3B parameters. The 8B slot uses Llama 3.1; the 70B slot uses Llama 3.3, the most recent 70B instruction model.

### Image Generation

Three models are tested at 1024×1024 and 1536×1536. Any model whose checkpoint is absent from `ComfyUI/models/checkpoints/` is skipped automatically; `setup_check.py` downloads them on first run.

| Model | Checkpoint | Steps | Size | HuggingFace login |
|---|---|---|---|---|
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~6.5 GB | No |
| SD3.5 Large | `sd3.5_large.safetensors` | 28 | ~16.5 GB | Yes (free) |
| Flux.1-dev | `flux1-dev.safetensors` | 20 | ~24 GB | Yes (free) |

SD3.5 Large and Flux.1-dev require a free HuggingFace account and license acceptance at:
- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/black-forest-labs/FLUX.1-dev

`setup_check.py` finds your HF token in this order:

1. `HF_TOKEN` environment variable
2. `hf.txt` in the repo root (token on a single line)
3. Interactive prompt — offers to save to `hf.txt` for future runs

### Embeddings

`nomic-embed-text` via Ollama — 5,000 sentences across batch sizes 32, 128, and 512. Ollama uses the GPU on all supported platforms (Metal, CUDA, ROCm), so results are directly comparable across machines.

---

## CLI Reference

```
python benchmark.py [options]

--tests llm emb img     Tests to run (default: all three)
--runs N                Measured runs per test (default: 5)
--warmup N              Warmup runs before measuring (default: 2)
--timeout N             Seconds per run before skipping model (default: 300)
--small-only            Run only small-tier LLM models (≤16GB VRAM)
--medium-only           Run only medium-tier LLM models (16–32GB VRAM)
--large-only            Run only large-tier LLM models (42GB+ VRAM)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results_<hostname>.json)
```

```bash
# Full run — large models skipped automatically if they don't fit
python benchmark.py

# LLM only, quick check
python benchmark.py --tests llm --runs 3

# Skip image generation
python benchmark.py --tests llm emb

# Small models only
python benchmark.py --small-only

# Give slow hardware more time per run
python benchmark.py --timeout 600
```

A full run takes 2–4 hours on a Mac; faster on the Spark and Ryzen.

### Comparing results

Copy result files from all machines to one machine, then:

```bash
python compare.py results_*.json
# or explicitly:
python compare.py results_mac.json results_dgx.json results_ryzen.json
```

Output is color-coded: green = best, red = slowest. A `compare_results.json` is also saved.

---

## Dashboard

An interactive results explorer for visualising and exporting benchmark output.

```bash
python launch_dashboard.py
python launch_dashboard.py --port 8080   # use a different port
python launch_dashboard.py --rebuild     # force a fresh npm build
```

Requires Node.js/npm. On first run, installs npm dependencies and builds the app, then starts a local server on port 3000 and opens the browser automatically.

### Loading results

Drag one or more `results_*.json` files onto the drop zone in the top-right corner, or click to open a file picker. Up to six files can be loaded at once. Dropping a single file when fewer than six are loaded adds it to the current set; dropping multiple at once replaces all. Sample files for testing are in `samples/`.

### Sections

| Section | Charts |
|---|---|
| LLM | Two charts per model — Tokens/sec and TTFT — across context lengths (8K / 32K / 64K) |
| Embeddings | Sentences per second across batch sizes (32 / 128 / 512) |
| Images | One grouped bar chart per resolution — all image models side by side per host |

### Multi-file comparison

Each file is assigned a colour (blue → orange → green → purple → red → teal). All charts use that colour to identify the host, making results from different machines directly comparable. The **Models** filter shows or hides individual models.

### Exporting

Drop a logo image onto the **Logo** drop zone to embed it in the bottom-right corner of every chart. Click **Save PNG** to export all visible charts as individual files:

```
llama3.1-8b-q4_tps_hostname1_vs_hostname2.png
llama3.1-8b-q4_ttft_hostname1_vs_hostname2.png
hostname1_vs_hostname2_embeddings.png
1024x1024_images_hostname1_vs_hostname2.png
```

The **Chart Width** field (default 708 px) controls the capture width — increase for wider exports.

### Development

```bash
cd dashboard
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

---

## How It Works

### Execution

LLM benchmarks run first (small → medium → large), followed by embeddings. Before starting ComfyUI for image tests, `unload_all_models()` performs a hard sweep of `/api/ps` to ensure GPU memory is clear.

Each LLM model follows this pattern:

```
warmup (2 runs) → measure (5 runs at 2K / 8K / 32K / 64K) → unload → confirm gone
```

Any run that exceeds the 300-second timeout causes the model to be skipped immediately. Only one model is ever in memory at a time: after each model completes, a `keep_alive: 0` request to Ollama forces eviction, and `/api/ps` is polled until the model is confirmed unloaded before the next one loads.

**Ollama** is started if not already running. If the benchmark started it, it is shut down at exit; if it was already running, it is left running.

**ComfyUI** is started just before the image tests and shut down immediately after. A signal handler and `finally` block ensure clean shutdown on Ctrl-C or crash.

### Parameters

| Parameter | Value |
|---|---|
| LLM context lengths | 2K, 8K, 32K, 64K |
| LLM warmup runs | 2 (discarded) |
| LLM measured runs | 5 (averaged) |
| Run timeout | 300s per run — model skipped if exceeded |
| LLM metrics | TTFT, tokens/sec (TPS) |
| Embedding model | `nomic-embed-text` (via Ollama) |
| Embedding corpus | 5,000 sentences |
| Embedding batch sizes | 32, 128, 512 |
| Image models | SDXL (20 steps), SD3.5 Large (28 steps), Flux.1-dev (20 steps) |
| Image resolutions | 1024×1024, 1536×1536 |
| Image seed | 42 (fixed) |
| Image metrics | Seconds per image, per model, per resolution |

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
