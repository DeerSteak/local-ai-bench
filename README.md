# Local AI Bench v1.1

Cross-platform benchmarking for LLM generation, image generation, and embeddings. Designed to run on any hardware from an 8GB GPU up to high-memory unified-memory systems — models that don't fit are skipped automatically with no configuration needed.

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) — free to use, fork, modify, and redistribute for non-commercial purposes. For commercial licensing, contact [beatclikr@gmail.com](mailto:beatclikr@gmail.com).

---

## Quick Start

```bash
git clone https://github.com/DeerSteak/local-ai-bench
cd local-ai-bench
```

| Platform | Script | What it can install |
|---|---|---|
| macOS | `bash setup.sh` | Homebrew, Python |
| Linux / DGX Spark | `bash setup.sh` | Python, Ollama |
| Windows | `setup.bat` | Python, Ollama, ComfyUI portable |

`setup.sh` / `setup.bat` show exactly what they need to install (Homebrew and/or Python) and ask before doing it — nothing happens silently.

Each script then creates a virtual environment and hands off to `setup_check.py`, which detects your hardware and shows a numbered list of every LLM and image model — everything selected by default. Type numbers to toggle individual models (`2 4 7-9`), a size tier (`xs`/`s`/`m`/`l`) to toggle every model at that tier — LLM and image checkpoints together, e.g. `s` toggles the small-tier LLMs and SDXL as a group — `emb`/`img` to toggle a whole section, `a` to select/deselect all, or just press Enter to install everything shown; `q` or Ctrl-C cancels at any point with nothing installed yet. If you selected a gated image model (SD3.5 Large, Flux.1-dev, Flux.2-dev), it asks for a HuggingFace token next. After that, everything you picked — Ollama, pip packages, models, image checkpoints — installs with no further prompts. When setup is complete:

```bash
# Linux / macOS
bash run_linux_mac.sh

# Windows
run_windows.bat
```

These scripts activate the virtual environment automatically and forward any arguments to `benchmark.py`.

### Platform notes

Close other apps before running — GPU memory contention affects results.

**macOS** — Plug in power and disable sleep (System Settings → Battery) before a long run. For 70B models, watch Activity Monitor → Memory: if pressure turns red and TPS drops between runs, the system is swapping — use `--timeout 600` or `--maxtier medium`.

**Linux (NVIDIA)** — Python 3.11 is installed via apt if missing (you'll be asked to confirm first); on non-Debian distros, install it manually. Verify Ollama sees your GPU before running: `ollama run llama3.1:8b-instruct-q4_K_M "hello"` and confirm it loads on GPU in `nvidia-smi`.

**DGX Spark** — Ollama is installed via snap if missing (`setup_check.py` asks before installing it). If RAM looks full outside a benchmark run: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

**Note for Mac and Linux: If the script fails with a permissions error, run `sudo bash setup.sh` instead. 

**Windows (NVIDIA)** — The setup script detects the GPU and downloads the latest official ComfyUI NVIDIA portable build (bundled Python environment). No manual CUDA Toolkit install required.

**Windows (AMD)** — The setup script downloads the latest official ComfyUI AMD portable build. No manual ROCm install required.

**Windows (Intel Arc)** — The setup script downloads the latest official ComfyUI Intel portable build with XPU support.

**Windows (all)** — If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Workloads

### LLM

Twelve models across four tiers are benchmarked by default. If any warmup or measured run exceeds the 300-second timeout, the model is skipped and the benchmark moves on — small GPUs naturally skip the large models without any flags.

Every model is run through **two separate LLM tests**, back to back, both measured at the same four context lengths (2K / 8K / 32K / 64K):

- **Single-shot** — a large prompt, padded to the target size and sent fresh (with unique content) on every run, so it's always a genuine cold prefill of that many tokens with nothing cached. This simulates dropping a large document, codebase, or transcript into a single prompt and asking one question about it.
- **Conversation** — a real multi-turn chat that grows to each target depth: the model explains Plato's Allegory of the Cave, then each following turn asks for more detail on a section. Because the message history grows turn over turn, this measures the cost of processing the *next* turn against an already-filled context — the pattern of a long chat session or an agent loop, not a one-shot cold start.

These two tests measure genuinely different things, and their TTFT numbers are **not** comparable at face value — see [What the charts mean](#what-the-charts-mean) for why the conversation test's TTFT is typically far lower than the single-shot test's at the same nominal context length.

A model is only excluded from the conversation test if it timed out or was skipped in the single-shot test (e.g. too large to fit in memory) — that check only runs when the LLM test ran earlier in the same session; running `--tests conv` on its own has no single-shot data to check against, so every model is tested.

#### Extra-small tier (<6B params)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.2 3B Q4_K_M | `llama3.2:3b-instruct-q4_K_M` | ~2.0 GB |
| Phi 4 Mini | `phi4-mini` | ~2.5 GB |
| Qwen3.5 4B | `qwen3.5:4b` | ~2.6 GB |

#### Small tier (≤20B params)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.1 8B Q4_K_M | `llama3.1:8b-instruct-q4_K_M` | ~4.9 GB |
| Gemma 4 E4B | `gemma4:e4b` | ~9.6 GB |
| GPT-OSS 20B (MXFP4) | `gpt-oss:20b` | ~14 GB |

#### Medium tier (26–35B params)

| Model | Ollama tag | Size |
|---|---|---|
| Gemma 4 26B (MoE, 4B active) | `gemma4:26b` | ~18 GB |
| DeepSeek-R1 32B | `deepseek-r1:32b` | ~20 GB |
| Qwen3.6 35B-A3B (MoE) | `qwen3.6:35b-a3b` | ~22 GB |

#### Large tier (70B+ params)

| Model | Ollama tag | Size |
|---|---|---|
| Llama 3.3 70B Q4_K_M | `llama3.3:70b-instruct-q4_K_M` | ~43 GB |
| DeepSeek-R1 70B | `deepseek-r1:70b` | ~43 GB |
| GPT-OSS 120B (MXFP4) | `gpt-oss:120b` | ~65 GB |

**DeepSeek-R1** is a reasoning model that generates internal thinking tokens before its answer. TPS includes this thinking output, and TTFT captures the first thinking token rather than the first answer token — an accurate reflection of real-world inference.

**Llama versions:** Llama 3.2 tops out at 3B parameters. The 8B slot uses Llama 3.1; the 70B slot uses Llama 3.3, the most recent 70B instruction model.

### Image Generation

Five models are tested at 1024×1024 and 1536×1536 — except Stable Diffusion 1.5, which uses 512×512 and 768×768 instead (see below). Any model whose checkpoint is absent from `ComfyUI/models/checkpoints/` is skipped automatically; `setup_check.py` downloads them on first run.

Each of the 3 measured runs uses a different seed (`seed + run index`) — an identical seed and workflow would let ComfyUI cache every node and return a cached result almost instantly instead of actually re-running generation. The warmup run before those 3 also uses its own distinct seed, for the same reason.

| Model | Checkpoint | Steps | Size | Tier | HuggingFace login |
|---|---|---|---|---|---|
| Stable Diffusion 1.5 | `v1-5-pruned-emaonly.safetensors` | 20 | ~2.1 GB | xsmall | No |
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~6.5 GB | small | No |
| SD3.5 Large | `sd3.5_large.safetensors` | 28 | ~16.5 GB | medium | Yes (free) |
| Flux.1-dev | `flux1-dev.safetensors` | 20 | ~24 GB | large | Yes (free) |
| Flux.2-dev | `flux2-dev.safetensors` | 28 | ~64 GB | large | Yes (free) |

**Stable Diffusion 1.5** was trained at 512×512; testing it at the other models' 1024/1536 resolutions produces visibly degraded (duplicated-subject) output, so it gets its own native-range pair — 512×512 and 768×768 (the same 1.5x step used for everything else) — instead of the shared resolution list.

`--maxtier` caps image models the same way it caps LLMs — see [CLI Reference](#cli-reference).

SD3.5 Large, Flux.1-dev, and Flux.2-dev require a free HuggingFace account and license acceptance at:
- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.2-dev

If you select one of these in the model picker, `setup_check.py` finds your HF token in this order:

1. `HF_TOKEN` environment variable
2. `hf.txt` in the repo root (token on a single line)
3. Interactive prompt — offers to save to `hf.txt` for future runs

### Embeddings

Two models via Ollama — Nomic Embed Text and MixedBread Embed Large — measured on a single real-world task: chunking a real multi-chapter document (`sample_document.txt`, ~19 chapters) into paragraph-sized pieces (capped at 150 words each) and embedding every chunk from it in one call, the way a RAG ingestion pipeline actually embeds a document — rather than sweeping arbitrary batch sizes that don't correspond to real client behavior. The chunk cap also keeps every chunk safely under any embedding model's context length, regardless of the source document's formatting. Ollama uses the GPU on all supported platforms (Metal, CUDA, ROCm), so results are directly comparable across machines.

If you see repeated connection errors or crashes during the embedding tests (some GPU backends are unstable or immature under batched embedding workloads), try `--emb-cpu-only` to force CPU-only inference instead — in some cases this is also faster or just more stable than a flaky GPU path. This restarts Ollama with GPU devices hidden for the duration of the embedding tests, then restores normal GPU mode afterward.

| Model | Ollama tag | Size |
|---|---|---|
| Nomic Embed Text | `nomic-embed-text` | ~274 MB |
| MixedBread Embed Large | `mxbai-embed-large` | ~670 MB |

---

## CLI Reference

```
run_linux_mac.sh [options]  # Linux / macOS
run_windows.bat [options]   # Windows

--tests llm conv emb img  Tests to run (default: all four)
--emb-cpu-only          Force CPU-only inference for the embedding tests
                        (restarts Ollama with GPU devices hidden, then
                        restores normal GPU mode afterward)
--warmup N              Warmup runs before measuring (default: 2)
--timeout N             Seconds per run before skipping model (default: 300)
--maxtier TIER          Cap LLM models (single-shot + conversation) AND image
                        models at this tier and below: xsmall (<6B / +SD1.5),
                        small (≤20B / +SDXL), medium (26–35B / +SD3.5 Large),
                        large (70B+ / +Flux.1-dev, Flux.2-dev — default, no cap)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results_<hostname>_<timestamp>.json)
```

Every test always measures 3 runs per checkpoint and averages them.

```bash
# Full run — large models skipped automatically if they don't fit
bash run_linux_mac.sh

# LLM only, quick check
bash run_linux_mac.sh --tests llm

# Skip image generation
bash run_linux_mac.sh --tests llm conv emb

# Conversation benchmark only
bash run_linux_mac.sh --tests conv

# Cap at small-tier models and below — skips medium/large LLMs and
# medium/large-tier image models (SD3.5 Large, Flux.1-dev, Flux.2-dev),
# leaving SD1.5 and SDXL for the image test
bash run_linux_mac.sh --maxtier small

# Give slow hardware more time per run
bash run_linux_mac.sh --timeout 600
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
| LLM | Two charts per model — Tokens/sec and TTFT — across context lengths (2K / 8K / 32K / 64K), single-shot cold-prefill test |
| LLM Conversation | Same two charts per model and context lengths, but from the multi-turn conversation test |
| Embeddings | Chunks per second embedding one real document in a single call |
| Images | One grouped bar chart per resolution — all image models side by side per host |

The **Models** filter and **Machine** labels are shared between the LLM and LLM Conversation sections, so switching between them keeps the same models/files selected.

### What the charts mean

**LLM → Tokens/sec.** Decode throughput (tokens generated per second) for the single-shot test, at each context length. Higher is better. This is generation speed *after* the prompt has already been processed — it answers "once the model starts responding, how fast does text come out?"

**LLM → TTFT.** Time to process the single-shot prompt before the first token comes back — a genuine cold prefill, since every run sends fresh, never-before-seen prompt content. Lower is better. This answers "if I paste a large document and hit send, how long do I wait before anything happens?" TTFT rises sharply with context length here, since the model has to run every one of those tokens through the network with nothing cached.

**LLM Conversation → Tokens/sec.** The same decode-throughput metric, but measured mid-conversation instead of after a single cold prompt. Generally close to the single-shot number for the same model — decode speed doesn't depend much on how the context got filled.

**LLM Conversation → TTFT.** Time to process just the *next* turn in an already-long conversation, relying on the backend's KV-cache reuse (llama.cpp/Ollama's slot cache) so only the new turn's tokens need to be run through the network, not the entire history again. This is **why conversation TTFT at, say, 32K is typically a small fraction of single-shot TTFT at 32K** — they're not measuring the same thing. Single-shot TTFT is "cold start with a huge prompt"; conversation TTFT is "one more message in a chat that's already this long." Both are real workloads; which one matters more depends on whether your use case looks like one-shot document Q&A or an ongoing chat/agent session.

**Embeddings → Chunks/sec.** Throughput embedding one real document's chunks in a single call. Higher is better.

**Images → Sec/image.** Wall-clock time to generate one image at a given resolution, per model. Lower is better.

### Multi-file comparison

Each file is assigned a colour (blue → orange → green → purple → red → teal). All charts use that colour to identify the host, making results from different machines directly comparable. The **Models** filter shows or hides individual models.

### Exporting

Drop a logo image onto the **Logo** drop zone to embed it in the bottom-right corner of every chart. Click **Save PNG** to export all visible charts as individual files:

```
llama3.1-8b-q4_tps.png
llama3.1-8b-q4_ttft.png
llama3.1-8b-q4_conv_tps.png       # LLM Conversation section
llama3.1-8b-q4_conv_ttft.png      # LLM Conversation section
embeddings.png
1024x1024_images.png
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

LLM benchmarks run first (small → medium → large), followed by embeddings. Before starting ComfyUI for image tests, Ollama is stopped entirely (not just unloaded) to free up its memory for image generation, since image tests always run last.

Each LLM model follows this pattern:

```
warmup (2 runs)
  → measure single-shot   (3 runs at 2K / 8K / 32K / 64K)
  → measure conversation  (3 runs at 2K / 8K / 32K / 64K, one continuous growing chat)
  → unload → confirm gone
```

The single-shot test builds an independent padded prompt for every run. The conversation test keeps a single chat growing across all four checkpoints — it grows past each depth, takes 3 measured turns there, then keeps growing toward the next one — with `num_ctx` sized generously above the largest checkpoint so growth never hits the context ceiling mid-session (hitting it would force a full reprocess and corrupt the "incremental turn" measurement).

Any run that exceeds the 300-second timeout causes the model to be skipped immediately. Only one model is ever in memory at a time: after each model completes both tests, a `keep_alive: 0` request to Ollama forces eviction, and `/api/ps` is polled until the model is confirmed unloaded before the next one loads.

A model is only excluded from the conversation test if it timed out or was skipped in the single-shot test.

**Ollama** is started if not already running. If the benchmark started it, it is shut down at exit; if it was already running, it is left running.

**ComfyUI** is started just before the image tests and shut down immediately after. A signal handler and `finally` block ensure clean shutdown on Ctrl-C or crash.

### Parameters

| Parameter | Value |
|---|---|
| LLM context lengths | 2K, 8K, 32K, 64K |
| LLM test modes | Single-shot (cold prefill), Conversation (incremental turn in a growing chat) |
| LLM warmup runs | 2 (discarded) |
| LLM measured runs | 3 per context length per test mode, averaged |
| Run timeout | 300s per run — model skipped if exceeded |
| LLM metrics | TTFT, tokens/sec (TPS) |
| Conversation test exclusion | Model excluded if it timed out or was skipped in the single-shot test |
| Embedding models | `nomic-embed-text`, `mxbai-embed-large` (via Ollama) |
| Embedding corpus | `sample_document.txt` chunked into ~150-word paragraph-sized pieces (~190 chunks), embedded in one call |
| Embedding measured runs | 3, averaged |
| Image models | SD1.5 (20 steps), SDXL (20 steps), SD3.5 Large (28 steps), Flux.1-dev (20 steps), Flux.2-dev (28 steps) |
| Image resolutions | 1024×1024, 1536×1536 (SD1.5: 512×512, 768×768) |
| Image seed | 42 (fixed) |
| Image metrics | Seconds per image, per model, per resolution |
| Image measured runs | 3 per resolution, averaged |

---

## Files

| File | Purpose |
|---|---|
| `setup.sh` | One-shot setup for macOS and Linux |
| `setup.bat` | One-shot setup for Windows |
| `setup_check.py` | Called by setup scripts — detects hardware, lets you pick which models to install, then installs everything unattended |
| `run_linux_mac.sh` | Activates the venv and runs `benchmark.py` on Linux / macOS |
| `run_windows.bat` | Activates the venv and runs `benchmark.py` on Windows |
| `benchmark.py` | Main benchmark — produces `results_<hostname>.json` |
| `compare.py` | Comparison — takes all result JSONs and prints a ranked summary table |
| `launch_dashboard.py` | Builds and serves the dashboard, opens browser automatically |
