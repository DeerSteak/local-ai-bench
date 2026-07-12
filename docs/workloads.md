[← Back to README](../README.md)

# Workloads

Three workload types are benchmarked: LLM generation (two test modes), image generation, and embeddings. Every workload skips models automatically when they don't fit in available memory — no configuration needed on smaller hardware.

Every "Size" figure below is the model's actual on-disk download size, rounded **up** to the next 0.1 GB (not nearest) — the same convention `setup_check.py` uses for its own disk-space check, so an estimate never undersells how much room a model actually needs.

**Contents**
- [LLM](#llm)
  - [Extra-small tier (<6B params)](#extra-small-tier-6b-params)
  - [Small tier (≤20B params)](#small-tier-20b-params)
  - [Medium tier (26–35B params)](#medium-tier-2635b-params)
  - [Large tier (70B+ params)](#large-tier-70b-params)
  - [Dense vs. Mixture-of-Experts (MoE)](#dense-vs-mixture-of-experts-moe)
- [Image Generation](#image-generation)
- [Embeddings](#embeddings)

## LLM

Twelve models across four tiers are benchmarked by default. If any warmup or measured run exceeds the 300-second timeout, the model is skipped and the benchmark moves on — small GPUs naturally skip the large models without any flags.

Every model is run through **two separate LLM tests**, back to back:

- **Single-shot** — a large prompt, padded to the target size and sent fresh (with unique content) on every run, measured at four context lengths (2K / 8K / 32K / 64K), so it's always a genuine cold prefill of that many tokens with nothing cached. This simulates dropping a large document, codebase, or transcript into a single prompt and asking one question about it.
- **Conversation** — a real multi-turn chat, measured at up to eight depths (0 / 2K / 4K / 8K / 16K / 32K / 64K / 96K, whichever the model's own context window reaches): the model explains Plato's Allegory of the Cave, then each following turn asks for more detail on a section, growing the conversation from a blank slate toward 96K. This test is expensive (many turns growing all the way to the sampling ceiling), so it always runs a single conversation regardless of `--runs` — that flag only repeats the other tests (see [CLI Reference](cli-reference.md)).

The context window handed to a model for the conversation test is still the full 128K if it supports at least that much, otherwise its own real maximum — looked up live from Ollama for the exact model that's actually pulled, not assumed or hardcoded. Sampling deliberately stops at 96K rather than 128K so the conversation always has real headroom left against that context window instead of scraping the ceiling — most models' real maximum is exactly 128K, with no slack to spare for the growth loop's final turns.

These two tests measure genuinely different things, and their TTFT numbers are **not** comparable at face value — see [What the charts mean](dashboard.md#what-the-charts-mean) for why the conversation test's TTFT is typically far lower than the single-shot test's at the same nominal context length.

If a model's single-shot decode speed drops below 15 tok/s at some context length, the single-shot test stops there — deeper context lengths are skipped for that model, since a slower run would just be a longer wait for a data point nobody needs. A model is excluded from the conversation test *entirely* if it timed out or was already marked too slow in the single-shot test (e.g. too large to fit in memory) — that pre-flight check only runs when the single-shot test ran earlier in the same session; running `--tests conv` on its own has no single-shot data to check against, so every model is tested there.

Separately, *within* the conversation test itself: if the decode speed at any history depth drops below the slow-model cutoff, the conversation exits early and records results to that point. Pass `--force-all` to ignore this cutoff and always run every context length (see [CLI Reference](cli-reference.md)).

### Extra-small tier (<6B params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| Llama 3.2 3B Q4_K_M | `llama3.2:3b-instruct-q4_K_M` | ~2.1 GB | Dense |
| Phi 4 Mini | `phi4-mini` | ~2.5 GB | Dense |
| Qwen3.5 4B | `qwen3.5:4b` | ~3.4 GB | Dense |

### Small tier (≤20B params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| Llama 3.1 8B Q4_K_M | `llama3.1:8b-instruct-q4_K_M` | ~5.0 GB | Dense |
| Gemma 4 E4B | `gemma4:e4b` | ~9.7 GB | Dense (Per-Layer Embeddings — see below) |
| GPT-OSS 20B (MXFP4) | `gpt-oss:20b` | ~13.8 GB | MoE — 3.6B active of ~20.9B total |

### Medium tier (26–35B params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| Gemma 4 26B | `gemma4:26b` | ~18.0 GB | MoE — 4B active of ~26B total |
| DeepSeek-R1 32B | `deepseek-r1:32b` | ~19.9 GB | Dense |
| Qwen3.6 35B-A3B | `qwen3.6:35b-a3b` | ~24.0 GB | MoE — 3B active of 35B total |

### Large tier (70B+ params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| Llama 3.3 70B Q4_K_M | `llama3.3:70b-instruct-q4_K_M` | ~42.6 GB | Dense |
| DeepSeek-R1 70B | `deepseek-r1:70b` | ~42.6 GB | Dense |
| GPT-OSS 120B (MXFP4) | `gpt-oss:120b` | ~65.4 GB | MoE — 5.1B active of ~116.8B total |

### Dense vs. Mixture-of-Experts (MoE)

A **dense** model runs every one of its parameters for every token it generates. A **Mixture-of-Experts (MoE)** model instead routes each token through only a small subset of specialized "expert" sub-networks, out of many more it holds in total — so most of its parameters sit idle on any given token. Ollama tags spell this out for MoE variants with an `-aN` suffix (e.g. `qwen3.6:35b-a3b`) or in the model's own naming (e.g. Gemma's `26B-A4B`): the number after `a` is how many parameters actually activate per token ("active"), versus the number before it (total parameters, which is what drives memory/VRAM use).

Because decode speed tracks active parameters far more closely than total size or VRAM footprint, an MoE model can generate noticeably faster than a dense model of similar total size — in the medium tier here, Qwen3.6 35B-A3B activates only ~3B parameters per token versus DeepSeek-R1 32B's dense 32B, despite both landing in a similar VRAM footprint (~20–22 GB). Gemma 4 E4B is a dense model that uses a different technique, Per-Layer Embeddings (PLE), to shrink its loaded memory footprint (~8B total parameters, ~4.5B "effective") without changing how much compute each token costs — every layer's core weights still run for every token, unlike MoE's routing.

**DeepSeek-R1** is a reasoning model that generates internal thinking tokens before its answer, via Ollama's separate `thinking` field rather than mixing them into the answer text. Tokens/sec includes this thinking output — Ollama's reported generation count and duration cover the whole response, thinking included, with no separate accounting. TTFT reflects prompt-processing time only (Ollama's reported prompt-eval duration), which happens before generation starts, so it is not affected by how much the model reasons afterward.

**Llama versions:** Llama 3.2 tops out at 3B parameters. The 8B slot uses Llama 3.1; the 70B slot uses Llama 3.3, the most recent 70B instruction model.

`--maxtier` caps LLM models (and image models, see below) at a given tier and below — see [CLI Reference](cli-reference.md).

## Image Generation

Five models are tested at 1024×1024 and 1536×1536 — except Stable Diffusion 1.5, which uses 512×512 and 768×768 instead (see below). Any model whose checkpoint is absent from `ComfyUI/models/checkpoints/` is skipped automatically; `setup_check.py` downloads them on first run.

Each measured run (`--runs`, default 3) uses a different seed (`seed + run index`) — an identical seed and workflow would let ComfyUI cache every node and return a cached result almost instantly instead of actually re-running generation. The warmup run before those also uses its own distinct seed, for the same reason.

| Model | Checkpoint | Steps | Size | Tier | HuggingFace login |
|---|---|---|---|---|---|
| Stable Diffusion 1.5 | `v1-5-pruned-emaonly.safetensors` | 20 | ~4.3 GB | xsmall | No |
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~7.0 GB | small | No |
| SD3.5 Large | `sd3.5_large.safetensors` | 28 | ~16.5 GB | medium | Yes (free) |
| Flux.1-dev | `flux1-dev.safetensors` | 20 | ~23.9 GB | large | Yes (free) |
| Flux.2-dev | `flux2-dev.safetensors` | 28 | ~64.5 GB | large | Yes (free) |

**Stable Diffusion 1.5** was trained at 512×512; testing it at the other models' 1024/1536 resolutions produces visibly degraded (duplicated-subject) output, so it gets its own native-range pair — 512×512 and 768×768 (the same 1.5x step used for everything else) — instead of the shared resolution list.

`--maxtier` caps image models the same way it caps LLMs — see [CLI Reference](cli-reference.md).

SD3.5 Large, Flux.1-dev, and Flux.2-dev require a free HuggingFace account and license acceptance — see [HuggingFace token](setup.md#huggingface-token) in the setup guide.

Generated sample images are saved to `results/images_<hostname>_<timestamp>/`, alongside the matching results JSON — see [Project Structure](project-structure.md).

## Embeddings

Two models via Ollama — Nomic Embed Text and MixedBread Embed Large — measured on a single real-world task: chunking a real multi-chapter document (`sample_document.txt`, ~27 chapters) into paragraph-sized pieces (capped at 150 words each) and embedding every chunk from it in one call, the way a RAG ingestion pipeline actually embeds a document — rather than sweeping arbitrary batch sizes that don't correspond to real client behavior. The chunk cap also keeps every chunk safely under any embedding model's context length, regardless of the source document's formatting.

Like the other test types, each model gets `--warmup` discarded runs first — the very first embed call against a freshly-loaded model pays a one-time model-load cost that has nothing to do with steady-state throughput, so it's absorbed before the `--runs` measured runs (default 3) rather than skewing them. Ollama uses the GPU on all supported platforms (Metal, CUDA, ROCm), so results are directly comparable across machines.

If you see repeated connection errors or crashes during the embedding tests (some GPU backends are unstable or immature under batched embedding workloads), try `--emb-cpu-only` to force CPU-only inference instead — in some cases this is also faster or just more stable than a flaky GPU path. This restarts Ollama with GPU devices hidden for the duration of the embedding tests, then restores normal GPU mode afterward.

| Model | Ollama tag | Size |
|---|---|---|
| Nomic Embed Text | `nomic-embed-text` | ~0.3 GB |
| MixedBread Embed Large | `mxbai-embed-large` | ~0.7 GB |

---

[← Setup](setup.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
