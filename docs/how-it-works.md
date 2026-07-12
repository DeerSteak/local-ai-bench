[← Back to README](../README.md)

# How It Works

**Contents**
- [Execution order](#execution-order)
- [Code organization](#code-organization)
- [Parameters](#parameters)

## Execution order

LLM benchmarks run first (small → medium → large), followed by embeddings. Before starting ComfyUI for image tests, Ollama is stopped entirely (not just unloaded) to free up its memory for image generation, since image tests always run last.

Each LLM model follows this pattern:

```
warmup (--warmup runs, default 2)
  → measure single-shot   (--runs runs at 2K / 8K / 32K / 64K, default 3)
  → measure conversation  (a single full conversation, --runs is ignored here,
                            sampled at 0 / 2K / 4K / 8K / 16K / 32K / 64K / 96K
                            up to the model's real context ceiling)
  → unload → confirm gone
```

The single-shot test builds an independent padded prompt for every run. The conversation test is different: it's a single conversation (this test is expensive enough — many turns growing all the way to the sampling ceiling — that it always runs once, ignoring `--runs`), started from a blank slate and grown toward a 96K sampling ceiling. The model is still given the full 128K context window (or its real maximum, whichever is lower — looked up live via Ollama's `/api/show`, not hardcoded), so there's always headroom left between the top checkpoint and the actual `num_ctx` limit. Growth happens in small steps (capped and scaled to the size of the gap being crossed) rather than one big jump per checkpoint, so the turn that actually crosses each threshold lands close to it instead of overshooting by a large margin.

If the run exceeds the 300-second timeout, it stops wherever it got to — whatever checkpoints it already reached are kept. Only one model is ever in memory at a time: after each model completes both tests, a `keep_alive: 0` request to Ollama forces eviction, and `/api/ps` is polled until the model is confirmed unloaded before the next one loads.

A model is excluded from the conversation test *entirely* if it timed out or was already marked too slow in the single-shot test. Within the conversation test itself there's no mid-conversation early exit — its one run always plays out to its natural end. See [LLM workload](workloads.md#llm) for the full skip logic.

**Ollama** is started if not already running. If the benchmark started it, it is shut down at exit; if it was already running, it is left running.

**ComfyUI** is started just before the image tests and shut down immediately after. A signal handler and `finally` block ensure clean shutdown on Ctrl-C or crash.

## Code organization

The benchmark implementation lives in `scripts/`, split by responsibility:

| Module | Responsibility |
|---|---|
| `scripts/benchmark.py` | CLI argument parsing and orchestration (`main()`) — calls each test class in order and writes results |
| `scripts/config.py` | Shared constants: URLs, paths (`SCRIPT_DIR`, `RESULTS_DIR`, `COMFYUI_DIR`), timeouts, run counts |
| `scripts/shared.py` | Cross-cutting helpers: logging, server lifecycle (start/stop Ollama and ComfyUI), machine profiling, low-level Ollama/ComfyUI HTTP clients |
| `scripts/llm_prefill_benchmark.py` | The single-shot LLM test |
| `scripts/llm_conversation_benchmark.py` | The multi-turn conversation test |
| `scripts/embedding_benchmark.py` | The embeddings test |
| `scripts/image_benchmark.py` | The image generation test (ComfyUI workflow builders + submission) |
| `scripts/models.py` | Single source of truth for every model definition (tags, checkpoints, tiers, sizes) |
| `scripts/setup_check.py` | Hardware detection, model picker, and unattended install — called by `setup.sh`/`setup.bat` |

Values that CLI flags can override at runtime (`RUN_TIMEOUT` via `--timeout`, `N_RUNS` via `--runs`) are read via `config.RUN_TIMEOUT`/`config.N_RUNS` (a dotted attribute lookup) everywhere, rather than imported by name — a plain `from config import RUN_TIMEOUT` would bind a stale copy at import time and silently ignore the CLI override.

## Parameters

| Parameter | Value |
|---|---|
| LLM single-shot context lengths | 2K, 8K, 32K, 64K |
| LLM conversation checkpoints | 0, 2K, 4K, 8K, 16K, 32K, 64K, 96K — capped per model at its real context ceiling (model is still given the full 128K context window, or its real max if lower) |
| LLM test modes | Single-shot (cold prefill), Conversation (a single full conversation, `--runs` ignored) |
| LLM warmup runs | `--warmup` (default: 2, discarded) |
| LLM measured runs | `--runs` — repeated context lengths for single-shot (default: 3, range: 1–10); ignored by the conversation test, which always runs once |
| Run timeout | `--timeout` per run (default: 300s) — that run stops wherever it got to if exceeded |
| LLM metrics | TTFT, tokens/sec (TPS) |
| Conversation test exclusion | Model excluded entirely if it timed out or was already marked too slow in the single-shot test |
| Embedding models | `nomic-embed-text`, `mxbai-embed-large` (via Ollama) |
| Embedding corpus | `sample_document.txt` chunked into ~150-word paragraph-sized pieces (~290 chunks), embedded in one call |
| Embedding warmup runs | `--warmup` (default: 2, discarded) |
| Embedding measured runs | `--runs`, averaged (default: 3, range: 1–10) |
| Image models | SD1.5 (20 steps), SDXL (20 steps), SD3.5 Large (28 steps), Flux.1-dev (20 steps), Flux.2-dev (28 steps) |
| Image resolutions | 1024×1024, 1536×1536 (SD1.5: 512×512, 768×768) |
| Image seed | 42 (fixed) |
| Image metrics | Seconds per image, per model, per resolution |
| Image measured runs | `--runs` per resolution, averaged (default: 3, range: 1–10) |

---

[← Dashboard](dashboard.md) · [Back to README](../README.md) · [Project Structure →](project-structure.md)
