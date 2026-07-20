[← Back to README](../README.md)

# How It Works

**Contents**
- [Execution order](#execution-order)
- [Code organization](#code-organization)
- [Parameters](#parameters)

## Execution order

LLM benchmarks run first (small → medium → large), followed by the accuracy-style tests (MCQ, then math, then code — same models, same running engine server), then embeddings. Before starting ComfyUI for image tests, the engine is stopped entirely (not just unloaded) to free up its memory for image generation, since image tests always run last. See [Engines](engines.md) for `--engine <name>|all` — everything on this page describes one engine's pass; `all` just repeats it once per registered engine (only llama.cpp is registered today, so `all` is currently a no-op). On a multi-engine `all` run, image generation still runs only once (on the first pass) rather than once per engine, since it doesn't depend on `--engine` — a separate ComfyUI call, not an LLM inference call.

Each LLM model follows this pattern:

```
warmup (--warmup runs, default 2)
  → measure single-shot   (--runs runs at 512 / 2K / 8K / 32K / 64K, default 3)
  → measure conversation  (a single full conversation, --runs is ignored here,
                            sampled at 0 / 2K / 4K / 8K / 16K / 32K / 48K / 64K /
                            80K / 96K up to the model's real context ceiling)
  → unload → confirm gone
```

The single-shot test builds an independent padded prompt for every run. The conversation test is different: it's a single conversation (this test is expensive enough — many turns growing all the way to the sampling ceiling — that it always runs once, ignoring `--runs`), started from a blank slate and grown toward a 96K sampling ceiling. The model is still given the full 128K context window (or its real maximum, whichever is lower — read live from the downloaded GGUF's own metadata, not hardcoded), so there's always headroom left between the top checkpoint and the actual `num_ctx` limit. Growth happens in small steps (capped and scaled to the size of the gap being crossed) rather than one big jump per checkpoint, so the turn that actually crosses each threshold lands close to it instead of overshooting by a large margin.

If the run exceeds the 300-second timeout, it stops wherever it got to — whatever checkpoints it already reached are kept. Only one model is ever in memory at a time: after each model completes both tests, the engine unloads it and confirms it's gone before the next one loads — a process stop for `LlamaCppEngine` (see [Engines](engines.md)).

A model is excluded from the conversation test *entirely* if it timed out or was already marked too slow in the single-shot test. Within the conversation test itself, if the decode speed at any history depth drops below the slow-model cutoff, it exits early. See [LLM workload](workloads.md#llm) for the full skip logic.

The MCQ and math accuracy tests each follow a simpler pattern per model: warmup, then one chat call per question in the bank (temperature 0, so a single pass is representative), scored right/wrong against the dataset's known answer, then unload. Both reuse the same crash-cache machinery as the other tests, applied per question instead of per measured run — but the per-question timeout is `--acc-timeout` (default 60s), not `--timeout`, and a timeout only affects that one question: it's scored wrong (using whatever partial text streamed before the cutoff, run through a loop-detection heuristic) and the bank continues to the next question. See [Accuracy workload](workloads.md#accuracy) for the full timeout/loop-detection behavior. The code accuracy test follows the same per-model pattern, but scoring each reply is more involved: the generated function is run against that problem's visible and hidden test cases in an isolated Python subprocess (with its own wall-clock timeout, separate from the engine call timeout), and the problem only counts as correct if every test case passes.

**The engine** (llama-server) is started if not already running. If the benchmark started it, it is shut down at exit; if it was already running, it is left running.

**ComfyUI** is started just before the image tests and shut down immediately after. A signal handler and `finally` block ensure clean shutdown on Ctrl-C or crash.

## Code organization

The benchmark implementation lives in `scripts/`, split by responsibility:

| Module | Responsibility |
|---|---|
| `scripts/benchmark.py` | CLI argument parsing and orchestration (`main()`) — calls each test class in order and writes results |
| `scripts/config.py` | Shared constants: URLs, paths (`SCRIPT_DIR`, `RESULTS_DIR`, `COMFYUI_DIR`), timeouts, run counts |
| `scripts/shared.py` | Cross-cutting helpers: logging, ComfyUI server lifecycle/HTTP client, machine profiling, engine-agnostic run/crash-cache orchestration |
| `scripts/engines/base.py` | `InferenceEngine` interface — start/stop, model lifecycle, generate/chat/embed, see [Engines](engines.md) |
| `scripts/engines/llamacpp.py` | `LlamaCppEngine` — the low-level HTTP/process client for llama-server |
| `scripts/llm_prefill_benchmark.py` | The single-shot LLM test |
| `scripts/llm_conversation_benchmark.py` | The multi-turn conversation test |
| `scripts/embedding_benchmark.py` | The embeddings test |
| `scripts/image_benchmark.py` | The image generation test (ComfyUI workflow builders + submission) |
| `scripts/mcq_benchmark.py` | The MCQ accuracy test |
| `scripts/math_benchmark.py` | The math accuracy test |
| `scripts/code_benchmark.py` | The code accuracy test |
| `scripts/models.py` | Single source of truth for every model definition (tags, checkpoints, tiers, sizes) |
| `scripts/setup_check.py` | Hardware detection, model picker, and unattended install — called by `setup.sh`/`setup.bat` |

Values that CLI flags can override at runtime (`RUN_TIMEOUT` via `--timeout`, `ACC_TIMEOUT` via `--acc-timeout`, `N_RUNS` via `--runs`) are read via `config.RUN_TIMEOUT`/`config.ACC_TIMEOUT`/`config.N_RUNS` (a dotted attribute lookup) everywhere, rather than imported by name — a plain `from config import RUN_TIMEOUT` would bind a stale copy at import time and silently ignore the CLI override.

## Parameters

| Parameter | Value |
|---|---|
| LLM single-shot context lengths | 512, 2K, 8K, 32K, 64K — capped per model at its real context ceiling |
| LLM conversation checkpoints | 0, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 80K, 96K — capped per model at its real context ceiling (model is still given the full 128K context window, or its real max if lower) |
| LLM test modes | Single-shot (cold prefill), Conversation (a single full conversation, `--runs` ignored) |
| LLM warmup runs | `--warmup` (default: 2, discarded) |
| LLM measured runs | `--runs` — repeated context lengths for single-shot (default: 3, range: 1–10); ignored by the conversation test, which always runs once |
| Run timeout | `--timeout` per run (default: 300s) — applies to warmup and to `llm`/`conv`/`emb`/`img`; that run stops wherever it got to if exceeded |
| Accuracy question timeout | `--acc-timeout` per question (default: 60s), for `mcq`/`math`/`code` only — that question is scored wrong and the bank continues, rather than stopping the model's run |
| LLM metrics | TTFT, tokens/sec (TPS) |
| Conversation test exclusion | Model excluded entirely if it timed out or was already marked too slow in the single-shot test |
| Embedding models | `nomic-embed-text`, `mxbai-embed-large` |
| Embedding corpus | `sample_document.txt` chunked into ~150-word paragraph-sized pieces (~290 chunks), embedded in one call |
| Embedding warmup runs | `--warmup` (default: 2, discarded) |
| Embedding measured runs | `--runs`, averaged (default: 3, range: 1–10) |
| Image models | SD1.5 (20 steps), SDXL (20 steps), SD3.5 Large (28 steps), Flux.1-dev (20 steps), Flux.2-dev (28 steps) |
| Image resolutions | 1024×1024, 1536×1536 (SD1.5: 512×512, 768×768) |
| Image seed | 42 (fixed) |
| Image metrics | Seconds per image, per model, per resolution |
| Image measured runs | `--runs` per resolution, averaged (default: 3, range: 1–10) |
| MCQ question bank | `scripts/data/mcq_questions.json` — 150 questions across 8 categories (science, history, geography, logic, literature, arithmetic, commonsense, language), with A–D answer positions balanced |
| MCQ warmup runs | `--warmup` (default: 2, discarded) |
| MCQ measured runs | Always 1 pass through the full question bank — `--runs` is ignored (temperature 0, so repeats wouldn't change the answers) |
| MCQ metrics | Overall accuracy (%), plus accuracy (%) per category; `timed_out_count`/`timed_out_ids` and `likely_loop_count`/`likely_loop_ids` when any question hit `--acc-timeout` |
| Math question bank | `scripts/data/math_questions.json` — 150 numeric-answer problems across 30 categories, including calculus, combinatorics, linear algebra, number theory, probability, and statistics |
| Math warmup runs | `--warmup` (default: 2, discarded) |
| Math measured runs | Always 1 pass through the full question bank — `--runs` is ignored (temperature 0, so repeats wouldn't change the answers) |
| Math metrics | Overall accuracy (%), plus accuracy (%) per category; `timed_out_count`/`timed_out_ids` and `likely_loop_count`/`likely_loop_ids` when any question hit `--acc-timeout` |
| Code question bank | `scripts/data/code_problems.json` — 60 problems across 13 categories, including dynamic programming, graph, interval, divide-and-conquer, and advanced stateful structures |
| Code warmup runs | `--warmup` (default: 2, discarded) |
| Code measured runs | Always 1 pass through the full question bank — `--runs` is ignored (temperature 0, so repeats wouldn't change the answers) |
| Code metrics | Overall accuracy (%), plus accuracy (%) per category — a problem counts as correct only if every one of its visible and hidden test cases passes; `timed_out_count`/`timed_out_ids` and `likely_loop_count`/`likely_loop_ids` when any question hit `--acc-timeout` |

---

[← Dashboard](dashboard.md) · [Back to README](../README.md) · [Engines →](engines.md)
