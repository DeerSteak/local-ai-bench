[← Back to README](../README.md)

# How It Works

**Contents**
- [Execution order](#execution-order)
- [Code organization](#code-organization)
- [Parameters](#parameters)

## Execution order

With no wrapper arguments, `run_bench.sh`/`run_bench.bat` starts `scripts/benchmark_frontend.py`. The frontend performs read-only inventory discovery using the same setup-managed `config.COMFYUI_DIR`, restores applicable entries from the gitignored `.benchmark_frontend_state.json` (falling back to current defaults when it is absent, invalid, or stale), presents the tests, collects one engine/test/model selection, and launches `benchmark.py` as a child process with explicit `--engine`, `--comfyui`, `--tests`, and applicable model-selector arguments. A confirmed selection is saved atomically before launch; cancellation saves nothing, and persistence failure never blocks execution. The frontend never calls benchmark orchestration directly. Any wrapper argument bypasses the frontend and is forwarded to `benchmark.py`, which remains completely non-interactive; direct CLI users can still override `--comfyui`.

Before profiling hardware or creating an output path, the CLI expands test groups, applies `--maxtier`, resolves the LLM/embedding/image selectors, and performs a validation-only pre-pass for every selected engine. Local inventory is read only when custom LLM matching or concurrency scoping needs it; no model is loaded and no inference server is started. An explicit selector that leaves one of its selected workload families empty aborts the invocation at this point. With `--engine all`, every engine must validate before any runs; the resolved scopes are cached so hardware profiling and the filename timestamp still happen once and all per-engine files share the same stem.

`--list-models` uses the same read-only inventory helpers and exits before profiling. It reports catalog LLMs, embeddings, custom LLM folders, and catalog image checkpoints from the effective `--comfyui` directory.

Selected tests run in a fixed stage order, independent of the order passed to `--tests`:

```
single-shot LLM (all selected models, xsmall → large)
  → conversation LLM (all eligible selected models)
  → accuracy (MCQ → math → reasoning → code → tool)
  → embeddings
  → concurrency (tool → chat)
  → images
```

See [Engines](engines.md) for `--engine <name>|all`. Each engine gets one pass through the selected engine-backed stages. Images run only on the first pass because they use ComfyUI rather than the selected inference engine.

Within each stage, only one model is loaded at a time. `LlamaCppEngine` runs a model-specific llama-server process and restarts it whenever the requested model, context allocation, GPU mode, or concurrency shape changes. Each workload unloads or stops that model before advancing.

The single-shot test builds an independent padded prompt for every measured call. Conversation instead grows one chat from a blank slate and samples it once at each eligible checkpoint. Medium/large catalog models and custom models can grow toward 128K and sample through 96K; xsmall/small catalog models use the shorter 64K plan and sample through 48K. Both are further capped by the GGUF's real context ceiling. Growth uses larger steps while far from a checkpoint and finer steps within 8K, stopping at 99.5% of the target to avoid expensive tiny turns.

When single-shot and conversation are selected together, conversation excludes models with no usable single-shot result, a repeatable runner crash, a timeout at the first 512-token checkpoint, or a slow marker there. A deeper single-shot timeout alone does not exclude it. Conversation also stops after recording any sampled checkpoint below the slow-TPS cutoff. `--force-all` bypasses these speed gates, not actual failures. See [LLM workload](workloads.md#llm).

Each accuracy test warms a model, makes one deterministic call per question, scores it, and unloads the model. A question that reaches `--acc-timeout` keeps and normally scores its partial response, records the timeout, and continues. Periodic loop detection can stop a likely loop before that timeout; it is a separate diagnostic. MCQ and reasoning use confidence-ordered choice parsing so explicit final answers and later self-corrections take precedence over incidental reasoning text; reasoning deliberately disables MCQ's last-resort unstructured-letter fallback. Math accepts only completed scalar conclusions or same-clause results stated after `=`, while a leading numeric line must be corroborated by the response's final number or final completed equality result. Code answers run visible and hidden cases in one isolated Python subprocess, streaming per-test diagnostics so completed results survive a later timeout. Tool answers use `chat_tools` and require either exactly one matching call or a correct decline; question metadata can opt free-text arguments into limited normalization while identifiers remain exact.

Before images, the active inference engine is stopped entirely to free memory. ComfyUI is started only if it is not already reachable; processes managed by the benchmark are shut down after images or on exit, while a pre-existing external ComfyUI process is left running. Its loaded models and queue are still cleared during cleanup.

## Code organization

The benchmark implementation lives in `scripts/`, split by responsibility:

| Module | Responsibility |
|---|---|
| `scripts/benchmark.py` | CLI argument parsing and orchestration (`main()`) — calls each test class in order and writes results |
| `scripts/benchmark_frontend.py` | Interactive installed-model/test picker that constructs and launches the public benchmark CLI |
| `scripts/config.py` | Shared constants: URLs, paths (`SCRIPT_DIR`, `RESULTS_DIR`, `COMFYUI_DIR`), timeouts, run counts |
| `scripts/model_inventory.py` | Read-only catalog/custom/embedding/image inventory classification shared by CLI listing and launch preflight |
| `scripts/shared.py` | Cross-cutting helpers: logging, ComfyUI server lifecycle/HTTP client, machine profiling, engine-agnostic run/crash-cache orchestration |
| `scripts/hardware.py` | GPU/system-memory detection and model-fit classification shared by setup and concurrency snapshots |
| `scripts/engines/base.py` | `InferenceEngine` interface — start/stop, model lifecycle, generate/chat/embed, see [Engines](engines.md) |
| `scripts/engines/llamacpp.py` | `LlamaCppEngine` — the low-level HTTP/process client for llama-server |
| `scripts/llm_prefill_benchmark.py` | The single-shot LLM test |
| `scripts/llm_conversation_benchmark.py` | The multi-turn conversation test |
| `scripts/embedding_benchmark.py` | The embeddings test |
| `scripts/image_benchmark.py` | The image generation test (ComfyUI workflow builders + submission) |
| `scripts/concurrency_benchmark.py` | The tool-style and chat-server concurrency sweeps |
| `scripts/mcq_benchmark.py` | The MCQ accuracy test |
| `scripts/math_benchmark.py` | The math accuracy test |
| `scripts/reasoning_benchmark.py` | The knowledge-light reasoning accuracy test and validated bank loader |
| `scripts/code_benchmark.py` | The code accuracy test |
| `scripts/tool_benchmark.py` | The tool-calling accuracy test |
| `scripts/models.py` | Single source of truth for every model definition (tags, checkpoints, tiers, sizes) |
| `scripts/setup_check.py` | Hardware detection, model picker, and unattended install — called by `setup.sh`/`setup.bat` |

Values that CLI flags can override at runtime (`RUN_TIMEOUT` via `--timeout`, `ACC_TIMEOUT` via `--acc-timeout`, `N_RUNS` via `--runs`) are read via `config.RUN_TIMEOUT`/`config.ACC_TIMEOUT`/`config.N_RUNS` (a dotted attribute lookup) everywhere, rather than imported by name — a plain `from config import RUN_TIMEOUT` would bind a stale copy at import time and silently ignore the CLI override.

The frontend uses `Shared.plain_output`, native `cls` clearing on Windows, and ANSI clearing elsewhere, keeping selection prompts compact and untimestamped. It preserves the welcome banner through the initial single-engine test menu and the final model choices through confirmation, while clearing between screens and before subsequent redraws. Restored menus say which local state file supplied their selections and how to reset it. Benchmark execution output goes through `Shared.output` and the existing severity helpers, which prefix each independently emitted status or progress message with local `[HH:MM:SS]` time. This display layer does not touch result JSON, captured model responses, answer sidecars, caches, or generated artifacts.

## Parameters

| Parameter | Value |
|---|---|
| LLM single-shot context lengths | 512, 2K, 8K, 32K, 64K — capped per model at its real context ceiling |
| LLM conversation checkpoints | 0, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 80K, 96K — medium/large/custom sample through 96K with up to a 128K growth plan; xsmall/small stop at 48K with a 64K growth plan; all are capped by the GGUF's real context ceiling |
| LLM test modes | Single-shot (cold prefill), Conversation (a single full conversation, `--runs` ignored) |
| LLM warmup runs | `--warmup` at each context-specific server configuration (default: 2, discarded) |
| LLM measured runs | `--runs` — repeated context lengths for single-shot (default: 3, range: 1–10); ignored by the conversation test, which always runs once |
| Run timeout | `--timeout` is a total model-load-plus-generation/chat deadline and also bounds engine warmup (default: 300s); image generation uses 2× this value. Embeddings retain a fixed 120s request timeout |
| Accuracy question timeout | `--acc-timeout` per question (default: 60s), for `mcq`/`math`/`reasoning`/`code`/`tool`; partial output is scored normally, the timeout is recorded, and the bank continues |
| LLM metrics | TTFT, tokens/sec (TPS) |
| Conversation pre-flight | When single-shot also ran: excludes no/failed data, repeatable crashes, first-checkpoint timeouts, and first-checkpoint slow markers; deeper timeouts alone do not exclude conversation |
| Embedding models | `nomic-embed-text`, `mxbai-embed-large` |
| Embedding corpus | `sample_document.txt` chunked into ~150-word paragraph-sized pieces (~290 chunks), embedded in one call |
| Embedding warmup runs | `--warmup` (default: 2, discarded) |
| Embedding measured runs | `--runs`, averaged (default: 3, range: 1–10) |
| Image models | SD1.5 (20 steps), SDXL (20 steps), SD3.5 Large (28 steps), Flux.1-dev (20 steps), Flux.2-dev (28 steps) |
| Image resolutions | 1024×1024, 1536×1536 (SD1.5: 512×512, 768×768) |
| Image seed | Warmup 41; measured runs 42, 43, ... so ComfyUI cannot reuse a cached graph |
| Image warmup runs | Always 1 at the model's first resolution; `--warmup` is ignored |
| Image metrics | Seconds per image, per model, per resolution |
| Image measured runs | `--runs` per resolution, averaged (default: 3, range: 1–10) |
| MCQ question bank | `scripts/data/mcq_questions.json` — 150 questions across 8 categories (science, history, geography, logic, literature, arithmetic, commonsense, language), with A–D answer positions balanced |
| MCQ warmup runs | `--warmup` (default: 2, discarded) |
| MCQ measured runs | Always 1 pass through the full question bank — `--runs` is ignored (temperature 0, so repeats wouldn't change the answers) |
| MCQ metrics | Overall and per-category accuracy; timeout and likely-loop counts/IDs when those separate diagnostics occur |
| Math question bank | `scripts/data/math_questions.json` — 150 numeric-answer problems across 30 categories, including calculus, combinatorics, linear algebra, number theory, probability, and statistics |
| Math warmup runs | `--warmup` (default: 2, discarded) |
| Math measured runs | Always 1 pass through the full question bank — `--runs` is ignored (temperature 0, so repeats wouldn't change the answers) |
| Math metrics | Overall and per-category accuracy; timeout and likely-loop counts/IDs when those separate diagnostics occur |
| Reasoning question bank | `scripts/data/reasoning_questions.json` — 60 original A–D questions across 10 knowledge-light categories, with a 20-question `very_hard` tail |
| Reasoning warmup/measured runs | `--warmup` discarded warmups, then 1 pass through the bank; `--runs` is ignored |
| Reasoning metrics | Overall, per-category, and per-difficulty accuracy; timeout and likely-loop counts/IDs when those separate diagnostics occur |
| Code question bank | `scripts/data/code_problems.json` — 60 problems across 13 categories, including dynamic programming, graph, interval, divide-and-conquer, and advanced stateful structures |
| Code warmup runs | `--warmup` (default: 2, discarded) |
| Code measured runs | Always 1 pass through the full question bank — `--runs` is ignored (temperature 0, so repeats wouldn't change the answers) |
| Code metrics | Overall and per-category accuracy — all visible and hidden cases must pass; timeout and likely-loop counts/IDs when those separate diagnostics occur |
| Tool question bank | `scripts/data/tool_questions.json` — 100 tool-calling questions across 20 categories, from basic selection/extraction through close distractors, conversions, structured arguments, strict optional omission, adversarial content, corrections/negations, and nuanced decline cases |
| Tool warmup/measured runs | `--warmup` discarded warmups, then 1 pass through the bank; `--runs` is ignored |
| Tool metrics | Overall and per-category accuracy — exactly one matching call or a correct decline; timeout and likely-loop counts/IDs when those separate diagnostics occur |
| Tool-style concurrency | 1, 2, 4, 6, 8, 12, 16 requests; 4,096-token prompts; no slow-TPS soft exit |
| Chat concurrency | 1, 2, 4, 8, 16, 24, 32 requests; 16,384-token prompts; soft exit after a measured level ≥8 falls below 15 tok/s unless `--force-all` |
| Concurrency measurement | `--warmup` discarded batches then one measured batch per level; up to 512 output tokens per request; `--runs` is ignored |

---

[← Dashboard](dashboard.md) · [Back to README](../README.md) · [Engines →](engines.md)
