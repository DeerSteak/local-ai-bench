[← Back to README](../README.md)

# How It Works

**Contents**
- [Execution order](#execution-order)
- [Code organization](#code-organization)
- [Configuration sources](#configuration-sources)

## Execution order

With no wrapper arguments, `run_bench.sh`/`run_bench.bat` opens the interactive launcher (`scripts/benchmark_frontend.py`), which launches `benchmark.py` as a child process with explicit CLI flags — see [Launch modes](cli-reference.md#launch-modes) for the launcher's own behavior. The frontend never calls benchmark orchestration directly; any wrapper argument bypasses it and forwards straight to `benchmark.py`, which is always non-interactive.

Before profiling hardware or creating an output path, the CLI expands test groups, applies `--maxtier`, resolves the LLM/embedding/image selectors, and performs a validation-only pre-pass for every selected engine. Local inventory is read only when custom LLM matching or concurrency scoping needs it; no model is loaded and no inference server is started. An explicit selector that leaves one of its selected workload families empty aborts the invocation at this point. With `--engine all`, every engine must validate before any runs; the resolved scopes are cached so hardware profiling and the filename timestamp still happen once and all per-engine files share the same stem.

`--list-models` uses the same read-only inventory helpers and exits before profiling. It reports catalog LLMs, embeddings, custom LLM folders, and catalog image checkpoints from the effective `--comfyui` directory.

Selected tests run in a fixed stage order, independent of the order passed to `--tests`:

```
single-shot LLM (all selected models, xsmall → large)
  → conversation LLM (all eligible selected models)
  → llama-bench (opt-in)
  → llama-bench concurrency (opt-in)
  → embeddings
  → accuracy (MCQ → math → reasoning → code → tool)
  → concurrency (tool → chat)
  → images
```

See [Engines](engines.md) for `--engine <name>|all`. Each engine gets one pass through the selected engine-backed stages. Images run only on the first pass because they use ComfyUI rather than the selected inference engine.

Within each stage, only one model is loaded at a time. `LlamaCppEngine` runs a model-specific llama-server process and restarts it whenever the requested model, context allocation, GPU mode, or concurrency shape changes. Each workload unloads or stops that model before advancing.

The single-shot test builds an independent padded prompt for every measured call. Conversation instead grows one chat from a blank slate and samples it once at each eligible checkpoint, growing toward 128K and sampling through 96K, capped by the GGUF's real context ceiling. Growth uses larger steps while far from a checkpoint and finer steps within 8K, stopping at 99.5% of the target to avoid expensive tiny turns.

When single-shot and conversation are selected together, conversation excludes models with no usable single-shot result, a repeatable runner crash, a timeout at the first 512-token checkpoint, or a slow marker there. A deeper single-shot timeout alone does not exclude it. Conversation also stops after recording any sampled checkpoint below the slow-TPS cutoff. `--force-all` bypasses these speed gates, not actual failures. See [LLM workload](workloads.md#llm).

If `llamabench` is selected, the active inference engine's server is stopped entirely first (not just unloaded) to free memory for `llama-bench`'s own subprocess, which it drives directly rather than through `LlamaCppEngine`'s HTTP interface — see [Workloads](workloads.md#llama-bench). One model's failure (missing binary, timeout, crash) records that model's error and moves on rather than stopping the run. `llamabenchconc` runs immediately afterward on the same terms, driving `llama-batched-bench` instead — see [Workloads](workloads.md#llama-bench-concurrency).

Each accuracy test warms a model, makes one deterministic first pass per question, scores it, and unloads the model. A literal length stop uses the remaining 40% of `--acc-token-budget` for one concise final-answer request; only that replacement is graded. Both passes share one `--acc-timeout` deadline. Token exhaustion, timeout, and periodic loop detection preserve and score the graded pass's partial output, record separate diagnostics, and continue the bank. MCQ and reasoning use confidence-ordered choice parsing so explicit final answers and later self-corrections take precedence over incidental reasoning text; reasoning deliberately disables MCQ's last-resort unstructured-letter fallback. Math accepts only completed scalar conclusions or same-clause results stated after `=`, while a leading numeric line must be corroborated by the response's final number or final completed equality result. Code answers run visible and hidden cases in one isolated Python subprocess, streaming per-test diagnostics so completed results survive a later timeout. Tool answers use `chat_tools` and require either exactly one matching call or a correct decline; question metadata can opt free-text arguments into limited normalization while identifiers remain exact.

Before images, the active inference engine is stopped entirely to free memory. ComfyUI is started only if it is not already reachable; processes managed by the benchmark are shut down after images or on exit, while a pre-existing external ComfyUI process is left running. Its loaded models and queue are still cleared during cleanup.

## Code organization

The implementation has four layers:

| Layer | Responsibility |
|---|---|
| Entry points | `benchmark_frontend.py` builds a public CLI command; `benchmark.py` validates it, selects engines/models, orders stages, checkpoints results, and handles cleanup |
| Workloads | One module per workload or closely related workload family; each receives an engine and returns its section of the results schema |
| Engine adapters | `engines/base.py` defines the interface and `engines/llamacpp.py` owns llama-server process/HTTP details; the two native llama.cpp benchmark modules intentionally bypass this interface |
| Shared definitions | `config.py`, `models.py`, `model_inventory.py`, `hardware.py`, and `shared.py` own defaults, catalog data, discovery, fit estimates, logging, retries, statistics, and ComfyUI lifecycle |

See [Project Structure](project-structure.md#scripts-in-detail) for the complete module-by-module map and [Engines](engines.md) for the adapter contract.

Values that CLI flags can override at runtime (`RUN_TIMEOUT`, `ACC_TIMEOUT`, `ACC_TOKEN_BUDGET`, and `N_RUNS`) are read through dotted `config.*` lookups everywhere, rather than imported by name, so CLI assignments remain visible after import.

The frontend uses `Shared.plain_output`, native `cls` clearing on Windows, and ANSI clearing elsewhere, keeping selection prompts compact and untimestamped. It preserves the welcome banner through the initial single-engine test menu and the final model choices through confirmation, while clearing between screens and before subsequent redraws. Restored menus say which local state file supplied their selections and how to reset it. Benchmark execution output goes through `Shared.output` and the existing severity helpers, which prefix each independently emitted status or progress message with local `[HH:MM:SS]` time. This display layer does not touch result JSON, captured model responses, answer sidecars, caches, or generated artifacts.

## Configuration sources

The detailed workload shapes, checkpoints, model lists, and metrics live in [Workloads](workloads.md). Public flags and their effective defaults live in the [CLI Reference](cli-reference.md). Keeping those facts in one place prevents this architecture guide from becoming a second, stale parameter reference.

Runtime defaults are defined in `scripts/config.py`; model metadata is defined in `scripts/models.py`; the conversation checkpoint plan is defined by `LLMConversationBenchmark.CONV_CHECKPOINTS`; and dashboard context ordering is defined by `CTX_ORDER` in `dashboard/src/constants.js`. CLI-overridable values are read through `config.*` at use sites so an argument applied after import is still honored.

---

[← Dashboard](dashboard.md) · [Back to README](../README.md) · [Engines →](engines.md)
