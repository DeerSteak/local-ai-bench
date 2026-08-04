[← Back to README](../README.md)

# How It Works

**Contents**
- [Execution order](#execution-order)
- [Code organization](#code-organization)
- [Configuration sources](#configuration-sources)

## Execution order

With no wrapper arguments, `run_bench.sh`/`run_bench.bat` uses `benchmark_launcher.py` to select the Tk GUI on a usable local desktop or the terminal frontend over SSH/headless sessions. Both launch `benchmark.py` as a child process with explicit CLI flags — see [Launch modes](cli-reference.md#launch-modes). The frontends never call benchmark orchestration directly; benchmark arguments bypass them and forward straight to `benchmark.py`, which is always non-interactive.

Before profiling hardware or creating an output path, the CLI expands test groups, applies `--maxtier`, resolves the LLM/embedding/image selectors, and performs a validation-only pre-pass for every selected engine. Local inventory is read only when custom LLM matching or concurrency scoping needs it; no model is loaded and no inference server is started. An explicit selector that leaves one of its selected workload families empty aborts the invocation at this point. With `--engine all`, every engine must validate before any runs; the resolved scopes are cached so hardware profiling and the filename timestamp still happen once and all per-engine files share the same stem.

`--list-models` uses the same read-only inventory helpers and exits before profiling. It reports catalog LLMs, embeddings, custom LLM folders, and catalog image checkpoints from Local AI Bench's managed `models/comfyui/` directory; the resolved `--comfyui` path selects the program used to run those models, not their storage location.

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

The selected keys are resolved through the fixed `STAGE_ORDER` registry in `orchestration.py`; a registered or selected key missing from that order is an error rather than a silently skipped stage. Each `StageDefinition` names its result section, selected-model count, runner, and only the preparation or cleanup hooks it actually needs. Native llama.cpp stages explicitly stop the HTTP engine, images restore normal engine mode before stopping it for ComfyUI, and ordinary stages have no special hooks.

The main JSON is owned by `ResultStore`, which atomically checkpoints when a run or stage changes state and after every model reaches a terminal outcome. Ctrl+C first records `interrupted`, then checkpoints again after workload `finally` callbacks flush their latest in-memory model data; the interrupted run is never relabeled complete. Unhandled stage-hook failures use the documented reasons `stage_preparation_failed`, `stage_execution_failed`, and `stage_cleanup_failed`; a cleanup failure secondary to a preparation or execution failure is retained in that stage's `cleanup_failure` metadata without replacing the primary reason. Final lifecycle teardown runs before terminal exception persistence, so a failed checkpoint cannot prevent server cleanup.

Stages currently end as `complete`, `failed`, or `interrupted`. The schema reserves `partial` for a future intentionally handled early-stop outcome; usable measurements retained before an unhandled failure remain visible through the failed stage's section and coverage counts rather than relabeling that stage partial.

The single-shot test builds an independent padded prompt for every measured call. Conversation instead grows one chat from a blank slate and samples it once at each eligible checkpoint, growing toward 128K and sampling through 96K, capped by the GGUF's real context ceiling. Growth uses larger steps while far from a checkpoint and finer steps within 8K, stopping at 99.5% of the target to avoid expensive tiny turns.

When single-shot and conversation are selected together, conversation excludes models with no usable single-shot result, a repeatable runner crash, a timeout at the first 512-token checkpoint, or a slow marker there. A deeper single-shot timeout alone does not exclude it. Conversation also stops after recording any sampled checkpoint below the slow-TPS cutoff. `--force-all` bypasses these speed gates, not actual failures. See [LLM workload](workloads.md#llm).

If `llamabench` is selected, the active inference engine's server is stopped entirely first (not just unloaded) to free memory. One prefill and one decode subprocess run the full matrix and all repetitions with the model retained; completed JSONL cases are checkpointed as they stream, so a timeout preserves every earlier case without reloading the model for every matrix cell. `llamabenchconc` likewise retains and checkpoints each JSONL row as it streams. One model's failure moves on rather than stopping the run — see [Workloads](workloads.md#llama-bench).

Each accuracy test warms a model, makes one deterministic first pass per question, scores it, and unloads the model. A literal length stop uses the remaining 40% of `--acc-token-budget` for one concise final-answer request; only that replacement is graded. Both passes share one `--acc-timeout` deadline. Token exhaustion, timeout, and periodic loop detection preserve and score the graded pass's partial output, record separate diagnostics, and continue the bank. MCQ and reasoning use confidence-ordered choice parsing so explicit final answers and later self-corrections take precedence over incidental reasoning text; reasoning deliberately disables MCQ's last-resort unstructured-letter fallback. Math accepts only completed scalar conclusions or same-clause results stated after `=`, while a leading numeric line must be corroborated by the response's final number or final completed equality result. Code answers run visible and hidden cases in one isolated Python subprocess, streaming per-test diagnostics so completed results survive a later timeout. Tool answers use `chat_tools` and require either exactly one matching call or a correct decline; question metadata can opt free-text arguments into limited normalization while identifiers remain exact.

Before images, the active inference engine is stopped entirely to free memory. ComfyUI is started only if it is not already reachable; processes managed by the benchmark are shut down after images or on exit, while a pre-existing external ComfyUI process is left running. Its loaded models and queue are still cleared during cleanup.

## Code organization

The implementation has four layers:

| Layer | Responsibility |
|---|---|
| Entry points | `benchmark_frontend.py` builds a public CLI command; `benchmark.py` parses and validates it, resolves engines/models, and creates each run specification |
| Orchestration | `run_plan.py` owns the immutable serializable execution identity; `orchestration.py` owns fixed stage order/execution, local paths, and lifecycle policy; `result_store.py` owns schema mutation, legal state transitions, and durable checkpoints |
| Workloads | One module per workload or closely related workload family; each receives an engine and returns its section of the results schema |
| Engine adapters | `engines/base.py` defines the interface and `engines/llamacpp.py` owns llama-server process/HTTP details; the two native llama.cpp benchmark modules intentionally bypass this interface |
| Shared definitions | `config.py`, `models.py`, `model_inventory.py`, `hardware.py`, and `shared.py` own defaults, catalog data, discovery, fit estimates, logging, retries, statistics, and ComfyUI lifecycle |

See [Project Structure](project-structure.md#scripts-in-detail) for the complete module-by-module map and [Engines](engines.md) for the adapter contract.

Values that CLI flags can override at runtime (`RUN_TIMEOUT`, `ACC_TIMEOUT`, `ACC_TOKEN_BUDGET`, and `N_RUNS`) are read through dotted `config.*` lookups everywhere, rather than imported by name, so CLI assignments remain visible after import.

The frontend uses `Shared.plain_output`, native `cls` clearing on Windows, and ANSI clearing elsewhere, keeping selection prompts compact and untimestamped. It preserves the welcome banner through the initial single-engine test menu and the final model choices through confirmation, while clearing between screens and before subsequent redraws. The test menu keeps its number/range and group-shortcut legend on screen during every redraw; restored menus say which local state file supplied their selections and how to reset it. Benchmark execution output goes through `Shared.output` and the existing severity helpers, which prefix each independently emitted status or progress message with local `[HH:MM:SS]` time. This display layer does not touch result JSON, captured model responses, answer sidecars, caches, or generated artifacts.

## Configuration sources

The detailed workload shapes, checkpoints, model lists, and metrics live in [Workloads](workloads.md). Public flags and their effective defaults live in the [CLI Reference](cli-reference.md). Keeping those facts in one place prevents this architecture guide from becoming a second, stale parameter reference.

Runtime defaults are defined in `scripts/config.py`; model metadata is defined in `scripts/models.py`; the conversation checkpoint plan is defined by `LLMConversationBenchmark.CONV_CHECKPOINTS`; and dashboard context ordering is defined by `CTX_ORDER` in `dashboard/src/constants.js`. CLI-overridable values are read through `config.*` at use sites so an argument applied after import is still honored.

Before creating `RunContext`, the CLI canonicalizes the selected engine, tests, stage order, safe model identities, and effective measurement settings into an immutable `RunPlan`. Its SHA-256 `plan_id` is deterministic for equivalent plans and changes when a measurement-affecting input changes. Schema-3 results embed that plan and identity while retaining the existing manifest fields for compatibility. Output and ComfyUI paths live in a separate `RunPaths` object and cannot enter the shareable plan; unknown configuration or model-identity fields are rejected rather than risking a secret or user path in exported metadata.

---

[← Dashboard](dashboard.md) · [Back to README](../README.md) · [Engines →](engines.md)
