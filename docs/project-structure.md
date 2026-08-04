[← Back to README](../README.md)

# Project Structure

**Contents**
- [`scripts/` in detail](#scripts-in-detail)
- [`results/` in detail](#results-in-detail)
  - [Main results JSON](#main-results-json)

| File / Folder | Purpose |
|---|---|
| `setup.sh` | One-shot setup for macOS and Linux |
| `setup.bat` | One-shot setup for Windows |
| `run_bench.sh` | Activates the venv; opens the interactive frontend with no arguments or forwards arguments directly to `scripts/benchmark.py` on Linux / macOS |
| `run_bench.bat` | Windows equivalent of `run_bench.sh` |
| `launch_dashboard.sh` | Builds and serves the dashboard on Linux / macOS, opens browser automatically |
| `launch_dashboard.bat` | Builds and serves the dashboard on Windows, opens browser automatically |
| `tests.sh` | Activates the venv and runs unit/integration tests on Linux / macOS — see [Testing](testing.md) |
| `tests.bat` | Activates the venv and runs unit/integration tests on Windows — see [Testing](testing.md) |
| `scripts/` | Benchmark implementation — see [How It Works](how-it-works.md#code-organization) for what each module does |
| `results/` | Default benchmark output — `results_*.json`, generated-image folders, and one `answers_<test>_*` JSON sidecar per selected accuracy workload |
| `dashboard/` | The results-explorer web app (React + Vite) |
| `tests/` | The unit and integration test suite — see [Testing](testing.md) |
| `tests/fixtures/` | Immutable compatibility results that freeze commercially important application/schema behavior before execution-kernel migration |
| `docs/result-compatibility-v4.1.md` | Export and dashboard behavior the commercial execution-kernel rewrite must preserve or version explicitly |
| `samples/` | Sample `results_*.json` files for trying the dashboard without running a benchmark |
| `models/` | Downloaded LLM/embedding GGUF files, namespaced per engine (`models/llamacpp/<tag-slug>/`) — created by `setup_check.py`, gitignored |
| `models.py` (in `scripts/`) | Single source of truth for every model definition — imported by `benchmark.py`, `setup_check.py`, and `shared.py` |
| `requirements.txt` | Python dependencies, installed by the setup scripts |
| `sample_document.txt` | The corpus chunked and embedded by the embeddings test |
| `scripts/data/` | Active accuracy banks—`mcq_questions.json` (150 questions), `math_questions.json` (150 questions), `reasoning_questions.json` (60 questions), `code_problems.json` (60 problems), and `tool_questions.json` (100 tool-calling questions) |
| `scripts/data/reasoning_questions.json` | Versioned, validated reasoning bank across ten categories, including a 20-question `very_hard` tail |
| `hf.txt` | Optional saved HuggingFace token (see [Setup](setup.md#huggingface-token)) — not tracked in git |
| `local_ai_bench_config.json` | Versioned, gitignored setup handoff containing validated non-secret ComfyUI and llama.cpp tool paths |
| `.benchmark_frontend_state.json` | Gitignored last-confirmed interactive engine/test/model/max-prompt-tokens/tg-tokens selection; stale or invalid values fall back to current defaults |
| `.coveragerc` | Coverage config for the test suite — omits `setup_check.py` (unsafe to import) and excludes live-server/subprocess code marked `# pragma: no cover`, so `pytest --cov` reports coverage of the unit-testable code only |
| `.llm_crash_cache.json` | Records LLM models that crashed the active engine's runner repeatedly during the single-shot test, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry |
| `.conv_crash_cache.json` | Same as above, for the conversation test |
| `.embed_crash_cache.json` | Records model/document combos that crashed the active engine's runner repeatedly, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry |
| `.mcq_crash_cache.json` | Same as above, for the MCQ accuracy test. Also records which question-bank version (a short content hash) the crash happened against, so a crash recorded on an old/smaller bank doesn't skip a model forever once the bank changes — see [bank versioning](workloads.md#bank-versioning) |
| `.math_crash_cache.json` | Same as above, for the math accuracy test |
| `.reasoning_crash_cache.json` | Same as above, for the reasoning accuracy test |
| `.code_crash_cache.json` | Same as above, for the code accuracy test |
| `.tool_crash_cache.json` | Same as above, for the tool-calling accuracy test |
| `.concurrency_tool_crash_cache.json` | Records repeatable engine crashes from the tool-style concurrency sweep; safe to delete to retry |
| `.concurrency_chat_crash_cache.json` | Same as above, for the chat concurrency sweep |

The old `compare.py` CLI tool has been dropped — it's been replaced by the [dashboard](dashboard.md).

## `scripts/` in detail

| Module | Purpose |
|---|---|
| `benchmark.py` | CLI entry point — argument parsing, scope resolution, and workload-stage wiring |
| `benchmark_frontend.py` | Interactive installed-model/test picker; launches `benchmark.py` with explicit public CLI flags |
| `run_plan.py` | Immutable, serializable, path-free execution plan and deterministic plan identity |
| `interface_mode.py` | Pure GUI/terminal/noninteractive selection for local desktop, SSH, and headless sessions |
| `orchestration.py` | Local run paths, fixed stage ordering/execution, and engine/ComfyUI lifecycle coordination |
| `result_store.py` | Atomic JSON writer plus the narrow result-section and run/stage transition API |
| `llamacpp_tools.py` | System-first discovery shared by setup, llama-server, llama-bench, and llama-batched-bench |
| `config.py` | Shared constants (URLs, paths, timeouts, run counts) |
| `model_inventory.py` | Installed-model discovery/classification plus narrowly scoped non-catalog llama.cpp folder cleanup |
| `setup_selection.py` | Pure setup-picker state rules, including destructive-cleanup isolation from broad model toggles |
| `shared.py` | Cross-cutting helpers: plain frontend and timestamped benchmark console output, machine profiling, engine-agnostic run/crash orchestration, ComfyUI server lifecycle/HTTP client |
| `hardware.py` | GPU/system-memory detection, shared-memory classification, and model-fit estimates |
| `engines/base.py`, `engines/llamacpp.py` | `InferenceEngine` interface and `LlamaCppEngine` — server lifecycle + HTTP/process client, see [Engines](engines.md) |
| `llm_prefill_benchmark.py` | Single-shot LLM test |
| `llm_conversation_benchmark.py` | Multi-turn conversation LLM test |
| `embedding_benchmark.py` | Embeddings test |
| `image_benchmark.py` | Image generation test (ComfyUI workflow builders + submission) |
| `concurrency_benchmark.py` | Shared implementation for the tool-style and chat concurrency sweeps |
| `mcq_benchmark.py` | MCQ accuracy test |
| `math_benchmark.py` | Numeric-answer math accuracy test |
| `reasoning_benchmark.py` | Knowledge-light A–D reasoning accuracy test and validated bank loader |
| `code_benchmark.py` | Isolated Python code-generation accuracy test |
| `tool_benchmark.py` | Tool-calling accuracy test |
| `regrade.py` | Offline utility that reapplies current accuracy graders to matching raw-answer sidecars and writes separate `regraded_*.json` copies |
| `llamabench_benchmark.py` | Opt-in `llamabench` test — llama.cpp's own separate prefill and depth-aware decode sweeps across installed models, bypassing the HTTP engine (see [Workloads](workloads.md#llama-bench)) |
| `llamabench_concurrency_benchmark.py` | Opt-in `llamabenchconc` test — llama.cpp's own `llama-batched-bench` decode-throughput-vs-concurrency sweep, bypassing the HTTP engine (see [Workloads](workloads.md#llama-bench-concurrency)) |
| `models.py` | Model definitions (tags, checkpoints, tiers, sizes) |
| `comfyui_installation.py` | ComfyUI program discovery, Python selection, saved path, and managed extra-model configuration |
| `setup_check.py` | Hardware detection, model picker, unattended install |
| `setup_config.py` | Atomic loading and persistence for the non-secret setup handoff |
| `data/` | Question banks used by accuracy tests (see above) |

## `results/` in detail

By default, each benchmark run produces one main results file plus one separately named file or folder for workloads with bulky output:

```
results/
  results_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  images_Mac_Studio_M4_Max_64_GB_20260711_090000/
    sdxl_1024x1024.png
    sdxl_1536x1536.png
    flux-dev_1024x1024.png
    ...
  answers_mcq_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  answers_math_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  answers_reasoning_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  answers_code_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  answers_tool_Mac_Studio_M4_Max_64_GB_20260711_090000.json
```

Each auxiliary name is derived from the main results filename's stem by swapping `results_` for `images_` or `answers_<test>_` (`mcq`, `math`, `reasoning`, `code`, or `tool`). If the stem does not begin with `results_`, the auxiliary prefix is prepended instead. With the default output, this preserves the hostname and timestamp across the set. If `--out` places the main JSON elsewhere, only that main file follows the custom directory; images and answer sidecars still go under the repository's `results/` directory. See [CLI Reference](cli-reference.md).

`--engine all` (see [Engines](engines.md)) appends the engine name to the results filename's stem for each pass, so a run of the example above would produce `results_..._090000_llamacpp.json` (and one more per additional engine, once a second one is registered) side by side, each tagged internally with `"engine"`.

The `answers_*.json` sidecars hold every question's answer for that accuracy test, keyed by model, each with the model's full graded response text and a `correct` flag. They stay outside the main results JSON because raw answers are much larger than summary scores and diagnostics. The main results JSON's own `incorrect` list (per model, per test) is unaffected and still covers only wrong answers.

`python scripts/regrade.py results/results_...json` reapplies every accuracy grader with stored model results when the bank hashes exactly match the banks in `scripts/data/`; empty blocks from unselected tests require no sidecar. This keeps pre-reasoning result files regradeable while new files can include reasoning. It writes a complete `regraded_results_...json` plus matching `regraded_answers_*_...json` sidecars and never edits the source files. `--dry-run` validates and scores without writing. Code answers run again through the existing timeout harness, which provides process isolation and timeout recovery rather than a security sandbox.

### Main results JSON

The main file is checkpointed throughout a run, so completed stages and models survive an interruption. Its top level contains:

| Key | Contents |
|---|---|
| `version`, `engine` | Application release and inference-engine name |
| `run` | Schema version, run ID, source revision, effective non-secret configuration, selected model identities, overall completion state, and per-stage state/coverage |
| `profile` | Host description, OS/release, architecture, Python version, RAM, UTC timestamp, effective inference backend (`cuda`, `rocm`, `metal`, `xpu`, `vulkan`, or `cpu`), and separately detected `hardware_backend` |
| `bank_versions` | Content hashes for the MCQ, math, reasoning, code, and tool banks |
| `sample_ids` | Exact per-bank IDs only when `--sample` was used |
| `llm`, `llm_conversation` | Per-model context/checkpoint measurements and any timeout, crash, slow-TPS, or skip markers |
| `accuracy_settings` | Effective accuracy timeout, completion-token budget, and first-pass fraction used by the run |
| `mcq`, `math`, `reasoning`, `code`, `tool` | Per-model overall/category scores plus nudge, exhausted-budget, timeout, and likely-loop diagnostics when present; reasoning also includes `by_difficulty` |
| `embeddings`, `images` | Per-model throughput or per-resolution generation-time measurements |
| `concurrency_tool`, `concurrency_chat` | Per-model/per-level TTFT, per-request and aggregate throughput, token/batch timing, memory snapshots, and stop markers |
| `llamabench` | Opt-in — per-model raw `llama-bench -o json` `prefill_entries` and depth-aware `decode_entries` arrays (or an `error` string) — see [Workloads](workloads.md#llama-bench) |
| `llamabenchconc` | Opt-in — per-model raw `llama-batched-bench` JSONL entries plus the effective `pp`/`ctx_size` used (or an `error` string), one entry per pp/tg/concurrency-level combination — see [Workloads](workloads.md#llama-bench-concurrency) |

Generation, conversation, concurrency, and embedding aggregates add explicit `requested_runs`, `completed_runs`, and `valid_runs` counts. Generation-family entries retain legacy aggregate fields while adding client-TTFT, server-prompt, wall/decode, token, finish-reason, valid-sample, and invalid-diagnostic fields; `n_runs` remains the completed-call count for compatibility.

Performance workloads retain means, standard deviations, run counts, and—where applicable—the individual measured values. New files use `run.schema_version` for results-schema compatibility; the top-level `version` remains the application release, while older files without `run` remain supported. Main results, answer sidecars, crash caches, and regraded outputs use same-directory temporary files plus atomic replacement so a failed checkpoint leaves the prior valid file intact. Missing keys and empty sections are valid because the dashboard supports partial runs and older schema versions.

`results/` is gitignored — nothing under it is tracked. Load its contents into the [dashboard](dashboard.md) to compare across machines.

---

[← Engines](engines.md) · [Back to README](../README.md) · [Testing →](testing.md)
