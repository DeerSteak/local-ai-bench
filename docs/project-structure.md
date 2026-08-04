[← Back to README](../README.md)

# Project Structure

**Contents**
- [`scripts/` in detail](#scripts-in-detail)
- [`results/` in detail](#results-in-detail)
  - [Main results JSON](#main-results-json)

| File / Folder | Purpose |
|---|---|
| `setup.sh` | One-shot setup for macOS and Linux |
| `Setup Local AI Bench.command` | Double-clickable macOS launcher for the graphical setup wizard |
| `Setup Local AI Bench.desktop` | Double-clickable Linux desktop launcher for the graphical setup wizard |
| `Setup Local AI Bench.bat` | Double-clickable Windows launcher for the graphical setup wizard |
| `setup.bat` | One-shot setup for Windows |
| `Run Local AI Bench.command` | Double-clickable macOS launcher for the graphical benchmark screen |
| `Run Local AI Bench.desktop` | Double-clickable Linux desktop launcher for the graphical benchmark screen |
| `Run Local AI Bench.bat` | Double-clickable Windows launcher for the graphical benchmark screen |
| `run_bench.sh` | Activates the venv; auto-selects GUI/terminal with no arguments or forwards benchmark arguments directly on Linux / macOS |
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
| `docs/architecture-decisions.md` | Simplicity gate, accepted architecture decisions, and compatibility-layer deletion ledger |
| `docs/methodology-contract.md` | Neutral 4.1 metric, cache, retry, timeout, validity, aggregation, acceptance, and change-control contract |
| `docs/platform-tuning.md` | Neutral runtime settings, platform compatibility workarounds, and tuning-profile change rules |
| `docs/acceptance-policies.md` | Versioned explicit threshold policy, evidence, and rejection semantics |
| `docs/projects.md` | Local project workflows, portable configuration, baseline, and acceptance-policy behavior |
| `docs/result-history.md` | Filesystem-owned result discovery, filtering, comparison, and policy evaluation |
| `docs/outbound-review.md` | Embargo-safe identity preview, private aliases, and source verification |
| `docs/limitations.md` | Benchmark representativeness, variance, compatibility, and recommendation limitations |
| `docs/data-lifecycle.md` | Local retention, deletion, portability, and artifact-handling behavior |
| `docs/coordinator-api.md` | Versioned future localhost coordinator API, authentication, validation, lifecycle, and compatibility contract |
| `docs/extension-contracts.md` | Versioned workload SDK, conformance-vector format, and capability-negotiated engine adapter contract |
| `docs/security-and-privacy.md` | Working threat model, data classifications, embargo policy, controls, and verification gaps |
| `samples/` | Sample `results_*.json` files for trying the dashboard plus reviewed HTML/PDF decision-report examples |
| `models/` | Downloaded LLM/embedding GGUF files, namespaced per engine (`models/llamacpp/<tag-slug>/`) — created by `setup_check.py`, gitignored |
| `models.py` (in `scripts/`) | Single source of truth for every model definition — imported by `benchmark.py`, `setup_check.py`, and `shared.py` |
| `requirements.txt` | Python dependencies, installed by the setup scripts |
| `sample_document.txt` | The corpus chunked and embedded by the embeddings test |
| `scripts/data/` | Active accuracy banks—`mcq_questions.json` (150 questions), `math_questions.json` (150 questions), `reasoning_questions.json` (60 questions), `code_problems.json` (60 problems), and `tool_questions.json` (100 tool-calling questions) |
| `scripts/data/reasoning_questions.json` | Versioned, validated reasoning bank across ten categories, including a 20-question `very_hard` tail |
| `hf.txt` | Optional saved HuggingFace token (see [Setup](setup.md#huggingface-token)) — not tracked in git |
| `local_ai_bench_config.json` | Versioned, gitignored setup handoff containing validated non-secret ComfyUI and llama.cpp tool paths |
| `.benchmark_frontend_state.json` | Gitignored Custom GUI/terminal selection and execution settings; stale or invalid values fall back to current defaults |
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
| `benchmark_options.py` | Typed public-option metadata shared by CLI parsing, GUI defaults, validation, and option coverage |
| `benchmark_frontend.py` | Interactive installed-model/test picker; launches `benchmark.py` with explicit public CLI flags |
| `benchmark_gui.py` | Single-screen Tk benchmark configuration, subprocess log, and safe cancellation interface |
| `benchmark_presets.py` | Versioned portable benchmark preset validation, persistence, duplication, and comparison |
| `benchmark_project.py` | Versioned local decision projects combining portable configuration with optional baseline and policy |
| `result_history.py` | Local result summaries, filters, named metric extraction, and compatibility-aware comparison |
| `outbound_metadata.py` | Exact outbound identity preview, private aliases, and stable source-identity digests |
| `result_bundle.py` | Deterministic portable result bundles, digest verification, safe import, methodology checks, and aggregate reproduction |
| `result_bundle_cli.py` | Command-line `export`, `verify`, and `import` interface for portable result bundles |
| `decision_report.py` | Deterministic self-contained HTML/PDF decision-report model and renderers |
| `decision_report_cli.py` | Command-line decision-report generator for validated result JSON |
| `acceptance_policy.py` | Validates and evaluates versioned per-case evidence-threshold policies |
| `acceptance_policy_cli.py` | Machine-readable command-line acceptance evaluator with distinct decision exit codes |
| `support_bundle.py` | Allowlisted, deterministic support diagnostics with private-path and credential redaction |
| `benchmark_launcher.py` | Automatic GUI/terminal benchmark frontend dispatcher |
| `run_plan.py` | Immutable, serializable, path-free execution plan and deterministic plan identity |
| `methodology_profile.py` | Resolves the neutral profile and records selected workloads' effective runtime settings |
| `event_store.py` | Transactional append-only SQLite job events, immutable plan loading, digest verification, and rebuildable projections |
| `resume_policy.py` | Content-based plan, artifact, runtime, and methodology identity plus safe case-boundary resume/fork decisions |
| `content_store.py` | Atomic content-addressed storage and verified references for large local artifacts |
| `runner_supervisor.py` | Fixed-command internal runner protocol, heartbeat monitoring, process ownership, and cancellation escalation |
| `workload_runner.py` | Owned internal single-shot runner; reconstructs its immutable plan from the journal and exposes no general command surface |
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
| `llm_event_stage.py` | Journal-owned generation/conversation/concurrency samples, stage/model-family isolation, and compatible JSON projections |
| `conversation_selection.py` | Pure conversation preflight selection shared by the coordinator tests and child runner |
| `native_bench_event_stage.py` | Journal-owned streamed llama-bench rows, partial markers, repetition counts, and compatible projection |
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
| `setup_gui.py` | Tkinter setup wizard that produces the same pre-download setup plan as the terminal interface |
| `tk_utils.py` | Shared cross-platform Tk mouse-wheel normalization |
| `close_terminal_tab.applescript` | Closes only the macOS Terminal tab that launched a cancelled graphical setup |
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
