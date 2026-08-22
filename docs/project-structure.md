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
| `Launch Local AI Bench Dashboard.command` | Double-clickable macOS launcher that builds, serves, and opens the dashboard |
| `Launch Local AI Bench Dashboard.desktop` | Double-clickable Linux desktop launcher that builds, serves, and opens the dashboard |
| `Launch Local AI Bench Dashboard.bat` | Double-clickable Windows launcher that builds, serves, and opens the dashboard |
| `run_bench.sh` | Activates the venv; auto-selects GUI/terminal with no arguments or forwards benchmark arguments directly on Linux / macOS |
| `run_qualification.sh` / `run_qualification.bat` | Thin platform launchers that run normal setup and the normal smallest-model benchmark |
| `run_telemetry_trials.sh` | Resumable alternating telemetry-off/on qualification runner for memory and power on Linux / macOS |
| `run_sustained_qualification_linux.sh` | Repeated ten-minute sustained-load evidence wrapper for Linux small systems |
| `run_temperature_qualification_linux.sh` | Unattended Linux temperature observer-effect matrix across latency/sustained screens and all candidate intervals |
| `run_bench.bat` | Windows equivalent of `run_bench.sh` |
| `launch_dashboard.sh` | Builds and serves the dashboard on Linux / macOS, optionally stages selected `--result` files, and opens the browser automatically |
| `launch_dashboard.bat` | Windows equivalent of `launch_dashboard.sh` |
| `tests.sh` | Activates the venv and runs unit/integration tests on Linux / macOS — see [Testing](testing.md) |
| `tests.bat` | Activates the venv and runs unit/integration tests on Windows — see [Testing](testing.md) |
| `scripts/` | Packaged implementation grouped by application, runtime, workloads, results, setup, and release responsibilities |
| `.githooks/pre-commit` | Version-sync hook — see [Release policy](release-policy.md#version-sync-hook); enable per clone with `git config core.hooksPath .githooks` |
| `.github/CODEOWNERS` | Default and sensitive-boundary review ownership used by GitHub rulesets |
| `.github/dependabot.yml` | Scheduled Python and dashboard dependency updates targeting `develop` |
| `SECURITY.md` | Private vulnerability-reporting route, response targets, supported versions, and safe-testing policy |
| `results/` | Default benchmark output — `results_*.json`, generated-image folders, and one `answers_<test>_*` JSON sidecar per selected accuracy workload |
| `dashboard/` | The results-explorer web app (React + Vite), including the temporary selected-result staging utility used by local launchers |
| `tests/` | The unit and integration test suite — see [Testing](testing.md) |
| `tests/fixtures/` | Immutable compatibility results that freeze commercially important application/schema behavior before execution-kernel migration |
| `docs/result-compatibility-v4.1.md` | Export and dashboard behavior the commercial execution-kernel rewrite must preserve or version explicitly |
| `docs/README.md` | Audience- and status-oriented documentation index |
| `docs/architecture-decisions.md` | Simplicity gate, accepted architecture decisions, and compatibility-layer deletion ledger |
| `docs/methodology-contract.md` | Neutral 4.1 metric, cache, retry, timeout, validity, aggregation, acceptance, and change-control contract |
| `docs/product-requirements.md` | Primary pre-launch hardware-validation decision, scope, outcomes, and product qualities |
| `VERSION_6_PLAN.md` | Root-level implementation, qualification, pilot, rollback, and release plan for Version 6 |
| `docs/version-6-foundation.md` | Frozen schema-5 map, vocabulary, telemetry screening rules, and qualification set |
| `docs/telemetry-qualification.md` | Supervised memory observer-screen procedure and analyzer manifest format |
| `docs/user-journey.md` | Complete discovery, project, execution, recovery, review, export, and escalation path |
| `docs/recommendation-policy.md` | Consumer goals, evidence eligibility, fit, ranking, conflicts, and GPU/Mac workflows |
| `docs/platform-tuning.md` | Neutral runtime settings, platform compatibility workarounds, and tuning-profile change rules |
| `docs/model-catalog-audit-v6.md` | Precommitted Milestone 9 selection rubric, incumbent roles, candidate register, and compatibility-screen contract |
| `docs/strix-halo-troubleshooting-ubuntu-24.04.md` | Ubuntu OEM-kernel, inbox-driver, ROCm, DKMS recovery, and qualification runbook for Strix Halo |
| `docs/acceptance-policies.md` | Versioned explicit threshold policy, evidence, and rejection semantics |
| `docs/projects.md` | Local project workflows, portable configuration, baseline, and acceptance-policy behavior |
| `docs/result-history.md` | Filesystem-owned result discovery, filtering, dashboard launch, and policy evaluation |
| `docs/vendor-diagnostics.md` | First-divergence diagnostic content, verification, and engineer workflow |
| `docs/outbound-review.md` | Embargo-safe identity preview, private aliases, and source verification |
| `docs/offline-mode.md` | Loopback-only execution policy, inherited controls, and qualification boundary |
| `docs/limitations.md` | Benchmark representativeness, variance, compatibility, and recommendation limitations |
| `docs/data-lifecycle.md` | Local retention, deletion, portability, and artifact-handling behavior |
| `docs/coordinator-api.md` | Versioned future localhost coordinator API, authentication, validation, lifecycle, and compatibility contract |
| `docs/extension-contracts.md` | Versioned workload SDK, conformance-vector format, and capability-negotiated engine adapter contract |
| `docs/security-and-privacy.md` | Working threat model, data classifications, embargo policy, controls, and verification gaps |
| `samples/` | Sample results, recommendation constraints/artifact, trial artifact, and reviewed HTML/PDF decision-report examples |
| `models/` | Downloaded LLM/embedding GGUF files, namespaced per engine (`models/llamacpp/<tag-slug>/`) — created by `setup/setup_check.py`, gitignored |
| `requirements.txt` | Python dependencies, installed by the setup scripts |
| `scripts/workloads/data/` | Active accuracy banks plus `sample_document.txt`, the real-world corpus used by the embeddings workload |
| `scripts/workloads/data/reasoning_questions.json` | Versioned, validated reasoning bank across ten categories, including a 20-question `very_hard` tail |
| `hf.txt` | Optional saved HuggingFace token (see [Setup](setup.md#huggingface-token)) — not tracked in git |
| `local_ai_bench_config.json` | Versioned, gitignored setup handoff containing validated non-secret ComfyUI and llama.cpp tool paths plus detected NVIDIA or ROCm GPU topology |
| `.benchmark_frontend_state.json` | Gitignored GUI/terminal selection and execution settings plus the last GUI preset name; stale or invalid values fall back to current defaults |
| `.resume_digest_cache.json` | Gitignored local path/metadata cache for previously computed model/runtime content identities; portable journals contain only size and SHA-256 |
| `results_*.events.sqlite3.local.json` | Owner-only run-local execution paths required to recover isolated image workloads; never portable or bundleable and deleted with the run |
| `.coveragerc` | Coverage config for the test suite — excludes live-server/subprocess code marked `# pragma: no cover`, so `pytest --cov` reports coverage of the unit-testable code only |
| `.llm_crash_cache.json` | Records LLM models that crashed the active engine's runner repeatedly during the single-shot test, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry. Keyed `{engine_name: {tag: detail}}`: a crash is scoped to the engine that produced it, since the same catalog tag is a different runtime and a different weight file per engine (see [Engines](engines.md)) |
| `.conv_crash_cache.json` | Same as above, for the conversation test |
| `.embed_crash_cache.json` | Records model/document combos that crashed the active engine's runner repeatedly, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry. Same per-engine keying as `.llm_crash_cache.json` |
| `.mcq_crash_cache.json` | Same as above, for the MCQ accuracy test. Also records which question-bank version (a short content hash) the crash happened against, so a crash recorded on an old/smaller bank doesn't skip a model forever once the bank changes — see [bank versioning](workloads.md#bank-versioning) |
| `.math_crash_cache.json` | Same as above, for the math accuracy test |
| `.reasoning_crash_cache.json` | Same as above, for the reasoning accuracy test |
| `.code_crash_cache.json` | Same as above, for the code accuracy test |
| `.tool_crash_cache.json` | Same as above, for the tool-calling accuracy test |
| `.concurrency_tool_crash_cache.json` | Records repeatable engine crashes from the tool-style concurrency sweep; safe to delete to retry. Same per-engine keying as `.llm_crash_cache.json` |
| `.concurrency_chat_crash_cache.json` | Same as above, for the chat concurrency sweep |

The old `compare.py` CLI tool has been dropped — it's been replaced by the [dashboard](dashboard.md).

## `scripts/` in detail

The package boundaries are deliberately broad and practical: `app/` owns user entry points and coordination, `runtime/` owns engine and process infrastructure, `workloads/` owns benchmark definitions and bundled data, `results/` owns persistence, recovery, and reporting, `setup/` owns installation, and `release/` owns shipping gates. Public shell and batch wrappers remain at the repository root, so existing automation does not depend on internal file locations.

| Module | Purpose |
|---|---|
| `app/benchmark.py` | CLI entry point — argument parsing, scope resolution, and workload-stage wiring |
| `app/benchmark_options.py` | Typed public-option metadata shared by CLI parsing, GUI defaults, validation, and option coverage |
| `app/benchmark_frontend.py` | Interactive installed-model/test picker; launches `app/benchmark.py` with explicit public CLI flags |
| `app/benchmark_gui.py` | Tk application bootstrap, shared run state, controller wiring, and cross-screen orchestration |
| `app/macos_sudo_askpass.sh` | Native macOS password prompt used only to authorize opt-in GUI power telemetry before launch |
| `app/benchmark_gui_screens/` | Dedicated Configuration, Run Log, Result History, Engine Management, and live progress screens with screen-specific action controllers |
| `app/benchmark_gui_support.py` | Pure GUI configuration, planning, and progress-state helpers |
| `app/benchmark_gui_process.py` / `app/benchmark_gui_resources.py` | GUI subprocess coordination plus CPU, RAM, GPU, and VRAM monitoring |
| `app/model_import_dialog.py` | Non-blocking Hugging Face custom-model inspection and import dialog |
| `app/engine_management.py` | Engine status, runtime updates, model verification, and mirrored operation output |
| `app/benchmark_presets.py` | Versioned portable benchmark preset validation, persistence, duplication, and comparison |
| `app/benchmark_project.py` | Versioned local decision projects combining portable configuration with optional baseline and policy |
| `results/result_history.py` | Local result summaries, filters, named metric extraction, and compatibility-aware comparison |
| `results/significance.py` | Practical-threshold, uncertainty-availability, and repeated-trial comparison mathematics |
| `results/trial_set.py` / `results/trial_set_cli.py` | Compatible independent-trial aggregation, drift detection, uncertainty intervals, verdicts, and artifact CLI |
| `results/trial_set_report.py` | Auditable Markdown presentation of repeated-trial evidence and inconclusive states |
| `results/recommendation.py` / `results/recommendation_cli.py` | Constraint validation, evidence eligibility, hard filtering, qualified trial ranking, and versioned recommendation artifacts |
| `results/outbound_metadata.py` | Exact outbound identity preview, private aliases, and stable source-identity digests |
| `runtime/network_policy.py` | Loopback classification, offline environment, and Python socket enforcement |
| `results/vendor_diagnostic.py` | Deterministic first-divergence package, raw evidence selection, and source verification |
| `results/vendor_diagnostic_cli.py` | Reviewed diagnostic creation and source-pair verification commands |
| `results/result_bundle.py` | Deterministic portable result bundles, digest verification, safe import, methodology checks, and aggregate reproduction |
| `results/result_bundle_cli.py` | Command-line `export`, `verify`, and `import` interface for portable result bundles |
| `results/decision_report.py` | Deterministic self-contained HTML/PDF decision-report model plus acceptance and recommendation-artifact rendering |
| `results/decision_report_cli.py` | Command-line decision-report generator for validated result JSON and source-verified recommendation artifacts |
| `results/acceptance_policy.py` | Validates and evaluates versioned per-case evidence-threshold policies |
| `results/acceptance_policy_cli.py` | Machine-readable command-line acceptance evaluator with distinct decision exit codes |
| `results/support_bundle.py` | Allowlisted, deterministic support diagnostics with private-path and credential redaction |
| `app/benchmark_launcher.py` | Automatic GUI/terminal benchmark frontend dispatcher |
| `results/run_plan.py` / `results/canonical_json.py` | Immutable execution plans plus the single canonical JSON and digest contract for durable identities |
| `workloads/methodology_profile.py` | Resolves the neutral profile and records selected workloads' effective runtime settings |
| `results/event_store.py` | Transactional append-only SQLite job events, immutable plan loading, digest verification, and rebuildable projections |
| `results/resume_policy.py` | Content-based plan, artifact, runtime, and methodology identity plus safe case-boundary resume/fork decisions |
| `results/recovery_inspector.py` | Read-only resume/fork eligibility, identity revalidation, and durable coverage report |
| `results/recovery_executor.py` | State-changing ordered recovery for plans composed entirely of journal-owned stages |
| `results/accuracy_event_stage.py` | Per-question accuracy journal ownership plus scored-result and raw-answer projections shared by all five banks |
| `results/embedding_event_stage.py` | Per-model embedding-batch journal ownership, corpus identity, and compatible throughput projection without retained vectors |
| `results/image_event_stage.py` | Journal-owned per-resolution image attempts, compatible projection, merged telemetry, and content-addressed artifact metadata |
| `results/vllm_bench_event_stage.py` | Journal-owned vLLM latency/throughput size cases and compatible native-result projection |
| `results/native_concurrency_event_stage.py` | Journal-owned streamed llama-batched-bench rows and compatible concurrency projection |
| `results/local_execution_context.py` | Strict owner-only local-path sidecar bound to a journal job and excluded from outbound artifacts |
| `results/retry_executor.py` | Explicit selected-case retry for eligible stopped journal context/level cases |
| `results/fork_executor.py` | Reviewed new-job execution of a saved journal-owned plan without changing its source result |
| `runtime/pause_control.py` | Short-lived cooperative pause state plus schema-4 pause-transition evidence shared across GUI-launched parent and workload processes |
| `runtime/telemetry.py` | Shared resource queries, background memory sampler, aggregation, and headroom classification |
| `workloads/sustained_benchmark.py` / `workloads/sustained_analysis.py` | Continuous-generation soak, aligned time windows, pure retention/onset classification, and sensor correlation |
| `results/sustained_event_stage.py` | Request-level sustained journal and whole-soak recovery projection |
| `results/content_store.py` | Atomic content-addressed storage and verified references for large local artifacts |
| `runtime/runner_supervisor.py` / `runtime/process_tree.py` | Fixed-command internal runner protocol, heartbeat monitoring, and descendant-aware cancellation escalation |
| `runtime/workload_runner.py` / `runtime/supervised_stage.py` | Owned internal stage runner plus the parent supervisor service shared by normal and recovery execution |
| `release/qualification.py` / `release/qualification_docs.py` | Evidence-derived platform support policy plus generated published matrices |
| `release/qualification_run.py` / `release/qualification_targets.py` / `release/qualification_targets.ps1` / `release/qualification_setup.ps1` / `release/qualification_coverage.py` | Explicit platform selection, dependency-free Windows target parsing, setup evidence capture, normal benchmark launch, and smallest-model result validation |
| `app/interface_mode.py` | Pure GUI/terminal/noninteractive selection for local desktop, SSH, and headless sessions |
| `stage_registry.py` | Authoritative workload order, result section, model family, label, category, and native-engine ownership |
| `app/orchestration.py` | Local run paths, stage execution, and engine/ComfyUI lifecycle coordination |
| `app/result_actions.py` / `app/recovery_actions.py` | GUI-facing result/log/dashboard commands and recovery review/command construction |
| `results/result_store.py` | Atomic JSON writer plus the narrow result-section and run/stage transition API |
| `runtime/llamacpp_tools.py` | System-first discovery shared by setup, llama-server, llama-bench, and llama-batched-bench |
| `runtime/config.py` | Shared constants (URLs, paths, timeouts, run counts) |
| `runtime/model_identity.py` | Filesystem-safe normalization shared by engine and setup model paths |
| `setup/model_inventory.py` | Installed-model discovery/classification plus narrowly scoped non-catalog llama.cpp folder cleanup |
| `setup/custom_models.py` | Gitignored engine-specific custom-model provenance registry |
| `setup/model_import.py` / `setup/model_download.py` | Hugging Face repository inspection and engine-specific artifact downloads |
| `setup/runtime_identity.py` | Read-only engine runtime ownership classification and version inspection |
| `setup/runtime_status.py` | Combined engine health, backend, dependency-stack, and WSL runtime status records |
| `setup/model_compatibility.py` | Imported-model architecture metadata and read-only vLLM registry compatibility probes |
| `setup/engine_selection.py` | Engine-picker rules and terminal interaction, including disabled engines and installation needs |
| `setup/cuda_install.py` | Native Ubuntu NVIDIA driver bootstrap and WSL2-only CUDA toolkit installation, so qualification cannot silently build llama.cpp CPU-only |
| `setup/rocm_install.py` | Pinned native-Linux and WSL2 ROCm qualification plans, platform gates, and privileged execution |
| `setup/intel_xpu_install.py` | Native Ubuntu Intel GPU compute/oneAPI prerequisite plan, SYCL environment loading, and XPU probe |
| `setup/vllm_install.py` | vLLM platform-support matrix, launcher/server discovery, interpreter/venv resolution, and the optional installer |
| `setup/runtime_update.py` / `setup/directory_transaction.py` | Platform runtime updates plus the shared staged-directory swap and rollback transaction |
| `setup/setup_selection.py` | Terminal model picker, fit-based defaults, input parsing, and destructive-cleanup isolation |
| `runtime/shared.py` | Remaining cross-workload console, machine-profile, measured-run, accuracy-runner, and ComfyUI helpers pending narrow ownership |
| `runtime/crash_cache.py` | Engine-scoped crash-cache persistence, retry policy, and cleanup |
| `runtime/progress_events.py` | Structured cross-process progress events consumed by the graphical launcher |
| `runtime/generation_guard.py` / `runtime/failure_handling.py` | Generation-loop detection and consistent unexpected per-model failure records |
| `runtime/hardware.py` | GPU/system-memory detection, shared-memory classification, and model-fit estimates |
| `runtime/engines/base.py`, `runtime/engines/llamacpp.py`, `runtime/engines/vllm.py` | Engine interface and engine-specific lifecycle/transport clients, see [Engines](engines.md) |
| `runtime/engines/chat_flow.py` | Engine-neutral bounded chat finalization and measurement aggregation |
| `workloads/llm_prefill_benchmark.py` | Single-shot LLM test |
| `results/llm_event_stage.py` | Journal-owned generation/conversation/concurrency samples, stage/model-family isolation, and compatible JSON projections |
| `workloads/conversation_selection.py` | Pure conversation preflight selection shared by the coordinator tests and child runner |
| `results/native_bench_event_stage.py` | Journal-owned streamed llama-bench rows, partial markers, repetition counts, and compatible projection |
| `workloads/llm_conversation_benchmark.py` | Multi-turn conversation LLM test |
| `workloads/embedding_benchmark.py` | Embeddings test |
| `workloads/image_benchmark.py` | Image generation test (ComfyUI workflow builders + submission) |
| `workloads/concurrency_benchmark.py` | Shared implementation for the tool-style and chat concurrency sweeps |
| `workloads/mcq_benchmark.py` | MCQ accuracy test |
| `workloads/math_benchmark.py` | Numeric-answer math accuracy test |
| `workloads/reasoning_benchmark.py` | Knowledge-light A–D reasoning accuracy test and validated bank loader |
| `workloads/code_benchmark.py` | Restricted Python code-generation accuracy test |
| `workloads/code_sandbox.py` | Generated-Python child boundary with static policy and bounded resources/output |
| `workloads/tool_benchmark.py` | Tool-calling accuracy test |
| `workloads/accuracy_scoring.py` | Shared accuracy-bank aggregation with workload-owned correctness callbacks |
| `workloads/accuracy_registry.py` | Shared accuracy class and question-bank metadata for supervised execution and recovery identity |
| `results/regrade.py` | Offline utility that reapplies current accuracy graders to matching raw-answer sidecars and writes separate `regraded_*.json` copies |
| `workloads/llamabench_benchmark.py` | Opt-in `llamabench` test — llama.cpp's own separate prefill and depth-aware decode sweeps across installed models, bypassing the HTTP engine (see [Workloads](workloads.md#llama-bench)) |
| `workloads/vllm_benchmark.py` | Opt-in `vllmbench` test — vLLM's own `vllm bench latency`/`throughput` sweep, bypassing the HTTP engine (see [Workloads](workloads.md#vllm-bench)) |
| `workloads/llamabench_concurrency_benchmark.py` | Opt-in `llamabenchconc` test — llama.cpp's own `llama-batched-bench` decode-throughput-vs-concurrency sweep, bypassing the HTTP engine (see [Workloads](workloads.md#llama-bench-concurrency)) |
| `workloads/models.py` | Model definitions (tags, checkpoints, tiers, sizes) |
| `runtime/comfyui_installation.py` | ComfyUI program discovery, Python selection, saved path, and managed extra-model configuration |
| `setup/setup_check.py` | Import-safe setup entrypoint and explicit orchestration across focused setup services |
| `setup/setup_console.py` | Terminal status formatting, hyperlinks, and confirmation prompts |
| `setup/setup_discovery.py` | Read-only host identity and memory discovery for setup |
| `setup/llamacpp_install.py` | llama.cpp tool discovery and platform-specific installation execution |
| `setup/hf_credentials.py` | Hugging Face token discovery, prompting, caching, and optional persistence |
| `setup/comfyui_assets.py` | Selected checkpoint, encoder, and VAE provisioning for ComfyUI |
| `setup/comfyui_runtime.py` | ComfyUI Python requirements, managed model paths, and accelerator-specific PyTorch preparation |
| `setup/comfyui_install.py` | ComfyUI source/portable installation and Windows CUDA-wheel compatibility |
| `setup/setup_gui.py` | Tkinter setup wizard that produces the same pre-download setup plan as the terminal interface |
| `setup/setup_progress.py` | Isolated Tk setup-progress window and its temporary status-file protocol |
| `app/tk_utils.py` | Shared cross-platform Tk mouse-wheel normalization |
| `setup/setup_config.py` | Atomic loading and persistence for the non-secret setup handoff |
| `workloads/data/` | Question banks used by accuracy tests (see above) |

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

Each auxiliary name is derived from the main results filename's stem by swapping `results_` for `images_` or `answers_<test>_` (`mcq`, `math`, `reasoning`, `code`, or `tool`). If the stem does not begin with `results_`, the auxiliary prefix is prepended instead. With the default output, this preserves the hostname and timestamp across the set. If `--out` places the main JSON elsewhere, its journal, local context, images, and answer sidecars remain beside it. See [CLI Reference](cli-reference.md).

`--engine all` (see [Engines](engines.md)) appends the engine name to the results filename's stem for each pass, so a run of the example above would produce `results_..._090000_llamacpp.json` (and one more per additional engine, once a second one is registered) side by side, each tagged internally with `"engine"`.

The `answers_*.json` sidecars hold every question's answer for that accuracy test, keyed by model, each with the model's full graded response text and a `correct` flag. They stay outside the main results JSON because raw answers are much larger than summary scores and diagnostics. The main results JSON's own `incorrect` list (per model, per test) is unaffected and still covers only wrong answers.

`python -m scripts.results.regrade results/results_...json` reapplies every accuracy grader with stored model results when the bank hashes exactly match the banks in `scripts/workloads/data/`; empty blocks from unselected tests require no sidecar. This keeps pre-reasoning result files regradeable while new files can include reasoning. It writes a complete `regraded_results_...json` plus matching `regraded_answers_*_...json` sidecars and never edits the source files. `--dry-run` validates and scores without writing. Code answers run again through the existing timeout harness, which provides process isolation and timeout recovery rather than a security sandbox.

### Main results JSON

The main file is checkpointed throughout a run, so completed stages and models survive an interruption. Its top level contains:

| Key | Contents |
|---|---|
| `version`, `engine`, `engine_version` | Application release, inference-engine name, and the local runtime version when it can be inspected |
| `run` | Schema version, run ID, source revision, effective non-secret configuration, selected model identities, overall completion state, per-stage state/coverage, and optional memory/power summaries |
| `profile` | Host description, OS/release, architecture, Python version, RAM, UTC timestamp, effective inference backend (`cuda`, `rocm`, `metal`, `xpu`, `vulkan`, or `cpu`), and separately detected `hardware_backend` |
| `bank_versions` | Content hashes for the MCQ, math, reasoning, code, and tool banks |
| `sample_ids` | Exact per-bank IDs only when `--sample` was used |
| `llm`, `llm_conversation` | Per-model context/checkpoint measurements, optional same-timeline memory/power evidence, and any timeout, crash, slow-TPS, or skip markers |
| `accuracy_settings` | Effective accuracy timeout, completion-token budget, and first-pass fraction used by the run |
| `mcq`, `math`, `reasoning`, `code`, `tool` | Per-model overall/category scores plus nudge, exhausted-budget, timeout, and likely-loop diagnostics when present; reasoning also includes `by_difficulty` |
| `embeddings`, `images` | Per-model throughput or per-resolution generation-time measurements |
| `concurrency_tool`, `concurrency_chat` | Per-model/per-level TTFT, per-request and aggregate throughput, token/batch timing, memory snapshots, and stop markers |
| `llamabench` | Opt-in — per-model raw `llama-bench -o json` `prefill_entries` and depth-aware `decode_entries` arrays (or an `error` string) — see [Workloads](workloads.md#llama-bench) |
| `vllmbench` | Opt-in — per-model `latency_entries`/`throughput_entries` from `vllm bench`, each carrying its `input_len`/`output_len` (or `error`/`timed_out` diagnostics) — see [Workloads](workloads.md#vllm-bench) |
| `llamabenchconc` | Opt-in — per-model raw `llama-batched-bench` JSONL entries plus the effective `pp`/`ctx_size` used (or an `error` string), one entry per pp/tg/concurrency-level combination — see [Workloads](workloads.md#llama-bench-concurrency) |

Generation, conversation, concurrency, and embedding aggregates add explicit `requested_runs`, `completed_runs`, and `valid_runs` counts. Generation-family entries retain legacy aggregate fields while adding client-TTFT, server-prompt, wall/decode, token, finish-reason, valid-sample, and invalid-diagnostic fields; `n_runs` remains the completed-call count for compatibility.

Performance workloads retain means, standard deviations, run counts, and—where applicable—the individual measured values. New files use `run.schema_version` for results-schema compatibility; the top-level `version` remains the application release, while older files without `run` remain supported. Main results, answer sidecars, crash caches, and regraded outputs use same-directory temporary files plus atomic replacement so a failed checkpoint leaves the prior valid file intact. Missing keys and empty sections are valid because the dashboard supports partial runs and older schema versions.

`results/` is gitignored — nothing under it is tracked. Load its contents into the [dashboard](dashboard.md) to compare across machines.

---

[← Engines](engines.md) · [Back to README](../README.md) · [Testing →](testing.md)
