[← Back to README](../README.md)

# Testing

The repository has two independent automated suites: pytest for the Python benchmark and Vitest for the React dashboard. Neither suite starts a real inference server, downloads models, or runs the benchmark.

**Contents**
- [Python tests](#python-tests)
- [Coverage and safety boundaries](#coverage-and-safety-boundaries)
- [Suite map](#suite-map)
- [Dashboard tests](#dashboard-tests)
- [What to run before submitting](#what-to-run-before-submitting)

## Python tests

Use the platform wrapper so the tests run in the same `bench-env/` environment as the benchmark:

```bash
# Linux / macOS
bash tests.sh

# One file, one test-name pattern, or verbose output
bash tests.sh tests/test_config.py
bash tests.sh -k "select_tier"
bash tests.sh -v
```

```cmd
:: Windows
tests.bat
tests.bat tests/test_config.py
tests.bat -k "select_tier"
```

The wrapper installs or updates [tests/requirements.txt](../tests/requirements.txt) before invoking pytest. If `bench-env/` does not exist, run the appropriate setup script first. Do not import or execute `scripts/setup_check.py` as a test shortcut: importing it starts the real interactive installation flow.

[tests/conftest.py](../tests/conftest.py) adds `scripts/` to `sys.path`, matching the benchmark's own top-level imports such as `import config`.

## Coverage and safety boundaries

Install `pytest-cov` into `bench-env/`, then run:

```bash
bench-env/bin/pip install pytest-cov
bash tests.sh --cov=scripts --cov-report=term-missing
```

On Windows, use `bench-env\Scripts\pip.exe` and `tests.bat`.

[.coveragerc](../.coveragerc) omits `scripts/setup_check.py`, which is unsafe to import, and live orchestration functions are marked `# pragma: no cover`. The excluded functions start real subprocesses, poll llama.cpp or ComfyUI, or drive an entire benchmark run. Pure decisions and calculations are extracted and tested instead. Treat the missing-line report as the useful signal; a fixed percentage is not a project target.

HTTP and process boundaries are mocked where there is a clean seam. Tests may run generated Python in an isolated subprocess for the code grader, but they do not contact a live inference server or ComfyUI instance.

The immutable complete and interrupted 4.1 result fixtures are shared by pytest and Vitest. They enforce the [4.1 result compatibility contract](result-compatibility-v4.1.md) across producer state/count semantics and dashboard chart/reliability behavior before the commercial execution-kernel migration. A later producer adds a new fixture for a schema or methodology boundary rather than editing the 4.1 files in place.

The retained-behavior characterization gate is covered explicitly: immutable result fixtures freeze complete and interrupted/partial exports; `test_mcq_benchmark.py`, `test_math_benchmark.py`, `test_reasoning_benchmark.py`, `test_code_benchmark.py`, and `test_tool_benchmark.py` freeze parsers and scoring; engine/shared tests freeze measurement validators and aggregates; `test_run_accuracy_benchmark.py` freezes partial-response scoring and interruption flushing; and journal/result-store tests freeze partial-output recovery. A behavior in this set changes only with a reviewed methodology or compatibility boundary and a new fixture, never by editing the old expectation.

## Suite map

The Python modules are grouped by responsibility below. The test files themselves are the authoritative, executable detail; this guide intentionally summarizes them instead of duplicating every assertion.

### CLI, selection, and inventory

| Area | Test modules |
|---|---|
| Public option schema, accuracy options, and test-group expansion | [test_benchmark_options.py](../tests/test_benchmark_options.py), [test_benchmark_accuracy_options.py](../tests/test_benchmark_accuracy_options.py), [test_benchmark_expand_tests.py](../tests/test_benchmark_expand_tests.py) |
| Prompt/tg caps | [test_benchmark_max_prompt_tokens_cap.py](../tests/test_benchmark_max_prompt_tokens_cap.py), [test_benchmark_tg_tokens_override.py](../tests/test_benchmark_tg_tokens_override.py) |
| Tier, model, and engine selection | [test_benchmark_select_tier.py](../tests/test_benchmark_select_tier.py), [test_benchmark_filter_models.py](../tests/test_benchmark_filter_models.py), [test_benchmark_model_selectors.py](../tests/test_benchmark_model_selectors.py), [test_benchmark_resolve_custom_models.py](../tests/test_benchmark_resolve_custom_models.py), [test_benchmark_downloaded_models.py](../tests/test_benchmark_downloaded_models.py), [test_benchmark_resolve_engine_names.py](../tests/test_benchmark_resolve_engine_names.py) |
| Conversation eligibility and output paths | [test_benchmark_conv_skip.py](../tests/test_benchmark_conv_skip.py), [test_benchmark_sidecar_path.py](../tests/test_benchmark_sidecar_path.py) |
| Interactive launchers, portable presets, and wrappers | [test_benchmark_frontend.py](../tests/test_benchmark_frontend.py), [test_benchmark_gui.py](../tests/test_benchmark_gui.py), [test_benchmark_presets.py](../tests/test_benchmark_presets.py), [test_run_bench_wrappers.py](../tests/test_run_bench_wrappers.py), [test_shared_console.py](../tests/test_shared_console.py) |
| GUI/terminal/headless mode selection | [test_interface_mode.py](../tests/test_interface_mode.py) |
| Inventory and setup-picker rules | [test_model_inventory.py](../tests/test_model_inventory.py), [test_setup_selection.py](../tests/test_setup_selection.py) |
| Configuration, catalog, and hardware | [test_config.py](../tests/test_config.py), [test_models.py](../tests/test_models.py), [test_hardware.py](../tests/test_hardware.py) |

These tests cover exact and wildcard matching, cumulative tier caps, custom-model discovery, validation before orchestration, saved launcher state, safe non-catalog cleanup targeting, platform wrapper behavior, and hardware memory-fit calculations.

The typed public-option schema supplies the CLI's shared choices and numeric constraints and generates the GUI's defaults, validation metadata, classification, and coverage inventory. The frontend suite also parses `benchmark.py`'s public argparse declarations and requires every flag to appear in that schema as exposed, represented by a more precise UI equivalent, or intentionally excluded. This prevents new CLI controls or changed constraints from disappearing silently.

Run-plan round-trip tests load both standalone plans and plans embedded in CLI results, convert supported values into frontend state, and assert that the shared command builder emits the same measurement-affecting controls. Developer-only sampled plans are tested as an explicit rejection so the GUI cannot silently hide or discard a CLI value.

Run-plan identity tests prove that hierarchical IDs are deterministic, distinct by entity kind and input, and reject entities outside the plan or invalid ordinals. Schema-3 tests cover deterministic workload, runtime, privacy, retry, timeout, and output identities, reject tampering, and preserve schema-2 reads. Execution-validation tests require the complete resolved configuration and model inventory and adversarially reject invalid ranges, types, duplicate sweep values, and empty identities before runtime preparation. A schema-1 golden result is deserialized and reserialized exactly and must reproduce its recorded `plan_id`, preventing the identity upgrade from rewriting historical plans.

Execution-progress tests verify the structured event parser, idempotent model coverage, retry/invalid counters, remaining-time estimates, and recursive child-process resource totals. Runner tests additionally reconstruct an immutable journal plan with a fake engine, record a case through the child execution seam, and verify its committed-event notification and compatible projection. Failure tests reopen and verify prior committed measurements after a runner crash, coordinator interruption, read-only JSON export, and simulated SQLite disk-full abort. Workload tests also assert that an implausible token-rate retry emits recovered or invalid measurement events without changing the one-retry policy.

### Workloads and graders

| Area | Test modules |
|---|---|
| LLM throughput and conversation growth | [test_llm_prefill_benchmark.py](../tests/test_llm_prefill_benchmark.py), [test_llm_conversation_benchmark.py](../tests/test_llm_conversation_benchmark.py) |
| Embeddings and images | [test_embedding_benchmark.py](../tests/test_embedding_benchmark.py), [test_image_benchmark.py](../tests/test_image_benchmark.py) |
| HTTP concurrency | [test_concurrency_benchmark.py](../tests/test_concurrency_benchmark.py) |
| llama.cpp native benchmarks | [test_llamabench_benchmark.py](../tests/test_llamabench_benchmark.py), [test_llamabench_concurrency_benchmark.py](../tests/test_llamabench_concurrency_benchmark.py) |
| MCQ, math, and reasoning | [test_mcq_benchmark.py](../tests/test_mcq_benchmark.py), [test_math_benchmark.py](../tests/test_math_benchmark.py), [test_reasoning_benchmark.py](../tests/test_reasoning_benchmark.py), [test_reasoning_questions.py](../tests/test_reasoning_questions.py) |
| Code and tool use | [test_code_benchmark.py](../tests/test_code_benchmark.py), [test_tool_benchmark.py](../tests/test_tool_benchmark.py) |
| Offline regrading | [test_regrade.py](../tests/test_regrade.py) |

The workload tests emphasize the pure behavior behind orchestration: context planning, prompt construction, output parsing, scoring, timeout recovery, command construction, result-schema shaping, ComfyUI workflow graphs, and question-bank validation. The llama-bench suites mock `subprocess.Popen`, so they can verify exact command matrices and incremental output parsing without loading a model.

### Engines and shared helpers

| Area | Test modules |
|---|---|
| Engine registry, shared llama.cpp tool discovery, and adapter | [test_engines_registry.py](../tests/test_engines_registry.py), [test_llamacpp_tools.py](../tests/test_llamacpp_tools.py), [test_llamacpp_engine.py](../tests/test_llamacpp_engine.py) |
| Measurement contracts and validation | [test_engine_measurements.py](../tests/test_engine_measurements.py) |
| Measured-call and accuracy orchestration | [test_shared_run_measured_calls.py](../tests/test_shared_run_measured_calls.py), [test_run_accuracy_benchmark.py](../tests/test_run_accuracy_benchmark.py) |
| Crash caches and bank versions | [test_shared_crash_cache.py](../tests/test_shared_crash_cache.py), [test_shared_bank_versioning.py](../tests/test_shared_bank_versioning.py) |
| Statistics, prompts, and scoring | [test_shared_stats.py](../tests/test_shared_stats.py), [test_shared_tally_accuracy_entry.py](../tests/test_shared_tally_accuracy_entry.py), [test_shared_looks_like_loop.py](../tests/test_shared_looks_like_loop.py) |
| ComfyUI Python discovery | [test_shared_find_comfyui_python.py](../tests/test_shared_find_comfyui_python.py) |
| ComfyUI installation and managed-model path resolution | [test_comfyui_installation.py](../tests/test_comfyui_installation.py) |
| Versioned setup configuration and path handoff | [test_setup_config.py](../tests/test_setup_config.py) |
| Setup wizard defaults and plan validation | [test_setup_gui.py](../tests/test_setup_gui.py) |
| Atomic results, run state, and 4.1 compatibility | [test_result_store.py](../tests/test_result_store.py), [test_result_compatibility.py](../tests/test_result_compatibility.py) with immutable fixtures in `tests/fixtures/` |
| Serializable plan identity and redaction | [test_run_plan.py](../tests/test_run_plan.py) |
| Transactional event journal, transition safety, and projections | [test_event_store.py](../tests/test_event_store.py) |
| Journal-owned single-shot/conversation/concurrency projection, stage/model-family isolation, batch fields, depth retention, and golden compatibility | [test_llm_event_stage.py](../tests/test_llm_event_stage.py) |
| Journal-owned native llama-bench streamed rows, partial timeouts, and export-failure retention | [test_native_bench_event_stage.py](../tests/test_native_bench_event_stage.py) |
| Resume identity and case-boundary resume/fork policy | [test_resume_policy.py](../tests/test_resume_policy.py) |
| Content-addressed artifact storage, integrity, limits, and failure cleanup | [test_content_store.py](../tests/test_content_store.py) |
| Runner command confinement, event authentication, heartbeat, cleanup escalation, conversation preflight, and parent stage checkpoint dispatch | [test_runner_supervisor.py](../tests/test_runner_supervisor.py), [test_benchmark_runner.py](../tests/test_benchmark_runner.py) |
| Portable result bundles, verification, and CLI | [test_result_bundle.py](../tests/test_result_bundle.py), [test_result_bundle_cli.py](../tests/test_result_bundle_cli.py) |
| Redacted support-bundle allowlist and preview | [test_support_bundle.py](../tests/test_support_bundle.py) |
| Stage ordering and lifecycle policy | [test_orchestration.py](../tests/test_orchestration.py) |

`LlamaCppEngine` HTTP behavior is tested with mocked requests and streams. Measurement tests cover named records, separate timing sources, invalid-sample exclusion, completed-versus-valid counts, medians, and coefficients of variation. Stage tests use fake runners and engines to cover fixed ordering, selection, preparation/execution/cleanup classification, state transitions, engine exclusivity, CPU-mode restoration, and cleanup after failure. Shared workload orchestration covers retries, partial responses, token budgets, loop detection, timeouts, crash-cache behavior, and result diagnostics without network access.

## Dashboard tests

The dashboard uses Vitest and ESLint from its own `node_modules`:

```bash
cd dashboard
npm test
npx vitest -t "getBarStatusLabel"
npm run lint
```

The Vitest suite covers pure transformations in `dashboard/src/utils/*.js` and registry invariants in `dashboard/src/constants.js`: chart data, status labels, sorting, formatting, historical-schema compatibility, model ordering, and color contrast. It deliberately does not mount React components; chart and layout changes also need a rendered dashboard check against a sample or relevant results file.

## What to run before submitting

```bash
bash tests.sh
cd dashboard
npm test
npm run lint
```

Run the dashboard commands whenever `dashboard/src` changed. For a Python-only change, the pytest suite is sufficient unless the results schema or documented dashboard behavior also changed.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
