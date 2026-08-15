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

The wrapper installs or updates [tests/requirements.txt](../tests/requirements.txt) before invoking pytest. If `bench-env/` does not exist, run the appropriate setup script first. Importing `scripts/setup/setup_check.py` is safe, but never call its `main()` as a test shortcut because that starts the real interactive installation flow.

The scripts tree is a package rather than a flat import directory. A structural test parses every Python module and rejects bare imports of another project module, preventing delayed GUI/setup paths from passing unit collection and then failing only when launched through `python -m`.

Tk controller tests use fake widgets and variables on every platform. Screen-construction and full-application smoke tests use real Tk, skip when no display is available locally, and run under Xvfb in the Linux `Python tests (Tk/Xvfb)` CI job.

[tests/conftest.py](../tests/conftest.py) prevents llama.cpp discovery tests from reading the machine's real saved setup configuration, so running setup cannot change mocked discovery outcomes. Project modules use package-qualified imports, matching the `python -m scripts.<package>.<module>` entry points.

`conftest.py` also provides the `symlink_or_skip` fixture. Tests covering symlink-escape defenses need a real symlink, which Windows refuses without Developer Mode or administrator rights, so the fixture creates one and skips the test when the platform will not. Use it instead of calling `Path.symlink_to` directly. The skip is limited to platforms that cannot create the link: on Linux and macOS these tests always run, and a Windows skip is not a silent hole because the behavior under test is POSIX symlink semantics, which Windows junctions do not reproduce faithfully.

## Coverage and safety boundaries

Install `pytest-cov` into `bench-env/`, then run:

```bash
bench-env/bin/pip install pytest-cov
bash tests.sh --cov=scripts --cov-report=term-missing
```

On Windows, use `bench-env\Scripts\pip.exe` and `tests.bat`.

Live orchestration functions are marked `# pragma: no cover`. The excluded functions start real subprocesses, poll llama.cpp or ComfyUI, or drive an entire benchmark run. Pure decisions and calculations are extracted and tested instead. Treat the missing-line report as the useful signal; a fixed percentage is not a project target.

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
| Quick and dry-run plan resolution, formatting, and exact-history ETA | [test_benchmark_quick_preset.py](../tests/test_benchmark_quick_preset.py), [test_benchmark_dry_run.py](../tests/test_benchmark_dry_run.py) |
| Tier, model, engine selection, and custom imports | [test_benchmark_select_tier.py](../tests/test_benchmark_select_tier.py), [test_benchmark_filter_models.py](../tests/test_benchmark_filter_models.py), [test_benchmark_model_selectors.py](../tests/test_benchmark_model_selectors.py), [test_benchmark_resolve_custom_models.py](../tests/test_benchmark_resolve_custom_models.py), [test_benchmark_downloaded_models.py](../tests/test_benchmark_downloaded_models.py), [test_benchmark_resolve_engine_names.py](../tests/test_benchmark_resolve_engine_names.py), [test_custom_models.py](../tests/test_custom_models.py), [test_model_import.py](../tests/test_model_import.py), [test_model_download.py](../tests/test_model_download.py) |
| Conversation eligibility, output paths, interrupt exit status, and exact-plan fork provenance/overwrite guards | [test_benchmark_conv_skip.py](../tests/test_benchmark_conv_skip.py), [test_benchmark_sidecar_path.py](../tests/test_benchmark_sidecar_path.py), [test_benchmark_run_state.py](../tests/test_benchmark_run_state.py) |
| Interactive launchers, GUI controllers/screens, portable presets, recovery presentation/commands, and wrappers | [test_benchmark_frontend.py](../tests/test_benchmark_frontend.py), [test_benchmark_gui.py](../tests/test_benchmark_gui.py), [test_benchmark_gui_controllers.py](../tests/test_benchmark_gui_controllers.py), [test_benchmark_gui_tk_screens.py](../tests/test_benchmark_gui_tk_screens.py), [test_benchmark_presets.py](../tests/test_benchmark_presets.py), [test_run_bench_wrappers.py](../tests/test_run_bench_wrappers.py), [test_shared_console.py](../tests/test_shared_console.py) |
| GUI/terminal/headless mode selection | [test_interface_mode.py](../tests/test_interface_mode.py) |
| Cross-process pause state, transition evidence, lost-control fallback, launch cleanup, blocking, and measured-call boundaries | [test_pause_control.py](../tests/test_pause_control.py), [test_benchmark_gui.py](../tests/test_benchmark_gui.py), [test_shared_run_measured_calls.py](../tests/test_shared_run_measured_calls.py) |
| Inventory, model identity, and setup-picker rules | [test_model_inventory.py](../tests/test_model_inventory.py), [test_model_identity.py](../tests/test_model_identity.py), [test_setup_selection.py](../tests/test_setup_selection.py) |
| Configuration, catalog, dashboard-boundary invariants, and hardware | [test_config.py](../tests/test_config.py), [test_models.py](../tests/test_models.py), [test_cross_boundary_invariants.py](../tests/test_cross_boundary_invariants.py), [test_hardware.py](../tests/test_hardware.py) |

These tests cover exact and wildcard matching, cumulative tier caps, custom-model discovery, validation before orchestration, saved launcher state, safe non-catalog cleanup targeting, platform wrapper behavior, and hardware memory-fit calculations.

The typed public-option schema supplies the CLI's shared choices and numeric constraints and generates the GUI's defaults, validation metadata, classification, and coverage inventory. The frontend suite also parses `benchmark.py`'s public argparse declarations and requires every flag to appear in that schema as exposed, represented by a more precise UI equivalent, or intentionally excluded. This prevents new CLI controls or changed constraints from disappearing silently.

Run-plan round-trip tests load both standalone plans and plans embedded in CLI results, convert supported values into frontend state, and assert that the shared command builder emits the same measurement-affecting controls. Developer-only sampled plans are tested as an explicit rejection so the GUI cannot silently hide or discard a CLI value.

Run-plan identity tests prove that hierarchical IDs are deterministic, distinct by entity kind and input, and reject entities outside the plan or invalid ordinals. Schema-3 tests cover deterministic workload, runtime, privacy, retry, timeout, and output identities, reject tampering, and preserve schema-2 reads. Execution-validation tests require the complete resolved configuration and model inventory, accept the image catalog's short-only identity, and adversarially reject invalid ranges, types, duplicate sweep values, and missing family-specific identities before runtime preparation. A schema-1 golden result is deserialized and reserialized exactly and must reproduce its recorded `plan_id`, preventing the identity upgrade from rewriting historical plans.

Execution-progress tests verify the structured event parser, idempotent model coverage, retry/invalid counters, remaining-time estimates, and recursive child-process resource totals. Runner tests additionally reconstruct an immutable journal plan with a fake engine, record a case through the child execution seam, and verify its committed-event notification and compatible projection. Failure tests reopen and verify prior committed measurements after a runner crash, coordinator interruption, read-only JSON export, and simulated SQLite disk-full abort. Workload tests also assert that an implausible token-rate retry emits recovered or invalid measurement events without changing the one-retry policy.

### Workloads and graders

| Area | Test modules |
|---|---|
| LLM throughput and conversation growth | [test_llm_prefill_benchmark.py](../tests/test_llm_prefill_benchmark.py), [test_llm_conversation_benchmark.py](../tests/test_llm_conversation_benchmark.py) |
| Embeddings and images | [test_embedding_benchmark.py](../tests/test_embedding_benchmark.py), [test_image_benchmark.py](../tests/test_image_benchmark.py) |
| HTTP concurrency | [test_concurrency_benchmark.py](../tests/test_concurrency_benchmark.py) |
| llama.cpp native benchmarks | [test_llamabench_benchmark.py](../tests/test_llamabench_benchmark.py), [test_llamabench_concurrency_benchmark.py](../tests/test_llamabench_concurrency_benchmark.py) |
| vLLM native benchmark | [test_vllm_benchmark.py](../tests/test_vllm_benchmark.py) |
| Non-catalog vLLM cache cleanup | [test_vllm_cleanup.py](../tests/test_vllm_cleanup.py) |
| MCQ, math, and reasoning | [test_mcq_benchmark.py](../tests/test_mcq_benchmark.py), [test_math_benchmark.py](../tests/test_math_benchmark.py), [test_reasoning_benchmark.py](../tests/test_reasoning_benchmark.py), [test_reasoning_questions.py](../tests/test_reasoning_questions.py) |
| Code and tool use | [test_code_benchmark.py](../tests/test_code_benchmark.py), [test_tool_benchmark.py](../tests/test_tool_benchmark.py) |
| Offline regrading | [test_regrade.py](../tests/test_regrade.py) |

The workload tests emphasize the pure behavior behind orchestration: context planning, prompt construction, output parsing, scoring, timeout recovery, command construction, result-schema shaping, ComfyUI workflow graphs, and question-bank validation. The llama-bench suites mock `subprocess.Popen`, so they can verify exact command matrices and incremental output parsing without loading a model.

### Engines and shared helpers

| Area | Test modules |
|---|---|
| Engine registry, shared llama.cpp tool discovery, OpenAI-compatible HTTP/SSE parsing, and adapters | [test_engines_registry.py](../tests/test_engines_registry.py), [test_llamacpp_tools.py](../tests/test_llamacpp_tools.py), [test_openai_api.py](../tests/test_openai_api.py), [test_llamacpp_engine.py](../tests/test_llamacpp_engine.py) |
| Measurement contracts and validation | [test_engine_measurements.py](../tests/test_engine_measurements.py) |
| Measured-call and accuracy orchestration | [test_shared_run_measured_calls.py](../tests/test_shared_run_measured_calls.py), [test_run_accuracy_benchmark.py](../tests/test_run_accuracy_benchmark.py) |
| Crash caches and bank versions | [test_shared_crash_cache.py](../tests/test_shared_crash_cache.py), [test_shared_bank_versioning.py](../tests/test_shared_bank_versioning.py) |
| Statistics, prompts, scoring, and loop detection | [test_shared_stats.py](../tests/test_shared_stats.py), [test_accuracy_scoring.py](../tests/test_accuracy_scoring.py), [test_shared_looks_like_loop.py](../tests/test_shared_looks_like_loop.py) |
| ComfyUI Python discovery | [test_shared_find_comfyui_python.py](../tests/test_shared_find_comfyui_python.py) |
| ComfyUI installation and managed-model path resolution | [test_comfyui_installation.py](../tests/test_comfyui_installation.py) |
| ComfyUI setup services and directory transactions | [test_comfyui_assets.py](../tests/test_comfyui_assets.py), [test_comfyui_install.py](../tests/test_comfyui_install.py), [test_comfyui_runtime.py](../tests/test_comfyui_runtime.py), [test_directory_transaction.py](../tests/test_directory_transaction.py) |
| Versioned setup configuration and path handoff | [test_setup_config.py](../tests/test_setup_config.py) |
| Setup wizard defaults and plan validation | [test_setup_gui.py](../tests/test_setup_gui.py) |
| Setup console, discovery, coordinator safety, credentials, and llama.cpp installation | [test_setup_console.py](../tests/test_setup_console.py), [test_setup_discovery.py](../tests/test_setup_discovery.py), [test_setup_coordinator_structure.py](../tests/test_setup_coordinator_structure.py), [test_hf_credentials.py](../tests/test_hf_credentials.py), [test_llamacpp_install.py](../tests/test_llamacpp_install.py) |
| CUDA toolkit plan gating and install execution | [test_cuda_install.py](../tests/test_cuda_install.py) |
| vLLM platform support, interpreter resolution, install commands | [test_vllm_install.py](../tests/test_vllm_install.py) |
| Managed runtime update validation, replacement, rollback, and cancellation | [test_runtime_update.py](../tests/test_runtime_update.py) |
| Engine picker defaults, disabled engines, install fan-out | [test_engine_selection.py](../tests/test_engine_selection.py) |
| Atomic results, run/recovery state, terminal-history retention, and 4.1 compatibility | [test_result_store.py](../tests/test_result_store.py), [test_result_compatibility.py](../tests/test_result_compatibility.py) with immutable fixtures in `tests/fixtures/` |
| Serializable plan identity and redaction | [test_run_plan.py](../tests/test_run_plan.py) |
| Canonical JSON encoding and durable identity hashing | [test_canonical_json.py](../tests/test_canonical_json.py) |
| Neutral methodology profile and effective optimization inventory | [test_methodology_profile.py](../tests/test_methodology_profile.py) |
| Local project workflows, portability boundaries, validation, and round trips | [test_benchmark_project.py](../tests/test_benchmark_project.py) |
| Local result discovery, filtering, metric extraction, comparison compatibility, and noise-aware evidence labels | [test_result_history.py](../tests/test_result_history.py), [test_significance.py](../tests/test_significance.py) |
| Repeated-trial pooling, intervals, drift, pairing, verdicts, and artifact CLI | [test_trial_set.py](../tests/test_trial_set.py) |
| Memory/power qualification runner argument validation and alternating dry-run order | [test_telemetry_trial_runner.py](../tests/test_telemetry_trial_runner.py) |
| Power parser fixtures, discovery/permission states, shared-timeline sampling, uneven-timestamp integration, scope-safe run totals, efficiency derivation, and unavailable handling | [test_power_telemetry.py](../tests/test_power_telemetry.py) with captured fixtures in `tests/fixtures/power/` |
| Outbound identity preview, private aliases, source digests, and export acknowledgement | [test_outbound_metadata.py](../tests/test_outbound_metadata.py), [test_result_bundle_cli.py](../tests/test_result_bundle_cli.py), [test_decision_report.py](../tests/test_decision_report.py) |
| Offline loopback classification, socket blocking, environment controls, CLI inventory, concrete frontend control bindings, and plan identity | [test_network_policy.py](../tests/test_network_policy.py), [test_benchmark_frontend.py](../tests/test_benchmark_frontend.py), [test_run_plan.py](../tests/test_run_plan.py) |
| Generated-code static restrictions, kernel/parent memory bounds, output/time limits, and filesystem denial | [test_code_sandbox.py](../tests/test_code_sandbox.py) |
| Chunked release secret scanning, credential containers, log redaction, and non-disclosure of matched values | [test_security_gate.py](../tests/test_security_gate.py), [test_log_redaction.py](../tests/test_log_redaction.py) |
| Vendor diagnostic first divergence, raw invalidity, determinism, review, and source verification | [test_vendor_diagnostic.py](../tests/test_vendor_diagnostic.py) |
| Transactional event journal, job/stage terminal and recovery transitions, selected recovery scope, attempt abandonment, and projections | [test_event_store.py](../tests/test_event_store.py) |
| Journal-owned single-shot/conversation/concurrency projection, failed-case classification, pending-level selection, all/selected recovery attempts, stage/model-family isolation, batch fields, depth retention, and golden compatibility | [test_llm_event_stage.py](../tests/test_llm_event_stage.py), [test_llm_conversation_benchmark.py](../tests/test_llm_conversation_benchmark.py), [test_concurrency_benchmark.py](../tests/test_concurrency_benchmark.py) |
| Journal-owned native llama-bench streamed rows, grouped remaining sweeps, partial timeouts, and export-failure retention | [test_native_bench_event_stage.py](../tests/test_native_bench_event_stage.py) |
| Resume identity, startup-cache behavior, cache-bypassing recovery hashes, and case-boundary resume/fork policy | [test_resume_policy.py](../tests/test_resume_policy.py) |
| Read-only recovery eligibility, ordered retry candidates, coverage, completed-result rejection, identity drift, and non-mutation | [test_recovery_inspector.py](../tests/test_recovery_inspector.py) |
| State-changing journal-only recovery/selected retry/fork, attempt continuation, all-stages-complete finalization, journal/JSON terminal agreement, unselected/source preservation, overwrite rejection, truthful interruption, atomic completion, and pre-mutation rejection | [test_recovery_executor.py](../tests/test_recovery_executor.py) |
| Content-addressed artifact storage, integrity, limits, and failure cleanup | [test_content_store.py](../tests/test_content_store.py) |
| Runner command confinement, event authentication, heartbeat, cleanup escalation, conversation preflight, and parent stage checkpoint dispatch | [test_runner_supervisor.py](../tests/test_runner_supervisor.py), [test_benchmark_runner.py](../tests/test_benchmark_runner.py) |
| Portable result bundles, verification, and CLI | [test_result_bundle.py](../tests/test_result_bundle.py), [test_result_bundle_cli.py](../tests/test_result_bundle_cli.py) |
| Decision-report evidence, escaping, deterministic HTML/PDF, and CLI | [test_decision_report.py](../tests/test_decision_report.py) |
| Acceptance-policy schema, evidence resolution, methodology compatibility, and CLI | [test_acceptance_policy.py](../tests/test_acceptance_policy.py) |
| Redacted support-bundle allowlist and preview | [test_support_bundle.py](../tests/test_support_bundle.py) |
| Restricted planning-file privacy and ignore policy | [test_private_plan_privacy.py](../tests/test_private_plan_privacy.py) |
| Fixed release scan commands, evidence, findings, and missing-tool failures | [test_release_scans.py](../tests/test_release_scans.py) |
| Version-mirror sync, mirror-edit rejection, and drift repair for the pre-commit hook | [test_version_sync.py](../tests/test_version_sync.py) |
| Stage registry, ordering, and lifecycle policy | [test_stage_registry.py](../tests/test_stage_registry.py), [test_orchestration.py](../tests/test_orchestration.py) |

`LlamaCppEngine` HTTP behavior is tested with mocked requests and streams. Measurement tests cover named records, separate timing sources, invalid-sample exclusion, completed-versus-valid counts, medians, and coefficients of variation. Stage tests use fake runners and engines to cover fixed ordering, selection, preparation/execution/cleanup classification, state transitions, engine exclusivity, CPU-mode restoration, and cleanup after failure. Shared workload orchestration covers retries, partial responses, token budgets, loop detection, timeouts, crash-cache behavior, and result diagnostics without network access.

Telemetry tests cover empty and failed channels, retained lifecycle sub-windows, weighted case aggregation, every headroom threshold boundary, sampler failure containment, window changes, thread cleanup after exceptions, normalized power-source discovery, captured platform formats, shared timestamps, RAPL deltas, one persistent macOS reader, uneven integration, scope rejection, and workload efficiency. Existing GUI resource-query tests exercise the same functions through imports from the shared runtime module.

## Dashboard tests

The dashboard is TypeScript (see [Dashboard](dashboard.md)) and uses Vitest, ESLint, and `tsc` from its own `node_modules`:

```bash
cd dashboard
npm test
npx vitest -t "getBarStatusLabel"
npm run lint
npx tsc --noEmit
```

The Vitest suite covers pure transformations in `dashboard/src/utils/*.ts`, selected-result staging, and registry invariants in `dashboard/src/constants.ts`: chart data, status labels, sorting, formatting, memory/power telemetry and missing states, mixed-scope power rejection, sample-validity inspection, historical-schema compatibility, model ordering, color contrast, build-time suite-version parsing, and the bounded local-file autoload handoff. `samples.test.ts` loads every bundled result and verifies that each populated section still produces rows, catching schema drift at the compatibility boundary. The validity tests prove that invalid runs remain distinct from zero, rejection reasons survive, and aggregate-only historical files are labeled rather than assigned invented samples. The suite deliberately does not mount React components; chart and layout changes also need a rendered dashboard check against a sample or relevant results file.

Run `npm run test:coverage` to measure statement, branch, function, and line coverage across `src/constants.ts` and the pure utilities in `src/utils/`. CI publishes the text report for every protected-target pull request; coverage is diagnostic rather than a fixed percentage gate.

## What to run before submitting

```bash
bash tests.sh
cd dashboard
npm test
npm run lint
npx tsc --noEmit
```

Run the dashboard commands whenever `dashboard/src` changed. For a Python-only change, the pytest suite is sufficient unless the results schema or documented dashboard behavior also changed.

Pull requests targeting `develop`, `release/**`, or `main` run the full Python suite with Tk enabled under Xvfb, Pyright, and the dashboard's Vitest, ESLint, TypeScript, and `any`-ratchet checks. The jobs have stable names so repository rules can require each check independently.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
