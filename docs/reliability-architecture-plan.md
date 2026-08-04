[← Back to README](../README.md)

# Reliability Architecture Plan

**Status:** All three increments implemented during v4.1 development.

## Purpose

This plan improves confidence that a benchmark result is durable, reproducible, semantically unambiguous, and produced by orchestration that can be tested without launching real inference services. It deliberately preserves the existing workload definitions, model tiers, early-exit policies, engine behavior, dashboard comparisons, and default run counts unless a later reviewed change explicitly says otherwise.

## Goals

- Prevent interrupted writes from leaving a corrupt result, answer sidecar, or crash cache.
- Make every result self-describing enough to reproduce or explain the run later.
- Distinguish complete, partial, skipped, timed-out, crashed, interrupted, and failed work without forcing consumers to infer state from missing keys.
- Give every timing value one stable meaning across engines and workloads.
- Separate orchestration policy from CLI parsing and process lifecycle so stage transitions can be unit tested.
- Keep older result files readable and comparable in the dashboard.

## Non-goals

- Changing prompts, question banks, scoring rules, context checkpoints, concurrency levels, model tiers, or slow-model thresholds.
- Increasing the default number of measured runs or adding outlier removal.
- Replacing raw samples with aggregate statistics.
- Making live model or server execution part of the unit test suite.
- Building a general workflow framework outside the needs of this benchmark.

## Compatibility and rollout rules

All new result fields are additive. Existing workload section names and measurement fields remain readable, and dashboard code continues to treat missing metadata as an older result rather than an error. The top-level `version` remains the application release version mirrored by `config.VERSION` and the README title. New results add `run.schema_version` as the only results-schema compatibility axis; schema changes bump that value without requiring an application-version bump, while an application release still bumps `config.VERSION` and the README together. Consumers use `run.schema_version` when present and fall back to the historical top-level `version` only for files that predate it.

Each increment is independently releasable and must pass the Python suite plus dashboard tests and lint when dashboard code changes. A compatibility fixture representing a pre-change result must remain loadable after every increment. No migration rewrites existing files in place.

## Increment 1: Durable, self-describing results

### Outcome

A saved result survives process interruption at write boundaries, says exactly what was requested and completed, and exposes partial-run state without changing the meaning of existing benchmark measurements.

### 1.1 Central atomic JSON persistence

Add a shared JSON persistence helper used by main results, accuracy answer sidecars, crash caches, and `regrade.py` outputs. The helper serializes with `allow_nan=False`, creates a temporary file in the destination directory, flushes and `fsync`s it, applies `os.replace()` to the destination, and syncs the parent directory where the platform supports it. Creating the temporary file beside the destination keeps replacement on the same filesystem and therefore atomic on filesystems that provide atomic replacement.

Parent-directory creation is explicit. A failed checkpoint must be logged and raised for the main results or answer sidecars because continuing would falsely imply that progress is safely recorded; crash-cache persistence may retain its current best-effort warning behavior because the cache is an optimization rather than the benchmark record. There is no non-atomic overwrite fallback: if replacement or durability operations fail or are unsupported by the mounted filesystem, the writer reports failure and leaves the last valid destination checkpoint in place where the filesystem permits. This protects recoverability on external, network, and exFAT-like volumes without claiming power-loss guarantees that the filesystem cannot provide.

Non-finite numeric values are data-integrity failures, not values to coerce silently. Serialization reports the offending result path, rejects that section update, preserves the prior valid checkpoint, and attempts to record the stage and run as `failed` with reason `invalid_numeric_value` without retaining the invalid payload. If even the terminal-state checkpoint cannot be persisted, the process exits unsuccessfully and reports that the prior checkpoint is the recovery point.

Focused tests cover valid replacement, replacement of an existing file, strict rejection of NaN, cleanup after serialization/write/replace failures, preservation of the prior destination when replacement fails, Unicode content, and parent-directory creation. Platform-specific implementation details are hidden behind the helper rather than duplicated at callers.

### 1.2 Run manifest

Add a top-level `run` object to new results. It records a generated `run_id`, `schema_version`, `started_at` and eventual `finished_at` UTC timestamps, overall `status`, requested tests in user-facing order, effective stage order, selected engine, selected model identifiers by workload family, effective CLI-controlled settings, warmup count, and benchmark source identity.

Source identity contains the Git commit when discoverable and a `dirty` boolean without including diffs, filenames, tokens, environment variables, or other repository contents. If Git metadata is unavailable, both values are `null` and the run remains valid. Effective settings include only non-secret parameters that affect measurements or scoring. Secret values and secret-bearing sources such as `hf.txt`, environment variables, credentials, authorization headers, and connector configuration are always excluded; user-specific filesystem paths, including input, model, ComfyUI, and output paths, are excluded or reduced to non-sensitive logical identifiers. Display-only frontend choices are also excluded.

Model identity initially records the catalog tag, short key, and configured size metadata already available without reading large files. Artifact byte size and a cheap stable identity supplied by the engine inventory may be added when available. Full multi-gigabyte model hashing is explicitly deferred because it would add substantial startup cost; the manifest must label any non-content identity honestly rather than call it a checksum.

Existing `profile`, `accuracy_settings`, `bank_versions`, and `sample_ids` remain in place for compatibility. Increment 1 may duplicate their essential values inside `run.effective_config`, but they are not removed or renamed.

### 1.3 Explicit run and section state

Use a small closed vocabulary for top-level and stage state: `running`, `complete`, `partial`, `interrupted`, and `failed`. A selected stage gets an entry in `run.stages`; an unselected stage gets no stage record. Each selected stage records `status`, `started_at`, `finished_at`, and an optional short machine-readable `reason` from a documented vocabulary.

`complete` means the stage orchestrator reached its normal end, even when individual models were legitimately skipped under existing policy. `partial` means usable data exists but the stage did not reach its normal end. `interrupted` is reserved for signal or keyboard interruption. `failed` means an unhandled error or data-integrity failure prevented normal completion. Existing per-model fields such as `skipped`, `skip_reason`, `timed_out`, and `crashed` remain authoritative for whether an individual model result is usable; stage state is authoritative only for whether stage orchestration finished.

Every terminal stage record includes explicit `selected_models`, `models_with_results`, `models_skipped`, and `models_failed` counts so consumers do not derive stage coverage from section shape. A stage with every selected model skipped is therefore `complete` with zero result models and a nonzero skipped count. Consumer precedence is fixed: top-level run state answers whether the invocation finished, stage state and counts answer whether the selected stage finished and how much coverage it produced, and per-model state answers whether a particular model measurement is usable. A broader `complete` state never overrides a narrower skip, timeout, or crash record.

The result is checkpointed with overall status `running` before the first stage. A stage transitions to `running` before invocation and to its terminal state afterward, with a checkpoint at each transition. Normal completion sets `finished_at` and `complete`. Signal cleanup first checkpoints `interrupted`; after stack unwinding runs workload `finally` callbacks, orchestration checkpoints the interrupted result again so their latest in-memory model data is durable. An unhandled exception records `failed` and propagates. If some selected stages finish and later execution fails, the overall state is `partial` only for an intentionally handled early stop; otherwise it is `failed`, while completed stage records remain intact.

Within every model-based stage, completing one model is a required checkpoint boundary. The stage writes the accumulated section and refreshed coverage counts after each model reaches a terminal outcome, including a successful result, policy skip, timeout, crash, or handled failure; with three selected LLMs, for example, the main results file is durably replaced at least three times during that stage. Accuracy stages checkpoint the matching answer sidecar at the same boundary so its answers cannot lag the main result's completed-model state. The checkpoint occurs after model cleanup is attempted, and a cleanup warning is recorded without discarding the model's already completed measurements.

Per-model checkpointing is the minimum durability contract, not a prohibition on finer checkpoints. A workload may also checkpoint after a naturally expensive subunit such as an LLM context depth, accuracy-question batch, concurrency level, or image resolution when losing that subunit would be costly, but those finer boundaries require workload-specific state semantics and tests. Increment 1 standardizes the existing per-model callback behavior first and does not introduce per-question writes or a configurable checkpoint-frequency flag.

The two native llama.cpp workloads have an additional partial-result contract. `llamabench` runs one prefill and one decode `-o jsonl` sweep per model, passing the configured repetition count through so the model remains loaded across cases and repetitions. Each completed JSONL case row is checkpointed immediately, so a timeout loses only the active and later cases in that sweep rather than every earlier result. Completed repetitions, completed cases, and any completed prefill work remain in the model result alongside requested/completed repetition and case counts, `timed_out`, and `timed_out_at` metadata; a timeout must never replace successful entries with an error-only object.

The suite retains llama-bench's internal repetition samples and existing per-case `avg_ts` and `stddev_ts` fields without outlier removal. llama-bench does not emit an unfinished case, so completed internal repetitions from the active timed-out case cannot be recovered; every earlier completed case remains chartable with explicit requested and completed repetition counts. The manifest records the streamed internal-repetition mode, and comparisons with the briefly used per-case-process mode receive a methodology warning rather than being presented as strictly equivalent.

`llamabenchconc` has no repetition flag and streams one JSONL entry per completed matrix case. Its runner retains parsed entries as they arrive and returns or raises a structured partial outcome containing them when an idle timeout or later process failure occurs. The model result saves those entries with the requested and completed matrix-case counts plus terminal diagnostics. A timeout before the first valid entry remains an error-only outcome, while a timeout after one or more entries is partial usable data.

Native benchmark aggregation and dashboard consumers distinguish completeness from usability: completed cases and entries remain chartable, model-level timeout metadata triggers an incomplete warning, and no missing repetition or matrix case is synthesized as zero. Tests cover successful prefill followed by decode timeout, successful early matrix cases followed by timeout in the same streamed sweep, streamed concurrency rows followed by timeout, timeout before any result, preservation of llama-bench's internal repetition samples, methodology comparison warnings, and checkpoint calls after each newly durable native result.

### 1.4 Dashboard behavior

Older files with no `run` object load exactly as today. New files with `running`, `partial`, `interrupted`, or `failed` state display a compact warning near file import or file identity; charts still render all valid measurements present. Comparison warnings distinguish unknown legacy state from an explicitly incomplete new run without blocking comparison.

Pure helpers determine a file's reliability warning and receive Vitest coverage for legacy, complete, partial, interrupted, failed, and malformed metadata. Any visible warning is verified in the running dashboard with a sample legacy file and synthetic incomplete file.

### 1.5 Documentation and acceptance criteria

Update `project-structure.md` with the additive schema, `how-it-works.md` with checkpoint state transitions, `dashboard.md` with incomplete-run warnings, and `testing.md` with atomic-writer and compatibility coverage. Update sample data only if needed to exercise the visible dashboard state; do not convert every historical sample.

Increment 1 is accepted when all JSON-producing paths use the central writer, a forced write failure cannot destroy an existing valid destination in unit tests, every model terminal outcome checkpoints the accumulated stage data and coverage counts, accuracy results and answer sidecars advance together at that boundary, native llama.cpp timeouts preserve and checkpoint every completed case or streamed matrix entry, a synthetic interruption leaves parseable JSON marked `interrupted`, a normal mocked stage sequence ends `complete`, an unhandled mocked stage failure ends `failed` while preserving earlier stage and model state, legacy files still load, and all required test and lint commands pass.

## Increment 2: Unambiguous measurement contracts

### Outcome

Workloads consume named measurement records rather than positional tuples, every timing field has one documented clock and interval, and results preserve enough raw information to audit aggregates.

### 2.1 Typed engine results

Introduce focused dataclasses or typed records in `engines/base.py`, such as `GenerationMeasurement`, `ChatMeasurement`, and `EmbeddingMeasurement`. They use named fields for client-observed TTFT, server-reported prompt evaluation duration when available, client wall duration, decode duration, generated token count, prompt token count, finish reason, response payload, and engine diagnostics.

The base interface returns these records. `LlamaCppEngine` is adapted first, and workload code accesses names rather than tuple positions. Optional engine-reported fields remain `None` when unavailable; adapters never substitute one timing source into a differently named field.

### 2.2 Timing definitions

Client TTFT is measured with `time.perf_counter()` from immediately before the HTTP request is opened until the first generated content, reasoning content, or tool-call fragment is observed. Model loading and server restart time are excluded and reported separately when relevant. Server prompt time comes only from the engine's timing payload. Client wall duration spans request start through stream completion. Decode duration uses server-reported generation time when present and otherwise a clearly labeled client estimate.

The single-shot and conversation workloads retain their intentionally different cache semantics, but both identify which TTFT source they publish. Existing public fields remain during a compatibility window; new explicit fields are added beside them, and the dashboard prefers explicit fields while falling back to legacy fields.

### 2.3 Validation and aggregation

Move timing sanity checks into pure validation functions: finite non-negative durations, non-negative integral token counts, TTFT not greater than wall duration within a documented tolerance, and decode-rate consistency. Invalid samples are retained in diagnostics but excluded from aggregates and counted explicitly.

Each aggregate records `requested_runs`, `completed_runs`, `valid_runs`, and raw valid samples. Existing `n_runs` keeps its historical meaning as the completed-sample count; it does not become an alias for `valid_runs`. New consumers use the explicit counts, while old consumers continue to see the same `n_runs` semantics. Partial call failures and invalid measurements no longer require a consumer to infer completeness or validity from array length.

Add median and coefficient of variation only where at least two valid samples exist. Means and standard deviations remain unchanged, no outliers are dropped, and no automatic instability rejection is introduced. An `unstable` flag may be added only after a separately reviewed threshold is documented and tested; it is not part of the initial increment.

### 2.4 Tests and migration

Engine tests assert exact clock boundaries with mocked clocks and streams, including empty first chunks, reasoning-only output, tool fragments, missing server timings, malformed timings, timeouts, and model reloads. Workload tests assert named-field use and aggregation of complete, partial, and invalid samples. Dashboard tests cover new fields and legacy fallback.

Increment 2 is accepted when positional measurement tuples are gone from the engine interface, client and server timing sources cannot overwrite one another, invalid measurements are visible but excluded from aggregates, requested/completed/valid counts are explicit, legacy results still render, and all required test and lint commands pass.

## Increment 3: Testable orchestration boundaries

### Outcome

The CLI translates arguments into an immutable run specification, a stage runner executes declarative workload stages, and result/checkpoint policy is testable with fake stages and engines while live lifecycle methods remain excluded from unit coverage.

### 3.1 Run specification and context

Create an immutable run specification containing validated engines, selected workloads and models, effective configuration, paths, and flags. This initially landed as `RunSpec`; commercial-kernel work subsequently split it into a serializable, path-free `RunPlan` and local-only `RunPaths`. `RunContext` contains those objects with the current engine, hardware profile, result store, and lifecycle services. CLI parsing and model resolution build the plan before any server starts. Existing workload calls may continue to receive explicit resolved inputs from their local runner closures rather than forcing a second dependency-injection layer.

Neither object becomes a general dependency container. It includes only data currently threaded through `main()` or captured by nested checkpoint functions, and workload classes continue to accept explicit inputs where that keeps them independently testable.

### 3.2 Result store

Route stage transitions, checkpoints, terminal run state, and atomic persistence through a `ResultStore`. The CLI retains the readable top-level result schema construction because it already assembles the profile, bank versions, effective settings, and empty workload sections in one place. Once constructed, the store owns persistence and validates legal state transitions through narrow operations such as `start_stage`, `update_section`, `complete_stage`, and `finish`.

The store does not know how a workload runs and does not swallow workload exceptions. Existing workload `save_fn` callbacks preserve Increment 1's per-model and finer-grained checkpoint contracts by passing the accumulated section through `update_section`; adding a second model-terminal API would duplicate that working boundary. Unit tests cover legal and illegal transitions, validation before mutation, coverage-count refreshes, cleanup-failure metadata, and preservation of completed sections after failure.

### 3.3 Declarative stage registry

Represent the fixed execution order as `StageDefinition` entries with a key, result section, selected-model count, engine requirement, preparation hook, runner, and cleanup hook. Accuracy workloads may share a stage factory because their wiring is already symmetric. `llamabench`, `llamabenchconc`, and images retain their special lifecycle behavior through explicit hooks rather than hidden conditionals: both native llama.cpp stages stop the HTTP engine and bypass `InferenceEngine`, while images stop it and use ComfyUI.

`benchmark_frontend.py` is a pre-run command builder and launcher, not a benchmark stage, so it stays outside the stage registry. Its responsibility ends after producing and launching the validated CLI command; registry tests separately confirm that the resulting selected test keys map to the intended stages.

The registry must remain easy to read as the authoritative execution order; it should not introduce dynamic plugin discovery or configuration-driven imports. Engine validation and multi-engine image-once behavior remain explicit policies with focused tests.

### 3.4 Lifecycle and failure policy

Introduce a small lifecycle coordinator for stopping competing engines, switching CPU-only mode, unloading models, freeing ComfyUI resources, and shutting down processes started by the suite. Ownership is tracked so cleanup does not terminate unrelated processes except where current documented behavior intentionally stops a selected inference engine.

Stage preparation, execution, and cleanup failures are classified separately in result metadata. Cleanup runs through `finally`, signal handlers request interruption through the coordinator, and unhandled-exception teardown completes before the terminal checkpoint attempt so persistence failure cannot strand managed processes. Live subprocess and server methods remain `# pragma: no cover`; coordinator decisions are tested with fakes.

### 3.5 Orchestration tests and documentation

Add fake-engine and fake-stage tests for fixed ordering, selection, multi-engine passes, images running once, server-stop boundaries around native llama.cpp tools and images, stage checkpoint transitions, unhandled exceptions, interruption, and cleanup after every path. `partial` remains reserved for a future intentionally handled early-stop outcome; this increment does not manufacture it from failed execution. Do not invoke `benchmark.py`, setup entrypoints, real servers, or real model files.

Update `how-it-works.md`, `project-structure.md`, `engines.md`, and `testing.md` after the refactor. Increment 3 is accepted when CLI/bootstrap retains readable local workload wiring while stage sequencing and lifecycle policy move behind tested boundaries, results pass through `ResultStore`, existing workload outputs remain compatible, and all required test and lint commands pass. Moving every runner closure and resolved input out of `main()` is explicitly not required by this increment because it would add indirection without changing execution policy.

## Cross-increment verification matrix

| Concern | Increment 1 | Increment 2 | Increment 3 |
|---|---|---|---|
| Corrupt output after interrupted write | Atomic writer tests | No regression | ResultStore integration tests |
| Reproduce effective run configuration | Run manifest | Measurement-source metadata | Immutable `RunPlan` plus local-only `RunPaths` |
| Explain incomplete data | Run/stage state | Sample counts and validity | Tested failure transitions |
| Compare old result files | Additive fields and warning fallback | Legacy metric fallback | Stable section names |
| Avoid live side effects in tests | Mocked persistence/orchestration seams | Mocked clocks and streams | Fake stages, engines, and lifecycle |
| Dashboard trust signals | Run-state warning | Explicit timing/count preference | No consumer-facing rewrite required |

## Review questions

1. Is `failed` preferable to `partial` for a run that saved valid earlier stages but ended on an unhandled exception? This plan chooses `failed` and lets stage records show the usable completed work.
2. Should interrupted results remain at the requested output path, or should the filename gain an `.incomplete` marker? This plan keeps the stable path and relies on explicit metadata plus dashboard warnings.
3. Is Git commit plus dirty state sufficient source identity, or should a later implementation also hash selected source and configuration files? This plan avoids an expensive or fragile source-tree hash initially.
4. Should model identity include a full GGUF checksum despite startup cost? This plan records available inventory identity first and defers full hashing.
5. During Increment 2, how long should legacy aggregate fields remain? This plan keeps them for at least one public version and removes them only through a separately reviewed migration.
6. Should a future `unstable` measurement flag be purely informational or exclude a result from dashboard comparison? This plan adds no threshold or exclusion without separate evidence and review.

## Recommended review sequence

Review the state vocabulary and failure semantics first because they shape the schema. Review durability and privacy of manifest fields second. Review measurement timing definitions before approving typed records. Review the stage registry last, using the accepted result-state transitions as its contract.

---

[← Testing](testing.md) · [Back to README](../README.md) · [How It Works →](how-it-works.md)
