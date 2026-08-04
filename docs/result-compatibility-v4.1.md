[← Testing](testing.md) · [Back to README](../README.md) · [How It Works →](how-it-works.md)

# Version 4.1 Result Compatibility Contract

## Purpose

This contract freezes the commercially important result behavior at the boundary between the 4.1 benchmark and the planned execution-kernel rewrite. The rewrite may change internal orchestration, persistence, process isolation, and storage, but it must either preserve this exported behavior or identify a new schema/methodology boundary explicitly. The executable source of truth is `tests/fixtures/results_v4_1_complete.json` and `tests/fixtures/results_v4_1_interrupted.json`, consumed by both pytest and the dashboard's Vitest suite.

## Version identities

Top-level `version` is the application release and remains `4.1` for work performed under this development version. `run.schema_version` is the result-schema compatibility axis. The original complete and interrupted fixtures use schema 2. Schema 3 adds the immutable `run.plan` and deterministic `run.plan_id` in `results_v4_1_schema3_plan.json` without editing the schema-2 fixtures. A future schema change likewise adds fixtures rather than rewriting an older producer contract to make it appear backward compatible.

The embedded plan has its own compatibility axis. Existing golden results retain run-plan schema 1 and reproduce their original `plan_id`. Newly created 4.1 results use run-plan schema 2, which adds the `sha256-v1` hierarchical identity scheme and a unique serialized `job_id` without changing the surrounding result schema. `plan_id` excludes `job_id`, so separate executions of the same plan remain comparable; descendant stage/model/case/attempt/sample IDs are job-scoped and cannot collide across those executions.

Workload methodology has its own identity. In particular, `run.llamabench_repetition_mode` identifies the streamed internal-repetition behavior. A compatible export retains this identity, and a consumer warns rather than treating results produced by different methodologies as strictly equivalent.

## Required result envelope

A current result contains `version`, `engine`, `profile`, `accuracy_settings`, `bank_versions`, `sample_ids`, `run`, and every workload section, even when a section is empty. The workload sections are `llm`, `llm_conversation`, `embeddings`, `images`, `mcq`, `math`, `reasoning`, `code`, `tool`, `concurrency_tool`, `concurrency_chat`, `llamabench`, and `llamabenchconc`.

Older files may lack the application version, run manifest, or newer workload sections. Dashboard readers continue to treat absent legacy fields as unknown or empty rather than rejecting the file. This contract freezes the current producer shape without revoking that legacy-reader behavior.

## Run and stage state

`run.status` is authoritative for invocation completion. Selected stages appear in `run.stages`; each records status, timestamps, selected-model count, and coverage counts. Per-model fields remain authoritative for a model's skip, timeout, crash, or usable partial data.

A complete run has `run.status: complete`, a terminal timestamp, and terminal selected stages. An interrupted run has `run.status: interrupted`, retains its terminal reason and timestamp, and leaves every completed measurement chartable. The dashboard must display the interruption warning without suppressing valid data.

Stage coverage is not inferred from section length. `models_with_results`, `models_skipped`, and `models_failed` retain their meanings and must agree with the exported model payloads. A model with usable measurements followed by a later crash or timeout remains a model with results; diagnostics do not erase its earlier data.

## Checkpoint identities

Context keys are public display/schema identities rather than arbitrary formatting. Binary-K labels include `0.5K`, `2K`, `4K`, `8K`, and later configured depths; conversation additionally uses `0K` for the opening checkpoint. A producer must use the canonical labels understood by `CTX_ORDER`. For example, a 512-token result exported under `512` would be valid JSON but would not be a compatible dashboard checkpoint.

The maximum-prompt cap changes which checkpoints are planned; it does not rename retained checkpoints or synthesize missing ones. Missing, skipped, timed-out, and invalid work is never emitted as a numeric zero.

The journal-owned single-shot implementation is required to reproduce every LLM field in the schema-3 golden fixture from recorded samples. It may add current validity/client-timing diagnostics, but preserves the legacy TTFT/TPS means, counts, sample payload, timeout/crash/slow markers, context labels, and missing-data semantics. JSON is now an export projection for this section; the sibling SQLite journal is its live recovery source.

## LLM measurement contract

Each checkpoint keeps compatibility aggregates such as `ttft_mean_sec` or explicit `client_ttft_mean_sec`, `tps_mean`, standard deviations where available, and run counts. `n_runs` and `completed_runs` are completed sample counts. `requested_runs` is the requested count. `valid_runs` is the number eligible for aggregates. `invalid_runs` identifies rejected attempts and their reasons. Invalid measurements do not enter means or `valid_samples`.

Each valid generation sample retains named fields for `client_ttft_sec`, `server_prompt_sec`, `client_wall_sec`, `decode_sec`, `generated_tokens`, `tokens_per_sec`, `finish_reason`, and `model_load_sec`. Client TTFT and server prompt duration remain different measurements; neither overwrites the other. Generated-token rate must be consistent with decode duration under the measurement validator.

An implausible server TPS value may produce a wall-clock diagnostic, but that corrected diagnostic is not a valid benchmark sample. The workload retries once under the current methodology; a second implausible measurement is retained as invalid evidence and excluded from aggregates.

## Partial-result contract

Checkpointing preserves every completed model and expensive subunit required by the workload. Interruption after one of three requested LLM samples exports `requested_runs: 3`, `completed_runs: 1`, `valid_runs: 1`, and the valid sample; it does not discard the checkpoint or pretend three samples completed.

Timeout, crash, interruption, and later-case failure do not replace earlier measurements with an error-only object. Consumers render valid retained measurements and separately disclose incomplete run, stage, model, case, repetition, or sample coverage.

## llama-bench contract

Native llama-bench stores prefill and decode data separately in `prefill_entries` and `decode_entries`. Prefill entries identify prompt work through `n_prompt` with zero generation. Decode entries identify generation through `n_gen` and the prefilled depth through `n_depth`; they do not mislabel that depth as `n_prompt`.

Entries retain llama-bench's own `avg_ts` and `stddev_ts` outputs. Requested/completed cases and repetitions disclose coverage. Streamed cases completed before timeout remain available. Dashboard consumers build prefill and decode charts from the explicit arrays while retaining fallback support for older unambiguous combined `entries` files.

## Numeric and persistence integrity

Exported JSON contains no `NaN`, positive infinity, or negative infinity. A non-finite measurement is rejected before section mutation so the prior valid checkpoint remains the recovery point. Export replacement remains atomic where the filesystem provides the required semantics.

The rewrite may use SQLite or another transactional internal store, but portable export must remain deterministic, finite JSON with the state and raw measurements necessary to reproduce its aggregates.

## Consumer acceptance matrix

| Behavior | Python contract | Dashboard contract |
|---|---|---|
| Complete 4.1 result | Finite JSON, schema 2, required sections, matching stage coverage | No reliability warning; LLM, conversation, and llama-bench data remain chartable |
| Interrupted 4.1 result | Terminal interrupted state and retained partial sample counts | Interruption warning plus chartable completed sample |
| Context labels | Canonical binary-K identities | Ordered through `CTX_ORDER` without hidden data |
| Named measurements | Exact raw valid-sample field set | Explicit TTFT preferred with legacy fallback |
| Native llama-bench | Separate prefill/decode entries and methodology identity | Correct prompt-depth and tg series |
| Legacy result | Missing newer fields tolerated | Missing run metadata and sections do not block loading |
| Schema-3 plan result | Embedded plan reproduces `plan_id` and compatibility manifest fields | Existing charts render unchanged while plan metadata remains available |

## Rewrite acceptance

Before a new execution path replaces the 4.1 path, it must pass the Python and dashboard compatibility tests against these immutable fixtures, export equivalent retained measurements for characterized runs, and document every intentional difference. An intentional change to prompts, scoring, timing, cache behavior, retry rules, checkpoint identity, model lifecycle, validity, or aggregation is a methodology change, not an internal refactor.

New result fields should be additive when practical. Removing or changing an existing field requires a schema-version change, a migration/reader strategy, new golden fixtures, dashboard coverage, and a visible compatibility note. No migration step may rewrite the 4.1 golden files.
