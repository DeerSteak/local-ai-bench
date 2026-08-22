[← Back to README](../README.md)

# Methodology contract

This document defines the Local AI Bench 4.1 neutral methodology. A result is comparable only when its application/result schema, workload behavior, model/runtime identity, effective configuration, and relevant question-bank versions are compatible. The exact export boundary is frozen in [Result compatibility v4.1](result-compatibility-v4.1.md).

The Version 6 active comparison profile is `neutral-v2`; `neutral-v1` remains the immutable 4.1 historical profile. The active profile records its effective runtime optimizations and fully resolved `deterministic-baseline-v1` sampling controls in the run plan and report, so repository defaults and engine defaults cannot silently alter comparable decoding. See [Platform Tuning Profiles](platform-tuning.md).

Qualified 0.5-second memory sampling is enabled by default and does not change the performance methodology identity. A telemetry-on result remains comparable with a telemetry-off result when every other identity-bearing setting matches; other sampling intervals retain distinct identities unless separately qualified.

Power telemetry is opt-in until each source and the combined sampler pass the shared repeated-trial observer-effect policy. Its interval, normalized source, and measurement scope are identity-bearing in run-plan schema 4; results with different power sources or scopes cannot be pooled merely because both report watts or joules.

Run-plan schema 5 adds the resolved sampler profile to methodology identity. Text generation pins temperature `0`, top-k `0`, top-p `1`, min-p `0`, repetition penalty `1`, presence and frequency penalties `0`, seed `0`, and empty logit bias across llama.cpp and vLLM; llama.cpp additionally pins its engine-specific samplers to neutral values. Managed vLLM servers ignore repository `generation_config.json` defaults. Changing any control or selecting a future publisher profile creates a distinct methodology identity and cannot pool with this baseline.

Publisher-recommended sampling is opt-in through `--publisher-sampling-profile`. Its schema-1 JSON must contain `name`, an exact `source.repo` and `source.revision`, and supported `controls`; unspecified supported controls resolve to the deterministic neutral values rather than engine defaults. The plan records the publisher controls, their canonical digest, the fully resolved semantic and engine controls, and methodology `publisher-v1`, so these results never pool with `neutral-v2`.

## Supported workload scope

The commercially supported default workload set is single-shot LLM, conversation, embeddings, image generation, MCQ, math, reasoning, code, and tool accuracy. Native llama-bench throughput, native batched concurrency, HTTP tool concurrency, and HTTP chat concurrency remain opt-in diagnostic workloads: they are useful cross-checks and capacity evidence, but a default result is not incomplete merely because they were not selected. Developer `--sample` accuracy runs are non-comparable diagnostics and are not eligible for decision-grade acceptance.

## Measurement definitions

| Measurement | Definition and clock boundary | Unit |
|---|---|---|
| Client TTFT | Wall time from sending the client request until the first generated token reaches the client; model-load time is recorded separately | seconds |
| Server prompt time | llama.cpp-reported prompt evaluation duration when available; it is not substituted for client TTFT | seconds |
| Decode time | Client wall time after first token until request completion; an implausible server timing report remains diagnostic and is not substituted into the measurement | seconds |
| Decode throughput | Generated tokens divided by decode time for an accepted measurement | tokens/second |
| Model load | Time spent loading the requested model before the measured request boundary | seconds |
| Embedding throughput | Fixed document chunks successfully embedded divided by client wall time | chunks/second |
| Image latency | Client-observed time from accepted ComfyUI submission through completed history result for one image | seconds/image |
| Accuracy | Correct scored questions divided by total scored questions; a timed-out partial answer is scored by the same parser | percent and counts |
| Concurrent throughput | Sum of accepted generated tokens divided by batch wall time, with per-request TTFT and request coverage retained | tokens/second |
| Native llama-bench rate | `llama-bench`/`llama-batched-bench` reported prompt or generation throughput for the recorded native-tool case | tokens/second |
| Power | Mean instantaneous source reading on the shared sample timeline, without idle subtraction | watts |
| Energy | Trapezoidal integration of adjacent valid power samples over actual timestamps inside measured windows only | joules |
| Energy efficiency | Completed valid generated tokens, images, or embedding chunks divided by measured-window energy; the work count and unit remain attached | tokens, images, or embeddings per joule |

The authoritative field-level names and old/new aliases are listed in [Result compatibility v4.1](result-compatibility-v4.1.md). Generation durations are retained to six decimal places so short measurements do not lose sub-millisecond resolution; presentation layers may display fewer digits. Conversation client TTFT includes request transport and cached-turn handling; server prompt time separately isolates prompt evaluation. Single-shot and conversation TTFT therefore answer different questions and must not be compared as if they share cache state.

## Cache and load state

Single-shot prompts use stable source text across comparable invocations and are measured as cold prompt processing through request-level cache bypass: llama.cpp disables prompt caching and vLLM uses a fresh cache salt. Conversation intentionally retains the same server slot/KV cache while growing one chat. Accuracy, embedding, and ordinary request workloads may reuse the loaded model but do not reuse a prior question's prompt as a methodology input. Native llama-bench cases execute in one per-model matrix so a case timeout preserves earlier cases without introducing a per-case model reload. Image models are unloaded between model families. Any future change to these states is a methodology boundary.

A GUI pause is a methodology-visible interval between measured cases, never part of a case's recorded latency. The current case completes and checkpoints before the next supported boundary waits; the loaded model, KV cache, and process remain resident where the workload lifecycle permits, but thermal and operating-system cache state may change during an unbounded pause. Schema-4 results retain pause and resume request timestamps under `run.pause.control_transitions`; comparisons must not assume uninterrupted thermal or cache continuity when that field is present.

## Warmups, attempts, retries, and timeouts

Warmups exercise the same resolved model and workload shape but are never samples and never enter aggregates. `--runs` controls measured repetitions only for single-shot LLM, embeddings, images, and native llama-bench aggregation; conversation, accuracy, and concurrency retain their documented single pass/batch behavior.

An implausible server token rate causes one reattempt. Single-shot and conversation retry that request; HTTP concurrency retries the entire batch so contention semantics stay intact. If the second observation is still implausible, it is retained as an invalid completed measurement, excluded from aggregates, and execution moves on. Engine connection crashes use the separately bounded recovery policy; a recovered retry targets the same measured run. Ordinary failed measured calls consume that attempt and do not invent a replacement sample.

Generation/chat timeouts stop the affected measured sequence while preserving prior successful samples. Accuracy timeouts retain and score partial text, then continue with the bank. Native llama-bench idle timeout preserves every previously streamed case/repetition for that model. Image timeout interrupts and clears ComfyUI work before continuing. Timeout markers never become numeric zeroes.

## Validity, aggregation, and exclusion

A performance sample is valid only when required numeric values are finite, non-negative where permitted and strictly positive where duration requires it, TTFT does not exceed wall time, token counts are integral, decode/token rate agrees within the documented tolerance, and no implausible-server-timing marker remains. Invalid samples remain visible with machine-readable reasons but are excluded from means, medians, standard deviations, coefficients of variation, charts, and acceptance calculations.

`n_runs` and `completed_runs` count completed returned samples, including invalid samples; `valid_runs` counts aggregate-eligible samples; `invalid_runs` identifies excluded attempts. Means use every valid sample with no outlier removal. Standard deviation is sample standard deviation when at least two valid samples exist and zero for one valid sample where the workload emits that field. Median and coefficient of variation are emitted only where supported and sufficiently sampled. A missing aggregate means there was no valid basis; it is never interpreted as zero.

Accuracy parsing and scoring use the immutable bank version recorded in the result. Unanswered, malformed, or incorrect answers count according to the workload scorer; timeout and likely-loop markers are diagnostics and do not independently override the parser's score. A sampled bank result is not comparable with a full-bank result.

## Decision-grade acceptance

Acceptance is evaluated per required workload/model/case, never by one hidden composite score. A policy must name its required cases, direction, threshold or baseline tolerance, minimum valid repetitions, and permitted partial coverage before execution.

Version 6 uses the distinct comparison terms and predeclared practical-threshold derivation in [Version 6 Foundation](version-6-foundation.md). Within-case or within-run dispersion may describe available evidence but cannot establish a reproducibility verdict; that verdict requires qualified compatible independent trials, and insufficient evidence is inconclusive.

Repeated-trial artifacts require at least five compatible trials per side before producing a 95% uncertainty interval. Matching case sequences use paired per-trial relative changes with a Student-t interval; unequal sequences or counts use a Welch interval over independent means. The artifact always states the method and trial counts. A monotonic increase or decline by trial ordinal is flagged and forces an inconclusive verdict rather than being absorbed into dispersion. Qualified default practical thresholds are 8% for TTFT, 3% for throughput, and 3% for wall time; accuracy retains its provisional 1% floor. `N_RUNS` remains three because qualification found that its averaging materially reduces dispersion at far less cost than three model reloads; independent trials remain the unit of reproducibility and do not reinterpret repeated requests within one loaded run as separate trials.

- Missing required data fails acceptance as insufficient evidence; it does not equal zero and cannot silently pass.
- A required case with zero valid samples fails as invalid evidence.
- Partial coverage may pass only when the policy explicitly marks omitted cases optional and the report lists them.
- Interrupted or failed runs may supply usable evidence for completed required cases, but the run status and all missing work remain prominent.
- Incompatible methodology, bank, runtime, model artifact, or measurement identity blocks numerical threshold comparison.
- Ties remain ties within the declared tolerance; no undocumented tiebreaker is applied.
- A recommendation or approval must cite the underlying measurements, coverage, uncertainty, and limitations.

## Change control

The 4.1 methodology baseline consists of this contract, [Workloads](workloads.md), [Engines](engines.md), [Result compatibility v4.1](result-compatibility-v4.1.md), the recorded question-bank hashes, and their golden fixtures. A change to prompts, scoring, cache state, timing boundaries, retry/exclusion policy, checkpoint definitions, or aggregation creates a reviewed methodology boundary and new immutable fixtures; documentation and dashboard handling ship in the same change.
