[← Back to README](README.md)

# Version 6 Plan

Version 5.1 has broad workload coverage, durable evidence, and heavy governance machinery. Its remaining gaps are not "more benchmarks" — they fall into two groups: measurements the suite never takes (memory, power, thermal), and the missing steps between a measurement and a trustworthy decision (noise handling, preflight validation, recommendations).

This plan orders eleven features on one rule: **anything that makes existing conclusions more trustworthy outranks anything that adds new conclusions.** A faster number nobody can trust is worth less than an existing number that can now be defended.

This is a single-maintainer project. “Release owner,” “methodology owner,” “code owner,” and “qualification owner” are responsibilities held by the same person, not separate approvers; controls that would normally rely on separation of duties instead rely on committed predefinitions, automated gates, durable evidence, and reproducible real-hardware records.

## The plan in plain English

Build Version 6 in eleven numbered milestones. Finish and verify one milestone before starting anything that depends on it. For every milestone, follow the same simple loop:

1. Write down exactly what the feature will measure or decide.
2. Write the tests that prove the important rules and failure cases.
3. Build the smallest working backend path.
4. Save the result without breaking older files.
5. Show the result in every user-facing place that needs it.
6. Update the documentation while the behavior is still fresh.
7. Run all required automated checks.
8. Try to break the feature with missing, malformed, interrupted, and incompatible inputs.
9. Test parsers with captured fixtures and test user-facing measurement claims on the smallest real machine that can prove them.
10. Mark the milestone complete only when every acceptance box is checked and the old workflow still works.

Do not start with the dashboard. A feature exists only after the measurement or decision is correct, durable, backward-compatible, and tested. The dashboard is the final presentation of that capability.

## What “complete” means

A milestone is complete only when all six layers below are finished. If one layer is missing, the milestone remains in progress.

| Layer | Child-simple question | Required proof |
|---|---|---|
| Definition | What exactly are we measuring or deciding? | Units, boundaries, unavailable states, and compatibility rules are written down. |
| Collection | How do we get the input? | Supported sources work; failure becomes explicit `unknown`, never a believable zero. |
| Storage | Will it survive a crash and remain readable later? | Durable event/JSON representation, provenance, schema compatibility, and migration tests exist. |
| Interpretation | What conclusion may we draw? | Pure, tested calculations state their assumptions and can return indeterminate. |
| Presentation | Can a person understand and audit it? | CLI, GUI/dashboard, report, and raw evidence agree wherever each is applicable. |
| Qualification | Has the claim worked outside a mock? | Captured fixtures plus required real-hardware evidence and a recorded support status exist. |

## Work that happens before milestone 1

Complete this short foundation phase first. It prevents eleven features from inventing eleven incompatible meanings for the same data.

1. Create one Version 6 tracking issue that links to the eleven milestone sections in this document and to implementation changes. **This document is the sole authoritative checklist.** The tracking issue reports status and links here; it never copies acceptance or release boxes.
2. Freeze the Version 5.1 compatibility fixtures and add representative schema-1 through schema-4 results wherever the supported test set has a gap.
3. Write the schema-5 field map before code: field name, unit, scope, source, availability state, owning event, JSON location, and old-file fallback.
4. Write the telemetry vocabulary: `sample`, `channel`, `source`, `scope`, `measured window`, `idle window`, `unknown`, and `unsupported` must have one meaning everywhere.
5. Write the comparison vocabulary: within-case variation, within-run variation, between-trial variation, practical threshold, uncertainty interval, paired trial, and inconclusive must not be used interchangeably.
6. Choose and document the minimum real-hardware qualification set for each telemetry source. Captured command output proves parsers; it does not prove timing, permissions, process ownership, or sensor meaning.
7. Define a deliberately conservative observer-effect bound and the practical-threshold derivation method before seeing telemetry results, then commit both definitions to Git before generating qualification data. Use a temporary prototype and many alternating telemetry-off/on repetitions as a **coarse screen** for obviously intrusive sources. Report descriptive distributions only; do not call this a reproducibility verdict or use it to approve default-on telemetry.
8. Record the provisional telemetry modes and coarse-screen failures in [architecture-decisions.md](docs/architecture-decisions.md). Final telemetry-on/off comparability, methodology identity, and default-on approval remain open until item 3's repeated-trials machinery re-qualifies every source and the combined sampler.
9. Record the remaining Version 6 compatibility and telemetry decisions in [architecture-decisions.md](docs/architecture-decisions.md) before the first schema or persistence change.
10. Run the existing Python and dashboard suites and save the clean baseline. A pre-existing failure must be resolved or explicitly recorded before Version 6 work begins.

## Sequencing and dependencies

Items 1, 4, and 5 share one background telemetry sampler. Item 1 builds the light version (memory only, no elevated permissions); items 4 and 5 extend that same sampler rather than adding parallel instrumentation. Do not implement them as three separate efforts.

Item 6 (recommendations) depends on item 3's completed repeated-trials verdicts, because a recommendation that ranks models on differences it cannot distinguish from run-to-run noise is worse than no recommendation. Item 11 (workspace) is deliberately last: it consolidates capabilities rather than adding any, and it is easier to design once the decision it must display has been defined by item 6.

Items 2, 7, 8, and 9 are independent and may be resequenced against team capacity. Item 10 must wait for items 1, 3, and 9; its energy comparison waits for item 4.

| # | Feature | Depends on | Sampler group |
|---|---------|-----------|---------------|
| 1 | Measured memory footprint and headroom | 3 for default-on/comparability approval | ● |
| 2 | Pre-run model compatibility preflight | — | |
| 3 | Noise-aware comparison and repeated trials | — | |
| 4 | Power and energy per run | 1 | ● |
| 5 | Sustained-load and thermal degradation | 1, 4 | ● |
| 6 | Goal-driven recommendation view | 3 | |
| 7 | Complete case-level resume for every workload | — | |
| 8 | Evidence-backed platform qualification | — | |
| 9 | Model catalog audit and refresh | — | |
| 10 | Quantization comparison workflow | 1, 3, 9; 4 for energy comparisons | |
| 11 | Unified results and decision workspace | 6 | |

### Coarse effort estimates

These are planning sizes, not promises: **S** is a focused change, **M** spans several modules or one platform boundary, **L** changes persistence/methodology or several product surfaces, and **XL** is a multi-stage cross-platform program. Re-estimate after each slice and split any milestone that grows beyond its size.

| # | Feature | Size | Main effort driver |
|---|---|---|---|
| 1 | Measured memory footprint and headroom | L | Cross-platform sampler, event persistence, schema, and dashboard |
| 2 | Pre-run model compatibility preflight | M | Static checks, clean-state runtime probe, and workload gates |
| 3 | Noise-aware comparison and repeated trials | XL | Trial identity, statistics, policy semantics, UI, and qualification |
| 4 | Power and energy per run | XL | Platform sensors, permissions, observer effect, scope, and energy integration |
| 5 | Sustained-load and thermal degradation | L | New workload, aligned time series, classification, and sensors |
| 6 | Goal-driven recommendation view | L | Authoritative policy engine plus CLI, report, and UI artifact flow |
| 7 | Complete case-level resume | XL | Workload-by-workload persistence migration and interruption parity |
| 8 | Platform qualification matrix | M | Evidence process, release gating, generated docs, and UI labels |
| 9 | Model catalog audit and refresh | M | Role coverage, candidate qualification, lifecycle compatibility, and migration evidence |
| 10 | Quantization comparison workflow | L | Catalog/identity migration, setup, storage estimates, and comparison |
| 11 | Unified results and decision workspace | XL | Cross-platform architecture, shared state, packaging, and offline UX |

## Conventions that apply to every item

Each item below is only complete when all of the following are true, per [AGENTS.md](AGENTS.md) and [docs/release-policy.md](docs/release-policy.md).

- `bash tests.sh` passes, with real unit tests for new business logic — extracted to a testable function first if it lands inside a `run()` or `main()`.
- Any change to `dashboard/src` passes `npm test`, `npm run lint`, and `npx tsc --noEmit`, with new Vitest tests for new pure functions in `utils/*.ts` or `constants.ts`.
- Any new results-JSON field is treated as optional on read. Dashboard code uses optional chaining and `JsonRecord`, never a bare `any` and never an assumption of presence — older files must keep rendering.
- Docs are updated in the same change, not afterward: flags and defaults to [cli-reference.md](docs/cli-reference.md), workload behavior to [workloads.md](docs/workloads.md), algorithms and execution order to [how-it-works.md](docs/how-it-works.md), new files to [project-structure.md](docs/project-structure.md), new test files to [testing.md](docs/testing.md).
- `VERSION` in [config.py](scripts/runtime/config.py) is the only place the version is edited; the pre-commit hook rewrites every mirror.
- Chart changes are visually previewed against a file in `samples/`, not merely traced in code.

### One telemetry observer-effect policy

This policy is authoritative for items 1, 4, and 5; those items name only their additional sources and evidence.

1. The foundation prototype performs a coarse descriptive screen only. It may reject an obviously intrusive source, but it cannot approve default-on telemetry or establish comparability.
2. After item 3 phase two exists, the maintainer uses its independent-trial machinery to re-qualify memory-only sampling, every power source, every temperature source, and the combined sampler at each proposed interval.
3. Before generating qualification data, the maintainer commits the perturbation bound and practical-threshold derivation method to Git in the relevant methodology/architecture document. Every qualification record cites that commit hash, then records TTFT, throughput, wall-clock effects, trial count, pairing/order, drift, interval, platform, and source versions. Changing a committed definition requires a later commit explaining why; history is never rewritten to make the change look predeclared.
4. A mode inside the bound may be default-on only after the maintainer records the telemetry-on/off comparability decision and its evidence. A mode outside the bound is redesigned, opt-in under a distinct methodology identity, or unsupported; overhead is never averaged away.
5. Comparisons enforce the recorded methodology identity. Matching schemas never imply that telemetry-on and telemetry-off measurements are scientifically comparable.

### One telemetry privacy and outbound policy

This policy is authoritative for items 1, 4, and 5.

1. Parse sensor output in memory and persist only allowlisted normalized measurements and provenance.
2. Never persist raw command output, serial numbers, device UUIDs, host identity, process arguments, private paths, or unrelated hardware inventory.
3. Route every exported telemetry field through [outbound_metadata.py](scripts/results/outbound_metadata.py), preview it before bundles, reports, and diagnostics, and apply aliases where the existing policy permits.
4. Test each parser and outbound path with fixtures containing forbidden identity data; the export must omit it.

### Minimum telemetry outcome for Version 6

Telemetry is not complete merely because unsupported states render correctly.

- Version 6 must qualify measured host/process memory and the platform's applicable accelerator or unified-memory channel on at least one discrete-GPU platform class and one unified-memory platform class.
- Version 6 must qualify energy measurement on at least one platform class with a clearly stated scope. Whole-system energy is preferred, but GPU/package energy is acceptable only when the UI and reports prohibit broader claims.
- Version 6 must qualify one sustained-load configuration with throughput and temperature on at least one thermally constrained platform class; power correlation may be unavailable if no qualified source exists.
- If these floors are not met, the affected capability is removed from the Version 6 feature claim and acceptance checklist rather than shipped as universally unavailable scaffolding.
- Because power and temperature add work to the memory sampler, a platform whose memory-only mode cannot qualify as default-on cannot have power or thermal default-on. Heavier modes may still be opt-in under distinct methodology identities if independently qualified.

### Three gates that close a milestone, then one slice pilot

The implementation and test conventions above are ordinary development work, not eleven separately tracked process transitions. A milestone closes after the first three surprise-finding gates pass; the fourth gate closes the slice before dependent work begins:

1. **Old-file compatibility:** supported historical results, plans, journals, bundles, reports, and dashboard paths still behave according to their contracts.
2. **Adversarial review:** exercise malformed and missing inputs, denied permissions, cancellation, persistence failure, incompatible methodology, and every feature-specific boundary named in the milestone.
3. **Real-hardware qualification:** a human follows the safe feature-specific procedure and records evidence; automated tests never run `setup.sh`, `setup.bat`, or a real benchmark to claim hardware support.
4. **Slice pilot:** after the slice's milestones close, the applicable pilot below passes before the next slice begins.

Use a lightweight commit-message tag instead of a prose status block: `v6:<item>/<gate> [schema] [methodology] [cli] [privacy]`, including only changed bracketed surfaces. Example: `v6:1/adversarial [schema] [privacy]`. The tracking issue links commits and this document's boxes; it does not duplicate either.

### Schema versioning for this release

Items 1, 2, 4, and 5 add result fields, while items 3, 6, 8, 10, and 11 add derived artifacts or interpretation metadata. Introduce **result schema 5** once, in the first merged change that writes a new result field, rather than bumping per item. Store it in the existing `run.schema_version` location, keep every new field optional on read, and extend [result-compatibility-v4.1.md](docs/result-compatibility-v4.1.md) with a Version 6 section stating exactly which fields are new, which are optional, and what a schema-4 file renders as. Runs from schema 4 and earlier must remain loadable, comparable where methodology permits, and visibly distinct from runs that carry the new telemetry — missing telemetry is not zero.

Do not reuse the result-schema number for plans, policies, projects, bundles, journals, or qualification records. Each format changes its own schema only when its serialized shape changes. A methodology-affecting change also receives a new methodology identity even when the JSON remains readable; schema compatibility and scientific comparability are separate decisions.

---

# 1. Measured memory footprint and headroom

## Why this is first

`model_fits()` near line 231 of [hardware.py](scripts/runtime/hardware.py) decides whether a model fits by parsing the catalog's `download_size` string against a computed ceiling. That is a static estimate of weight-file size, made before anything runs. The suite never records what a model actually consumed, so it cannot answer the question its own [limitations.md](docs/limitations.md) raises — that fitting once does not guarantee safe capacity — nor the practical version of it: does this model fit *at the context depth and concurrency I intend to use*, and with how much headroom.

The plumbing already exists and is being discarded. [benchmark_gui_resources.py](scripts/app/benchmark_gui_resources.py) already implements `process_resource_usage`, `system_memory_usage`, `query_vram_usage`, and `query_gpu_process_memory` — but only to paint rows on the GUI progress screen. None of it reaches the result store.

This ranks first because it is the cheapest of the four measurement items, needs no elevated permissions, answers the most common user question, and builds the sampler that items 4 and 5 extend.

## Implementation outline

1. **Create `scripts/runtime/telemetry.py`** as the home of the shared sampler. Define an immutable per-sample record (timestamp, host RAM used, process RSS, GPU VRAM used and total) and a `TelemetrySampler` that runs on a background thread at a fixed interval, appending samples to an in-memory buffer. Give it explicit `start()`/`stop()` and a context-manager form so a measured window cannot leak a thread on an exception path.
2. **Move the existing query functions out of the GUI module** into `telemetry.py`, leaving `benchmark_gui_resources.py` importing from the new home. These are already pure and already unit-tested — preserve their tests and extend them, do not rewrite them. This makes the sampler usable from a headless CLI run, which is the actual blocker today.
3. **Define hierarchical aggregation as pure functions**, separate from the thread. A lifecycle window may contain fixed or workload-defined sub-windows; compute peak, mean, final, sample count, and duration per sub-window first, then derive the case-level summary from those results without discarding the sub-window series. Short cases may have one sub-window. This data model must support item 5's long soak without storing one meaningless ten-minute peak as its only evidence.
4. **Add a headroom calculation** that compares peak observed usage against the ceiling from `compute_memory_ceiling_gb()`. Report absolute headroom in GB and as a fraction, and classify into a small set of named states (comfortable, tight, exceeded, unknown). Classification thresholds live in `config.py` as named constants, not inline literals. `unknown` is a real state and must not collapse to zero or to comfortable — a machine with no readable VRAM counter is a coverage gap, not a pass.
5. **Define three explicit windows:** idle baseline before model load, model load, and measured case. Keep the sampler alive across those windows and tag every sample with its window. This captures the load peak without charging load time or energy to the measured request, and it prevents a run-wide peak from being smeared across unrelated cases. Use existing workload lifecycle seams rather than assuming one hook in `workload_runner.py` covers every engine and workload.
6. **Persist per-case telemetry through the event journal.** Extend the case payload written by [llm_event_stage.py](scripts/results/llm_event_stage.py) and the native stage with a memory block, so telemetry survives interruption on the same terms as the measurements it describes. Telemetry must never be the reason a case fails — a sampler error records `unknown` and the case proceeds.
7. **Export a memory block per model per case** into the results JSON under schema 5, alongside a run-level summary (peak across the whole run, and the tightest headroom observed with the case that produced it).
8. **Record sampler provenance**: interval, source of each channel (`psutil`, `nvidia-smi`, `rocm-smi`, unavailable), and the count of failed samples. Without this a reader cannot tell a low peak from a broken counter.
9. **Apply the shared telemetry privacy and outbound policy** to memory source names, intervals, availability, and normalized measurements.
10. **Dashboard**: add memory columns to `StatsTable`, a headroom indicator on `RunSummaryCards`, and a per-model peak-memory chart. Add the new keys to `constants.ts`. Follow the existing color conventions — this is a per-model series, so `MODEL_COLORS`, not `CATEGORY_COLORS`. Files without telemetry render as an explicit "not recorded" state, never a zero bar.
11. **Tests**: pure aggregation over empty, single-sample, all-failed, one-sub-window, and many-sub-window inputs; headroom boundaries; GPU parser failures; sampler cleanup; and the shared outbound-policy fixtures. Vitest tests cover the new dashboard chart builders including missing telemetry. The foundation coarse screen rejects obvious observer problems; post-item-3 qualification supplies the final evidence required by the shared observer-effect policy.
12. **Docs**: [workloads.md](docs/workloads.md) for what is measured and when the window opens; [how-it-works.md](docs/how-it-works.md) for the sampler's place in execution; [limitations.md](docs/limitations.md) for observer effect and what a peak does and does not prove — notably that peak RSS on unified memory is not the same quantity as discrete VRAM occupancy and the two must not be compared across machine classes.

## Acceptance criteria

- [x] A completed run records, for every model and case, peak and mean host RAM, process RSS, and GPU VRAM where a counter is readable.
- [x] Idle, model-load, and measured-case windows are stored separately; loading is never silently included in request efficiency.
- [x] Every recorded figure carries its source, sample count, and sampler interval; an unreadable channel records `unknown` and is visually distinct from zero in the dashboard.
- [x] Run-level summary reports the peak observed and the tightest headroom, naming the case that produced it.
- [x] Headroom classification is a pure, unit-tested function with tests at every threshold boundary.
- [x] A sampler failure never fails, aborts, or invalidates a case; a test asserts this by injecting a raising query function.
- [x] Sub-window evidence is retained and case summaries are derived from it; a many-window test proves the series is not collapsed prematurely.
- [x] Memory-only sampling passes the shared observer-effect policy before it is default-on or treated as comparable.
- [x] Memory telemetry passes the shared privacy and outbound policy.
- [x] Telemetry survives interruption and resume on the same terms as the measurements it accompanies.
- [x] A schema-4 results file loads and renders with memory columns explicitly marked not recorded.
- [x] `benchmark_gui_resources.py` consumes the shared sampler rather than duplicating query logic; no query function exists in two places.

---

# 2. Pre-run model compatibility preflight

## Why this is second

[engines.md](docs/engines.md) already documents the hole: `-jinja` renders the model's embedded `tokenizer.chat_template`, not every GGUF has that metadata, and there is no setup-time check that warns when it is missing. The failure mode is the worst kind a benchmark can have — not an error, but hours of well-formatted, confidently-reported, *invalid* results that measure a configuration mistake instead of a model.

Partial machinery exists. `ModelCompatibility` in [model_compatibility.py](scripts/setup/model_compatibility.py) already carries architecture and a load-probe status, with `inspect_llamacpp_model`, `probe_llamacpp_load`, and `architecture_from_gguf` behind it. But the record has no chat-template, context, or tool-call fields, and its only callers are [engine_management.py](scripts/app/engine_management.py) and a GUI engines screen — it is a manual inspection tool, never a gate on a run.

This is an extension of existing, already-tested code, and it protects the validity of every other item on this list.

## Implementation outline

1. **Extend the `ModelCompatibility` record** with the checks that are missing: chat-template presence and source, declared versus readable maximum context, tool-call support, weight-file completeness, and a formatting-probe result. Keep `status` as an overall verdict but add a per-check breakdown, so a caller can distinguish "cannot run at all" from "cannot run the tool workload."
2. **Implement chat-template detection** by reading `tokenizer.chat_template` from GGUF metadata, reusing the existing GGUF reader seam from `architecture_from_gguf` rather than opening a second parsing path. Distinguish three outcomes: template embedded, absent (llama.cpp will fall back to heuristic guessing), and unreadable. Absent is a warning, not a hard failure — it is a known-quality-risk state that must be recorded in the results profile so a reader can attribute a quality difference to it later.
3. **Implement a context-capacity check** comparing the model's declared training context against the value the run plan intends to use and against `max_context_length()` on the engine. Flag a plan that requests more context than the model declares; this silently degrades quality rather than erroring.
4. **Implement a tool-call support check** gating the tool workload specifically, building on the engine's existing `supports_tool_calls()`. A model that fails this is excluded from the tool bank with a recorded skip reason — it does not fail the whole run.
5. **Implement a weight-completeness check** for multipart weights, verifying every declared shard is present and non-truncated before a multi-hour run begins.
6. **Add a deterministic formatting and tokenization probe** — a single fixed short prompt at temperature zero, asserting the response is non-empty, terminates, and does not emit raw template markup. Run it only after the engine has loaded that model and immediately before its first measured case. Clear the slot/KV cache or restart through the engine's supported clean-state path afterward, so the probe cannot warm or otherwise contaminate the benchmark. This is a validity check, not a benchmark, and its result never enters a performance metric.
7. **Add preflight to the resolved execution plan**, after plan validation and before each model's first measured case. Static checks run before any runtime starts; the load/format probe runs after model load. Emit one report per model with per-check detail. Hard failures exclude that model with a recorded reason, while workload-specific failures exclude only that workload. `--force-all` may bypass quality-risk warnings and existing speed cutoffs, but it must not bypass corrupt/incomplete artifacts or a runtime that cannot load the model. Never silently drop a model.
8. **Record the full preflight report in the results JSON** under schema 5, including passing checks. A reader auditing a suspicious accuracy number needs to see that the template check passed, not merely the absence of a complaint.
9. **Surface preflight in both frontends**: a pre-run summary in the GUI listing excluded models and warnings with a chance to cancel, and equivalent CLI output through `Shared.log/warn/err`.
10. **Dashboard**: show preflight warnings in the existing `ValidityInspector`, and mark a model carrying a template warning wherever its accuracy is displayed. A missing template is precisely the kind of thing that explains an anomalous accuracy result, so it must be visible at the point of interpretation, not buried in a profile block.
11. **Tests**: each check independently against representative GGUF metadata fixtures including absent, malformed, and unreadable; the exclusion-versus-warning policy including `--force-all`; the tool-gate producing a skip reason rather than a run failure; and a regression test that a model failing preflight never appears in any measured section of the exported results.
12. **Docs**: update [engines.md](docs/engines.md) to state the check now exists and what it does, [workloads.md](docs/workloads.md) for the tool-workload gate, [cli-reference.md](docs/cli-reference.md) for any new flag, and [troubleshooting.md](docs/troubleshooting.md) with each failure and its remedy.

## Acceptance criteria

- [x] Every model is checked for chat template, context capacity, tool support, weight completeness, and formatting round-trip before the first measured case runs.
- [x] A missing chat template produces a visible warning recorded in the results and shown next to that model's accuracy in the dashboard.
- [x] A hard failure excludes only that model, with a recorded reason, and never aborts the run or discards other models' evidence.
- [x] Preflight results are recorded for passing checks too, not only failures.
- [x] `--force-all` may bypass documented warnings but cannot bypass corrupt artifacts or an unloadable model; tests cover both paths.
- [x] The formatting probe contributes to no performance metric; a test asserts its measurements are absent from every exported section.
- [x] The formatting probe cannot warm the measured request; a test verifies the engine's clean-state seam runs after the probe.
- [x] Total preflight time for a full catalog is small relative to a run and is reported before execution begins. Qualified on an RTX 5090 WSL run across llama.cpp and vLLM: 25m 56s combined preflight against 6h 44m total execution (6.4%), compared with 87m 48s of existing workload warmup time.
- [x] [engines.md](docs/engines.md) no longer describes this as an unimplemented gap.

---

# 3. Noise-aware comparison and repeated trials

## Why this is third

`compare_results()` in [result_history.py](scripts/results/result_history.py) computes `delta` and `percent_change` per metric and stops. A 3% regression and a 30% regression are presented identically, and nothing tells a reviewer whether either is reproducible. Meanwhile `ttft_stdev_sec` and `tps_stdev` are already recorded in [llm_prefill_benchmark.py](scripts/workloads/llm_prefill_benchmark.py), and `Shared`'s coefficient-of-variation helper in [shared.py](scripts/runtime/shared.py) exists and is never called by comparison. `acceptance_policy.py` has no concept of tolerance or dispersion at all.

[limitations.md](docs/limitations.md) already says a single run is evidence rather than a variance study and that vendor claims should disclose dispersion. Today that is a disclaimer standing in for a capability. Phase one is nearly free because the data is already on disk.

## Implementation outline

### Phase one — expose the within-run uncertainty already recorded

1. **Create `scripts/results/significance.py`** holding the comparison mathematics as pure functions, kept out of `result_history.py` so it is testable and reusable by acceptance policy, recommendations, and reports alike.
2. **Extend `extract_comparable_metrics()`** to carry each metric's dispersion and sample count alongside its mean, rather than the bare float it returns today. Preserve behavior for metrics that have no recorded dispersion — they yield an explicitly unknown dispersion, not zero, because zero dispersion would falsely imply perfect reproducibility.
3. **Show a within-run uncertainty label on every eligible comparison row.** Given two means, their within-run dispersions, and sample counts, classify the available evidence as recorded or insufficient. Do not call the difference statistically significant: repeated requests inside one loaded run do not measure day-to-day, thermal, setup, or machine-state variation and therefore cannot prove that a cross-run difference is real.
4. **Introduce a provisional practical-change threshold** configurable per metric family and recorded with the comparison. The maintainer applies the precommitted derivation method and documents the provisional values; phase one may say that a delta clears this user-relevant size threshold but may not say it is reproducible.
5. **Extend acceptance policy** so a rule may express a practical tolerance and an evidence requirement. A single-run rule may still pass or fail its literal threshold, but any rule asking for a reproducible improvement returns inconclusive until a qualified trial set exists. Add the corresponding outcome to policy validation and [acceptance-policies.md](docs/acceptance-policies.md).
6. **Surface the distinction in reports and dashboard:** show the raw delta, practical threshold, within-run dispersion, and the sentence “repeated trials required for a regression verdict.” Never hide a small or uncertain delta.

### Phase two — repeated trials

7. **Add a repeated-trials project mode** that groups compatible runs of the same plan using the existing [hardware_identity()](scripts/results/result_history.py) definition, reusing the compatibility gate from `compare_results()` so incompatible runs cannot be silently pooled. Do not invent a second hardware-identity rule.
8. **Compute per-metric aggregate statistics across independent trials:** mean, median, between-trial dispersion, an uncertainty interval, and the trial count. Choose the interval method only after documenting its assumptions and minimum trial count; below that count, show descriptive values and return inconclusive rather than printing a fragile interval.
9. **Detect order and drift effects** by testing each metric against trial ordinal. A trial set whose measurements decline monotonically is a thermal or background-load artifact, not a variance estimate, and must be flagged as such — this is the direct link to item 5.
10. **Support paired comparison** when both sides ran the identical case sequence, which is materially more sensitive than comparing independent means and is the common case for a driver, BIOS, or runtime upgrade.
11. **Produce a regression verdict only for the trial set** — improved, regressed, unchanged, or inconclusive — against the user's practical threshold, with the underlying numbers, interval method, pairing mode, drift status, and trial count always shown.
12. **Re-evaluate `N_RUNS` using evidence rather than intuition.** Compare equal-time designs such as three measured requests in one trial versus one measured request across three independent trials, including setup/load overhead and metric stability. Decide whether the default should change, whether repeated-trial projects need their own run-count default, and how accuracy/conversation workloads with fixed one-pass behavior fit the design. Record the chosen time-budget tradeoff in methodology docs; do not change the default without qualification data.
13. **Derive the Version 6 default practical thresholds from qualified between-trial evidence.** Apply the derivation method committed before data generation to repeated-trial distributions from telemetry observer studies and additional telemetry-off baselines. A default must not sit below the demonstrated noise floor; document the derivation, platform coverage, rounding, and any intentionally larger product-relevance floor. Revisit the threshold when methodology or supported platform evidence changes through a new explanatory commit.
14. **Create and schedule telemetry re-qualification using the shared observer-effect policy.** Replace the foundation coarse screen for memory-only sampling during Slice A. Re-run the same protocol for each power source, temperature source, and combined mode as those sources arrive in Slice B. No telemetry mode becomes default-on before its own independent-trial verdict.
15. **Dashboard**: a trial-set view showing per-metric distribution across trials, the interval, the drift flag, and the verdict. A single-run selection continues to work unchanged.
16. **Tests**: the repeated-trial verdict function across improved, regressed, unchanged, and inconclusive inputs; practical-threshold boundaries; unequal trial counts; zero and missing dispersion; aggregation over one, two, and many trials; the drift detector against a synthetic declining series and a flat noisy one; the compatibility gate rejecting a mismatched pool; and run-plan tests for any selected `N_RUNS`/trial-count behavior. Vitest covers the trial-set builders.
17. **Docs**: [limitations.md](docs/limitations.md) revised so the dispersion paragraph points at a real capability; [reports.md](docs/reports.md), [acceptance-policies.md](docs/acceptance-policies.md), and [recommendation-policy.md](docs/recommendation-policy.md) updated for the new outcome; [cli-reference.md](docs/cli-reference.md) for the trial-set flags; and [methodology-contract.md](docs/methodology-contract.md) for threshold derivation, telemetry re-qualification, and the final `N_RUNS` versus independent-trials decision.

## Acceptance criteria

- [x] Every comparison row carries the raw delta, practical threshold, within-run dispersion availability, and sample counts behind it.
- [x] No single-run comparison is labeled statistically significant or reproducible; tests search every output path for this invariant.
- [x] A comparison against a file lacking dispersion reports insufficient within-run uncertainty, never a false zero.
- [x] Acceptance policies support tolerance and evidence requirements and can return inconclusive; policy validation documents the outcome.
- [x] A sufficiently large trial set reports mean, median, between-trial dispersion, an uncertainty interval with its method stated, and the trial count; smaller sets return inconclusive.
- [x] Monotonic drift across trials is detected and flagged rather than absorbed into the dispersion estimate.
- [x] Paired comparison is used automatically when the case sequence matches, and the mode used is stated in the output.
- [x] The `N_RUNS` and independent-trial time-budget decision is supported by qualification evidence and documented; retaining the current default requires the same justification as changing it.
- [x] Each default practical threshold follows the derivation method committed before its data was generated, and no default is below its demonstrated noise floor.
- [x] Memory-only telemetry is re-qualified in Slice A, and the same required protocol is wired into the completion gate for every later telemetry source and combined mode.
- [x] Incompatible runs cannot be pooled into a trial set; a test asserts the gate rejects a methodology mismatch.
- [x] Reports and dashboard distinguish qualified repeated-trial verdicts from descriptive single-run deltas without hiding either.

---

# 4. Power and energy per run

## Why this is fourth

Nothing in `scripts/` measures power — a grep for `power`, `watt`, or `joule` hits only prose fixtures. For a product whose [product-requirements.md](docs/product-requirements.md) names hardware-vendor teams validating an upcoming small-system launch, performance per watt is frequently the entire launch claim, and it is the one number the vendor cannot get from `llama-bench`. Today the suite can say system A does 42 tok/s and system B does 38, and has no answer to the immediate follow-up: at what power draw.

It ranks below memory only because it needs elevated permissions on some platforms and platform-specific plumbing on all of them. It extends item 1's sampler; it does not build a new one.

## Implementation outline

1. **Add power channels to the existing sampler** in `telemetry.py`. Do not create a second sampler — power, memory, and temperature are one timeline and must share timestamps so they can be correlated in item 5.
2. **Implement a per-platform power source behind one interface**, each independently testable against captured output or driver structures: `powermetrics` on macOS, `nvidia-smi --query-gpu=power.draw` for NVIDIA, Adrenalin ADL on Windows AMD, `rocm-smi` on Linux AMD, and RAPL via sysfs for Intel CPU package power. Parse into a common unit and keep every parser a pure function over captured evidence, as the existing `parse_nvidia_gpus`-style functions already are.
3. **Make source discovery self-contained in `telemetry.py`.** `powermetrics` requires elevated privileges; RAPL sysfs may not be world-readable. A read-only availability function returns available or unavailable-with-reason before the run, without depending on item 2, prompting, or escalating privileges. When item 2 exists, preflight displays this result; the sampler remains independently usable and item 4 keeps only its dependency on item 1.
4. **Integrate power over each measured window** to produce energy in joules, using trapezoidal integration over actual sample timestamps rather than assuming a fixed interval — a sampler thread under load will not tick evenly, and assuming it will inflates or deflates the total.
5. **Derive efficiency metrics** per workload family: tokens per joule for generation, images per joule for image generation, embeddings per joule. Derive these as pure functions from recorded energy and recorded work, never sampled independently.
6. **Record what the measurement covers.** Package power, GPU-only power, and whole-system-at-the-wall are different quantities and are not comparable. Record scope explicitly per channel, and make the dashboard refuse to plot mixed scopes on one axis. This is the single most likely way for this feature to produce a confidently wrong cross-machine comparison.
7. **Record idle baseline power** before the run's first measured case, so a reader can distinguish incremental workload energy from total system draw. Report both; do not silently subtract.
8. **Supply item 3's re-qualification matrix** with every power source and interval; memory-only evidence does not qualify `powermetrics`, `nvidia-smi`, `rocm-smi`, or RAPL. Apply the shared observer-effect policy.
9. **Persist through the event journal and export** per case and per run, exactly as item 1 does, under the same schema-5 block.
10. **Apply the shared telemetry privacy and outbound policy** to normalized power source, scope, availability, interval, and derived measurements.
11. **Dashboard**: a tokens-per-joule chart per model, an efficiency column in `StatsTable`, and run-level total energy on `RunSummaryCards`. Scope is displayed on every power figure. Runs without power data show an explicit unavailable state with its reason.
12. **Tests**: each platform parser against captured real output including error, permission-denied, and truncated forms; integration over uneven timestamps including a single sample and an empty series; efficiency derivations including a zero-work guard; mixed-scope rejection; availability detection without item 2; and outbound redaction. Vitest covers the efficiency chart builders and unavailable path. Real-hardware qualification supplies the observer-effect evidence.
13. **Docs**: [platform-tuning.md](docs/platform-tuning.md) for permission requirements per platform; [workloads.md](docs/workloads.md) for what each metric means; [limitations.md](docs/limitations.md) for observer effect, scope non-comparability, and the fact that idle baseline varies with background load; [troubleshooting.md](docs/troubleshooting.md) for unavailable-power remedies.

## Acceptance criteria

- [x] A run on a supported platform records energy in joules per measured case and per run, with idle baseline recorded separately.
- [x] Tokens per joule, images per joule, and embeddings per joule are derived and exported where the corresponding workload ran.
- [x] Every power figure carries an explicit scope; the dashboard never plots mixed scopes on one axis, and a test asserts this.
- [x] Missing permissions are detected by telemetry source discovery and reported before the run starts, with the reason recorded in the results; item 2's preflight displays the same result when present.
- [x] Unavailable power never fails a run and never records zero; it records unavailable with a reason.
- [x] Integration uses real sample timestamps, verified by a test with deliberately uneven spacing.
- [ ] Power, memory, and temperature samples share one timeline and one set of timestamps.
- [x] Each power source and sampling interval passes the shared observer-effect policy or receives the resulting opt-in/unsupported methodology status.
- [x] Power-source discovery works without item 2; preflight only presents the result when available.
- [x] Power telemetry passes the shared privacy and outbound policy.

---

# 5. Sustained-load and thermal degradation

## Why this is fifth

[limitations.md](docs/limitations.md) lists thermal state, power mode, ambient temperature, and background load as confounders and asks users to begin from comparable conditions. That is a disclaimer substituting for a measurement. On the thermally-constrained small systems this product targets, sustained throughput after ten minutes is frequently well below the first-thirty-seconds figure — and that gap is often the actual product difference between a well-cooled mini-PC and a thin laptop carrying identical silicon. Every existing workload measures a burst.

This completes the sampler trio and converts the project's largest documented confounder into one of its most differentiating results.

## Implementation outline

1. **Add temperature channels to the shared sampler**: CPU package, GPU die, and any available hotspot, per platform, using the same source-behind-an-interface pattern and the same pure-parser discipline as item 4.
2. **Create `scripts/workloads/sustained_benchmark.py`** implementing a soak workload: continuous generation at a fixed context depth for a configurable duration, defaulting to a value long enough to reach steady state on typical hardware and documented as such in `config.py`. It is opt-in via `--tests`, like the existing concurrency diagnostics, and is not part of the default evidence set.
3. **Record a throughput time series** in fixed windows across the soak rather than a single aggregate. These throughput windows are the telemetry sub-windows defined by item 1, so throughput, memory, power, and temperature aggregate over identical boundaries and retain their aligned series.
4. **Extract the degradation analysis as pure functions** operating on the recorded series — this is the AGENTS.md extract-before-testing rule, and the analysis is the part with real logic. Compute: initial throughput over an early window, steady-state throughput over a late window, the retention ratio between them, and a throttle-onset point defined as the first sustained departure from initial performance beyond a configured tolerance. "Sustained" must be a real requirement of consecutive windows, not a single dip, or ordinary variance will be reported as throttling.
5. **Correlate degradation with the telemetry timeline.** Because power, temperature, and throughput share timestamps from one sampler, report whether throughput decline coincides with a temperature ceiling, a power-limit decline, both, or neither. Neither is an important and honest outcome — it points at background load or memory pressure rather than thermal limits, and must not be reported as thermal throttling.
6. **Classify performance and cause separately.** Performance is stable, mild degradation, significant degradation, or indeterminate based only on a sufficiently long throughput series. Cause is temperature-correlated, power-correlated, both, neither, or unavailable. Missing temperature must not erase a valid throughput-retention result, and a stable throughput result must not be relabeled thermal merely because the device is hot.
7. **Feed the drift detector from item 3.** A trial set flagged for monotonic decline and a soak showing degradation are the same physical phenomenon observed at different timescales; they should reference each other rather than being analyzed independently.
8. **Supply item 3's re-qualification matrix** with every temperature source and the combined sampler; individual-source evidence does not qualify the combined mode. Record soak-specific and latency-sensitive effects separately under the shared observer-effect policy.
9. **Persist and export** as its own results section with the full normalized time series, not only the summary, so the dashboard can plot the shape and a reviewer can audit the classification.
10. **Apply the shared telemetry privacy and outbound policy** to normalized sensor class, availability, interval, and derived temperature measurements.
11. **Dashboard**: add a sustained section following the existing panel pattern in `components/panels/`, with a throughput-over-time line chart overlaying temperature and power, plus a retention-ratio summary. Add the section to `SECTIONS` and `SECTION_LABELS` in `constants.ts`, and a `utils/sustained.ts` module with its own Vitest file, matching the one-module-per-section convention.
12. **Tests**: retention and onset detection against synthetic series — flat, monotonic decline, single-dip-then-recover, noisy-but-stable, and a series too short to classify; the correlation logic across each combination of temperature and power evidence including neither; classification at every threshold boundary; and outbound redaction. Vitest covers the chart builders including a series with missing temperature. Real-hardware qualification supplies observer-effect evidence for each source and the combined sampler.
13. **Docs**: [workloads.md](docs/workloads.md) for the new workload, its opt-in status, defaults, and duration; [limitations.md](docs/limitations.md) revised so the thermal paragraph points at a measurement and discloses observer effect; [cli-reference.md](docs/cli-reference.md), [how-it-works.md](docs/how-it-works.md), [dashboard.md](docs/dashboard.md), [project-structure.md](docs/project-structure.md), and [testing.md](docs/testing.md) for the new files.

## Acceptance criteria

- [x] A soak run records a throughput time series, not only an aggregate, and exports the full series.
- [x] Initial throughput, steady-state throughput, retention ratio, and throttle-onset point are computed by pure, unit-tested functions.
- [x] Throttle onset requires a sustained departure across consecutive windows; a test asserts a single-window dip does not trigger it.
- [x] Performance retention and suspected cause are separate outputs; missing sensors can make cause unavailable without erasing a valid throughput result.
- [x] Degradation is correlated with temperature and power from the shared timeline, and "neither" is reported honestly rather than defaulting to thermal.
- [x] A soak too short to reach steady state classifies as indeterminate, never as stable.
- [x] The workload is opt-in and absent from the default evidence set.
- [x] The dashboard plots throughput, temperature, and power on one aligned time axis.
- [x] Every temperature source and the combined sampler pass the shared observer-effect policy or receive its opt-in/unsupported methodology status.
- [x] Temperature telemetry passes the shared privacy and outbound policy.

---

# 6. Goal-driven recommendation view

## Why this is sixth

[recommendation-policy.md](docs/recommendation-policy.md) is an unusually careful document describing constraint-first ranking, and no recommendation engine or view implements it. Server-side pieces exist in `decision_report.py` and `evidence_policy.py`, but nothing in `dashboard/src` mentions recommendations. A project records a workflow name, baseline, preset, and optional policy without those materially guiding configuration or interpreting the result.

This is the item that changes the product from a benchmark viewer into a decision tool. It ranks sixth rather than first only because a recommendation built on mean-versus-mean comparison would confidently rank models on differences it cannot distinguish from noise — item 3 is what makes it defensible.

## Implementation outline

1. **Create `scripts/results/recommendation.py`** implementing the documented policy as pure functions over a loaded result set. Make Python the authoritative evaluator for CLI, reports, and workspace actions. Define a versioned recommendation JSON artifact so the standalone dashboard renders the authoritative verdict instead of re-implementing the policy in TypeScript.
2. **Define a constraint set** as an explicit record: workload type, context depth, minimum accuracy, maximum acceptable TTFT, minimum throughput, concurrency, memory ceiling, and — now available from items 1, 4, and 5 — a memory-headroom requirement and an optional efficiency constraint. Every field is optional; an absent constraint is not a zero-valued one.
3. **Implement evidence eligibility first.** A candidate is eligible only if it has compatible evidence for every constraint the user set, at the requested context depth, under a compatible methodology. Reuse the existing compatibility gate rather than writing a second one. A candidate with no evidence at 32K is not a poor candidate at 32K; it is an unevaluated one, and the two must never merge.
4. **Apply constraints as hard filters before any ranking**, per the documented constraint-first rule. A candidate failing a stated minimum is eliminated and reported as eliminated with the constraint and the measured value that eliminated it — never silently ranked low.
5. **Rank survivors without an opaque composite score.** Order by the user's stated primary objective. Use only item 3's qualified repeated-trial verdicts to break close comparisons; report unchanged candidates as tied and return insufficient evidence when the requested ranking requires reproducibility that the trial set cannot establish.
6. **Produce one of three verdicts** — recommended, tied, or insufficient evidence — and never a bare number. Insufficient evidence must state precisely what is missing and what run would resolve it, so the verdict is actionable rather than a dead end.
7. **Link every conclusion to its evidence**: each eliminated candidate to the measurement that eliminated it, each ranking to the chart rows and raw samples behind it. A recommendation a reviewer cannot audit is worth nothing in this product's stated setting.
8. **Render the versioned artifact in the standalone static dashboard**, showing recommended, tied, other eligible, eliminated (with reasons), and unevaluated (with what is missing) as five visually distinct groups. Do not pretend a browser-only page can call the Python evaluator, and do not make unevaluated look like a failure. Interactive constraint entry and integrated workspace placement remain item 11 work.
9. **Add a CLI path** that evaluates constraints and writes the versioned artifact for automation, reports, and standalone-dashboard review.
10. **Include the recommendation in the generated report** via `decision_report.py`, with its constraints, verdict, evidence links, and limitations, so an exported report is self-contained.
11. **Tests**: eligibility against complete, partial, and incompatible evidence; each constraint type as a hard filter at and across its boundary; tie detection driven by item 3's verdicts; insufficient-evidence output naming the specific gap; a regression test that an eliminated candidate never appears in the ranked list, and that no code path emits a composite score. Vitest coverage for the constraint form and each result group.
12. **Docs**: [recommendation-policy.md](docs/recommendation-policy.md) updated from describing intended behavior to describing implemented behavior; [user-journey.md](docs/user-journey.md) for the new flow; [dashboard.md](docs/dashboard.md), [reports.md](docs/reports.md), and [limitations.md](docs/limitations.md) for what a recommendation cannot establish.

## Acceptance criteria

- [x] A user can state workload, context, accuracy, latency, throughput, concurrency, memory, and efficiency constraints, leaving any of them unset.
- [x] Candidates without compatible evidence are reported as unevaluated, visually distinct from eliminated, with the missing evidence named.
- [x] Constraints are applied as hard filters before ranking; each eliminated candidate names the constraint and the measured value that eliminated it.
- [x] Candidates classified unchanged by qualified repeated trials are tied; inconclusive trial evidence never creates an ordering.
- [x] The output is always recommended, tied, or insufficient evidence — no opaque composite score exists on any code path, asserted by test.
- [x] Every conclusion links to its aggregate/chart row and raw evidence path.
- [x] Insufficient evidence states what run would resolve it.
- [x] Python computes the verdict once; CLI, report, and standalone dashboard render the same versioned artifact, verified with a shared conformance fixture. Integrated workspace rendering remains item 11 work.

---

# 7. Complete case-level resume for every workload

## Why this is seventh

Event-journal adoption is partial. Only [llm_event_stage.py](scripts/results/llm_event_stage.py) and [native_bench_event_stage.py](scripts/results/native_bench_event_stage.py) exist; embeddings, images, and the five accuracy workloads have no stage and fall back to JSON-level recovery. [architecture-decisions.md](docs/architecture-decisions.md) already identifies this dual ownership as a known split. The user-visible consequence is that an interruption late in a multi-hour accuracy or image run re-runs work that already succeeded.

It ranks below the items above because it costs users time rather than correctness — a lost run is recoverable by re-running; a wrong conclusion is not. It is nonetheless real reliability work that users feel immediately.

## Implementation outline

1. **Extract only proven common helpers** from the two existing stages: case identity, attempt tracking, legal transitions, and export checks. Keep workload-specific stages focused. Do not introduce a new base class unless an architecture decision first proves that direct functions and dataclasses are insufficient.
2. **Define case identity per remaining workload** — the unit that must not be re-run: one question for each accuracy bank, one prompt-and-resolution pair for images, one input batch for embeddings. Case identity must be stable across process restarts and must incorporate the question-bank hash, workflow identity, or input-corpus identity as applicable, so changed evidence invalidates rather than silently resumes onto mismatched work.
3. **Implement an accuracy event stage** covering all five banks. These share scoring and structure, so one stage parameterized by bank is correct; five near-duplicates are not. Preserve the existing partial-response and timeout semantics exactly — a timed-out question is scored from its partial text and the bank continues, and that must remain true across a resume boundary.
4. **Implement an embeddings event stage**, following the same pattern.
5. **Implement an image event stage**, with attention to the ComfyUI boundary: a case is complete only when its output is durably written, so an interruption mid-generation resumes rather than recording a partial success.
6. **Register all stages in the existing [scripts/stage_registry.py](scripts/stage_registry.py) and orchestration registry** so recovery, inspection, execution, and fork paths agree rather than special-casing workload types.
7. **Retire each workload's JSON-level recovery immediately after that workload proves event-store parity**, rather than waiting for one all-workload switch. A deferred workload stays wholly on its old authority and is recorded as a named exception with the failed parity gate; it must never shadow-write or expose two recovery paths.
8. **Verify resume identity gating** applies uniformly: plan, artifact, runtime, and methodology identity must all match before continuation, for the new stages exactly as for the existing ones. Preserve the safe-case-boundary rule from the PRD.
9. **Extend `recovery_inspector.py` and the GUI recovery view** so a user can see which cases completed and which remain for the newly covered workloads.
10. **Tests**: for each new stage — resume after interruption at the first, middle, and last case; a bank-hash change invalidating resume; a timed-out case resuming with its partial score intact; a duplicate case identity rejected; recovery inspection reporting accurate remaining work. Add a cross-cutting test that every workload in the registry has a stage, so a future workload cannot silently regress to JSON recovery.
11. **Docs**: [architecture-decisions.md](docs/architecture-decisions.md) records every completed migration and any named deferral with its failed parity gate; [reliability-architecture-plan.md](docs/reliability-architecture-plan.md), [data-lifecycle.md](docs/data-lifecycle.md), and [how-it-works.md](docs/how-it-works.md) describe the exact current authority for each workload.

## Acceptance criteria

- [ ] Every workload — accuracy, embeddings, images, and native — has a registered event stage.
- [ ] An interruption re-runs only unfinished cases, verified per workload by a test interrupting at first, middle, and last case.
- [ ] A question-bank change invalidates resume rather than continuing onto mismatched evidence.
- [ ] Timeout and partial-response semantics are preserved exactly across a resume boundary.
- [ ] JSON-level recovery is removed for every migrated workload; any deferral is a named exception with its parity failure recorded, remains wholly on the old authority, and exposes no second recovery path.
- [ ] A test asserts every registered workload has a stage, preventing silent regression.
- [ ] Recovery inspection reports accurate remaining work for every workload.

---

# 8. Evidence-backed platform qualification

## Why this is eighth

[limitations.md](docs/limitations.md) states plainly that nothing in this project currently runs vLLM and that the cross-engine weights have not been validated on real hardware. Intel Arc and parts of ROCm are similarly described as unverified. Yet [vllm.py](scripts/runtime/engines/vllm.py), [vllm_install.py](scripts/setup/vllm_install.py), and [vllm_benchmark.py](scripts/workloads/vllm_benchmark.py) carry substantial user-facing surface area, and the dashboard has a `VllmBenchPanel`. The honesty currently lives in documentation while the UI presents these as ordinary options.

This is mostly process rather than code, which is exactly why it is worth doing — the cost is low and the trust return is high.

## Implementation outline

1. **Define the qualification lifecycle** as an explicit ordered checklist a platform must pass end to end: install, discovery, first valid run, cancellation, resume, report generation, bundle export, upgrade, rollback, and uninstall. Partial passes are recorded as partial, not rounded up.
2. **Create `scripts/release/qualification.py`** holding the matrix as data — platform, runtime, version, GPU backend, date, suite version, lifecycle results, and known failures — with pure functions to validate an entry and to derive a support level from it. Support level is derived from evidence, never hand-set.
3. **Define three support levels** with explicit evidence requirements: supported (full lifecycle passed on a recorded date and suite version), experimental (partial evidence, specific known gaps recorded), and unverified (no qualification evidence). Absence of evidence yields unverified, which is the default for anything not deliberately qualified.
4. **Enforce support level in the UI**, not only in docs. An experimental or unverified engine is labeled at the point of selection, and choosing it records that choice in the run profile so any resulting evidence carries the caveat with it permanently.
5. **Gate vLLM behind an explicit experimental acknowledgment** until it passes qualification on real hardware, consistent with what [limitations.md](docs/limitations.md) already says. Do not remove the code; label it accurately.
6. **Mark WSL2 runs in the matrix** as their own platform row rather than a variant of Linux — the existing `wsl: true` profile flag and its dashboard tag already establish this distinction, and the matrix should follow it rather than contradict it.
7. **Generate the published matrix from the data**, so a documented claim cannot drift from recorded evidence. Add a release-readiness check that fails when a platform is presented as supported without a qualification record, wiring into the existing [release_readiness.py](scripts/release/release_readiness.py) gate.
8. **Add qualification date and suite version to every entry**, and treat an entry older than a defined number of releases as stale, surfacing it as such rather than silently trusting it.
9. **Dashboard**: display the support level of the engine that produced each loaded file, so a reader comparing two runs sees immediately that one came from an unverified path.
10. **Tests**: support-level derivation from complete, partial, and absent evidence; staleness detection at the boundary; the readiness gate failing on an unsupported claim; matrix generation matching the underlying data; a test that a result produced under an experimental engine carries the caveat in its profile.
11. **Docs**: a generated support matrix in [engines.md](docs/engines.md) and [setup.md](docs/setup.md); [governance.md](docs/governance.md) and [release-policy.md](docs/release-policy.md) for the qualification requirement; [limitations.md](docs/limitations.md) updated to reference the matrix rather than restating status in prose.

The maintainer builds validation and presentation, performs each real install/run/lifecycle checklist, records the evidence and failures, and applies the derived support rule without silently changing it. No automated test may claim that hardware was qualified.

## Acceptance criteria

- [ ] A machine-readable matrix records platform, runtime, version, backend, date, suite version, and per-lifecycle-step results.
- [ ] Support level is derived from evidence; no code path sets it manually, asserted by test.
- [ ] Anything without qualification evidence is unverified by default.
- [ ] Experimental and unverified paths are labeled in the UI at selection time, and the choice is recorded in the run profile.
- [ ] vLLM is presented as experimental until it passes the full lifecycle on real hardware.
- [ ] WSL2 is its own matrix row, consistent with the existing profile flag.
- [ ] Release readiness fails when a support claim lacks a qualification record.
- [ ] The published matrix is generated from the data and cannot drift from it.

---

# 9. Model catalog audit and refresh

## Why this is ninth

The catalog is a benchmark methodology, not a list of whatever models are newest. Each selected model consumes download space and hours of repeated workload time, represents a parameter tier or capability role, and becomes part of comparisons that users expect to remain intelligible across releases. The current lineup therefore needs a deliberate Version 6 audit rather than ad hoc additions whenever a compelling release appears.

The initial candidate register is deliberately broader than the expected set of accepted changes: Qwen 3.8 27B, Muse Glimmer 30B, Nemotron 3.5 Nano/Lightning 30B-A3B, Gemma 4 26B-A4B, and Nemotron Nano 9B v2 for the LLM tiers; EmbeddingGemma 300M, Qwen3 Embedding 0.6B, and Qwen3 Embedding 4B for embeddings; and FLUX.2 Klein 4B and Z-Image Turbo for image generation. These are audit inputs, not predetermined additions. Candidates that occupy roles already represented by the catalog are evaluated as possible replacements—not appended merely because they can run. The audit also covers every existing catalog entry so an older model is not retained only through inertia.

Repository entries below are starting points for the audit, not compatibility claims. The audit pins a revision and exact artifact only after testing; community GGUF conversions require provenance and architecture checks, and a safetensors repository must still pass the targeted vLLM lifecycle before it is considered supported.

| Candidate | Safetensors source for vLLM audit | GGUF source for llama.cpp audit | Repository status |
| --- | --- | --- | --- |
| Qwen 3.8 27B | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Upstream weights; community GGUF |
| Muse Glimmer 30B | [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) | [`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) | Publisher-hosted weights and GGUF; exact quantization still to select |
| Nemotron 3.5 Nano/Lightning 30B-A3B | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16) | [`ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | Candidate naming and base/instruct identity must be reconciled before testing |
| Gemma 4 26B-A4B | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | [`ggml-org/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) | Upstream weights; llama.cpp project GGUF |
| Nemotron Nano 9B v2 | [`nvidia/NVIDIA-Nemotron-Nano-9B-v2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | [`bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF`](https://huggingface.co/bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF) | Upstream weights; community GGUF |
| EmbeddingGemma 300M | [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) | [`cstr/embeddinggemma-300m-GGUF`](https://huggingface.co/cstr/embeddinggemma-300m-GGUF) | Upstream weights; provisional community GGUF |
| Qwen3 Embedding 0.6B | [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | [`Qwen/Qwen3-Embedding-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) | Upstream weights and GGUF |
| Qwen3 Embedding 4B | [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | [`Qwen/Qwen3-Embedding-4B-GGUF`](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF) | Upstream weights and GGUF |
| FLUX.2 Klein 4B | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Not applicable | Safetensors/ComfyUI candidate; not a llama.cpp workload |
| Z-Image Turbo | [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Not applicable | Safetensors/ComfyUI candidate; not a llama.cpp workload |

## Implementation outline

1. **Inventory the current catalog by purpose**, recording for each model its tier, architecture, dense or sparse parameter count, capability roles, context requirement, license, selected quantization, download size, supported engines, and the reason it earns benchmark time. Distinguish general instruction following, reasoning, code, tool use, long context, multimodal behavior, and architecture diversity rather than treating parameter count as the only axis.
2. **Define selection and retirement criteria before scoring candidates**: relevance to contemporary local inference, nonredundant role coverage, weights and license suitable for redistribution-by-reference and local use, a stable upstream identity, a supported GGUF conversion, clean llama.cpp lifecycle behavior, required context support, feasible resource coverage across the intended tier, and measurable value in this suite's workloads. Popularity or vendor benchmark claims alone are insufficient.
3. **Audit every incumbent under the same criteria.** Record whether it remains, moves tier, becomes legacy-only, or is proposed for replacement, with an explicit reason and named replacement candidate. Model tiers remain cumulative, and the audit must estimate the run-time and disk impact of the proposed lineup before changing it.
4. **Evaluate the registered LLM candidates**—Qwen 3.8 27B, Muse Glimmer 30B, Nemotron 3.5 Nano/Lightning 30B-A3B, Gemma 4 26B-A4B, and Nemotron Nano 9B v2—using their exact upstream model cards, licenses, architecture/configuration, context semantics, chat templates, and maintained GGUF artifacts. Verify current llama.cpp support rather than inferring compatibility from a community launch report; separately record optional multimodal projectors, speculative-draft dependencies, MTP, or custom flags so the baseline catalog does not silently depend on them.
5. **Freeze an explicit sampling policy across engines.** The comparable baseline sends temperature zero and explicit neutral values for every supported logit modifier, including presence and frequency penalties, rather than inheriting llama.cpp, vLLM, or model-repository defaults. Store the fully resolved sampler profile in the run plan and result methodology identity. Separately capture publisher-recommended settings from a pinned upstream model card as an opt-in named profile; those runs remain visibly distinct and may never pool or compare as baseline-equivalent evidence.
6. **Run a small compatibility screen before full evaluation**: setup discovery, download/inventory identity, model load and unload, 2K and the model's relevant deeper context, deterministic completion, chat template behavior, cancellation, and clean recovery. A model that requires an engine patch or unreleased runtime remains a documented candidate rather than entering the default catalog.
7. **Compare candidates with incumbents using existing evidence**, including accuracy categories, single-shot prefill/decode, conversation growth, tool and chat concurrency, memory headroom, sustained behavior, and supported context. Use item 3's repeated-trial verdicts for close performance claims; do not rank a noisy one-off delta as an improvement.
8. **Audit embedding and image candidates under workload-specific criteria.** Compare EmbeddingGemma 300M and Qwen3 Embedding 0.6B/4B with the current Nomic and MixedBread entries on runtime support, dimensionality, context, multilingual and instruction-aware behavior, throughput, memory, license/access friction, and retrieval quality once the suite has a defensible quality measure. Compare FLUX.2 Klein 4B and Z-Image Turbo with the current image lineup on ComfyUI lifecycle support, complete dependency size, peak memory, resolution, steps, latency, license, and prompt/image quality under a fixed workflow. Speed alone cannot justify replacing an embedding or image model.
9. **Make replacement decisions role-by-role.** Prefer the smallest lineup that preserves meaningful architecture and capability coverage. Adding a candidate without retiring an overlapping incumbent requires a documented distinct role that the existing workload suite can actually measure.
10. **Preserve historical rendering.** Removed catalog entries move to the dashboard's legacy model registry with their labels, colors, tier metadata, and result lookup intact. Existing result files remain readable and comparable where methodology permits; catalog retirement never rewrites old evidence.
11. **Apply accepted changes as separate reviewable commits** for catalog/setup identity, dashboard legacy/current registries, tests, and documentation. The audit report lands before any model change so reviewers can challenge the evidence without also reviewing implementation churn.
12. **Tests**: catalog uniqueness and tier consistency; every current entry has required audit metadata; accepted tags, repos, files, and sizes agree across setup and dashboard registries; baseline sampler payload parity across llama.cpp and vLLM; publisher-profile identity separation and unsupported-setting handling; retired models remain renderable through legacy mappings; cumulative tier selection remains unchanged; malformed or duplicate candidate identities are rejected.
13. **Docs**: publish the audit date, tested commit and engine version, selection rubric, incumbent decisions, candidate evidence, disk/run-time impact, sampler profiles, and unresolved compatibility gaps in [catalogs.md](docs/catalogs.md), [workloads.md](docs/workloads.md), and [methodology-contract.md](docs/methodology-contract.md). Vendor claims and community reports may motivate a candidate but are not recorded as suite qualification evidence.

## Acceptance criteria

- [ ] Every incumbent has a recorded role and an evidence-backed keep, replace, move, or legacy-only decision.
- [ ] Selection and retirement criteria were committed before candidate benchmark results were interpreted.
- [ ] Every registered LLM candidate has exact upstream identity, license, context, architecture, GGUF, and current llama.cpp compatibility recorded.
- [ ] Every candidate has a pinned safetensors and GGUF repository plus exact artifact and revision, or an explicit not-applicable entry, before compatibility testing begins.
- [ ] The baseline explicitly pins every supported sampling control across llama.cpp and vLLM; it never silently inherits engine or repository defaults.
- [ ] Publisher-recommended sampling is an opt-in, source-pinned profile with a distinct methodology identity and is never pooled with the deterministic baseline.
- [ ] EmbeddingGemma 300M and Qwen3 Embedding 0.6B/4B are compared with both current embedding entries, with quality limitations stated explicitly.
- [ ] FLUX.2 Klein 4B and Z-Image Turbo are compared with the current image lineup using complete pipeline size and a fixed quality/performance workflow.
- [ ] Candidate compatibility screens cover load/unload, context, completion, chat template, cancellation, and recovery without making setup or live benchmark execution automatic.
- [ ] Any accepted addition fills a measurable role or replaces an incumbent; novelty alone cannot expand the default lineup.
- [ ] Proposed catalog disk cost and representative run-time impact are reported before adoption.
- [ ] Close performance claims use item 3's repeated-trial verdicts and may remain inconclusive.
- [ ] Retired entries remain in the legacy dashboard registry and old results render unchanged.
- [ ] The audit report is reviewable before catalog implementation changes begin.

---

# 10. Quantization comparison workflow

## Why this is tenth

Quantization is fixed at one variant per catalog entry — every model in [models.py](scripts/workloads/models.py) carries a single `Q4_K_M` tag and `hf_repo`. Which quantization to run is one of the top questions a local-AI user faces, and this suite is unusually well-placed to answer it properly: it already has the accuracy banks to measure quality loss, the speed harness to measure the throughput gain, and — after items 1 and 4 — the memory and energy measurement to complete the tradeoff. Almost nothing else answers this with quality evidence attached.

The initial scope is llama.cpp and GGUF. A single base model may select several GGUF files—including multiple quantizations stored in one Unsloth or other Hugging Face repository—and execute them sequentially in one unattended, resumable run. Native vLLM quantization formats remain outside this milestone until a separate methodology defines which formats are comparable to GGUF and how their engine-specific effects should be reported.

It ranks tenth because it multiplies run time and disk consumption substantially, and it serves the enthusiast more directly than the hardware-vendor team the PRD names.

## Implementation outline

1. **Generalize the catalog entry** so a llama.cpp model may declare multiple GGUF variants, each with its own tag, repository, filename, size, and quantization label. Several variants may reference different files in the same Hugging Face repository. Preserve the existing single-variant shape as the default so no existing entry changes meaning and no existing results file becomes unreadable.
2. **Extend model identity** in [model_identity.py](scripts/runtime/model_identity.py) so a variant is a distinct identity for evidence purposes. Two quantizations of one base model must never pool into one evidence set — this is the central correctness requirement of the feature.
3. **Add explicit variant selection to every run surface**, defaulting to the single documented variant so ordinary runs are unchanged in time and disk. The desktop GUI shows the available GGUF variants for each selected base model as checkboxes, with the catalog default checked and disk size shown beside each option; select-all and clear controls make larger sweeps manageable. The CLI accepts repeatable model-qualified variant selectors so headless runs express the same set without prompts. An explicitly selected set runs sequentially in one invocation across every selected workload and model/quantization pair without another prompt or restart.
4. **Extend setup and download** to enumerate and fetch selected GGUF filenames from a shared or per-variant repository, reusing the existing resumable download and inventory paths rather than adding a second acquisition route. The run journal checkpoints every completed variant so an interrupted multi-quant sweep resumes at the next incomplete unit.
5. **Report the cost of a sweep before it starts** — added disk, added download, added run time — using the existing estimation path from [result_history.py](scripts/results/result_history.py). A user must not discover a 4x run time after committing to it.
6. **Extract the tradeoff analysis as pure functions**: for a base model across its variants, compute quality delta from the accuracy banks, throughput delta, memory delta from item 1, and energy delta from item 4, each relative to a chosen reference variant. Include item 3's qualified trial verdicts so an inconclusive or unchanged quality difference is not over-interpreted.
7. **Feed variants into the recommendation engine** from item 6 as ranked candidates, so "which quantization on this machine" is answered by the same constraint-first machinery rather than a parallel implementation.
8. **Dashboard**: a per-base-model variant comparison showing quality, speed, memory, and energy together, with unchanged and inconclusive quality differences visibly marked. Extend `constants.ts` with variant labels and ordering.
9. **Tests**: catalog parsing for single-variant and multi-variant entries including a malformed variant list; GUI and CLI selection normalization, defaults, duplicates, unknown model/variant pairs, and an empty selection; identity separation asserting two variants never pool; sweep cost estimation; each tradeoff computation including a missing-variant and a single-variant case; and unchanged/inconclusive trial verdicts. Vitest covers the variant comparison builders.
10. **Docs**: [workloads.md](docs/workloads.md) for the catalog shape and sweep behavior; [cli-reference.md](docs/cli-reference.md) for the flags; [setup.md](docs/setup.md) for variant download and disk cost; [catalogs.md](docs/catalogs.md) for the extended entry format; [limitations.md](docs/limitations.md) for what a cross-quantization comparison does and does not establish, extending the existing per-engine-weights reasoning.

## Acceptance criteria

- [ ] A catalog entry may declare multiple quantization variants; existing single-variant entries are unchanged in meaning.
- [ ] Each variant is a distinct evidence identity; a test asserts two variants of one base model never pool into one evidence set.
- [ ] One invocation runs every selected GGUF variant sequentially without further input and resumes without repeating completed variants.
- [ ] The GUI provides per-model quantization checkboxes with the default preselected, visible artifact sizes, and select-all/clear controls; only checked variants enter the run plan.
- [ ] The CLI can express the same model-qualified variant selection noninteractively, and invalid, duplicate, or empty selections fail before download or execution.
- [ ] Multiple selected GGUF filenames may resolve from one Hugging Face repository without duplicating repository metadata or downloads.
- [ ] A sweep is opt-in, and its added disk, download, and run time are reported before it starts.
- [ ] Variant comparison reports quality, throughput, memory, and energy deltas against a stated reference variant.
- [ ] Quality differences classified unchanged or inconclusive by item 3 are not presented as rankings.
- [ ] Variants are ranked by the item 6 engine rather than a parallel implementation.
- [ ] Older results files whose model identity has no explicit base-model/variant fields load and render unchanged through a documented legacy mapping.
- [ ] The initial implementation is explicitly llama.cpp/GGUF-only; vLLM-native quantization comparison is not implied by these results.

---

# 11. Unified results and decision workspace

## Why this is last

The friction is real: the desktop GUI owns configuration, execution, history, recovery, bundles, and reports, while chart exploration happens in a separately launched browser application, and acceptance-policy evaluation is a file-driven action rather than a persistent part of review. A user reviewing results moves between Result History, a launched dashboard, policy dialogs, and report creation, with a genuine risk that what was exported is not what was being viewed.

It is last because it is the largest build for the least new capability — it consolidates rather than adds — and because its central job is displaying a decision next to its evidence. Building the container before item 6 defines the decision means building it twice.

## Implementation outline

1. **Decide the integration approach explicitly and record it** in [architecture-decisions.md](docs/architecture-decisions.md) before writing code: embed the existing dashboard build in a webview inside the GUI, or drive a single shared selection state between the two surfaces. Either is defensible; an unrecorded drift between them is not. Do not fork the dashboard — a second chart implementation would immediately diverge from the tested `utils/*.ts` modules.
2. **Build a disposable proof of concept for both viable approaches** before committing to either. Verify Windows, macOS, Linux, offline startup, accessibility basics, packaging size, clean shutdown, and file handoff. Record the result and delete the rejected prototype; prototype code does not become a permanent compatibility path.
3. **Define one selection state** as the single source of truth for which runs are under review, which is the baseline, and which policy is applied — owned in one place and consumed by charts, comparison, policy evaluation, recommendation, report, and bundle alike. This is the actual fix for the export-mismatch risk; the visual consolidation is secondary.
4. **Surface compatibility and methodology warnings in the selection itself**, so an incompatible comparison is visible at selection time rather than discovered at export. Reuse the existing gate rather than re-implementing it.
5. **Make acceptance-policy evaluation persistent and visible** within the workspace: apply, view, and edit thresholds with results updating in place, rather than a one-shot file-driven action.
6. **Place the item 6 recommendation in the workspace** next to the evidence it draws on, which is the arrangement that makes the recommendation auditable in practice rather than only in principle.
7. **Generate reports and bundles from the same selection state**, so an exported artifact provably matches what was on screen. Record the selection state in the exported artifact so the match is verifiable after the fact, not merely asserted.
8. **Keep the standalone dashboard working.** `launch_dashboard.sh` remains supported for headless, remote, and sample-file review; the workspace is an additional surface, not a replacement, and the shared `utils/*.ts` modules stay the one implementation behind both.
9. **Preserve offline behavior** throughout, per [offline-mode.md](docs/offline-mode.md). An embedded webview must not introduce a network dependency; this is a hard requirement given the embargo constraints in the PRD.
10. **Tests**: selection-state transitions including incompatible selections and baseline changes; the invariant that report and bundle output derives from the same state that produced the on-screen view; policy edits re-evaluating in place; offline operation with no network available. Vitest coverage for shared state consumed by chart builders.
11. **Docs**: [dashboard.md](docs/dashboard.md), [user-journey.md](docs/user-journey.md), [architecture-decisions.md](docs/architecture-decisions.md), [reports.md](docs/reports.md), and [project-structure.md](docs/project-structure.md).

## Acceptance criteria

- [ ] Selecting runs, viewing charts and raw samples, setting a baseline, evaluating policy, reading the recommendation, and generating report and bundle happen without leaving one surface.
- [ ] One selection state drives every consumer; a test asserts report and bundle output derives from the same state as the displayed view.
- [ ] The exported artifact records the selection state that produced it.
- [ ] Compatibility and methodology warnings appear at selection time, not at export.
- [ ] Acceptance policy is persistent and editable in place, with results updating.
- [ ] The standalone dashboard continues to work unchanged for headless and sample review.
- [ ] No chart logic is duplicated; both surfaces consume the same `utils/*.ts` modules.
- [ ] The workspace operates fully offline.

---

## How the eleven milestones become one release

Version 6 is developed in small, usable slices. “All eleven features are coded” is not the release strategy.

### Slice A — trustworthy existing evidence

Complete milestones 1, 2, and 3 (**L + M + XL**). This is intentionally the largest foundational slice, so deliver it as A1 memory collection and coarse screen, A2 compatibility preflight, and A3 repeated trials plus final memory re-qualification. At the end, users can see qualified actual memory use, invalid model setups are caught before measurement, and repeated trials can support an honest regression verdict. Run a compatibility audit against old results, bundles, reports, policies, resume journals, and the standalone dashboard before moving on.

### Slice B — new efficiency evidence

Complete milestones 4 and 5 (**XL + L**). Deliver B1 power/energy before B2 soak/thermal. Qualify each supported sensor source separately on real hardware. A platform without a qualified source still runs benchmarks normally and reports telemetry unavailable, subject to the minimum Version 6 telemetry floor. Do not delay correctness on one platform by inventing a fake cross-platform lowest common denominator.

### Slice C — decisions and durability

Complete milestones 6 and 7 (**L + XL**). Deliver recommendations independently of the workload migrations. Recommendations remain disabled unless their required evidence is present. Migrate one remaining workload at a time to the event store, prove parity, then remove that workload's old ownership path; never switch all workloads in one unreviewable change.

### Slice D — scope and product integration

Complete milestones 8, 9, 10, and 11 (**M + M + L + XL**). Deliver D1 qualification matrix, D2 model-catalog audit, D3 quantization workflow, and D4 workspace separately; they are not assumed to fit one equal-length iteration. Publish only support claims backed by recorded qualification evidence. Review the model audit before changing the lineup, keep quantization sweeps opt-in, and choose the workspace architecture only after the disposable cross-platform prototypes have been reviewed.

### Pilot at every slice boundary

Do not start the next slice until a human completes this short pilot for the slice just finished. Use one real machine from every platform class the release currently claims; when a class is temporarily unavailable, record it as an open pilot gate rather than assuming another platform covers it.

1. Start from a clean copy and manually follow the normal documented installation or upgrade path. Do not automate the real setup entrypoint as a test.
2. Run one small representative project exercising the slice's new capabilities.
3. Interrupt the run once, inspect the preserved evidence, and perform the supported resume or fork action.
4. Load and compare one representative schema-4 result, checking explicit missing/new states.
5. Open the result in the dashboard and create every report, bundle, or derived artifact changed by the slice.
6. Record platform, version, result paths/digests, outcome, defects, and the evidence reference in the tracking issue.
7. Fix every release-blocking defect with a regression test and repeat the affected pilot step before beginning the next slice.

Slice A's pilot is mandatory before sampler expansion in Slice B because it validates the new schema, memory persistence, preflight, comparisons, and recovery on real systems. Slice B's pilot validates sensor permissions, observer effect, unavailable paths, and soak cleanup. Slice C's pilot validates recommendations and each migrated workload's interruption path. Slice D's pilot validates qualification labels, quantization identity, workspace packaging, and offline operation.

### Final pilot procedure

After all four slice pilots pass, run the complete controlled pilot. This is a human-operated release qualification, not a substitute for the earlier pilots or a unit test.

1. Choose at least one qualified machine from each support class the release intends to claim.
2. Start with a clean copy and follow the normal documented setup path manually.
3. Run a tiny safe smoke configuration first, then one representative Version 6 project with repeated trials and available telemetry.
4. During separate pilot runs, cancel once, pause once, resume once, deny one optional sensor permission, and load one schema-4 result.
5. Verify results, raw samples, event journal, dashboard, recommendation artifact, report, bundle export/import, support bundle preview, and deletion ownership.
6. Compare the report against the dashboard and raw JSON. Every displayed number must trace to the same stored evidence and units.
7. Record defects, fix them with regression tests, and repeat the affected pilot step.
8. Complete and record the qualification, security/privacy, documentation, and release checks required by [release-policy.md](docs/release-policy.md).

### Rollback and failure rules

- A new optional measurement that fails must become unavailable; it must not stop the benchmark.
- A model validity failure may exclude only the affected model or workload and must preserve the reason.
- A persistence or identity failure stops new measurement, preserves already durable evidence, and offers only the safe resume/fork action.
- A schema-5 writer bug blocks release. Readers must continue accepting older files; do not rewrite a user's source result to “fix” it.
- If a migrated workload cannot prove export parity and interruption safety, keep that workload on its old single authority for the current slice. Never ship shadow writes.
- If a platform telemetry source cannot be qualified, label it unsupported or experimental and ship the rest without it.
- If the integrated workspace fails cross-platform or offline qualification, keep the standalone dashboard and desktop GUI; do not block the measurement improvements on UI consolidation.

### Final handoff package

The maintainer assembles one review folder containing the final schema map, methodology identities, compatibility report, automated-test logs, dashboard screenshots, qualification matrix and evidence references, pilot checklist, known limitations, migration/rollback notes, SBOM and license outputs, security scan results, and draft release notes. The release is not concluded without these assembled artifacts; memory or verbal assurance is not a substitute for a reconstructable record.

---

## Release gate for Version 6

Version 6 ships when every item above meets its acceptance criteria and the following hold across the release as a whole.

- [ ] `bash tests.sh` passes; `npm test`, `npm run lint`, and `npx tsc --noEmit` pass from `dashboard/`.
- [ ] Schema-4 results files load, render, and compare correctly, with new telemetry shown as not recorded rather than zero.
- [ ] No feature reports a difference as reproducible without a repeated-trials verdict; a practical-threshold-only result is labeled exactly that.
- [ ] Single-run dispersion is never presented as proof of run-to-run reproducibility; regression verdicts use qualified repeated trials.
- [ ] Every new measurement records its source, scope, and availability, and degrades to an explicit unknown rather than to a plausible-looking zero.
- [ ] Every telemetry source claimed as supported has real-hardware qualification evidence; captured parser fixtures alone are insufficient.
- [ ] Memory-only, each power source, each temperature source, and the combined sampler meet their predeclared observer-effect bounds; telemetry modes that do not meet the bound are opt-in or unsupported with distinct methodology treatment.
- [ ] Comparisons enforce the recorded telemetry-on/off methodology decision instead of assuming schema compatibility means measurement comparability.
- [ ] The minimum telemetry outcome is met: qualified memory on one discrete-GPU and one unified-memory class, qualified scoped energy on one class, and qualified throughput-plus-temperature soak evidence on one thermally constrained class.
- [ ] Idle, model-load, and measured-case windows are distinct, and reports state which window each memory or energy value covers.
- [ ] Cancel, pause, resume, retry, fork, export/import, regrade, report, support preview, and owned deletion have been checked for every affected new artifact.
- [ ] Privacy review confirms that telemetry, qualification, trial-set, selection-state, and recommendation artifacts contain no credentials or unintended private identity.
- [ ] [limitations.md](docs/limitations.md) is revised so that every paragraph now backed by a real measurement says so, and every remaining disclaimer is one the suite genuinely cannot measure.
- [ ] The support matrix is generated from qualification evidence and no platform is presented as supported without it.
- [ ] The controlled pilot passes on every platform class claimed for the release, with evidence references recorded.
- [ ] Slice A, B, C, and D each passed their boundary pilot before work began on the following slice; evidence and repeated defect steps are linked from the tracking issue.
- [ ] The final handoff package is complete and every unresolved limitation appears in release notes.
- [ ] `VERSION` is bumped once in [config.py](scripts/runtime/config.py), with all mirrors rewritten by the pre-commit hook.

[← Back to README](README.md) · [Product Requirements](docs/product-requirements.md) · [Limitations](docs/limitations.md)
