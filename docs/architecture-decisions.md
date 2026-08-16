# Architecture decisions

This log records material implementation decisions that affect compatibility, data handling, distribution, or future deletion work. A decision is changed by adding a new entry that supersedes it; history is not rewritten.

## Simplicity gate

Every architecture slice must answer these questions before implementation:

1. Which current reliability or customer requirement needs the change?
2. Why are direct functions and focused dataclasses insufficient?
3. What new persistent state or compatibility path becomes authoritative?
4. Which old module or path becomes redundant, and what objective gate permits its deletion?
5. How will tests prove interruption, partial-result, and 4.1 export compatibility?

A new base class, manager, provider, repository, event bus, dependency-injection container, dynamically discovered plugin type, service boundary, or second live data store requires a written decision here. Speculative extension points are rejected. Temporary compatibility layers require an owner and deletion condition in the ledger below.

## Active decisions

### AD-001 — Preserve 4.1 application version during the reliability program

- Status: accepted
- Requirement: result comparisons must not confuse an internal schema change with a product-version change.
- Decision: keep the top-level application version at 4.1 while `run.schema_version` carries result-envelope compatibility.
- Evidence: `docs/result-compatibility-v4.1.md` and its Python/dashboard golden fixtures.

### AD-002 — Use one immutable plan and local-only paths

- Status: accepted
- Requirement: CLI and GUI runs need reproducible measurement identity without leaking private paths.
- Decision: `RunPlan` is the serializable measurement plan; `RunPaths` owns machine-local output and ComfyUI paths. Frontends translate into the same public CLI and plan representation.
- Rejected alternative: a second GUI-specific execution configuration hierarchy.
- Evidence: deterministic plan-ID, redaction, preset, and CLI/UI round-trip tests.

### AD-003 — Keep JSON authoritative until a workload moves transactionally

- Status: accepted
- Requirement: crash-safe resume ultimately needs transactions, but two indefinitely writable stores would create reconciliation failures.
- Decision: `ResultStore` and atomic JSON remain authoritative for unmigrated workloads. SQLite is authoritative only for the bounded single-shot, conversation, native llama-bench, and HTTP concurrency slices whose events reproduce their JSON sections; those slices no longer mutate the same live measurement state through JSON.
- Rejected alternative: a shadow SQLite journal beside mutable result JSON.
- Deletion gate: after every supported workload uses the transactional store, remove live JSON mutation and retain JSON only as deterministic export.

### AD-004 — Keep direct subprocess supervision until resume requires a coordinator

- Status: accepted
- Requirement: the current GUI needs streaming logs, cancellation, cleanup, progress, and preserved checkpoints.
- Decision: Tk supervises the existing CLI subprocess directly. Do not add a localhost service merely for current single-user execution.
- Reconsideration gate: transactional resume, concurrent clients, or team workflows require a persistent execution owner.

### AD-005 — Retry implausible server rates once and expose the outcome

- Status: accepted
- Requirement: physically implausible server timing must not contaminate aggregates or disappear silently.
- Decision: retry the request once, or the entire HTTP-concurrency batch once; mark a second implausible result invalid and continue. The GUI reports retry, recovery, and invalid-drop events.
- Evidence: engine, workload, progress-event, and dashboard validation tests.

### AD-006 — Derive execution identities from the immutable plan

- Status: accepted
- Requirement: transactional events and safe resume need stable identities across processes and restarts.
- Decision: run-plan schema 2 declares `sha256-v1` and serializes a unique stable job ID. `plan_id` hashes semantic plan fields while excluding that execution identity, and checked stage/model/case/attempt/sample inputs yield job-scoped hierarchical SHA-256 IDs. No database row ID defines domain identity.
- Compatibility: schema-1 plans retain their exact serialization and plan hash; schema-2 is additive within the existing result envelope.
- Extension: run-plan schema 3 adds deterministic policy identities for workloads, runtime adapters, privacy handling, retries, timeouts, and output schemas; schema-1 and schema-2 reads remain unchanged.
- Extension boundary: workload SDK v1 and engine adapter v1 are reviewed source contracts with finite JSON conformance vectors and fixed capabilities; they do not add dynamic plugin discovery or arbitrary executable loading.
- Evidence: deterministic, adversarial, and schema-1 golden round-trip tests in `tests/test_run_plan.py`.

### AD-007 — Introduce the event journal inactive before workload migration

- Status: superseded by AD-011, AD-013, and AD-014
- Requirement: safe resume requires a transactional append-only record, while simultaneous JSON and SQLite mutation would create two authorities.
- Decision: implement and adversarially verify the journal independently, but do not connect it to benchmark execution until one bounded workload stops owning that state in JSON. Events use stable plan-derived parentage, legal transitions, atomic batches, immutable rows, and a digest chain; projections and aggregates rebuild from events.
- Rejected alternative: shadow-write every current JSON checkpoint into SQLite and reconcile later.
- Deletion gate: when all workloads export from the journal, remove runtime JSON mutation according to the migration ledger.

### AD-008 — Resume only from exact content identity at case boundaries

- Status: accepted
- Requirement: a resumed result must not combine measurements from changed models, runtimes, methodology, or configuration.
- Decision: persist a path-free snapshot of plan ID, artifact/runtime SHA-256 and size, and methodology versions. Exact matches retain completed cases, terminalize abandoned running attempts, and allocate a new attempt ordinal for remaining cases; any mismatch or unknown case requires a fork.
- Rejected alternative: compare filenames/timestamps or resume directly inside a partially completed request.
- Evidence: content-change, privacy, completed-case, interrupted-attempt, next-attempt, unknown-case, and database-migration tests.

### AD-009 — Store large event artifacts by verified content digest

- Status: accepted
- Requirement: large logs, responses, images, and exports must not bloat SQLite or depend on private source paths.
- Decision: stream artifacts atomically into a local SHA-256 object tree and place only validated digest/size/media-type references in events. Reuse and reads reverify content; source filenames are excluded.
- Rejected alternative: BLOB storage in the event database or path references to mutable source files.
- Activation gate: a migrated workload adopts objects and journal references together; the store does not shadow-copy legacy files.

### AD-010 — Expose typed coordinator resources, never commands

- Status: accepted
- Requirement: persistent resume and concurrent local clients eventually require a process owner without turning localhost into remote shell access.
- Decision: the future `/api/v1` accepts versioned plans, job lifecycle intents, stable IDs, filters, and bounded artifact operations. It binds loopback, requires bearer authentication and Host/Origin checks, and contains no command, executable, module, environment, SQL, expression, arbitrary URL, or argument-vector field.
- Reconsideration gate: none; adding arbitrary command execution requires a new security review and is outside the product contract.
- Evidence: `docs/coordinator-api.md`; implementation and adversarial API tests remain required before activation.

### AD-011 — Migrate LLM workloads without shadow writes

- Status: accepted
- Requirement: prove the event path on a real workload while retaining 4.1 result compatibility and per-model durability.
- Decision: single-shot and conversation cases/attempts/samples commit to the sibling SQLite journal; JSON checkpoints and stage return values rebuild from stage-scoped events. Conversation commits each sampled checkpoint before further growth. Ephemeral in-memory values may guide an active loop but cannot checkpoint independently. Other workload sections remain JSON-owned until their bounded migration.
- Compatibility: schema-3 golden LLM fields are asserted value-for-value; conversation retains its depth and timing fields, partial checkpoints, selection rules, retry, and cache behavior; current additive validity diagnostics remain allowed.
- Deletion gate: after all workloads migrate, remove runtime JSON ownership and export the whole result from journal projections.

### AD-012 — Activate process isolation only after fixed-protocol proof

- Status: accepted
- Requirement: a runner crash must not kill the coordinator, but premature activation could regress a working migrated workload.
- Decision: the supervisor accepts only a fixed internal runner entrypoint and authenticated strict events, owns a process group, monitors monotonic heartbeat arrival, and escalates cleanup within bounds. The activated entrypoint reconstructs and executes only registered journal-owned stages; single-shot, conversation, native llama-bench, and both HTTP concurrency stages are supported.
- Rejected alternative: arbitrary subprocess commands or switching live execution before parity/crash tests.
- Activation gate: satisfied by fake-runner hang/crash/cancel/disk tests, schema-3 single-shot parity, and conversation stage-isolation/preflight tests.

### AD-013 — Persist native rows without changing model lifecycle

- Status: accepted
- Requirement: successful llama-bench cases must survive a later timeout without restoring the per-case subprocess reload that made the workload impractically slow.
- Decision: retain one prefill and one decode command per model and commit each streamed JSONL row transactionally from the output callback. The child runner owns the native process group; JSON is rebuilt from the native stage projection.
- Compatibility: row payloads, internal repetition samples, case/repetition counts, and timeout/error markers retain their 4.1 shape.
- Rejected alternative: one llama-bench process per case solely to create checkpoint boundaries.

### AD-014 — Preserve concurrency at the batch boundary

- Status: accepted
- Requirement: durable request samples must not change the contention represented by an HTTP concurrency level or mix an invalid batch with its retry.
- Decision: the supervised child resolves ladders and contexts from the immutable plan, executes the existing whole-batch retry policy unchanged, and atomically commits only the final batch's request samples plus batch-level metrics.
- Compatibility: level keys, per-request aggregates, aggregate throughput, memory, validity, and stop markers retain their 4.1 shape.
- Rejected alternative: retrying or checkpointing individual requests from a concurrent batch.

### AD-015 — Separate resume, selected retry, and fork truthfully

- Status: accepted
- Requirement: stopped work must preserve valid evidence without implying that aggregate-only workloads can resume cases they never journaled.
- Decision: exact-identity journal plans may resume remaining work; eligible measured context/level cases may be retried explicitly within one stage; a full-plan fork always creates a distinct run/job/output and retains source provenance. Journal job state terminalizes with every run outcome and reopens explicitly for recovery, including finalization after all stage evidence committed. Plans containing JSON-owned legacy stages replay through normal orchestration under an exact source-plan guard rather than claiming in-place case resume.
- Compatibility and data ownership: resume/retry update the original result only after the inspector gate and retain terminal/attempt history; fork never mutates source evidence. Native llama-bench resumes the remaining sweep because unstarted rows have no selectable case identity.
- Evidence: inspector, executor, event-stage, GUI command/presentation, exact-plan, overwrite, source-preservation, and interruption tests.

### AD-016 — Pause cooperatively at measurement boundaries

- Status: accepted
- Requirement: a user must be able to pause long local runs without corrupting an in-flight measurement, tripping runner liveness, or adding paused time to timing metrics.
- Decision: GUI-launched parent and child processes share one short-lived validated control file. Pause waits before the next measured request, conversation turn, concurrency level, native sweep, or batched-native model command; the current operation checkpoints normally, runner heartbeats continue, and Stop releases pause before interruption.
- Rejected alternative: suspending only the GUI-owned parent process, which would leave isolated runner/server processes active and could trigger supervisor timeouts.
- Evidence and deletion gate: pause-control validation/blocking and measured-boundary tests; replace the file only if a future authenticated coordinator becomes the execution owner.

### AD-017 — Retrospective simplicity review of the commercial engineering surface

- Status: accepted with consolidation triggers
- Requirement: the commercial slice added a large module surface, so per-slice approval is insufficient evidence that the aggregate remains the smallest practical architecture.
- Aggregate review: the added modules remain one modular Python application plus the existing dashboard. They separate current user capabilities (plans, presets, projects, history, reports, bundles, diagnostics), reliability boundaries (event journal, supervision, recovery, pause), and release/security checks. Persistence uses direct SQLite/JSON functions, execution uses focused dataclasses and explicit subprocesses, and the UI calls those modules directly. The review found no dependency-injection container, repository/service/controller stack, event-bus framework, generic workflow engine, dynamic plugin loader, or network service.
- Interfaces retained: `RunPlan`/`RunPaths` prevents private paths entering portable plans; `EventStore` supplies transactional recovery; `RunnerSupervisor` isolates owned workload children; narrowly named report, bundle, policy, and release modules give CLI and GUI code testable functions without adding runtime layers.
- Small wrappers retained: dedicated `*_cli.py`, recovery, retry, and fork entry points are process/user boundaries with distinct exit behavior, not domain layers. They must remain thin and may not acquire independent business rules.
- Consolidation triggers: merge a module when it becomes a pass-through with no independent contract or test seam; remove a CLI wrapper when no documented invocation or process boundary uses it; do not add a manager/provider/repository abstraction around the current functions; require another aggregate review before the first stable commercial release or after 15 additional `scripts/` modules, whichever comes first.
- Evidence: module dependency inspection, 4.1 compatibility fixtures, direct unit tests, and the absence of framework-style types named by the simplicity gate. This decision supplies the aggregate review that the original slice-by-slice ledger did not.

### AD-018 — Keep the current application boundaries and remove incidental duplication

- Status: accepted
- Requirement: a post-GUI simplicity audit must identify repeated logic and abstractions that do not provide a current reliability, security, compatibility, or user-facing benefit.
- Decision: retain the current package boundaries, process supervisor, event store, `RunPlan`/`RunPaths`, and thin CLI entry points because each has a live contract described in AD-017. Consolidate identical preparation/execution failure handling in stage orchestration and remove the unused hardware-profile field from `RunContext`; do not split the large Tk callback modules into controller/service classes solely to reduce file length.
- Rejected alternative: broad module merging or introducing generic UI controllers, repositories, lifecycle interfaces, or dependency injection. Those changes would increase coupling or abstraction count without removing a user-visible failure mode.
- Follow-up: runner closures and mutable result JSON remain governed by the deletion ledger below. Revisit the GUI module only when a cohesive reusable component or a testability need justifies a boundary, not at an arbitrary line-count threshold.
- Evidence: static module/import inspection, exact duplicate-body inspection, orchestration failure-path tests, and the full Python suite.

### AD-019 — Predeclare Version 6 telemetry meaning and screening

- Status: accepted
- Requirement: telemetry sources must not acquire incompatible meanings or be approved after observing favorable results.
- Decision: use the schema map, vocabulary, qualification set, observer bounds, and threshold derivation frozen in `docs/version-6-foundation.md`; schema 5 is additive and missing telemetry remains not recorded.
- Methodology: the foundation screen may reject intrusive sampling but cannot approve default-on use; milestone 3 independent trials decide comparability for every source, interval, and combined sampler.
- Privacy: persist only normalized allowlisted measurements and provenance through the outbound metadata policy; raw sensor output and private identity are excluded.
- Evidence: Apple unified-memory and native-Windows discrete-NVIDIA sources passed the predeclared 20-pair screens at 1.0, 0.5, and 0.25 seconds. Milestone 3 re-qualified the selected 0.5-second source on WSL discrete NVIDIA through llama.cpp and vLLM with fixed cold prompts; the same evidence derives the 8% TTFT, 3% throughput, and 3% wall-time practical thresholds. Detailed records are indexed under `docs/qualification/`.
- Decision: retain three measured requests per loaded model for within-run dispersion; regression verdicts continue to use independent invocations. On the WSL qualification series, three separate one-request invocations would cost 2.57–2.89 times one three-request invocation, while averaging three reduced between-invocation dispersion across every measured metric.
- Decision: enable 0.5-second memory telemetry by default in Version 6 and treat telemetry-on/off performance as comparable when all other methodology settings match. The fixed-prompt Milestone 3 series passed through both engines on discrete NVIDIA and on Apple unified memory; other intervals remain identity-bearing.
- Version 6.0-pre4 extension: keep power opt-in until source-specific repeated-trial qualification, collect it on the existing memory timeline, and make interval/source/scope identity-bearing in run-plan schema 4. Use a persistent non-interactive `powermetrics` reader on macOS; never run the benchmark as root or persist executable paths and raw sensor output.
- Scope rule: processor-package, accelerator, CPU-package, and whole-system energy are distinct claims. Mixed scopes have no aggregate run total or shared dashboard axis, and Apple `powermetrics` estimates remain within-device evidence because the tool warns against cross-device comparison.

## Migration and deletion ledger

| Temporary or superseded path | Current owner | Replacement gate | Required deletion |
|---|---|---|---|
| Atomic mutable result JSON | Engineering owner (`ResultStore`) | Every supported workload exports compatibly from the transactional store; deadline is the first stable commercial release | Remove runtime JSON mutation and keep deterministic export only. If migration is not complete by that release gate, remove journal-backed recovery from supported release scope or record a product-owner-approved superseding decision; do not ship indefinite dual ownership. |
| Runner closures inside `benchmark.py` | CLI orchestration | Typed runner jobs execute all workloads with parity fixtures | Remove closures and their duplicate argument capture |
| Direct Tk-to-CLI subprocess ownership | `benchmark_gui.py` | Persistent coordinator proves authenticated lifecycle, resume, and cancellation | Point Tk and CLI at the coordinator and remove duplicate supervision |
| Schema-2 compatibility reads | Dashboard and result utilities | Published support window expires with migration tooling available | Remove only in a separately announced compatibility boundary |

## Decision template

### AD-NNN — Title

- Status: proposed, accepted, superseded, or rejected
- Requirement:
- Decision:
- Rejected alternative:
- Compatibility and data ownership:
- Evidence and deletion gate:
