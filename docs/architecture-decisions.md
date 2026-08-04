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
- Decision: `ResultStore` and atomic JSON remain the only live source of truth today. SQLite may become authoritative only for a bounded workload slice whose events can reproduce its export; that slice must stop mutating the same state through JSON.
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
- Evidence: deterministic, adversarial, and schema-1 golden round-trip tests in `tests/test_run_plan.py`.

### AD-007 — Introduce the event journal inactive before workload migration

- Status: accepted
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

### AD-011 — Migrate single-shot LLM without shadow writes

- Status: accepted
- Requirement: prove the event path on a real workload while retaining 4.1 result compatibility and per-model durability.
- Decision: single-shot cases/attempts/samples commit to the sibling SQLite journal; JSON checkpoints and the stage return value rebuild from events. Ephemeral in-memory values may guide the active loop but cannot checkpoint independently. Other workload sections remain JSON-owned until their bounded migration.
- Compatibility: schema-3 golden LLM fields are asserted value-for-value; current additive validity diagnostics remain allowed.
- Deletion gate: after all workloads migrate, remove runtime JSON ownership and export the whole result from journal projections.

## Migration and deletion ledger

| Temporary or superseded path | Current owner | Replacement gate | Required deletion |
|---|---|---|---|
| Atomic mutable result JSON | `ResultStore` | Every supported workload exports compatibly from the transactional store | Remove runtime JSON mutation; keep deterministic export only |
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
