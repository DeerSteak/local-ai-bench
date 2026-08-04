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
- Decision: run-plan schema 2 declares `sha256-v1`; the canonical plan yields the job ID and checked stage/model/case/attempt/sample inputs yield hierarchical SHA-256 IDs. No mutable counter or database row ID defines domain identity.
- Compatibility: schema-1 plans retain their exact serialization and plan hash; schema-2 is additive within the existing result envelope.
- Evidence: deterministic, adversarial, and schema-1 golden round-trip tests in `tests/test_run_plan.py`.

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
