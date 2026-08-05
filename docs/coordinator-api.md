[← Back to README](../README.md)

# Local coordinator API contract

Status: design contract for the future persistent coordinator. The current 4.1 CLI/Tk application does not start an HTTP service. Implementation begins only when transactional resume or concurrent local clients justify replacing direct subprocess supervision.

## Compatibility

The API base is `/api/v1`. Every request and response is JSON except bounded artifact transfer and the server-sent event stream. Additive response fields are permitted within v1; removing/renaming a field, changing state semantics, accepting a previously rejected unsafe input, or changing plan/event schema compatibility requires a new API version. `RunPlan`, event, result, bundle, and API schema versions remain separate axes.

Clients send `X-Local-AI-Bench-API: 1` and receive the same header. An unsupported version returns `426` with supported versions. Mutation requests require `Idempotency-Key`; replay with the same authenticated principal, route, and canonical body returns the original response, while reuse with a different body returns `409`.

## Binding and authentication

The coordinator binds only explicit loopback sockets (`127.0.0.1` and, when enabled, `::1`), never wildcard interfaces. It runs without administrator privileges. On first start it creates a cryptographically random 256-bit bearer token in a user-private configuration file with owner-only permissions where the platform supports them. Tokens never appear in URLs, logs, results, support bundles, crash reports, or telemetry.

Every request except `GET /health` requires `Authorization: Bearer <token>`. The server rejects non-loopback peers, unrecognized `Host`, forwarded-host headers, browser requests with absent/unapproved `Origin`, credentialed cross-origin requests, and WebSocket upgrades. CORS is disabled by default; a packaged UI origin may be allowlisted exactly, never with `*`. Authentication comparison is constant-time. Repeated failures are rate-limited and recorded without the supplied credential.

## Resource model

| Route | Method | Purpose |
|---|---|---|
| `/health` | `GET` | Process readiness and API version only; no inventory or paths |
| `/inventory` | `GET` | Redacted hardware/runtime/model capabilities and support status |
| `/plans/validate` | `POST` | Validate one versioned `RunPlan` plus local path references without starting work |
| `/jobs` | `POST` | Create an immutable job from a validated plan and resume-identity snapshot |
| `/jobs` | `GET` | Paginated local history with bounded filters |
| `/jobs/{job_id}` | `GET` | Job projection, coverage, active ownership, and resumability |
| `/jobs/{job_id}/start` | `POST` | Start a pending validated job |
| `/jobs/{job_id}/pause` | `POST` | Request pause at the next declared safe case boundary |
| `/jobs/{job_id}/cancel` | `POST` | Request cancellation and ownership-aware cleanup |
| `/jobs/{job_id}/resume` | `POST` | Resume only after exact identity revalidation |
| `/jobs/{job_id}/fork` | `POST` | Create a new job from an explicitly reviewed changed plan |
| `/jobs/{job_id}/retry` | `POST` | Create a new attempt for selected retry-eligible cases |
| `/jobs/{job_id}/events` | `GET` | Bounded pagination or authenticated server-sent events after a sequence cursor |
| `/jobs/{job_id}/artifacts/{sha256}` | `GET` | Stream one job-referenced content object with range and size limits |
| `/bundles/export` | `POST` | Create a verified portable result bundle for an authorized job |
| `/bundles/import` | `POST` | Verify a bounded uploaded bundle before creating imported history |

There is no route that accepts a shell command, executable path, module name, environment-variable map, arbitrary URL, filesystem glob, SQL, Python expression, or subprocess argument vector. Workload and engine identifiers resolve through versioned internal registries; runner invocation is constructed entirely by trusted code from validated typed fields. Unknown fields are rejected.

## Validation and limits

Request bodies have route-specific byte limits, nesting limits, exact allowed keys, scalar types, ranges, and list cardinalities. IDs must match the declared stable identity scheme. Local paths are accepted only on validation routes that require them, normalized beneath approved roots, and never returned in portable plans or ordinary responses. Symlinks are resolved before root checks. Artifact reads require both a valid digest and a job reference to that digest; directory listing is impossible.

Pagination has a server maximum. Event streams require a sequence cursor, bounded replay window, heartbeat, idle expiry, and connection cap. Imports enforce compressed/uncompressed member limits, duplicate-name rejection, safe manifests, hashes, and aggregate verification. The coordinator applies per-route and per-principal rate limits and rejects work when configured storage/resource ceilings would be exceeded.

## Job and cancellation semantics

The transactional journal is the sole live owner for migrated job/stage/case/attempt/sample state. A mutation succeeds only after its event transaction commits. Clients may reconnect from the last sequence without inventing state. Runner stdout is diagnostic input, not authoritative completion.

Pause is cooperative and occurs only at a workload-declared safe boundary; the response states whether a model remains loaded. Cancel sends the documented graceful signal, checkpoints terminal events, performs ownership-aware cleanup, then escalates only against processes created and identified by that job. Resume rehashes plan/artifacts/runtimes/methodology and either continues remaining cases or returns `409 fork_required` with every mismatch. Completed cases are immutable. Retry creates a new attempt ID and never overwrites a prior sample.

## Errors and observability

Errors use `{code, message, resolution, request_id, details}`. `message` and `resolution` are safe for users; `details` is structured and redacted. Expected codes include `invalid_request`, `unauthenticated`, `forbidden_origin`, `not_found`, `conflict`, `fork_required`, `unsafe_boundary`, `resource_limit`, `artifact_invalid`, and `internal_error`. Responses state preserved durable state and cleanup status when execution was involved.

Logs contain request ID, authenticated local principal ID, route template, status, duration, and redacted error code. They exclude tokens, request authorization, prompts, responses, model content, private paths, and raw request bodies. Security-relevant authentication/origin/rate-limit failures are retained according to the local data policy.
