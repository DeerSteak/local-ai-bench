# Product analytics telemetry contract

Local AI Bench 6.0 sends no product telemetry, crash uploads, hosted sync, or background analytics. This contract governs possible future outbound product analytics, not local benchmark measurement such as the local memory sampler described in [Memory Telemetry Qualification](telemetry-qualification.md). The event dictionary below is the maximum initial contract for a future opt-in implementation; documenting an event does not mean it is currently collected.

## Consent and control

Telemetry defaults off for every installation and project. Enabling it requires a separate informed choice that names destination, controller, purpose, retention, event/field dictionary, and deletion/contact path; accepting a license, running setup, using a paid entitlement, or enabling update checks is not telemetry consent. Embargoed projects keep telemetry off regardless of an installation preference unless a future project-specific reviewed policy explicitly permits it.

Before transmission, events must enter a bounded local outbox the user can inspect as exact JSON, filter by project, export, and delete. One visible switch disables future collection and transmission; disablement tests must prove the outbox does not grow and the network transport is not called. Previously transmitted deletion follows the published service policy and cannot be represented as local forensic erasure.

## Initial event dictionary

| Event | Purpose | Allowed fields |
|---|---|---|
| `application_session` | Measure activation and interface reliability | event schema, application version, interface (`gui`/`terminal`), coarse OS family, start outcome, safe error code |
| `setup_outcome` | Measure onboarding completion and broad failure class | event schema, application version, success/cancel/failure, coarse component categories selected, duration bucket, safe error code |
| `benchmark_outcome` | Measure valid-run completion and recovery | event schema, application version, workload keys, complete/partial/interrupted/failed, aggregate case counts by state, valid/invalid counts, duration bucket, resume/retry used |
| `report_outcome` | Measure report creation | event schema, application version, HTML/PDF selection, success/failure, safe error code |
| `bundle_outcome` | Measure portability workflow | event schema, application version, export/import/verify, success/failure, safe error code |

Each event has a random installation-scoped identifier only if the user separately permits longitudinal measurement. It rotates on reset and is never derived from hardware, hostname, account, path, result, plan, model, or content hashes. There is no advertising identifier, fingerprint, precise timestamp requirement, free-form message, stack trace, or arbitrary property map.

## Forbidden fields

Telemetry never contains tokens, credentials, authorization/entitlement material, IP/MAC addresses, hostname, username, email, private paths, command lines, environment variables, serial numbers, exact/unreleased hardware identity, prompts, responses, datasets, generated code/images, logs, screenshots, raw samples, scores, model/checkpoint names, result/plan/bundle/report content or hashes, customer/project names, acceptance criteria, publication plans, support content, or free-form errors. A new field or event requires governance review, data classification, purpose/retention update, consent-copy update, local preview support, disablement/redaction tests, and a schema version.

Safe error codes come from a closed reviewed vocabulary and describe product stage/category rather than exception text. Durations use coarse documented buckets; counts are bounded and reveal no model identity. The application must keep operating when telemetry storage, consent, DNS, network, or service availability fails.

## Hosted sync and updates are separate

Project/result sync, support upload, crash reporting, entitlement checks, and update checks are separate data flows with separate contracts and controls. Telemetry consent cannot authorize any of them. Offline mode blocks telemetry transport and sets inherited third-party opt-out variables, but ordinary disabled telemetry must also make no transport attempt without relying on offline mode.
