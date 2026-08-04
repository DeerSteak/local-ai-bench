[← Back to README](../README.md)

# Security and privacy model

This is the working threat model and data-classification contract for the local product. Controls described as future coordinator or release requirements are not claims about functionality that has not shipped.

## Trust boundaries and assets

Protected assets include Hugging Face and future entitlement credentials; unreleased hardware identity; model artifacts; prompts, responses, question banks, generated code, generated images, raw samples, results and reports; local filesystem paths; runtime binaries; update packages; support bundles; and the integrity/availability of benchmark jobs. Trust boundaries exist at setup downloads, model/runtime processes, ComfyUI and llama.cpp localhost APIs, generated-code execution, imported bundles, exported artifacts, the future coordinator API, updates, telemetry/sync, and user-selected external destinations.

Local access is not automatically trusted: a malicious webpage can target localhost, another local process can race or impersonate a service, a model or imported artifact can be hostile, and a compromised runtime can emit crafted output. Pre-release evaluation increases impact because hardware names, inventory, screenshots, performance, and even update checks can reveal an embargoed product.

## Data classifications

| Class | Examples | Default handling |
|---|---|---|
| Public | Published docs, open-source code, public model identifiers | May be displayed/exported normally |
| Local operational | Installed runtime versions, non-identifying resource use, job state | Stored locally; exported only when selected by an allowlisted workflow |
| Benchmark confidential | Plans, raw measurements, scores, prompts, responses, images, reports, custom models | Local by default; explicit reviewed export; absent from telemetry |
| Secret | Access tokens, entitlement material, API bearer tokens, signing keys | Owner-restricted storage; never results/logs/telemetry/support bundles; redacted on detection |
| Embargoed | Unreleased hardware identity, inventory, logs, screenshots, results, reports, support data | Strongest local-only default; aliases in ordinary UI/export; explicit declassification/review before transfer |

Private paths are local metadata even when the referenced content is public. Content hashes can still be identifying and inherit the source data's classification. Deletion and retention follow [Local data lifecycle](data-lifecycle.md); ordinary deletion is not promised as forensic erasure.

## Threats and required controls

| Threat | Required controls | Verification status |
|---|---|---|
| Credential disclosure | No secret CLI arguments; protected files; log/result/support/telemetry exclusion; redaction tests | Hugging Face file and support export are covered; product-wide audit remains open |
| Localhost cross-origin attack | Loopback bind, bearer auth, Host/Origin validation, no permissive CORS, rate limits | Coordinator contract defined; implementation pending |
| Arbitrary code/command execution | No shell API, typed registry IDs, bounded generated-code sandbox, no untrusted interpolation | API contract defined; generated-code isolation pending |
| Malicious archive/path traversal | Exact manifests, duplicate rejection, size limits, digest verification, content-addressed extraction | Result bundles covered by tests |
| Result tampering | Atomic checkpoints, explicit validity, bundle digests, event digest chain, reproducible aggregates | Implemented for current JSON/bundles and the journal-owned single-shot workload |
| Runner escape or orphan process | Per-job ownership, process groups/job objects, heartbeats, graceful cancellation and bounded escalation | Supervised single-shot, conversation, native llama-bench, and HTTP concurrency runners implemented; remaining workloads pending migration |
| Dependency/update compromise | Locked dependencies, checksums, signatures, provenance, SBOM, staged rollback | Pending release work |
| Model/prompt exfiltration | Offline mode, denied network for generated code, opt-in reviewed exports, no content telemetry | Pending complete offline/isolation verification |
| Denial of service | Body/member/cardinality limits, timeouts, quotas, disk preflight, bounded artifact reads | Partial; coordinator enforcement pending |
| Embargo disclosure | Private aliases, no telemetry/sync/update metadata, reviewed export/report/support, offline qualification | Alias and export/report/support review implemented; offline verification pending |

Model weights and runtime files are untrusted input: malicious files, parser flaws, excessive resource use, dependency compromise, and unexpected native-process networking remain in scope even when their source is well known. Benchmark success never establishes that a checkpoint, generated response, or third-party runtime is safe. Current generated-code workloads score text and do not execute it; any future execution requires a separately reviewed sandbox with adversarial time, memory, filesystem, process, and network limits.

The future hosted service is outside the current shipped boundary. It requires proven demand, an exact sync-field policy, tenant isolation, authorization tests, encrypted transfer/storage, retention/deletion/export, backup/restore drills, and independent review before launch. Update and installer delivery likewise require signed packages and provenance, downgrade resistance, staged rollout, and tested rollback before stable-release claims.

## Embargoed evaluation policy

An evaluation is embargoed when the user or project marks any hardware, model, configuration, result, or relationship pre-release. The classification propagates to inventory, logs, screenshots, progress windows, result filenames/content, reports, bundles, support material, telemetry, sync, update requests, crash reports, and derived recommendations. A stable private alias is shown outside the restricted project while the internal identity remains available only to authorized local verification.

No embargoed artifact leaves the machine without an explicit review that names destination, files, fields, aliases, and intended audience. Result bundle and report creation now preview identity fields, support system/hardware aliases, and retain a source-identity digest; arbitrary attached artifacts still require separate inspection. Support bundles remain allowlisted but still require review. Telemetry, hosted sync, automatic crash upload, and metadata-bearing update checks default off for embargoed projects. [Offline Mode](offline-mode.md) enforces loopback-only application sockets and inherited offline controls; supported-platform packet-capture qualification remains open.

## Security ownership and response

Before a stable paid release, the project must name a security owner, publish a private vulnerability-reporting channel and response targets, inventory and scan dependencies/artifacts/secrets, generate an SBOM and notices, sign releases and updates, exercise rollback, commission independent assessment, and close or explicitly accept every high-severity finding. Security claims require evidence from the shipped build and supported platform matrix, not only this design.

A report receives an identifier, acknowledgement, severity and affected-version triage, an owner, a response target, and secure minimized reproduction instructions. Severity considers confidentiality, integrity, availability, exploitability, affected installs, crossed privilege boundaries, data class, and whether an offline guarantee failed. A credible active compromise suspends affected releases or updates, preserves evidence, revokes keys or artifacts where applicable, publishes containment guidance, and triggers assessment of notification duties.

Fixes require focused regression tests, security review, signed artifacts, affected/fixed version disclosure, and a rollback plan. Customer notices state observed versus possible impact, containment, remediation, verification, and the next update time; absence of telemetry is never presented as proof of no impact. Closure records root cause, detection gaps, response timing, corrective actions, owners, and deadlines. High-severity exceptions require a named owner, rationale, containment, expiry, and explicit product-owner acceptance; an independent assessment remains mandatory before enterprise claims or broad hosted deployment.
