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

## Embargoed evaluation policy

An evaluation is embargoed when the user or project marks any hardware, model, configuration, result, or relationship pre-release. The classification propagates to inventory, logs, screenshots, progress windows, result filenames/content, reports, bundles, support material, telemetry, sync, update requests, crash reports, and derived recommendations. A stable private alias is shown outside the restricted project while the internal identity remains available only to authorized local verification.

No embargoed artifact leaves the machine without an explicit review that names destination, files, fields, aliases, and intended audience. Result bundle and report creation now preview identity fields, support system/hardware aliases, and retain a source-identity digest; arbitrary attached artifacts still require separate inspection. Support bundles remain allowlisted but still require review. Telemetry, hosted sync, automatic crash upload, and metadata-bearing update checks default off for embargoed projects. [Offline Mode](offline-mode.md) enforces loopback-only application sockets and inherited offline controls; supported-platform packet-capture qualification remains open.

## Security ownership and response

Before a stable paid release, the project must name a security owner, publish a private vulnerability-reporting channel and response targets, inventory and scan dependencies/artifacts/secrets, generate an SBOM and notices, sign releases and updates, exercise rollback, commission independent assessment, and close or explicitly accept every high-severity finding. Security claims require evidence from the shipped build and supported platform matrix, not only this design.
