[← Back to README](../README.md)

# Documentation

This index separates current user guidance, scientific contracts, developer references, operational policy, and forward-looking plans. A document listed under **Plans and proposals** describes intended work rather than current product behavior unless it explicitly says otherwise.

## Start here

| Document | Purpose |
|---|---|
| [Setup](setup.md) | Installation, model selection, credentials, and platform notes |
| [User Journey](user-journey.md) | End-to-end path from setup through review and export |
| [Workloads](workloads.md) | What each benchmark measures |
| [CLI Reference](cli-reference.md) | Commands, flags, defaults, and examples |
| [Dashboard](dashboard.md) | Loading, comparing, interpreting, and exporting results |
| [Troubleshooting](troubleshooting.md) | Setup, execution, result, and report failures |
| [Limitations](limitations.md) | Representativeness, variance, compatibility, and interpretation bounds |

## Results and decisions

| Document | Purpose |
|---|---|
| [Projects](projects.md) | Local decision projects, portable configuration, and baselines |
| [Result History](result-history.md) | Result discovery, filtering, comparison, and policy evaluation |
| [Reports](reports.md) | Deterministic HTML and PDF evidence reports |
| [Acceptance Policies](acceptance-policies.md) | Explicit evidence thresholds and rejection semantics |
| [Recommendation Policy](recommendation-policy.md) | Fit, ranking, uncertainty, and conflict rules |
| [Reference Evidence](reference-evidence.md) | Requirements for the supported comparison corpus |
| [Vendor Diagnostics](vendor-diagnostics.md) | First-divergence evidence packages |
| [Outbound Review](outbound-review.md) | Embargo-safe identity review and aliases |

## Methodology and architecture

| Document | Purpose |
|---|---|
| [Methodology Contract](methodology-contract.md) | Metric, cache, retry, timeout, validity, and aggregation rules |
| [How It Works](how-it-works.md) | Execution order and orchestration |
| [Engines](engines.md) | Engine interface and adapter behavior |
| [Platform Tuning](platform-tuning.md) | Neutral settings and compatibility workarounds |
| [4.1 Result Compatibility](result-compatibility-v4.1.md) | Protected result and dashboard behavior |
| [Architecture Decisions](architecture-decisions.md) | Accepted decisions and deletion ledger |
| [Extension Contracts](extension-contracts.md) | Workload and engine extension boundaries |
| [Coordinator API](coordinator-api.md) | Versioned future localhost coordination contract |
| [Workload Packs](workload-packs.md) | Pack validation and execution contract |
| [Catalogs](catalogs.md) | Hardware and model catalog ownership |

## Security, privacy, and operations

| Document | Purpose |
|---|---|
| [Security and Privacy](security-and-privacy.md) | Threat model, classifications, controls, and gaps |
| [Security Gates](security-gates.md) | Automated and reviewed release gates |
| [Offline Mode](offline-mode.md) | Loopback-only enforcement and qualification boundary |
| [Data Lifecycle](data-lifecycle.md) | Retention, deletion, portability, and artifact handling |
| [Product Analytics Telemetry](telemetry.md) | Future opt-in outbound analytics contract; distinct from local measurement telemetry |
| [Support](support.md) | Intake, escalation, redaction, and retention runbooks |
| [Maintenance](maintenance.md) | Repair, upgrade, rollback, and uninstall boundaries |
| [Legal Readiness](legal-readiness.md) | Licensing, terms, notices, and approval gates |

## Product, release, and governance

| Document | Purpose |
|---|---|
| [Product Requirements](product-requirements.md) | Product outcomes, scope, and quality gates |
| [Governance](governance.md) | Change classes, evidence, and approval authority |
| [Contributor Workflow](contributor-workflow.md) | Branches, reviews, validation, and releases |
| [Release Policy](release-policy.md) | Support levels, qualification matrix, and stable gates |
| [Release Artifacts](release-artifacts.md) | Packaging, signing, provenance, and rollback |

## Developer reference

| Document | Purpose |
|---|---|
| [Project Structure](project-structure.md) | Repository and module ownership |
| [Testing](testing.md) | Test commands, boundaries, and suite map |

## Plans and proposals

| Document | Status and purpose |
|---|---|
| [Version 6 Plan](../VERSION_6_PLAN.md) | Active ordered implementation and qualification plan |
| [Version 6 Foundation](version-6-foundation.md) | Frozen definitions and predeclared qualification rules for Version 6 |
| [Memory Telemetry Qualification](telemetry-qualification.md) | Milestone-1 supervised observer-screen procedure |
| [macOS M4 Pro Memory Screen](qualification/memory-macos-m4-pro-1s.md) | Completed 1-second memory observer-effect evidence and limitations |
| [Reliability Architecture Plan](reliability-architecture-plan.md) | Forward-looking reliability migration plan |
| [vLLM Engine Plan](vllm-engine-plan.md) | Forward-looking vLLM adapter and packaging plan |
