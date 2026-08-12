# Local AI Bench v5.1

Cross-platform benchmarking for LLM generation, image generation, embeddings, accuracy (multiple-choice question answering, math, reasoning, code, and tool calling), and opt-in concurrency/load testing and llama.cpp-native throughput comparisons. Designed for hardware from 8GB GPUs to high-memory unified-memory systems; setup estimates model fit before download, while benchmark failures are isolated so completed measurements are preserved.

> **Stable and development branches:** `main` is the public default branch and always represents the latest stable release. Ongoing work integrates through `develop`, and versioned release branches move forward from `develop` to `main`. Contributors should read the [Contributor Workflow](docs/contributor-workflow.md) before opening a pull request.

---

## Quick Start

```bash
git clone https://github.com/DeerSteak/local-ai-bench
cd local-ai-bench
```

| Platform | Script | What it can install |
|---|---|---|
| macOS | `bash setup.sh` | Homebrew, Python, llama.cpp (includes llama-bench, llama-batched-bench), ComfyUI |
| Linux / DGX Spark | `bash setup.sh` | Python, llama.cpp source build (includes llama-bench, llama-batched-bench), ComfyUI |
| Windows | `setup.bat` | Python, llama.cpp (CUDA on NVIDIA, Vulkan otherwise; includes llama-bench and llama-batched-bench), ComfyUI portable |

`setup.sh` / `setup.bat` first ensure Python and create or reuse `bench-env/`. The setup assistant then shows its installation plan for approval before installing Python packages, llama.cpp, or models, and opens an interactive model picker so you choose every model download before the unattended installation phase begins.

Once setup finishes:

```bash
# Linux / macOS
bash run_bench.sh

# Windows
run_bench.bat
```

With no arguments, the benchmark launcher opens the graphical configuration screen on a usable local desktop and retains the terminal checklist over SSH or without a display. The GUI applies a selected preset immediately, switches the preset label to Custom whenever a setting is changed, and remembers the resulting configuration in `.benchmark_frontend_state.json`. Passing benchmark CLI arguments forwards them straight to the non-interactive benchmark CLI — see [Launch modes](docs/cli-reference.md#launch-modes) for the full behavior and defaults.

A full run takes several hours, depending on your hardware and which options you select. When it's done, explore the results in the [dashboard](docs/dashboard.md):

```bash
# Linux / macOS
bash launch_dashboard.sh

# Windows
launch_dashboard.bat
```

Desktop users can double-click **Launch Local AI Bench Dashboard** (`.command`, `.desktop`, or `.bat`) instead.

For platform-specific notes, the HuggingFace token flow, and what setup actually installs, see [Setup](docs/setup.md).

---

## Documentation

| Doc | Covers |
|---|---|
| [Setup](docs/setup.md) | What the setup scripts install, the model picker, HuggingFace tokens, platform-specific notes |
| [Workloads](docs/workloads.md) | What's tested — LLM tiers and modes, images, embeddings, MCQ/math/reasoning/code/tool accuracy, concurrency, and llama-bench |
| [Methodology Contract](docs/methodology-contract.md) | Supported scope, metric boundaries, cache/retry/timeout rules, validity, aggregation, and decision-grade acceptance |
| [Product Requirements](docs/product-requirements.md) | Primary pre-launch hardware-validation workflow and quality gates |
| [User Journey](docs/user-journey.md) | Complete discovery-to-report path, including cancellation, failure, resume, and review |
| [Consumer Recommendation Policy](docs/recommendation-policy.md) | Evidence, fit, uncertainty, ranking, conflicts, and GPU/Mac decision flows |
| [Platform Tuning Profiles](docs/platform-tuning.md) | Neutral runtime settings, compatibility workarounds, and profile change rules |
| [Acceptance Policies](docs/acceptance-policies.md) | Explicit per-case thresholds, evidence requirements, and rejection behavior |
| [Benchmark Projects](docs/projects.md) | Local decision workflows, portable configuration, baselines, and acceptance policies |
| [Local Result History](docs/result-history.md) | Filesystem-owned filtering, multi-file dashboard launch, and policy evaluation |
| [Vendor Diagnostics](docs/vendor-diagnostics.md) | First-divergence evidence and source-verified engineer reproduction package |
| [Outbound Metadata Review](docs/outbound-review.md) | Embargo review, private aliases, and source-identity verification |
| [Offline Mode](docs/offline-mode.md) | Loopback-only execution controls and qualification boundary |
| [Limitations](docs/limitations.md) | Representativeness, environmental variance, compatibility, and recommendation constraints |
| [Local Data Lifecycle](docs/data-lifecycle.md) | Local storage, retention, deletion, portability, and support-bundle handling |
| [CLI Reference](docs/cli-reference.md) | Every flag, with examples |
| [Dashboard](docs/dashboard.md) | Loading results, chart sections, what each chart means, exporting |
| [Decision Reports](docs/reports.md) | Deterministic self-contained HTML/PDF evidence summaries |
| [How It Works](docs/how-it-works.md) | Execution order, orchestration, and code organization |
| [Engines](docs/engines.md) | The `InferenceEngine` interface, `LlamaCppEngine`, `--engine`, and how to add a new engine |
| [Project Structure](docs/project-structure.md) | What every file and folder in the repo is for |
| [Testing](docs/testing.md) | How to run tests, coverage boundaries, and a concise suite map |
| [4.1 Result Compatibility](docs/result-compatibility-v4.1.md) | Export, partial-result, measurement, and dashboard behavior protected during the commercial rewrite |
| [Architecture Decisions](docs/architecture-decisions.md) | Simplicity gate, data ownership decisions, and migration deletion ledger |
| [Coordinator API Contract](docs/coordinator-api.md) | Future authenticated localhost API, compatibility, validation, lifecycle, and artifact boundaries |
| [Security and Privacy](docs/security-and-privacy.md) | Trust boundaries, data classifications, embargo handling, threats, controls, and open verification work |
| [Release Policy](docs/release-policy.md) | Platform support levels, qualification matrix, stable gates, channels, and compatibility notes |
| [Contributor Workflow](docs/contributor-workflow.md) | Branch roles, pull requests, validation, merge conventions, releases, hotfixes, and repository protection |
| [Installation Maintenance](docs/maintenance.md) | Repair, upgrade, rollback, and safe project-owned uninstall boundaries |
| [Product Governance](docs/governance.md) | Change classes, required evidence, approval authority, and decision records |
| [Troubleshooting](docs/troubleshooting.md) | Setup, execution, result, report, and privacy-safe support guidance |
| [Support Operations](docs/support.md) | Intake, severity, escalation, redaction, retention, and resolution runbooks |
| [Telemetry Contract](docs/telemetry.md) | Current no-telemetry state and future opt-in event/field boundaries |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for non-commercial use, forking, and modification. Commercial licensing: [beatclikr@gmail.com](mailto:beatclikr@gmail.com).
