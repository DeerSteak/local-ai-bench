# Local AI Bench v4.1

Cross-platform benchmarking for LLM generation, image generation, embeddings, accuracy (multiple-choice question answering, math, reasoning, code, and tool calling), and opt-in concurrency/load testing and llama.cpp-native throughput comparisons. Designed for hardware from 8GB GPUs to high-memory unified-memory systems; setup estimates model fit before download, while benchmark failures are isolated so completed measurements are preserved.

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

With no arguments, the benchmark launcher opens the graphical configuration screen on a usable local desktop and retains the terminal checklist over SSH or without a display. The GUI starts in locked Default mode; Custom exposes every practical benchmark setting and remembers it in `.benchmark_frontend_state.json`. Passing benchmark CLI arguments forwards them straight to the non-interactive benchmark CLI — see [Launch modes](docs/cli-reference.md#launch-modes) for the full behavior and defaults.

A full run takes several hours, depending on your hardware and which options you select. When it's done, explore the results in the [dashboard](docs/dashboard.md):

```bash
# Linux / macOS
bash launch_dashboard.sh

# Windows
launch_dashboard.bat
```

For platform-specific notes, the HuggingFace token flow, and what setup actually installs, see [Setup](docs/setup.md).

---

## Documentation

| Doc | Covers |
|---|---|
| [Setup](docs/setup.md) | What the setup scripts install, the model picker, HuggingFace tokens, platform-specific notes |
| [Workloads](docs/workloads.md) | What's tested — LLM tiers and modes, images, embeddings, MCQ/math/reasoning/code/tool accuracy, concurrency, and llama-bench |
| [Methodology Contract](docs/methodology-contract.md) | Supported scope, metric boundaries, cache/retry/timeout rules, validity, aggregation, and decision-grade acceptance |
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

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for non-commercial use, forking, and modification. Commercial licensing: [beatclikr@gmail.com](mailto:beatclikr@gmail.com).
