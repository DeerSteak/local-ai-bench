# Local AI Bench v6.0-pre3

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

Start with the [documentation index](docs/README.md), which separates current user guidance, methodology, developer references, operational policy, and forward-looking plans.

| Common destination | Covers |
|---|---|
| [Setup](docs/setup.md) | Installation, models, credentials, and platform notes |
| [Workloads](docs/workloads.md) | What each benchmark measures |
| [CLI Reference](docs/cli-reference.md) | Commands, flags, defaults, and examples |
| [Dashboard](docs/dashboard.md) | Loading, comparing, interpreting, and exporting results |
| [Methodology Contract](docs/methodology-contract.md) | Scientific boundaries and aggregation rules |
| [Troubleshooting](docs/troubleshooting.md) | Setup, execution, result, and report failures |
| [Version 6 Plan](VERSION_6_PLAN.md) | Active implementation, qualification, pilot, and rollback plan |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for non-commercial use, forking, and modification. Commercial licensing: [beatclikr@gmail.com](mailto:beatclikr@gmail.com).
