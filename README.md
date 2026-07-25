# Local AI Bench v4.0

Cross-platform benchmarking for LLM generation, image generation, embeddings, accuracy (multiple-choice question answering, math word problems, coding problems, and tool calling), and opt-in concurrency/load testing and llama-bench throughput comparisons. Designed to run on any hardware from an 8GB GPU up to high-memory unified-memory systems — models that don't fit are skipped automatically with no configuration needed.

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
| Windows | `setup.bat` | Python, llama.cpp Vulkan build (includes llama-bench, llama-batched-bench), ComfyUI portable |

`setup.sh` / `setup.bat` first ensure Python, create `bench-env/`, and install the project's Python packages. The setup assistant then shows its llama.cpp/model plan for approval and opens an interactive model picker, so you choose every model download before the unattended installation phase begins.

Once setup finishes:

```bash
# Linux / macOS
bash run_bench.sh

# Windows
run_bench.bat
```

With no arguments, the benchmark launcher shows an interactive checklist of installed tests and models, remembering your last confirmed selection (delete `.benchmark_frontend_state.json` to reset). Passing any CLI argument instead forwards straight to the non-interactive benchmark CLI — see [Launch modes](docs/cli-reference.md#launch-modes) for the full behavior and defaults.

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
| [CLI Reference](docs/cli-reference.md) | Every flag, with examples |
| [Dashboard](docs/dashboard.md) | Loading results, chart sections, what each chart means, exporting |
| [How It Works](docs/how-it-works.md) | Execution order, code organization, full parameter table |
| [Engines](docs/engines.md) | The `InferenceEngine` interface, `LlamaCppEngine`, `--engine`, and how to add a new engine |
| [Project Structure](docs/project-structure.md) | What every file and folder in the repo is for |
| [Testing](docs/testing.md) | How to run tests and detail on what each test file validates |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for non-commercial use, forking, and modification. Commercial licensing: [beatclikr@gmail.com](mailto:beatclikr@gmail.com).
