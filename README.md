# Local AI Bench v6.0

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
| macOS | `bash setup.sh` | Homebrew, Python, llama.cpp with Metal, and ComfyUI with MPS |
| Linux / WSL2 / DGX Spark | `bash setup.sh` | Python, llama.cpp with the detected CUDA, ROCm, or SYCL/XPU backend, ComfyUI, and vLLM where the platform matrix below records support |
| Windows | `setup.bat` | Python, llama.cpp with CUDA on NVIDIA, Vulkan on AMD, or SYCL/XPU on Intel, plus ComfyUI portable; see the Intel Smart App Control exception below |

`setup.sh` / `setup.bat` first ensure Python and create or reuse `bench-env/`. The setup assistant then shows its installation plan for approval before installing Python packages, llama.cpp, or models, and opens an interactive model picker so you choose every model download before the unattended installation phase begins.

Once setup finishes:

```bash
# Linux / macOS
bash run_bench.sh

# Windows
run_bench.bat
```

With no arguments, the benchmark launcher opens the graphical configuration screen on a usable local desktop and retains the terminal checklist over SSH or without a display. The GUI applies a selected preset immediately, switches the preset label to Custom whenever a setting is changed, and remembers the resulting configuration in `.benchmark_frontend_state.json`. Passing benchmark CLI arguments forwards them straight to the non-interactive benchmark CLI — see [Launch modes](docs/cli-reference.md#launch-modes) for the full behavior and defaults.

When llama.cpp is the only selected engine, every catalog LLM can run its default Q4_K_M artifact or an opt-in same-repository sweep across Q4_K_M, the catalog's preferred Q6-family artifact, and Q8_0. Setup and the benchmark GUI group those choices beneath one base-model checkbox; selecting vLLM hides the GGUF children and retains only the documented default because this workflow does not claim that native vLLM formats are comparable to GGUF. See [Product catalogs](docs/catalogs.md#llamacpp-quantization-variants) for the exact artifacts and [CLI Reference](docs/cli-reference.md) for headless selection.

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

## Qualified platforms

Local AI Bench publishes support only after the ordinary benchmark completes every workload required for that exact platform, runtime version, backend, and accelerator. Sixteen of the seventeen original runtime targets are qualified for v6.0-pre8. The remaining target is Intel Arc Pro B65 with llama.cpp SYCL/XPU on Windows: enforced Smart App Control blocked the official runtime DLLs on the tested host, so that path is not supported there and remains unverified on other Windows systems.

<!-- qualification-matrix:start -->
16 of 17 target runtime combinations are supported by current evidence.

| Target | Platform | Architecture | Runtime | Backend | Accelerator | Runtime support | ComfyUI images | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `macos-m5-pro-llamacpp-metal` | macos | arm64 | llamacpp | metal | M5 Pro | Supported | Supported | 0.1.2-dev, 2026-08-18, suite 6.0-pre8 |
| `geforce-windows-llamacpp-cuda` | windows | x86_64 | llamacpp | cuda | NVIDIA GeForce | Supported | Supported | 0.1.2-dev, 2026-08-19, suite 6.0-pre8 |
| `radeon-windows-llamacpp-vulkan` | windows | x86_64 | llamacpp | vulkan | AMD Radeon | Supported | Supported | 0.1.2-dev, 2026-08-20, suite 6.0-pre8 |
| `intel-arc-windows-llamacpp-sycl` | windows | x86_64 | llamacpp | xpu | B65 | Unverified | Unverified | No qualification record |
| `geforce-wsl2-llamacpp-cuda` | wsl2 | x86_64 | llamacpp | cuda | NVIDIA GeForce | Supported | Supported | 0.1.2-dev, 2026-08-19, suite 6.0-pre8 |
| `geforce-wsl2-vllm-cuda` | wsl2 | x86_64 | vllm | cuda | NVIDIA GeForce | Supported | Not applicable | 0.27.1, 2026-08-19, suite 6.0-pre8 |
| `radeon-wsl2-llamacpp-rocm` | wsl2 | x86_64 | llamacpp | rocm | Radeon RX 9060 XT | Supported | Supported | 0.1.2-dev, 2026-08-20, suite 6.0-pre8 |
| `nvidia-linux-llamacpp-cuda` | linux | x86_64 | llamacpp | cuda | NVIDIA | Supported | Supported | 0.1.2-dev, 2026-08-22, suite 6.0-pre8 |
| `nvidia-linux-vllm-cuda` | linux | x86_64 | vllm | cuda | NVIDIA | Supported | Not applicable | 0.27.1, 2026-08-22, suite 6.0-pre8 |
| `radeon-linux-llamacpp-rocm` | linux | x86_64 | llamacpp | rocm | Radeon RX 9060 XT | Supported | Supported | 0.1.2-dev, 2026-08-21, suite 6.0-pre8 |
| `radeon-linux-vllm-rocm` | linux | x86_64 | vllm | rocm | Radeon RX 9060 XT | Supported | Not applicable | 0.27.1+rocm723, 2026-08-21, suite 6.0-pre8 |
| `intel-arc-linux-llamacpp-sycl` | linux | x86_64 | llamacpp | xpu | 8086:e222 | Supported | Supported | 0.1.2-dev, 2026-08-21, suite 6.0-pre8 |
| `intel-arc-linux-vllm-xpu` | linux | x86_64 | vllm | xpu | 8086:e222 | Supported | Not applicable | 0.27.1+xpu, 2026-08-21, suite 6.0-pre8 |
| `ryzen-ai-halo-llamacpp-rocm` | linux | x86_64 | llamacpp | rocm | Radeon 8060S | Supported | Supported | 0.1.2-dev, 2026-08-21, suite 6.0-pre8 |
| `ryzen-ai-halo-vllm-rocm` | linux | x86_64 | vllm | rocm | Radeon 8060S | Supported | Not applicable | 0.27.1+rocm723, 2026-08-21, suite 6.0-pre8 |
| `dgx-spark-llamacpp-cuda` | linux | aarch64 | llamacpp | cuda | NVIDIA GB10 | Supported | Supported | 0.1.2-dev, 2026-08-19, suite 6.0-pre8 |
| `dgx-spark-vllm-cuda` | linux | aarch64 | vllm | cuda | NVIDIA GB10 | Supported | Not applicable | 0.27.1, 2026-08-19, suite 6.0-pre8 |
<!-- qualification-matrix:end -->

**Supported** means complete reviewed smallest-model evidence exists for the exact combination. **Experimental** means complete evidence exists but is stale. **Unverified** means no complete matching evidence has been recorded; it does not by itself mean the runtime is broken. ComfyUI image support is graded separately and is not applicable to vLLM.

Qualification exercises every applicable workload with the smallest compatible LLM, embedding model, and, for llama.cpp, Stable Diffusion 1.5. It proves functional completion of the shipped setup and benchmark path, not full-catalog performance or full-bank accuracy. See [Platform qualification](docs/qualification.md) for the procedure, pass contract, artifact policy, and exact scope.

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
