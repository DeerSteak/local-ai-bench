[← Back to README](../README.md)

# Setup

**Contents**
- [What the setup scripts do](#what-the-setup-scripts-do)
- [Memory-fit estimate](#memory-fit-estimate)
- [Disk space check](#disk-space-check)
- [HuggingFace token](#huggingface-token)
- [Platform notes](#platform-notes)

## What the setup scripts do

| Platform | Script | What it can install |
|---|---|---|
| macOS | `bash setup.sh` | Homebrew, Python, llama.cpp (includes llama-bench and llama-batched-bench), ComfyUI |
| Linux / DGX Spark | `bash setup.sh` | Python, llama.cpp source build (includes llama-bench and llama-batched-bench), ComfyUI, ROCm-enabled PyTorch on AMD, XPU-enabled PyTorch on Intel Arc (experimental) |
| Windows | `setup.bat` | Python, llama.cpp (CUDA on NVIDIA, Vulkan otherwise; includes llama-bench and llama-batched-bench), ComfyUI portable |

`setup.sh` / `setup.bat` first switch to the repository directory, locate Python 3.11+, and ask before installing Python or Homebrew when either is missing. They then create or reuse a valid `bench-env/` and hand off to `scripts/setup_check.py`, which presents a separate approval prompt before installing Python packages, llama.cpp, or models. The setup assistant then:

1. Detects your hardware (OS, GPU backend, RAM).
2. Shows a numbered list of all 12 LLMs, two embedding models, and five image models — everything selected by default except LLM/image models estimated not to fit in detected RAM/VRAM, which start unchecked with a note on how much they'd need. If `models/llamacpp/` contains GGUF model folders that do not belong to the current LLM or embedding catalog, the list also includes one optional cleanup row naming those folders; cleanup is always unchecked by default. Folders without a GGUF and loose files are not cleanup candidates. The estimate includes model weights, required image encoders, a 20% runtime allowance, and a small OS/driver reserve; it is guidance rather than a hard block.
3. Lets you toggle the selection interactively:
   - Numbers to toggle individual models (`2 4 7-9`)
   - A size tier (`xs`/`s`/`m`/`l`) to toggle every model at that tier — LLM and image checkpoints together, e.g. `s` toggles the small-tier LLMs and SDXL as a group
   - `emb`/`img` to toggle a whole section
   - `clean` to toggle deletion of the listed non-catalog model folders
   - `a` to select/deselect all models; it deliberately does not enable cleanup
   - Enter to install everything shown
   - `q` or Ctrl-C to cancel before the unattended installation phase; the bootstrap may already have installed Python or created `bench-env/`
4. If you selected any LLM, embedding, or image model, asks for a HuggingFace token next (see [HuggingFace token](#huggingface-token) below).
5. Installs everything you approved — llama.cpp, any ComfyUI dependencies, LLM/embedding GGUFs, and image checkpoints — with no further prompts. If cleanup was selected, it first deletes only the non-catalog folders shown in the picker; catalog folders and loose files are never cleanup targets.

Setup checks the system installation before planning any llama.cpp install. When `llama-server` is available through `PATH` or a standard macOS Homebrew location, setup does not install or build another llama.cpp copy. It independently detects `llama-bench` and `llama-batched-bench`; when all three system tools are present, benchmark runs use those exact system tools. Models remain managed by Local AI Bench under `models/llamacpp/`, and the selected system binary receives the downloaded GGUF's explicit path, so system installation and project model storage do not need to share a directory. If `llama-server` exists but either optional native benchmark tool is absent, setup leaves the existing installation untouched and reports that the corresponding opt-in test is unavailable rather than installing a second distribution behind the user's back.

When llama.cpp is genuinely absent, the installed copy also includes `llama-bench` and `llama-batched-bench`, llama.cpp's own throughput-benchmarking tools, needed only for the opt-in `llamabench` and `llamabenchconc` tests (`--tests llamabench llamabenchconc`) — see [Workloads](workloads.md#llama-bench) and [Workloads](workloads.md#llama-bench-concurrency).

Non-catalog cleanup is permanent rather than a move to Trash or Recycle Bin. It is deliberately excluded from the default selection and the `a` shortcut; select its numbered row or type `clean` only after checking the displayed folder names for models you want to keep.

When setup is complete, run the benchmark:

```bash
# Linux / macOS
bash run_bench.sh

# Windows
run_bench.bat
```

These scripts activate the virtual environment automatically and forward any arguments to `scripts/benchmark.py` — see the [CLI Reference](cli-reference.md) for available flags.

## Memory-fit estimate

`hardware.py`'s memory ceiling is VRAM (discrete GPU) or total system RAM (unified memory, integrated GPU, or CPU-only), minus a reserve — 1.0 GB for VRAM (driver/other GPU processes), 8.0 GB for RAM (OS, the inference server, and everything else sharing that pool). A model's estimated footprint is its download size plus a flat 20% runtime overhead (KV-cache for LLMs, activations for image models) — an approximation, not a per-context-length calculation.

Discrete-vs-integrated GPU classification is a naming-convention heuristic (AMD: `RX`/`PRO`/`INSTINCT` in the name; Intel: a model number like `A770`), not authoritative. An unknown or ambiguous name defaults to "integrated" — the more permissive failure mode, since it falls back to the system-RAM ceiling instead of wrongly capping to a VRAM number that doesn't apply.

## Disk space check

Before downloading anything, `setup_check.py` estimates how much space your selection still needs (skipping whatever's already downloaded) and compares it against free space on the volume containing the repository and its managed models:

- **Enough free space, plus a 10 GB buffer** — proceeds normally.
- **Enough for the downloads, but less than a 10 GB buffer left over** — prints a warning and continues.
- **Not enough free space at all** — stops before model downloads, explains that continuing could create a partial installation or fill the volume, and reports approximately how much additional space must be freed before rerunning setup.

Independently of that, if completing the downloads would leave less than 10% of your drive's total capacity free, it also prints a warning and pauses 5 seconds before continuing — just enough to notice, without stopping.

## HuggingFace token

Every LLM and embedding model is downloaded as a GGUF file from HuggingFace, resolved from the `hf_repo`/`hf_file` fields in `scripts/models.py` into `models/llamacpp/<tag-slug>/` (see [Engines](engines.md#llamacppengine)). Image checkpoints use the same HuggingFace download client but land in ComfyUI's model directories. Public repositories can be downloaded without an account or token. SD3.5 Large, Flux.1-dev, and Flux.2-dev are gated and require a free account, license acceptance, and an access token:

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.2-dev

If you select any LLM, embedding, or image model in the picker, `setup_check.py` finds your HF token in this order:

1. `HF_TOKEN` environment variable
2. `hf.txt` in the repo root (token on a single line)
3. Interactive prompt — saves to `hf.txt` by default for future runs, with an explicit opt-out

A token isn't required for non-gated models, but authenticated downloads generally receive better rate limits. `setup_check.py` therefore offers token authentication whenever any model is selected; pressing Enter skips it when no gated image model was selected. A saved token is written as a single line with user-only permissions on platforms that support POSIX file modes, and `hf.txt` remains excluded from Git.

## Platform notes

Close other apps before running — GPU memory contention affects results.

**macOS** — Plug in power and disable sleep (System Settings → Battery) before a long run. For 70B models, watch Activity Monitor → Memory: if pressure turns red and TPS drops between runs, the system is swapping — use `--timeout 600` or `--maxtier medium`.

**Linux (NVIDIA)** — Python 3.11 is installed with apt when missing on Debian-family systems, after confirmation; other distributions need a manual Python install. llama.cpp is built from source with CUDA when an NVIDIA GPU is detected.

**Linux (AMD/ROCm)** — `rocminfo` detection selects llama.cpp's HIP build. When image models are selected, setup replaces ComfyUI's default torch packages with the configured ROCm 6.4 wheels. This path is not verified on every newer APU architecture; if the wheel does not support the detected GPU, install a compatible PyTorch ROCm build manually.

**Linux (Intel Arc) — experimental** — `lspci` detection records `hardware_backend: "xpu"`, but setup does not build llama.cpp's SYCL backend, so LLM inference remains CPU unless you supply a manual `-DGGML_SYCL=ON` build. For image generation, setup checks for Intel's compute runtime and prints installation commands when it is absent; when image models are selected, it installs PyTorch's XPU wheels. This path has not been verified on real Arc hardware by the project maintainer.

**DGX Spark** — Uses the normal Linux NVIDIA source-build path; its ARM64 architecture does not require a separate prebuilt package.

**macOS and Linux** — If setup reports a permissions error, fix ownership or permissions for the named path and rerun it as your normal user. Avoid running the whole setup under `sudo`, which can leave the project environment and downloaded files owned by root.

**Windows (NVIDIA)** — Setup chooses the newest llama.cpp CUDA package supported by the installed driver and downloads ComfyUI's NVIDIA portable build; if no compatible CUDA package is available, llama.cpp falls back to Vulkan. It also checks whether portable PyTorch supports the GPU's compute capability and reinstalls the configured cu128 packages when required.

**Windows (AMD)** — The setup script downloads the latest official ComfyUI AMD portable build. No manual ROCm install required.

**Windows (Intel Arc) — experimental** — Setup downloads ComfyUI's Intel portable build and uses llama.cpp's Vulkan package. Results therefore report `backend: "vulkan"` while retaining `hardware_backend: "xpu"`; a manual SYCL build reports `xpu`. This path has not been verified on real Arc hardware by the project maintainer.

**Windows (all)** — If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

[← Back to README](../README.md) · [Workloads →](workloads.md)
