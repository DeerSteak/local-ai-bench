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

On macOS, double-click `Setup Local AI Bench.command` in Finder to open Terminal and launch the graphical wizard directly. The launcher switches to the repository directory automatically and leaves Terminal open when setup fails so the error can be reviewed. The first time the launcher closes its Terminal window, macOS asks whether it may control another application; choose **Allow** so future setup and benchmark launcher windows can close automatically. macOS may also require Control-click → **Open** the first time when the repository was downloaded rather than cloned.

On a Linux desktop, double-click `Setup Local AI Bench.desktop` to open a terminal and launch the graphical wizard. The launcher resolves the repository from its own location rather than assuming a fixed installation path. Some desktop environments require first enabling **Allow Launching**, **Trust and Launch**, or the executable permission in the file's Properties dialog; the file is shipped executable, but a downloaded archive may not preserve that bit.

On Windows, double-click `Setup Local AI Bench.bat` to launch the graphical wizard. It delegates to the existing `setup.bat`, so prerequisite installation, error handling, and command-line behavior remain in one implementation; cancelling the wizard closes cleanly without offering to run benchmarks.

`setup.sh` / `setup.bat` first switch to the repository directory, locate Python 3.11+, and ask before installing Python or Homebrew when either is missing. On a local macOS or Linux desktop, `setup.sh` also checks Tkinter and offers to install the platform package when it is missing; declining or an unavailable package falls back to terminal setup. The scripts then create or reuse a valid `bench-env/` and hand off to `scripts/setup_check.py`. A local graphical session uses the wizard by default, while SSH/headless sessions retain the terminal interface; pass `--interface gui` or `--interface terminal` to override automatic selection.

The setup wizard collects every decision before installation: a memory-aware model checklist, optional non-catalog cleanup, Hugging Face token and save preference, ComfyUI reuse/download choice, and a final review. Closing or cancelling the wizard performs no installation work, closes the double-click launcher's Terminal session without offering to run benchmarks, and downloads begin only after clicking **Install** on the review page. The terminal interface follows the same defaults and installation backend.

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
5. If image models were selected and no usable ComfyUI installation was detected, offers to download a managed copy by default or accept an existing ComfyUI directory, `main.py`, or Windows-portable launcher path. A valid entry is reused and saved; an invalid entry is reported and setup falls back to the managed download.
6. Installs everything you approved — llama.cpp, any ComfyUI dependencies, LLM/embedding GGUFs, and image checkpoints — with no further prompts. If cleanup was selected, it first deletes only the non-catalog folders shown in the picker; catalog folders and loose files are never cleanup targets.

Setup checks the system installation before planning any llama.cpp install. When `llama-server` is available through `PATH` or a standard macOS Homebrew location, setup does not install or build another llama.cpp copy. It independently detects `llama-bench` and `llama-batched-bench`; when all three system tools are present, benchmark runs use those exact system tools. Models remain managed by Local AI Bench under `models/llamacpp/`, and the selected system binary receives the downloaded GGUF's explicit path, so system installation and project model storage do not need to share a directory. If `llama-server` exists but either optional native benchmark tool is absent, setup leaves the existing installation untouched and reports that the corresponding opt-in test is unavailable rather than installing a second distribution behind the user's back.

Downloaded llama.cpp ZIP and ComfyUI 7z archives are fully inspected before extraction. Setup rejects absolute paths, drive-qualified paths, parent traversal, ambiguous or duplicate normalized paths, and ZIP symbolic links, then stops with the extraction error instead of writing an unsafe or only partially validated archive.

Direct runtime downloads use a sibling `.part` file, resume with an HTTPS Range request after interruption, validate the returned range and expected release-asset size, flush before atomically publishing the completed file, and retain an incomplete part for retry without replacing a known-good destination. If a server ignores the range, setup safely restarts that file instead of appending duplicate bytes. Python's standard proxy environment settings apply automatically. Hugging Face model downloads retain `huggingface_hub`'s own cache/resume behavior.

When llama.cpp is genuinely absent, the installed copy also includes `llama-bench` and `llama-batched-bench`, llama.cpp's own throughput-benchmarking tools, needed only for the opt-in `llamabench` and `llamabenchconc` tests (`--tests llamabench llamabenchconc`) — see [Workloads](workloads.md#llama-bench) and [Workloads](workloads.md#llama-bench-concurrency).

Setup resolves ComfyUI separately from its models. It prefers an explicit `--comfyui` path, then `COMFYUI_DIR`, a detectable running process, the path saved by an earlier setup, conventional manual or Windows-portable locations, and finally the repository-managed `ComfyUI/`; it downloads a managed copy only when none is usable. A valid path contains `main.py`, either directly or under a portable root's `ComfyUI/` child. Process inspection is best-effort and safely falls through when the operating system denies access. Arbitrary Desktop data locations can be supplied once with `--comfyui`; setup saves the resolved program path in the gitignored `local_ai_bench_config.json` file for later setup, frontend, and benchmark commands.

`local_ai_bench_config.json` is the versioned handoff between setup and benchmark execution. It records the validated ComfyUI program directory and resolved `llama-server`, `llama-bench`, and `llama-batched-bench` paths. Consumers validate saved paths and fall back to live discovery when an installation moves or is removed. Repository-managed model locations remain relative to the project rather than being persisted as machine-specific absolute paths, and credentials such as the Hugging Face token are never written to this file.

All image checkpoints, text encoders, and VAEs downloaded by Local AI Bench stay under `models/comfyui/`, even when the ComfyUI program comes from a system installation. Setup adds an idempotent, clearly marked entry to that installation's `extra_model_paths.yaml` without replacing existing entries, and Local AI Bench also passes its generated extra-path configuration whenever it launches ComfyUI. If ComfyUI is already running without that path loaded, stop it and retry; the benchmark detects this condition and does not submit a predictably broken image workflow. ComfyUI Desktop's protected application `resource/ComfyUI` directory should not be selected or modified.

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

Every LLM and embedding model is downloaded as a GGUF file from HuggingFace, resolved from the `hf_repo`/`hf_file` fields in `scripts/models.py` into `models/llamacpp/<tag-slug>/` (see [Engines](engines.md#llamacppengine)). Image checkpoints use the same HuggingFace download client but stay under `models/comfyui/`. Public repositories can be downloaded without an account or token. SD3.5 Large, Flux.1-dev, and Flux.2-dev are gated and require a free account, license acceptance, and an access token:

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
