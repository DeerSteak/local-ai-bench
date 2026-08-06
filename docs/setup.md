[← Back to README](../README.md)

# Setup

**Contents**
- [What the setup scripts do](#what-the-setup-scripts-do)
- [Choosing engines](#choosing-engines)
- [Memory-fit estimate](#memory-fit-estimate)
- [Disk space check](#disk-space-check)
- [HuggingFace token](#huggingface-token)
- [Platform notes](#platform-notes)

## What the setup scripts do

| Platform | Script | What it can install |
|---|---|---|
| macOS | `bash setup.sh` | Homebrew, Python, llama.cpp (includes llama-bench and llama-batched-bench), ComfyUI, optionally vLLM via the vllm-metal plugin (experimental) |
| Linux / DGX Spark | `bash setup.sh` | Python, llama.cpp source build (includes llama-bench and llama-batched-bench), ComfyUI, ROCm-enabled PyTorch on AMD, XPU-enabled PyTorch on Intel Arc (experimental), optionally vLLM on NVIDIA/ROCm |
| Windows | `setup.bat` | Python, llama.cpp (CUDA on NVIDIA, Vulkan otherwise; includes llama-bench and llama-batched-bench), ComfyUI portable |

On macOS, double-click `Setup Local AI Bench.command` in Finder to open Terminal and launch the graphical wizard directly. The launcher switches to the repository directory automatically and leaves Terminal open when setup fails so the error can be reviewed. Local AI Bench does not automate or close Terminal windows; what happens after the command exits follows the user's Terminal profile settings. macOS may require Control-click → **Open** the first time when the repository was downloaded rather than cloned.

On a Linux desktop, double-click `Setup Local AI Bench.desktop` to open a terminal and launch the graphical wizard. The launcher resolves the repository from its own location rather than assuming a fixed installation path. Some desktop environments require first enabling **Allow Launching**, **Trust and Launch**, or the executable permission in the file's Properties dialog; the file is shipped executable, but a downloaded archive may not preserve that bit.

On Windows, double-click `Setup Local AI Bench.bat` to launch the graphical wizard. It delegates to the existing `setup.bat`, so prerequisite installation, error handling, and command-line behavior remain in one implementation; cancelling the wizard closes cleanly without offering to run benchmarks.

`setup.sh` / `setup.bat` first switch to the repository directory, locate Python 3.11+, and ask before installing Python or Homebrew when either is missing. On a local macOS or Linux desktop, `setup.sh` also checks Tkinter and offers to install the platform package when it is missing; declining or an unavailable package falls back to terminal setup. The scripts then create or reuse a valid `bench-env/` and hand off to `scripts/setup/setup_check.py`. A local graphical session uses the wizard by default, while SSH/headless sessions retain the terminal interface; pass `--interface gui` or `--interface terminal` to override automatic selection.

The setup wizard collects every decision before installation: an engine checklist, a memory-aware model checklist, optional non-catalog cleanup, Hugging Face token and save preference, ComfyUI reuse/download choice, and a final review. Every page and conditional control group repaints immediately after navigation or a layout change without waiting for pointer movement. Gated image models show a keyboard-accessible **Accept license** button with the Hugging Face URL beside their selection; activating it opens the page in the default browser. Closing or cancelling the wizard performs no installation work and does not offer to run benchmarks; clicking **Install** closes the wizard process completely and opens a separate progress window while the unattended installation runs. The wizard hands its plan back through permission-restricted temporary files that are deleted immediately, keeping any entered token out of command-line arguments. The progress window remains responsive because neither it nor a dormant Tk process performs downloader work; it reports completion or action items and leaves detailed download output and errors visible in the terminal. The terminal interface follows the same defaults and installation backend.

1. Detects your hardware (OS, GPU backend, RAM).
2. Asks which engines to install — llama.cpp checked, vLLM unchecked, and vLLM disabled with a reason when this system can't run it. See [Choosing engines](#choosing-engines).
3. Shows a numbered list of all 12 LLMs, two embedding models, and five image models — everything selected by default except LLM/image models estimated not to fit in detected RAM/VRAM, which start unchecked with a note on how much they'd need, per selected engine. If `models/llamacpp/` contains GGUF model folders that do not belong to the current LLM or embedding catalog, the list also includes one optional cleanup row naming those folders; cleanup is always unchecked by default. Folders without a GGUF and loose files are not cleanup candidates. The estimate includes model weights, required image encoders, a 20% runtime allowance, and a small OS/driver reserve; it is guidance rather than a hard block.
4. Lets you toggle the selection interactively:
   - Numbers to toggle individual models (`2 4 7-9`)
   - A size tier (`xs`/`s`/`m`/`l`) to toggle every model at that tier — LLM and image checkpoints together, e.g. `s` toggles the small-tier LLMs and SDXL as a group
   - `emb`/`img` to toggle a whole section
   - `clean` to toggle deletion of the listed non-catalog model folders
   - `a` to select/deselect all models; it deliberately does not enable cleanup
   - Enter to install everything shown
   - `q` or Ctrl-C to cancel before the unattended installation phase; the bootstrap may already have installed Python or created `bench-env/`
5. If you selected any LLM, embedding, or image model, asks for a HuggingFace token next (see [HuggingFace token](#huggingface-token) below).
6. If image models were selected and no usable ComfyUI installation was detected, offers to download a managed copy by default or accept an existing ComfyUI directory, `main.py`, or Windows-portable launcher path. A valid entry is reused and saved; an invalid entry is reported and setup falls back to the managed download.
7. Installs everything you approved — llama.cpp, vLLM if you opted into it, any ComfyUI dependencies, LLM/embedding weights for every selected engine, and image checkpoints — with no further prompts. If cleanup was selected, it first deletes only the non-catalog folders shown in the picker; catalog folders and loose files are never cleanup targets.

Setup checks the system installation before planning any llama.cpp install. When `llama-server` is available through `PATH` or a standard macOS Homebrew location, setup does not install or build another llama.cpp copy. It independently detects `llama-bench` and `llama-batched-bench`; when all three system tools are present, benchmark runs use those exact system tools. Models remain managed by Local AI Bench under `models/llamacpp/`, and the selected system binary receives the downloaded GGUF's explicit path, so system installation and project model storage do not need to share a directory. If `llama-server` exists but either optional native benchmark tool is absent, setup leaves the existing installation untouched and reports that the corresponding opt-in test is unavailable rather than installing a second distribution behind the user's back.

Downloaded llama.cpp ZIP and ComfyUI 7z archives are fully inspected before extraction. Setup rejects absolute paths, drive-qualified paths, parent traversal, ambiguous or duplicate normalized paths, and ZIP symbolic links, then stops with the extraction error instead of writing an unsafe or only partially validated archive.

Direct runtime downloads use a sibling `.part` file, resume with an HTTPS Range request after interruption, validate the returned range and expected release-asset size, flush before atomically publishing the completed file, and retain an incomplete part for retry without replacing a known-good destination. If a server ignores the range, setup safely restarts that file instead of appending duplicate bytes. Python's standard proxy environment settings apply automatically. Hugging Face model downloads retain `huggingface_hub`'s own cache/resume behavior.

When llama.cpp is genuinely absent, the installed copy also includes `llama-bench` and `llama-batched-bench`, llama.cpp's own throughput-benchmarking tools, needed only for the opt-in `llamabench` and `llamabenchconc` tests (`--tests llamabench llamabenchconc`) — see [Workloads](workloads.md#llama-bench) and [Workloads](workloads.md#llama-bench-concurrency).

Setup resolves ComfyUI separately from its models. It prefers an explicit `--comfyui` path, then `COMFYUI_DIR`, a detectable running process, the path saved by an earlier setup, conventional manual or Windows-portable locations, and finally the repository-managed `ComfyUI/`; it downloads a managed copy only when none is usable. A valid path contains `main.py`, either directly or under a portable root's `ComfyUI/` child. Process inspection is best-effort and safely falls through when the operating system denies access. Arbitrary Desktop data locations can be supplied once with `--comfyui`; setup saves the resolved program path in the gitignored `local_ai_bench_config.json` file for later setup, frontend, and benchmark commands.

`local_ai_bench_config.json` is the versioned handoff between setup and benchmark execution. It records the validated ComfyUI program directory, resolved `llama-server`, `llama-bench`, and `llama-batched-bench` paths, the vLLM runtime when one is present (executable, platform launcher, an already-running server URL, the resolved model-cache location, and any extra arguments that launcher injects), and the name and VRAM reported for each detected NVIDIA CUDA or AMD ROCm GPU (plus NVIDIA driver identity). Consumers validate saved paths and fall back to live discovery when an installation moves or is removed. Repository-managed model locations remain relative to the project rather than being persisted as machine-specific absolute paths, and credentials such as the Hugging Face token are never written to this file.

All image checkpoints, text encoders, and VAEs downloaded by Local AI Bench stay under `models/comfyui/`, even when the ComfyUI program comes from a system installation. Setup adds an idempotent, clearly marked entry to that installation's `extra_model_paths.yaml` without replacing existing entries, and Local AI Bench also passes its generated extra-path configuration whenever it launches ComfyUI. If ComfyUI is already running without that path loaded, stop it and retry; the benchmark detects this condition and does not submit a predictably broken image workflow. ComfyUI Desktop's protected application `resource/ComfyUI` directory should not be selected or modified.

Non-catalog cleanup is permanent rather than a move to Trash or Recycle Bin. It is deliberately excluded from the default selection and the `a` shortcut; select its numbered row or type `clean` only after checking the displayed folder names for models you want to keep.

When setup is complete, run the benchmark:

```bash
# Linux / macOS
bash run_bench.sh

# Windows
run_bench.bat
```

These scripts activate the virtual environment automatically and forward any arguments to `scripts/app/benchmark.py` — see the [CLI Reference](cli-reference.md) for available flags.

## Choosing engines

Setup asks which inference engines to install before anything is downloaded, on the wizard's first page or as a numbered picker in the terminal. **llama.cpp is selected by default and [vLLM](https://docs.vllm.ai/) is not.** vLLM is an opt-in second engine: it downloads several GB and, unlike llama.cpp, its wheels pin their own PyTorch build, so it installs into its own `vllm-env/` virtual environment rather than into `bench-env/`. Nothing else in the project changes when it is skipped. At least one engine must stay selected.

Setup also looks for a platform launcher — AMD's Strix Halo image ships `vllm-launch`, which wraps `vllm serve` with the ROCm environment that hardware needs — and records its path alongside any `VLLM_EXTRA_ARGS` from `~/.local/share/vLLM/vllm-launch.conf`. Those injected arguments are printed as a warning during setup and stored in the configuration, because a flag that alters a run without appearing in the results file breaks the comparability the results format exists to provide.

Both engines follow the same system-first policy: if a working `vllm` is already on `PATH` (or in `~/.venv-vllm-metal` from the Metal plugin's own installer), setup uses it and installs nothing, exactly as it does for an existing `llama-server`. Setup also probes `http://localhost:8000/v1/models`, so a vLLM server that is already running — AMD's Strix Halo image ships one preconfigured, and a container or remote server looks the same from here — counts as present even when no host-side `vllm` executable exists.

**An already-present vLLM overrides the platform support gate entirely.** That gate only decides whether setup can *install* vLLM; when there is nothing to install, an "unsupported" or "experimental" verdict is irrelevant and the engine is offered as a normal, selectable option.

**Whatever models you select later are downloaded for every selected engine.** The model picker is not per-engine — the point is to compare the same models on the same hardware across engines.

The two engines store weights differently. llama.cpp's GGUFs are managed by this project under `models/llamacpp/<slug>/`. vLLM's are downloaded into the **HuggingFace cache vLLM itself reads**, so it resolves them by repo id and downloads nothing at run time; the engine never passes a filesystem path. Setup picks that cache in this order: a platform launcher's own cache (`~/.local/share/vLLM/models`, which AMD's `vllm-launch` bind-mounts into its container as `HF_HOME`), then `HF_HOME`, then `~/.cache/huggingface`. The resolved location is printed during setup and recorded in `local_ai_bench_config.json`.

This matters most for a containerised vLLM: the container cannot see an arbitrary host directory, so weights downloaded to a project folder would be invisible to it and silently re-downloaded at run time. Because that cache is shared with other tools on the machine, non-catalog cleanup never touches it — cleanup remains limited to `models/llamacpp/`.

vLLM's own platform support is much narrower than llama.cpp's, so setup decides what is possible here before offering anything, and states its reasoning rather than attempting an install that would fail after a multi-GB download:

| Platform | Offered as | Install path |
|---|---|---|
| Linux + NVIDIA CUDA | Supported | Prebuilt CUDA wheels; needs compute capability 7.5+ and Python 3.10–3.13 |
| Linux + AMD ROCm (gfx90a/942/950, RX 7900/9000) | Supported | Prebuilt wheels from `wheels.vllm.ai/rocm`; needs ROCm 6.3+ and a CPython 3.12 interpreter, which is the only version those wheels are published for |
| Linux + AMD ROCm, any other gfx target (e.g. gfx1151 / Strix Halo) | Experimental | Same wheels, but they ship no kernels for that target — see the Strix Halo platform note below |
| DGX Spark (GB10) | Experimental | CUDA 13 nightly wheels — the stock aarch64 wheels would silently install CPU-only PyTorch |
| macOS (Apple Silicon) | Experimental | The community-maintained `vllm-metal` plugin, via its own installer into `~/.venv-vllm-metal` |
| Windows | Not offered | vLLM has no upstream Windows support; run it under WSL2, where the Linux path applies |
| Linux + Intel XPU | Not offered | No prebuilt wheels exist; the source build is out of scope for this script |
| CPU-only | Not offered | This benchmark measures accelerated inference |

Support is decided from the OS, GPU vendor, architecture (CUDA compute capability or ROCm gfx target), ROCm version, and available Python. When vLLM cannot run on this system, its picker row is shown deselected and disabled, with the reason beside it, rather than being hidden or offered and then failing. The two experimental paths are unverified by this project's maintainers and are labelled as such in both the wizard and the terminal picker. Setup never installs the third-party native Windows fork of vLLM on your behalf.

### vLLM weights are not the same files

vLLM cannot use the GGUF files llama.cpp benchmarks against, so each catalog model carries a second set of weights (`vllm_repo` in `models.py`) downloaded as a whole HuggingFace snapshot into the cache described above. These are **4-bit AWQ, GPTQ, or compressed-tensors W4A16 safetensors** — chosen as the closest available analogue to the `Q4_K_M` GGUFs, so a cross-engine comparison is at least like-for-like on bit width.

The full per-model mapping is in [Workloads](workloads.md#per-engine-weights). It is not an identical quantization, and it cannot be: `Q4_K_M` is llama.cpp's own k-quant format with per-block mixed precision, while AWQ and GPTQ are different 4-bit algorithms with different calibration. Expect small quality differences alongside the performance ones, and read a cross-engine chart as "this model, quantized the way each engine does 4-bit" rather than "the same file, two runtimes".

Selecting vLLM roughly doubles the download for a given model set, and the two weight sets are usually different sizes — a 4-bit AWQ snapshot is not the same size as the equivalent `Q4_K_M` GGUF. The disk-space check accounts for both.

Setup installs the runtime and fetches these weights, but there is still no `VllmEngine` and no `--engine vllm` to select. See [the engine plan](vllm-engine-plan.md) for what remains.

## Memory-fit estimate

`hardware.py`'s memory ceiling is VRAM (discrete GPU) or total system RAM (unified memory, integrated GPU, or CPU-only), minus a reserve — 1.0 GB for each discrete NVIDIA or ROCm AMD GPU (driver/other GPU processes), 8.0 GB for RAM (OS, the inference server, and everything else sharing that pool). Multiple cards contribute their individually reserved capacities, so two detected 16GB cards produce an approximately 30GB model ceiling. A model's estimated footprint is its download size plus a flat 20% runtime overhead (KV-cache for LLMs, activations for image models) — an approximation, not a per-context-length calculation.

The estimate is **per engine**, because the two engines download different weights for the same model. Whichever engines you selected are the ones the picker sizes against: with only llama.cpp it shows the GGUF size, with only vLLM it shows the AWQ/W4A16 snapshot size, and with both it shows both (`llama.cpp ~6.2 GB · vLLM ~12.4 GB`) with a separate warning naming each engine the model won't fit on. A model is left checked by default when it fits **any** selected engine — a model that runs fine on llama.cpp is still worth downloading even when the vLLM copy is too large — so a warning beside a checked model means "this will be skipped on that one engine", not "this won't run at all". Changing the engine selection in the wizard re-labels and re-defaults the model list when you return to it.

This makes the vLLM footprint visible but does not model it exactly: vLLM also preallocates KV cache to `--gpu-memory-utilization` (0.9 by default), which the flat 20% runtime overhead above does not represent. Treat a vLLM row that only just fits as likely not to.

Discrete-vs-integrated GPU classification is a naming-convention heuristic (AMD: `RX`/`PRO`/`INSTINCT` in the name; Intel: a model number like `A770`), not authoritative. An unknown or ambiguous name defaults to "integrated" — the more permissive failure mode, since it falls back to the system-RAM ceiling instead of wrongly capping to a VRAM number that doesn't apply.

## Disk space check

Before downloading anything, `setup_check.py` estimates how much space your selection still needs — summed across every selected engine, and skipping whatever's already downloaded for each and compares it against free space on the volume containing the repository and its managed models:

- **Enough free space, plus a 10 GB buffer** — proceeds normally.
- **Enough for the downloads, but less than a 10 GB buffer left over** — prints a warning and continues.
- **Not enough free space at all** — stops before model downloads, explains that continuing could create a partial installation or fill the volume, and reports approximately how much additional space must be freed before rerunning setup.

Independently of that, if completing the downloads would leave less than 10% of your drive's total capacity free, it also prints a warning and pauses 5 seconds before continuing — just enough to notice, without stopping.

## HuggingFace token

Every LLM and embedding model is downloaded as a GGUF file from HuggingFace, resolved from the `hf_repo`/`hf_file` fields in `scripts/workloads/models.py` into `models/llamacpp/<tag-slug>/` (see [Engines](engines.md#llamacppengine)). Image checkpoints use the same HuggingFace download client but stay under `models/comfyui/`. Public repositories can be downloaded without an account or token. SD3.5 Large, Flux.1-dev, and Flux.2-dev are gated and require a free account, license acceptance, and an access token:

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.2-dev

If you select any LLM, embedding, or image model in the picker, `setup_check.py` finds your HF token in this order:

1. `HF_TOKEN` environment variable
2. `hf.txt` in the repo root (token on a single line)
3. Interactive prompt — saves to `hf.txt` by default for future runs, with an explicit opt-out

A token isn't required for non-gated models, but authenticated downloads generally receive better rate limits. `setup_check.py` therefore offers token authentication whenever any model is selected; pressing Enter skips it when no gated image model was selected. When the graphical setup detects an existing `HF_TOKEN` or `hf.txt` credential, it disables the token-entry and save controls by default; select **Override token** to enable them and supply a replacement. Whenever token entry is enabled, the wizard shows brief creation instructions and an **Open Hugging Face login** button that opens `https://huggingface.co/login` in the default browser. The graphical review identifies which credential source will be used without displaying its value. A saved token is written as a single line with user-only permissions on platforms that support POSIX file modes, and `hf.txt` remains excluded from Git.

## Platform notes

Close other apps before running — GPU memory contention affects results.

**macOS** — Plug in power and disable sleep (System Settings → Battery) before a long run. For 70B models, watch Activity Monitor → Memory: if pressure turns red and TPS drops between runs, the system is swapping — use `--timeout 600` or `--maxtier medium`.

**Linux (NVIDIA)** — Python 3.11 is installed with apt when missing on Debian-family systems, after confirmation; other distributions need a manual Python install. llama.cpp is built from source with CUDA when an NVIDIA GPU is detected.

**Linux (AMD/Strix Halo, Ryzen AI Max+ 395) — vLLM is experimental here** — llama.cpp is unaffected and uses its normal HIP build. For vLLM, setup reads the GPU's gfx target from `rocminfo` and compares it against the targets vLLM's prebuilt ROCm wheels actually ship kernels for (`gfx90a`, `gfx942`, `gfx950`, `gfx1100`, `gfx1200`, `gfx1201`). Strix Halo reports `gfx1151`, which is not among them, so vLLM is offered as experimental with the reason shown rather than as a plain supported option — **unless a vLLM is already present**, which on AMD's own Strix Halo image it is, in which case none of this applies and the engine is simply selectable. It is still installable — people do run vLLM on this hardware — but the known-working route is a TheRock-based container rather than the stock wheel index this script uses. The most actively maintained one is [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes), a Fedora Toolbx/Podman image built on TheRock ROCm nightlies, whose own tested-model list includes `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` — the same weights this project's catalog selects for Qwen3.6 35B-A3B.

Setup **points at that toolbox rather than installing it**, deliberately. A container is a different execution model from the `vllm-env/` virtualenv everything else here assumes: it needs podman or toolbox on the host, `--device /dev/dri --device /dev/kfd`, a relaxed seccomp profile, and a mounted model cache. Pulling a third-party image with those privileges is a decision to make yourself, not one an install script should make quietly — and the image carries no license file at the time of writing, which matters if you redistribute results or environments. The intended path for using it is the same as for any other externally managed vLLM: bring your own server, and point the benchmark at it. On the AWQ weights this project selects: [vLLM #37151](https://github.com/vllm-project/vllm/issues/37151) reports a segfault in `libhsa-runtime64.so` when loading an AWQ model on a Ryzen AI Max+ 395, so the risk is real — but AWQ has been observed loading successfully on AMD's own Strix Halo image (vLLM 0.21.0+rocm713), so it is not a blanket failure. Which vLLM build you are on matters more than the quantization.

**Linux (AMD/ROCm)** — `rocminfo` detection selects llama.cpp's HIP build. When image models are selected, setup replaces ComfyUI's default torch packages with the configured ROCm 6.4 wheels. This path is not verified on every newer APU architecture; if the wheel does not support the detected GPU, install a compatible PyTorch ROCm build manually.

**Linux (Intel Arc) — experimental** — `lspci` detection records `hardware_backend: "xpu"`, but setup does not build llama.cpp's SYCL backend, so LLM inference remains CPU unless you supply a manual `-DGGML_SYCL=ON` build. For image generation, setup checks for Intel's compute runtime and prints installation commands when it is absent; when image models are selected, it installs PyTorch's XPU wheels. This path has not been verified on real Arc hardware by the project maintainer.

**DGX Spark** — Uses the normal Linux NVIDIA source-build path for llama.cpp; its ARM64 architecture does not require a separate prebuilt package. The optional vLLM install is the exception: GB10 needs the CUDA 13 nightly wheels, since the stock aarch64 wheels pull CPU-only PyTorch — see [Choosing engines](#choosing-engines).

**macOS and Linux** — If setup reports a permissions error, fix ownership or permissions for the named path and rerun it as your normal user. Avoid running the whole setup under `sudo`, which can leave the project environment and downloaded files owned by root.

**Windows (NVIDIA)** — Setup chooses the newest llama.cpp CUDA package supported by the installed driver and downloads ComfyUI's NVIDIA portable build; if no compatible CUDA package is available, llama.cpp falls back to Vulkan. It also checks whether portable PyTorch supports the GPU's compute capability and reinstalls the configured cu128 packages when required.

**Windows (AMD)** — The setup script downloads the latest official ComfyUI AMD portable build. No manual ROCm install required.

**Windows (Intel Arc) — experimental** — Setup downloads ComfyUI's Intel portable build and uses llama.cpp's Vulkan package. Results therefore report `backend: "vulkan"` while retaining `hardware_backend: "xpu"`; a manual SYCL build reports `xpu`. This path has not been verified on real Arc hardware by the project maintainer.

**Windows (all)** — If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

[← Back to README](../README.md) · [Workloads →](workloads.md)
