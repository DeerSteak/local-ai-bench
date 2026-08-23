[← Back to README](../README.md)

# Setup

## Qualification status

This generated matrix reports the current evidence-backed runtime and ComfyUI image status of each platform path. One qualification invocation may exercise both components, but each is graded independently; installation availability does not by itself establish support.

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

### Explicitly unsupported paths

| Platform | Architecture | Runtime | Backend | Accelerator | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| windows with Smart App Control | x86_64 | llamacpp | vulkan / xpu | Intel Arc Pro B65 | Not supported | Enforced policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` blocked the official runtime's SYCL, Vulkan, and CPU DLLs. Qualification stops before downloads and does not bypass the policy. |

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
| macOS | `bash setup.sh` | Python, a project-managed official llama.cpp release (includes llama-bench and llama-batched-bench), ComfyUI (vLLM is not offered — see the platform table below) |
| Linux / WSL2 / DGX Spark | `bash setup.sh` | Python, llama.cpp source build (includes llama-bench and llama-batched-bench), ComfyUI, verified ROCm-enabled PyTorch on AMD, Intel oneAPI/SYCL llama.cpp plus XPU PyTorch on native Linux Arc, and optional vLLM on native Linux CUDA, ROCm, or Intel XPU and NVIDIA WSL2; explicit Radeon WSL2 qualification also installs and verifies the pinned ROCm WSL stack |
| Windows | `setup.bat` | Python, llama.cpp (CUDA on NVIDIA, SYCL on Intel, Vulkan otherwise; includes llama-bench and llama-batched-bench), ComfyUI portable |

On macOS, double-click `Setup Local AI Bench.command` in Finder to open Terminal and launch the graphical wizard directly. The launcher switches to the repository directory automatically and leaves Terminal open when setup fails so the error can be reviewed. Local AI Bench does not automate or close Terminal windows; what happens after the command exits follows the user's Terminal profile settings. macOS may require Control-click → **Open** the first time when the repository was downloaded rather than cloned.

On a Linux desktop, double-click `Setup Local AI Bench.desktop` to open a terminal and launch the graphical wizard. The launcher resolves the repository from its own location rather than assuming a fixed installation path. Some desktop environments require first enabling **Allow Launching**, **Trust and Launch**, or the executable permission in the file's Properties dialog; the file is shipped executable, but a downloaded archive may not preserve that bit.

On Windows, double-click `Setup Local AI Bench.bat` to launch the graphical wizard. It delegates to the existing `setup.bat`, so prerequisite installation, error handling, and command-line behavior remain in one implementation; cancelling the wizard closes cleanly without offering to run benchmarks.

`setup.sh` / `setup.bat` first switch to the repository directory, locate Python 3.11+, and ask before installing Python or Homebrew when either is missing. On Debian/Ubuntu, setup refreshes APT metadata before resolving prerequisites and prefers the distribution's unversioned `python3-venv` package; a missing package now stops with the repository issue instead of continuing into a guaranteed virtual-environment failure. On a local macOS or Linux desktop, `setup.sh` also checks Tkinter and offers to install the platform package when it is missing; declining or an unavailable package falls back to terminal setup. The scripts then create or reuse a valid `bench-env/` and hand off to `scripts/setup/setup_check.py`; on Windows, a broken environment or one tied to an unavailable or outdated interpreter is rebuilt automatically. A local graphical session uses the wizard by default, while SSH/headless sessions retain the terminal interface; pass `--interface gui` or `--interface terminal` to override automatic selection.

The setup wizard collects every decision before installation: an engine checklist, a memory-aware model checklist, optional non-catalog cleanup, Hugging Face token and save preference, ComfyUI reuse/download choice, and a final review. Every page and conditional control group repaints immediately after navigation or a layout change without waiting for pointer movement. Gated image models show a keyboard-accessible **Accept license** button with the Hugging Face URL beside their selection; activating it opens the page in the default browser. Closing or cancelling the wizard performs no installation work and does not offer to run benchmarks; clicking **Install** closes the wizard process completely and opens a separate progress window while the unattended installation runs. The wizard hands its plan back through permission-restricted temporary files that are deleted immediately, keeping any entered token out of command-line arguments. The progress window remains responsive because neither it nor a dormant Tk process performs downloader work; it reports completion or action items and leaves detailed download output and errors visible in the terminal. The terminal interface follows the same defaults and installation backend.

1. Detects your hardware (OS, GPU backend, RAM).
2. Asks which engines to use — llama.cpp checked, vLLM checked only if already installed, and vLLM disabled with a reason when this system can't run it. See [Choosing engines](#choosing-engines).
3. Shows a numbered list of all 12 LLMs, two embedding models, and five image models — everything selected by default except LLM/image models estimated not to fit in detected RAM/VRAM, which start unchecked with a note on how much they'd need, per selected engine. If `models/llamacpp/` contains GGUF model folders that do not belong to the current LLM or embedding catalog, the list also includes one optional cleanup row naming those folders; cleanup is always unchecked by default. Folders without a GGUF and loose files are not cleanup candidates. The estimate includes model weights, required image encoders, a 20% runtime allowance, and a small OS/driver reserve; it is guidance rather than a hard block.
4. Lets you toggle the selection interactively:
   - Numbers to toggle individual models (`2 4 7-9`)
   - A size tier (`xs`/`s`/`m`/`l`) to toggle every model at that tier — LLM and image checkpoints together, e.g. `s` toggles the small-tier LLMs and SDXL as a group
   - `emb`/`img` to toggle a whole section
   - `clean` to toggle deletion of the listed non-catalog llama.cpp model folders
   - `vclean` to toggle deletion of the listed cached vLLM repos, each of which is its own checkbox
   - `a` to select/deselect all models; it deliberately does not enable cleanup
   - Enter to install everything shown
   - `q` or Ctrl-C to cancel before the unattended installation phase; the bootstrap may already have installed Python or created `bench-env/`
5. If you selected any LLM, embedding, or image model, asks for a HuggingFace token next (see [HuggingFace token](#huggingface-token) below).
6. If image models were selected and no usable ComfyUI installation was detected, offers to download a managed copy by default or accept an existing ComfyUI directory, `main.py`, or Windows-portable launcher path. A valid entry is reused and saved; an invalid entry is reported and setup falls back to the managed download.
7. Installs everything you approved — llama.cpp, vLLM if you opted into it, any ComfyUI dependencies, LLM/embedding weights for every selected engine, and image checkpoints — with no further prompts. If cleanup was selected, it first deletes only the entries shown in the picker; catalog models and loose files are never cleanup targets.

Setup checks the system installation before planning any llama.cpp install on Linux and Windows. When all three tools are available through `PATH` and no complete project-managed set exists, benchmark runs use those system tools; models remain managed by Local AI Bench under `models/llamacpp/`, and the selected system binary receives the GGUF's explicit path. If setup must build or download a missing tool, the resulting managed `llama-server`, `llama-bench`, and `llama-batched-bench` are thereafter selected together so binaries from different releases cannot be mixed. On macOS, setup installs an architecture-matched official release under the project so version selection, transactional replacement, and rollback remain under application control. Existing system or Homebrew installations are neither removed nor modified.

Note that if you're using an AMD Ryzen AI Halo running the default Linux image, llama.cpp and llama-bench are installed by default, but the llama-batched-bench utility is not. Installation just requires running the following two-line script:

```
sudo apt update
sudo apt install llama.cpp-tools-extra
```

Downloaded llama.cpp and ComfyUI archives are fully inspected before extraction. Setup rejects absolute paths, drive-qualified paths, parent traversal, ambiguous or duplicate normalized paths, and ZIP symbolic links. A tar symbolic link is accepted only when it resolves to a regular file in the same archive and is materialized as a regular-file copy; other tar links and special members are rejected before anything is written.

Direct runtime downloads use a sibling `.part` file, resume with an HTTPS Range request after interruption, validate the returned range and expected release-asset size, flush before atomically publishing the completed file, and retain an incomplete part for retry without replacing a known-good destination. If a server ignores the range, setup safely restarts that file instead of appending duplicate bytes. Python's standard proxy environment settings apply automatically. Hugging Face model downloads retain `huggingface_hub`'s own cache/resume behavior.

When llama.cpp is genuinely absent, the installed copy also includes `llama-bench` and `llama-batched-bench`, llama.cpp's own throughput-benchmarking tools, needed only for the opt-in `llamabench` and `llamabenchconc` tests (`--tests llamabench llamabenchconc`) — see [Workloads](workloads.md#llama-bench) and [Workloads](workloads.md#llama-bench-concurrency).

Setup resolves ComfyUI separately from its models. It prefers an explicit `--comfyui` path, then `COMFYUI_DIR`, a detectable running process, the path saved by an earlier setup, conventional manual or Windows-portable locations, and finally the repository-managed `ComfyUI/`; it downloads a managed copy only when none is usable. A valid path contains `main.py`, either directly or under a portable root's `ComfyUI/` child. Process inspection is best-effort and safely falls through when the operating system denies access. Arbitrary Desktop data locations can be supplied once with `--comfyui`; setup saves the resolved program path in the gitignored `local_ai_bench_config.json` file for later setup, frontend, and benchmark commands.

`local_ai_bench_config.json` is the versioned handoff between setup and benchmark execution. It records the validated ComfyUI program directory, resolved `llama-server`, `llama-bench`, and `llama-batched-bench` paths, the vLLM runtime when one is present (executable, platform launcher, an already-running server URL, the resolved model-cache location, and any extra arguments that launcher injects), and the name and VRAM reported for each detected NVIDIA CUDA or AMD ROCm GPU (plus NVIDIA driver identity). Consumers validate saved paths and fall back to live discovery when an installation moves or is removed. Repository-managed model locations remain relative to the project rather than being persisted as machine-specific absolute paths, and credentials such as the Hugging Face token are never written to this file. Qualification requires the complete project-managed llama.cpp three-tool runtime even when an external `llama-server` is discoverable, just as vLLM qualification requires the project-managed native CLI.

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

Setup asks which inference engines to install before anything is downloaded, on the wizard's first page or as a numbered picker in the terminal. **llama.cpp is selected by default; [vLLM](https://docs.vllm.ai/) is selected only when it is already installed.** An engine already on the system starts checked so setup keeps maintaining it — headers, build tools, and weights — rather than reporting "already installed" beside a row it then ignores. An engine that would have to be downloaded starts unchecked. vLLM is an opt-in second engine: it downloads several GB and, unlike llama.cpp, its wheels pin their own PyTorch build, so it installs into its own `vllm-env/` virtual environment rather than into `bench-env/`. Nothing else in the project changes when it is skipped. At least one engine must stay selected.

Setup also looks for a platform launcher — AMD's Strix Halo image ships `vllm-launch`, which wraps `vllm serve` with its preconfigured ROCm environment. A project-managed executable takes precedence for both serving and `vllm bench`; the launcher and its `VLLM_EXTRA_ARGS` from `~/.local/share/vLLM/vllm-launch.conf` are recorded only when no managed executable exists. Active injected arguments are printed and stored because a flag that alters a run without appearing in the results file breaks comparability.

Both engines follow the same system-first policy: if a working `vllm` is already on `PATH`, setup uses it and installs nothing, exactly as it does for an existing `llama-server`. Setup also probes `http://localhost:8000/v1/models`, so a vLLM server that is already running — AMD's Strix Halo image ships one preconfigured, and a container or remote server looks the same from here — counts as present even when no host-side `vllm` executable exists.

**An already-present vLLM overrides the platform support gate entirely.** That gate only decides whether setup can *install* vLLM; when there is nothing to install, an "unsupported" or "experimental" verdict is irrelevant and the engine is offered as a normal, selectable option.

The benchmark GUI's Engine Management tab can update a project-owned `vllm-env/` without rerunning setup. It installs into a sibling staging environment, verifies that the runtime exposes the hardware backend required by its install method (CUDA, ROCm, or Intel XPU), and only then replaces the active environment; a failed hardware check or replacement restores the prior environment. UTF-8 operation output is mirrored below the engine panel so update and verification progress can be monitored without switching to the terminal. System installations, platform launchers, and external servers remain under their owner's control and are inspection-only. An external vLLM server's version is read from `/version` when available, and `/health` distinguishes an older reachable server from an offline endpoint when no version is returned.

Runtime updates expose a Cancel Operation action. It terminates the active installer or build process tree, cleans staging, and restores the prior runtime if replacement had started; closing the main window while an update is active offers cancellation but keeps the window open until that cleanup completes.

On Linux, that tab can update and rebuild a project-owned `llama.cpp/` checkout. It resolves the selected official `bNNNNN` release, checks out that exact source tag, embeds `NNNNN` as the upstream build number, configures the detected CPU, CUDA, ROCm, or Intel XPU backend, builds self-contained executables for `llama-server`, `llama-bench`, and `llama-batched-bench`, and requires `llama-server --list-devices` to expose that backend both before and after the rollback-capable replacement. A CPU-only or otherwise mismatched staged build is rejected without replacing the working checkout. On macOS and Windows, it selects the architecture-compatible official archive from that exact release, safely extracts and validates all three tools in staging, then performs the same final-path validation and rollback-capable replacement. Engine Management labels managed source checkouts `Built from source`, managed macOS/Windows archives `Binary download`, and external runtimes `System installation`.

An app-managed llama.cpp runtime also shows **Change…** beside its version. The dialog loads the ten most recent non-draft `bNNNNN` build releases and accepts a specific numeric build or `bNNNNN` tag, allowing either upgrades or downgrades on Linux, macOS, and Windows. GitHub labels llama.cpp's automated build releases as prereleases, so that label is accepted for numeric `bNNNNN` builds; unrelated and draft releases remain excluded. A Mac still using Homebrew can use the main update action once to install the latest managed release; subsequent arbitrary-version changes use **Change…**. The Homebrew formula remains untouched.

An app-managed vLLM runtime on a supported stable CUDA or ROCm wheel platform also shows **Change…** beside its version. The dialog loads the ten newest non-yanked stable releases from PyPI and accepts a specific stable PEP 440 version such as `0.10.2`; the exact `vllm[bench]==VERSION` environment is staged, validated, recreated at the final path, and rolled back on failure through the ordinary managed-vLLM update flow. Experimental DGX Spark does not expose arbitrary version selection; setup and qualification share one reviewed CUDA 13 version and wheel index instead.

**Whatever models you select later are downloaded for every selected engine.** The model picker is not per-engine — the point is to compare the same models on the same hardware across engines.

The two engines store weights differently. llama.cpp's GGUFs are managed by this project under `models/llamacpp/<slug>/`. vLLM's are downloaded into the **HuggingFace cache vLLM itself reads**, so it resolves them by repo id and downloads nothing at run time; the engine never passes a filesystem path. Setup picks that cache in this order: a platform launcher's own cache (`~/.local/share/vLLM/models`, which AMD's `vllm-launch` bind-mounts into its container as `HF_HOME`), then `HF_HOME`, then `~/.cache/huggingface`. The resolved location is printed during setup and recorded in `local_ai_bench_config.json`.

After at least one local inference engine is installed, the benchmark GUI can import a non-catalog Hugging Face model without rerunning Setup. Repository inspection groups complete multipart GGUF sets without mixing quantization subdirectories, excludes projection and drafter files from primary model choices, and recognizes vLLM snapshots only when root-level safetensors weights accompany `config.json`. A canonical `model.safetensors.index.json` selects its exact shard set even when another index is present; an unindexed snapshot must contain exactly one root-level weight file. Inspection resolves the selected revision to a commit, checks disk space when sizes are known, and uses `HF_TOKEN` or the gitignored `hf.txt` for private or gated access. Imports never start an inference server automatically. The Engine Management tab can verify an imported GGUF on demand by loading it CPU-only with a 512-token context on an ephemeral loopback port; the probe stops after three minutes, can be cancelled, and always terminates its temporary server. vLLM imports are checked against the installed runtime's architecture registry without loading weights.

An active model import can be cancelled from the dialog or by attempting to close it. Cancellation is cooperative through Hugging Face's download progress callbacks: llama.cpp partial destination files are removed, and vLLM removes the imported repository's new cache artifacts and incomplete blobs while preserving cache content that was complete before the import. Neither path writes a custom-model registry entry after cancellation.

This matters most for a containerised vLLM: the container cannot see an arbitrary host directory, so weights downloaded to a project folder would be invisible to it and silently re-downloaded at run time.

Cached vLLM repos that the catalog does not own can be deleted from setup, listed separately from the llama.cpp folders and with one checkbox each rather than a single group toggle. That separation is deliberate: `models/llamacpp/` belongs to this project outright, while the Hugging Face cache is shared with anything else on the machine that uses it, so each entry is a decision of its own and nothing is selected by default. Two guards apply. A catalog model's repo is never offered and is refused even if named directly, so a swap in `models.py` cannot make the model you are currently using deletable. And an entry is only listed when it actually contains `.safetensors` weights, which keeps tokenizer-only and dataset cache entries out of the list — the counterpart to the `*.gguf` check that guards llama.cpp cleanup. Symlinked cache entries are refused rather than followed. Deleting an imported llama.cpp folder or vLLM repo also removes its matching custom-model registry entry, allowing the same tag to be imported again later.

Replacing a model's `vllm_repo` strands the previous snapshot in that cache, since nothing removes it automatically. Setup's cleanup list is where those show up.

vLLM's own platform support is much narrower than llama.cpp's, so setup decides what is possible here before offering anything, and states its reasoning rather than attempting an install that would fail after a multi-GB download:

| Platform | Offered as | Install path |
|---|---|---|
| Linux + NVIDIA CUDA | Supported | Prebuilt CUDA wheels; needs compute capability 7.5+ and a Python 3.10–3.13 on `PATH`, which setup offers to provide when the system has none |
| Linux + AMD ROCm (gfx90a/942/950, RX 7900/9000) | Supported | Prebuilt wheels from `wheels.vllm.ai/rocm`; needs ROCm 6.3+ and a CPython 3.12 interpreter, which is the only version those wheels are published for |
| Linux + AMD ROCm (gfx1150/1151, including Strix Halo) | Supported install path | Current official ROCm wheels include these RDNA 3.5 targets; exact-build platform support still requires qualification |
| Linux + AMD ROCm, any other gfx target | Experimental | Setup may offer the ROCm wheel, but reports when its published kernel target list does not match the detected GPU |
| WSL2 + AMD ROCm | Not offered | vLLM's ROCm platform discovery requires AMD SMI interfaces that Radeon WSL2 does not expose; AMD's patched Docker workaround is not the normal environment this project ships |
| DGX Spark (GB10) | Experimental | Reviewed CUDA 13 wheels — the stock aarch64 wheels would silently install CPU-only PyTorch |
| macOS | Not offered | Out of scope for this project — see the design note below |
| Windows (native) | Not offered | vLLM has no upstream Windows support — see [vLLM on Windows via WSL2](#vllm-on-windows-via-wsl2) |
| Linux + Intel XPU | Experimental | Pinned vLLM 0.27.1 source checkout in `vllm-env/src/vllm`, built against its XPU requirements, PyTorch 2.13, `vllm-xpu-kernels` 0.1.12, and `triton-xpu` 3.7.2 with Python 3.12 |
| CPU-only | Not offered | This benchmark measures accelerated inference |

Support is decided from the OS, virtualization, GPU vendor, architecture (CUDA compute capability or ROCm gfx target), ROCm version, and available Python. When vLLM cannot run on this system, its picker row is shown deselected and disabled, with the reason beside it, rather than being hidden or offered and then failing. Installing vLLM also installs your distribution's Python development headers (`python3.X-dev`, `python3-devel`) when they are absent, because Triton compiles a small CUDA helper at import time and vLLM will not start without `Python.h`. That package install needs `sudo`: the terminal names it in the setup plan, and the wizard warns on both the engines page and the final review that you may be prompted for your password in the terminal behind it. On Linux the bootstrap installs these headers up front, alongside any other prerequisite, for both the interpreter it will use and CPython 3.12 (the version vLLM's ROCm, XPU, and DGX Spark builds require) — so the password prompt happens once, early, rather than midway through an unattended install. APT metadata is refreshed only when a missing package must be resolved, so an already-complete setup does not request sudo or fail because of an unrelated repository. `setup_check.py` then finds the headers present and does nothing. Its own header install remains as a fallback for a system whose interpreter arrived some other way. Setup's own bootstrap already installs the headers for a Python it installs itself; this covers the case where the interpreter came from the system instead. After creating a managed vLLM environment, setup imports PyTorch and vLLM, requires vLLM to identify the expected CUDA, ROCm, or XPU device, and checks that PyTorch uses the matching compute runtime; a CPU-only wheel, wrong CUDA/ROCm flavor, unresolved dependency, or unspecified platform blocks benchmark launch instead of producing partial qualification evidence. Qualification always requires and records the project-managed `vllm-env/bin/vllm`, even when another `vllm` is on `PATH`, so its evidence validates the environment setup actually ships.

### When the system Python is too new

vLLM's CUDA wheels stop at Python 3.13, and some distributions now ship something newer as their only `python3` — Ubuntu resolute, for instance, offers 3.14 and nothing else, so there is no in-range interpreter to install from apt at all. `bench-env` itself is unaffected by this, because vLLM builds its own separate virtual environment; the only requirement is that *some* interpreter in 3.10–3.13 exists on `PATH` for that environment to be created from.

Setup therefore looks for one before declaring vLLM unavailable, and treats the running interpreter as just one candidate among `python3.13` through `python3.10`. If none is found, it offers to install a private CPython 3.12 using [uv](https://astral.sh/uv), printing the exact commands first and defaulting to no; unattended qualification installs the required private interpreter automatically and stops before launching a benchmark if setup still has action items. Declining during interactive setup leaves vLLM reported as unsupported, exactly as before. Accepting downloads uv from `astral.sh` into `~/.local/bin` and fetches a standalone interpreter — it does not touch your system Python, your distribution's packages, or `bench-env`.

This is the one place interactive setup reaches outside your distribution's package manager, which is why it asks rather than assuming. If you would rather do it yourself, `uv python install 3.12` before running setup has the same effect, as does any other means of putting a compatible interpreter on `PATH`.

The two experimental paths are unverified by this project's maintainers and are labelled as such in both the wizard and the terminal picker. Setup never installs the third-party native Windows fork of vLLM on your behalf.

**macOS was briefly experimental via the community `vllm-metal` plugin, then deliberately dropped.** That plugin uses MLX rather than vLLM's normal AWQ/GPTQ kernels, so this project's catalog weights aren't loadable by it — supporting it would mean a third per-model repo and a per-backend catalog schema for one experimental platform. macOS setup now reports vLLM unsupported unconditionally; llama.cpp and ComfyUI are unaffected.

### vLLM weights are not the same files

vLLM cannot use the GGUF files llama.cpp benchmarks against, so each catalog model carries a second set of weights (`vllm_repo` in `models.py`) downloaded as a whole HuggingFace snapshot into the cache described above. These are **4-bit AWQ, GPTQ, or compressed-tensors W4A16 safetensors** — chosen as the closest available analogue to the `Q4_K_M` GGUFs, so a cross-engine comparison is at least like-for-like on bit width.

The full per-model mapping is in [Workloads](workloads.md#per-engine-weights). It is not an identical quantization, and it cannot be: `Q4_K_M` is llama.cpp's own k-quant format with per-block mixed precision, while AWQ and GPTQ are different 4-bit algorithms with different calibration. Expect small quality differences alongside the performance ones, and read a cross-engine chart as "this model, quantized the way each engine does 4-bit" rather than "the same file, two runtimes".

Selecting vLLM roughly doubles the download for a given model set, and the two weight sets are usually different sizes — a 4-bit AWQ snapshot is not the same size as the equivalent `Q4_K_M` GGUF. The disk-space check accounts for both.

## vLLM on Windows via WSL2

Native Windows cannot run vLLM, but NVIDIA can use the ordinary Linux CUDA path under WSL2 with the Windows driver's GPU passthrough. Radeon WSL2 can run the llama.cpp and ComfyUI paths through AMD's WSL-specific ROCm userspace stack, but vLLM is not offered because its AMD SMI platform discovery does not work through WSL2; the project does not substitute AMD's separately patched Docker environment for the normal managed runtime.

Treat it as a second machine. WSL2 gets its own clone, its own `bench-env/`, and its own HuggingFace cache — nothing is shared with a Windows-side installation, so the model set is downloaded again in full.

**1. Configure memory before installing anything.** WSL2 defaults to about half the host's RAM, and setup's memory-fit estimate believes what the OS reports — so on a 64GB machine it will silently filter out models that actually fit. Setup warns when it detects WSL2, reporting the RAM it can see, but it cannot check that against the Windows host's total from inside the VM — only you can tell whether the number it prints is the whole machine. Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=56GB
```

VRAM is unaffected; the GPU is passed through directly rather than partitioned.

**2. Install WSL2**, from an administrator PowerShell:

```powershell
wsl --install
```

**3. Install the NVIDIA driver on Windows only.** The host driver projects the GPU into WSL2 through `/dev/dxg`, and `nvidia-smi` works inside the distribution without any Linux driver. Installing an NVIDIA Linux driver inside WSL2 overwrites that passthrough and is the most common way this setup breaks.

For Radeon, use Ubuntu 22.04 or 24.04 and install AMD Software: Adrenalin Edition 26.1.1 for WSL2 on Windows, then reboot. Running a `radeon-wsl2-*-rocm` qualification target downloads AMD's ROCm 7.2 installer, runs its `wsl,rocm` use case with `--no-dkms`, and verifies `rocminfo` before any benchmark starts. Unsupported WSL distributions, a missing host driver, or unavailable GPU passthrough are hard failures; setup does not substitute Vulkan or CPU because neither would validate the requested ROCm target.

**4. Install the CUDA toolkit inside WSL2 — or let setup do it.** The Windows driver provides `libcuda.so` and a working `nvidia-smi`, but not `nvcc`, and llama.cpp is a source build on Linux. Without the toolkit the build is CPU-only. When setup detects WSL2 with an NVIDIA GPU and no `nvcc`, it offers to install this for you, printing the commands first and defaulting to no; declining just means a CPU-only llama.cpp. To do it yourself, use the **WSL-Ubuntu** repository, which ships the toolkit without a Linux driver:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install -y cuda-toolkit
```

Install `cuda-toolkit`, never the `cuda`, `cuda-12-x`, or `cuda-drivers` meta-packages — each of those pulls in the Linux driver and breaks the passthrough described in step 3. See NVIDIA's [CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

The toolkit installs to `/usr/local/cuda/bin`, which is not on `PATH` by default. Setup looks there directly, so a CUDA build works either way, but adding it (`export PATH=/usr/local/cuda/bin:$PATH`) makes `nvcc --version` work in your own shell and is the quickest way to confirm the install.

Setup passes the GPU's compute capability to cmake explicitly rather than letting it use `CMAKE_CUDA_ARCHITECTURES=native`. That default compiles and runs a small probe to detect the local GPU, and under WSL2 the probe reports "No CUDA devices found" even when `nvidia-smi` works, because `libcuda.so` lives in `/usr/lib/wsl/lib` outside the probe's link path. The resulting build reports a CUDA backend but carries no kernels for the card, which shows up as CPU-speed results from a run that looked like it configured correctly.

**5. Inside the WSL2 shell**, install the prerequisites, clone into the WSL2 filesystem, and run setup as normal:

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential git cmake
git clone <repo-url> ~/local-ai-bench && cd ~/local-ai-bench && bash setup.sh
```

Install the unversioned `python3-venv`/`python3-dev` rather than naming a series: recent Ubuntu releases carry exactly one Python, and asking for a version they do not ship fails outright. If your distribution's Python is newer than 3.13, setup handles vLLM's interpreter separately — see [When the system Python is too new](#when-the-system-python-is-too-new).

Clone into the WSL2 filesystem (`~/`), **not** a Windows drive under `/mnt/c`. That path crosses a 9p filesystem bridge slow enough to distort model-load timings, which matters when every benchmarked model is a multi-GB safetensors snapshot. Reusing a Windows checkout that way also exposes it to Windows line endings — if `./setup.sh` fails with ``env: 'bash\r': No such file or directory``, that checkout has CRLF shebangs and needs `git checkout` re-run under the repo's `.gitattributes`, which pins `*.sh` to LF.

Runs made this way record `wsl: true` in the results profile and are tagged `WSL2` in the dashboard. This is not cosmetic: GPU access under WSL2 is virtualized and carries real overhead, so a WSL2 result is not interchangeable with a bare-metal Linux result on identical hardware. See [Limitations](limitations.md).

For locally managed vLLM processes, the app enables vLLM's WSL2 pinned-memory opt-in to support the V2 Model Runner's UVA allocation. An explicitly configured `VLLM_WSL2_ENABLE_PIN_MEMORY` value is preserved, and external vLLM servers remain unmanaged.

Each exact WSL accelerator, backend, and engine combination still requires its own completed qualification record. Setup availability and a successful runtime probe do not create a support claim by themselves.

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

Every LLM and embedding model is downloaded as a GGUF file from HuggingFace, resolved from the `hf_repo`/`hf_file` fields in `scripts/workloads/models.py` into `models/llamacpp/<tag-slug>/` (see [Engines](engines.md#llamacppengine)). Setup also downloads any separate predictor GGUF required by that selected artifact's native-MTP configuration; embedded predictors add no second file, while Qwen 3.8 27B currently adds ~1.4 GB and the disk-space estimate includes it. Image pipelines use the same HuggingFace download client but keep their primary weights, text encoders, and VAEs in the matching managed subdirectories under `models/comfyui/`; setup and the benchmark GUI consider a pipeline installed only when every declared file is present. Public repositories, including the complete Z-Image Turbo pipeline, can be downloaded without an account or token. Flux.1-dev and Flux.2-dev are gated and require a free account, license acceptance, and an access token:

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

**Linux (NVIDIA)** — Python 3.11 is installed with apt when missing on Debian-family systems, after confirmation; other distributions need a manual Python install. On native Ubuntu 24.04 or 26.04 qualification hosts where PCI discovery finds an NVIDIA display adapter but `nvidia-smi` is unavailable, setup installs the current kernel headers and Canonical's hardware-selected signed NVIDIA driver through `ubuntu-drivers install`, then stops at an explicit reboot boundary; if nouveau is loaded, setup persistently blocklists it and rebuilds the initramfs before that reboot. The qualification launcher records this expected boundary as `reboot_required`, exits, and resumes setup only when the user reruns it after reboot. NVIDIA's selected open kernel-module flavor is an official CUDA-capable NVIDIA driver and is distinct from nouveau; setup does not override the distribution's hardware-specific choice. WSL2 never enters this driver path because its GPU driver belongs on Windows. After reboot, native x86-64 Ubuntu setup installs NVIDIA's driver-free `cuda-toolkit` package when `nvcc` is missing, then builds llama.cpp from source with CUDA; setup and qualification reject the result unless `llama-server --list-devices` actually exposes CUDA. Because that is a source build, `git` and `cmake` are installed up front alongside the other prerequisites when an existing llama.cpp binary is not already on `PATH` — the build itself runs in the unattended phase, so a tool discovered missing there would strand the run after its last approval prompt.

NVIDIA qualification uses NVIDIA's proprietary CUDA driver stack, never nouveau, because nouveau does not provide CUDA. On modern GPUs the stack may pair its proprietary CUDA userspace with NVIDIA's dual MIT/GPL open kernel module; `modinfo -F license nvidia` reports `Dual MIT/GPL` for that kernel flavor and `NVIDIA` for the proprietary kernel flavor. Linux AMD qualification instead uses the open-source `amdgpu` kernel driver with ROCm/HIP, and Intel qualification uses the open-source `xe` or `i915` kernel driver with Level Zero/oneAPI; describing those compute paths simply as Mesa would be inaccurate because Mesa is the graphics/Vulkan layer rather than the ROCm or SYCL compute runtime.

**Linux (AMD/Strix Halo, Ryzen AI Max+ 395)** — llama.cpp uses its normal HIP build. Native Ubuntu 24.04 setup installs the `linux-oem-24.04` inbox driver and ROCm with `--no-dkms`; it must never inherit the discrete-Radeon `amdgpu-dkms` path. See [Strix Halo Troubleshooting: Ubuntu 24.04](strix-halo-troubleshooting-ubuntu-24.04.md) for kernel, driver-recovery, reboot, peripheral, and qualification checks. Current upstream vLLM ROCm builds list the RDNA 3.5 `gfx1151` target and require Python 3.12; setup therefore uses the official ROCm wheel channel rather than PyPI's CUDA-oriented package. Qualification still applies to the exact vLLM local-version identifier, bundled ROCm/PyTorch stack, firmware, kernel, and model, so recognizing the wheel target does not by itself create a supported platform claim.

An older preinstalled `vllm-launch` remains discoverable as an external platform runtime, but normal setup installs the project's current ROCm environment when that engine is selected. AWQ behavior remains model- and build-specific, so the installed target build must pass the verified smallest-model workload matrix.

**Linux (AMD/ROCm)** — `rocminfo` detection selects llama.cpp's HIP build. A native discrete-Radeon qualification target installs AMD's pinned ROCm 7.2.1 graphics and compute stack when it is absent, after requiring an AMD display adapter, Ubuntu 24.04, and AMD-supported kernel 6.8 or 6.17; an unsupported kernel stops before any driver change. Strix Halo follows the separate OEM inbox-driver path above. A new driver or render/video group assignment may require one reboot and rerun. With ROCm 7.2 or newer and Python 3.12, setup replaces ComfyUI's default torch packages with AMD's pinned PyTorch 2.9.1 ROCm wheels: ROCm 7.2.1 wheels on native Linux and AMD's WSL-specific ROCm 7.2.0 set under WSL2. The compatible NumPy and SciPy versions are pinned with that stack. The WSL path also removes the wheel's bundled HSA runtime as AMD requires so it uses the installed WSL runtime. Older ROCm installations retain the generic PyTorch ROCm 6.4 fallback. Every path performs a real GPU tensor allocation after installation; failure blocks qualification rather than falling back to CPU.

**Linux (Intel Arc)** — `lspci -nn` detection records `hardware_backend: "xpu"` and retains the PCI device ID needed to distinguish exact Arc models. On Ubuntu 24.04 or 26.04, setup installs Intel's graphics compute packages, the pinned Intel Deep Learning Essentials 2026.1 meta-package, and Intel oneDNN 2026.0, supplying the DPC++ compiler, oneDPL, oneDNN, and oneMKL required by llama.cpp, then builds the managed llama.cpp tools with `GGML_SYCL=ON`. SYCL compilation uses two jobs with at most 20 GiB of installed memory, four jobs above 20 GiB, and eight jobs above 30 GiB; setup prints the selected count, and rerunning it resumes the existing checkout and completed build objects. Setup and benchmark execution prefer the pinned unified-layout `/opt/intel/oneapi/2026.1/oneapi-vars.sh` environment for llama-server, llama-bench, and llama-batched-bench, with `setvars.sh` retained for component-layout installations. Image setup installs PyTorch's XPU wheels and verifies a real GPU tensor allocation. The Intel Arc Pro B65 `8086:e222` llama.cpp, ComfyUI, and vLLM XPU paths are qualified as shown above. Optional vLLM setup creates a separate Python 3.12 environment, checks out the pinned upstream source, installs its pinned XPU and benchmark requirements, replaces NVIDIA Triton with the matching Intel XPU build, installs vLLM without dependency resolution changing that stack, and verifies TorchInductor can use XPU Triton before benchmarking. B65 needs kernel 6.17, including when Ubuntu 24.04 is used; after first-time setup, log out or reboot if the new `render` group membership is not active, then rerun the same command.

**DGX Spark** — Uses the normal Linux NVIDIA source-build path for llama.cpp; its ARM64 architecture does not require a separate prebuilt package. The optional vLLM install is the exception: GB10 needs the reviewed CUDA 13 wheel channel, since the stock aarch64 wheels pull CPU-only PyTorch — see [Choosing engines](#choosing-engines).

**macOS and Linux** — If setup reports a permissions error, fix ownership or permissions for the named path and rerun it as your normal user. Avoid running the whole setup under `sudo`, which can leave the project environment and downloaded files owned by root.

**Windows (NVIDIA)** — Setup chooses the newest llama.cpp CUDA package supported by the installed driver and downloads ComfyUI's NVIDIA portable build; if no compatible CUDA package is available, llama.cpp falls back to Vulkan. It also checks whether portable PyTorch supports the GPU's compute capability and reinstalls the configured cu128 packages when required.

**Windows (AMD)** — The setup script downloads the latest official ComfyUI AMD portable build. No manual ROCm install required.

**Windows (Intel Arc)** — Setup can download ComfyUI's Intel portable build and llama.cpp's official self-contained Windows SYCL package, which includes its required runtime DLLs and does not require a separate oneAPI installation. Results report `backend: "xpu"`. Intel Arc Pro B65 qualification is not supported when Smart App Control policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` is enforced: real attempts blocked the official SYCL, Vulkan, and CPU DLLs with Code Integrity event 3077. Qualification detects that policy and stops before runtime or model downloads without weakening the host security configuration. The target remains unverified on Windows without that policy; native Linux Intel Arc qualification is the supported next path.

**Windows (vLLM)** — Not available natively; run the benchmark inside WSL2, where the ordinary Linux CUDA path applies. See [vLLM on Windows via WSL2](#vllm-on-windows-via-wsl2). llama.cpp and image generation are unaffected and run natively.

**Windows (all)** — If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

[← Back to README](../README.md) · [Workloads →](workloads.md)
