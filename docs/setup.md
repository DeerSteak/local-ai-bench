[← Back to README](../README.md)

# Setup

**Contents**
- [What the setup scripts do](#what-the-setup-scripts-do)
- [Disk space check](#disk-space-check)
- [HuggingFace token](#huggingface-token)
- [Platform notes](#platform-notes)

## What the setup scripts do

| Platform | Script | What it can install |
|---|---|---|
| macOS | `bash setup.sh` | Homebrew, Python, llama.cpp |
| Linux / DGX Spark | `bash setup.sh` | Python, llama.cpp, XPU-enabled PyTorch (Intel Arc — experimental, see [Platform notes](#platform-notes)) |
| Windows | `setup.bat` | Python, llama.cpp, ComfyUI portable |

`setup.sh` / `setup.bat` show exactly what they need to install (Homebrew and/or Python) and ask before doing it — nothing happens silently.

Each script then creates a virtual environment (`bench-env/`) and hands off to `scripts/setup_check.py`, which:

1. Detects your hardware (OS, GPU backend, RAM).
2. Shows a numbered list of every LLM and image model — everything selected by default, except models estimated not to fit in your detected RAM/VRAM, which start unchecked with a note on how much they'd need.
3. Lets you toggle the selection interactively:
   - Numbers to toggle individual models (`2 4 7-9`)
   - A size tier (`xs`/`s`/`m`/`l`) to toggle every model at that tier — LLM and image checkpoints together, e.g. `s` toggles the small-tier LLMs and SDXL as a group
   - `emb`/`img` to toggle a whole section
   - `a` to select/deselect all
   - Enter to install everything shown
   - `q` or Ctrl-C to cancel at any point with nothing installed yet
4. If you selected any LLM, embedding, or image model, asks for a HuggingFace token next (see [HuggingFace token](#huggingface-token) below).
5. Installs everything you picked — llama.cpp, pip packages, LLM/embedding GGUFs, image checkpoints — with no further prompts.

When setup is complete, run the benchmark:

```bash
# Linux / macOS
bash run_bench.sh

# Windows
run_bench.bat
```

These scripts activate the virtual environment automatically and forward any arguments to `scripts/benchmark.py` — see the [CLI Reference](cli-reference.md) for available flags.

## Disk space check

Before downloading anything, `setup_check.py` estimates how much space your selection still needs (skipping whatever's already downloaded) and compares it against free space on your system drive:

- **Enough free space, plus a 10 GB buffer** — proceeds normally.
- **Enough for the downloads, but less than a 10 GB buffer left over** — prints a warning and continues.
- **Not enough free space at all** — prints a failure and adds it to the action-items list at the end (does not stop setup or block your model selection).

Independently of that, if completing the downloads would leave less than 10% of your drive's total capacity free, it also prints a warning and pauses 5 seconds before continuing — just enough to notice, without stopping.

## HuggingFace token

Every LLM and embedding model is downloaded as a GGUF file from HuggingFace, resolved from the `hf_repo`/`hf_file` fields in `scripts/models.py` into `models/llamacpp/<tag>/` (see [Engines](engines.md#llamacppengine)). Image checkpoints go through the same download path. SD3.5 Large, Flux.1-dev, and Flux.2-dev additionally require a free HuggingFace account and license acceptance at:

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.2-dev

If you select any LLM, embedding, or image model in the picker, `setup_check.py` finds your HF token in this order:

1. `HF_TOKEN` environment variable
2. `hf.txt` in the repo root (token on a single line)
3. Interactive prompt — offers to save to `hf.txt` for future runs

A token isn't required for non-gated models, but HuggingFace gives token holders faster downloads — `setup_check.py` will still offer to use one for those downloads if you have one available.

## Platform notes

Close other apps before running — GPU memory contention affects results.

**macOS** — Plug in power and disable sleep (System Settings → Battery) before a long run. For 70B models, watch Activity Monitor → Memory: if pressure turns red and TPS drops between runs, the system is swapping — use `--timeout 600` or `--maxtier medium`.

**Linux (NVIDIA)** — Python 3.11 is installed via apt if missing (you'll be asked to confirm first); on non-Debian distros, install it manually. Verify GPU acceleration after setup: run the benchmark and confirm llama-server loads on GPU in `nvidia-smi`.

**Linux (Intel Arc) — experimental, untested on real hardware** — this project's maintainers don't have access to an Intel Arc machine, so everything below is implemented against Intel's published documentation, not verified against a real run. Package names and version numbers may be wrong or out of date. If you have Arc hardware and try this, please report back (open an issue) with what did or didn't work — that's how this graduates out of experimental.

`setup_check.py` detects the GPU (via `lspci`) and labels it correctly in the hardware summary and results JSON (`"backend": "xpu"`). LLM tests need llama.cpp's SYCL backend for Intel Arc acceleration, which this script doesn't build — `setup_check.py` warns plainly that LLM tests will run on CPU unless you build llama.cpp yourself with `-DGGML_SYCL=ON`.

For image generation, ComfyUI's own [Intel XPU support](https://github.com/comfyanonymous/ComfyUI/pull/409) is already merged into the main repo this project clones — the same clone used for AMD/NVIDIA on Linux, no fork or custom node needed. Two things have to be true for it to actually use the GPU:

- **The Intel GPU compute runtime** (Level Zero/OpenCL) — `setup_check.py` checks for it via `dpkg` and, if missing, prints the exact commands rather than installing it for you: it requires adding [Intel's graphics APT repository](https://dgpu-docs.intel.com/driver/installation.html) and a GPG key, which is a bigger, harder-to-reverse system change than the plain-package installs (Python) this script automates from your distro's own repos.
- **An XPU-enabled PyTorch build** — `setup_check.py` *does* install this automatically (if an image model is selected): ComfyUI's `requirements.txt` normally pulls in a plain torch build, so after installing it, this script reinstalls `torch`/`torchvision`/`torchaudio` from [Intel's XPU wheel index](https://download.pytorch.org/whl/xpu). This is a plain `pip install` — no sudo, no new package source — so it's automated like every other pip install this script does. No IPEX involved: Intel is winding that down (EOL end of March 2026) in favor of PyTorch's native XPU support (mainline since PyTorch 2.5).

**DGX Spark** — Treated the same as any other Linux+NVIDIA box: llama.cpp is built from source (`git`/`cmake` required), same as elsewhere on Linux, since a source build has no prebuilt-binary architecture to match (Spark is ARM64). If RAM looks full outside a benchmark run: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

**macOS and Linux** — If the script fails with a permissions error, run `sudo bash setup.sh` instead.

**Windows (NVIDIA)** — The setup script detects the GPU and downloads the latest official ComfyUI NVIDIA portable build (bundled Python environment). No manual CUDA Toolkit install required.

**Windows (AMD)** — The setup script downloads the latest official ComfyUI AMD portable build. No manual ROCm install required.

**Windows (Intel Arc)** — The setup script detects the GPU, labels it correctly (`"backend": "xpu"`), and downloads the latest official ComfyUI Intel portable build with XPU support, so image generation is GPU-accelerated (this part is Intel's own official build, not something built for this project). LLM tests need llama.cpp's SYCL backend for Intel Arc acceleration, which this script doesn't build — it warns plainly that LLM tests will run on CPU unless you build llama.cpp yourself with `-DGGML_SYCL=ON`. **This warning is experimental** — like the Linux Intel Arc path above, this project's maintainers don't have Arc hardware to verify it against a real run.

**Windows (all)** — If `bench-env\Scripts\activate` gives a permissions error: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

[← Back to README](../README.md) · [Workloads →](workloads.md)
