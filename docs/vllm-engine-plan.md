[← Back to README](../README.md)

# Plan: adding vLLM as a second engine

**Contents**
- [Status](#status)
- [Platform support research](#platform-support-research)
- [Consequences that are not about installation](#consequences-that-are-not-about-installation)
- [Setup script plan](#setup-script-plan)
- [Engine plan](#engine-plan)
- [Config, catalog, and CLI changes](#config-catalog-and-cli-changes)
- [Tests](#tests)
- [Docs to update](#docs-to-update)
- [Suggested order of work](#suggested-order-of-work)
- [Open questions for the user](#open-questions-for-the-user)

## Status

**The setup half is complete** — [`scripts/setup/vllm_install.py`](../scripts/setup/vllm_install.py) plus detection, the opt-in prompt/checkbox, and the installer in `setup_check.py`/`setup_gui.py`; see [Setup](setup.md#optional-vllm-install). Scope as agreed: Linux CUDA and ROCm as supported, DGX Spark (nightly cu130) and macOS Apple Silicon (vllm-metal) as experimental, Windows/XPU/CPU-only reported unsupported with a reason. Model weights are included: setup now asks which engines to install up front (llama.cpp on by default, vLLM off and disabled where unsupported), and downloads the selected models for every selected engine. The quantization question below is answered — vLLM uses 4-bit AWQ/GPTQ/W4A16 safetensors as the closest analogue to this project's `Q4_K_M` GGUFs.

Everything else here — the engine, the catalog, the results/dashboard changes — is still plan only. Written August 2026 against vLLM's stable docs at the time; vLLM's install matrix moves fast, so every command below should be re-checked against [docs.vllm.ai](https://docs.vllm.ai/en/stable/getting_started/installation/) at implementation time rather than trusted as pinned truth.

## Platform support research

vLLM's own support matrix is much narrower than llama.cpp's. Linux is the only first-class OS; Windows is not supported at all upstream, and macOS is served by a separate plugin project rather than the main wheel.

| Platform | Upstream status | Install path | Python | Notes |
|---|---|---|---|---|
| Linux + NVIDIA CUDA | Officially supported, prebuilt wheels | `uv pip install vllm --torch-backend=auto` | 3.10–3.13 | Requires compute capability ≥ 7.5. Pinned CUDA variants (cu128/cu129/cu130) available from the GitHub release assets. Source build needs GCC ≥ 11.3. |
| Linux + AMD ROCm | Officially supported, prebuilt wheels | `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade` | **3.12 only** | ROCm ≥ 6.3. gfx90a (MI200), gfx942 (MI300), gfx950 (MI350, needs ROCm ≥ 7.0), Radeon RX 7900/9000. Source build sets `PYTORCH_ROCM_ARCH`. |
| Linux + Intel XPU | Supported, **no prebuilt wheels** | Source build: `pip install -r requirements/xpu.txt`, `triton-xpu`, then `VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e .` | **3.12 only** | Needs the separate `vllm-xpu-kernels` package (published for CPython 3.12 only). Realistically a long, fragile build. |
| Linux CPU (x86) | Supported, prebuilt wheels since v0.17.0 | wheel (AVX512 or AVX2 variant) | 3.10–3.13 | AVX512F recommended; AVX2 is feature-limited. Wants TCMalloc + Intel OpenMP in `LD_PRELOAD`. |
| Linux CPU (ARM aarch64) | Supported, prebuilt wheels since v0.11.2 | wheel | 3.10–3.13 | NEON required. |
| DGX Spark (GB10, sm_121, aarch64 + CUDA 13) | **Not covered by stock wheels** | Nightly cu130 wheels: `uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130`, or a vendor/community container | 3.12 | Plain `pip install vllm` on aarch64 pulls **CPU-only** PyTorch and silently produces a CPU run. Upstream tracking issue for SM121 is still open. |
| Windows (any GPU) | **Not supported upstream**, no roadmap | WSL2 (then it is exactly the Linux path), Docker Desktop + WSL2 backend, or the unofficial `SystemPanic/vllm-windows` wheels | 3.12 for the community wheels | The community fork is a third-party build; it should not be installed automatically by our setup script. |
| macOS Apple Silicon | Main wheel: source-build CPU backend only (no wheels, slow). Plugin: `vllm-metal` | `curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh \| bash` (installs into `~/.venv-vllm-metal`) | native arm64 **3.12** | `vllm-metal` is community-maintained under the vllm-project org and uses MLX as its compute backend. Requires Xcode Command Line Tools. Rosetta/x86_64 Python is unsupported. |
| macOS Intel | Effectively unsupported | — | — | Treat as "vLLM unavailable". |

The practical conclusion for this project: **vLLM is a supported engine on Linux (CUDA, ROCm, CPU), an experimental one on macOS Apple Silicon and DGX Spark, and an unsupported one on Windows and Intel XPU.** The setup script should say exactly that instead of attempting an install that will fail after downloading several GB of PyTorch.

Sources: [Installation overview](https://docs.vllm.ai/en/stable/getting_started/installation/) · [GPU install](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) · [CPU install](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/) · [vllm-metal](https://github.com/vllm-project/vllm-metal) · [vLLM SM121/DGX Spark issue](https://github.com/vllm-project/vllm/issues/31128) · [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)

## Consequences that are not about installation

These matter more than the install commands and should be settled before any code is written.

**vLLM does not use our GGUF files.** Its GGUF path exists but is experimental, slow, and not available for every architecture. A real vLLM engine wants HuggingFace safetensors repos — a different weight artifact per catalog tag, downloaded separately, at a different size, and at a *different quantization* (bf16/FP8/AWQ/GPTQ) than our `q4_K_M` GGUFs. That means a vLLM run is **not a like-for-like comparison against the llama.cpp run of the "same" model**, and the results file has to make that visible rather than implying the two bars are the same model. This is the single largest design decision in the whole effort.

**Quantization support is backend-specific.** FP8 needs Hopper/Blackwell (or MI300+); AWQ/GPTQ Marlin kernels are CUDA-centric; ROCm and Metal support a narrower set. A single `vllm_repo` per tag will not work everywhere — expect a per-backend choice, or a deliberately conservative default (bf16 where memory allows, AWQ otherwise) with a documented skip when the backend cannot run it.

**vLLM cannot share `bench-env/`.** It pins its own torch build (cu129/cu130/ROCm/XPU) which would collide with the ComfyUI/torch install already in the environment, and the ROCm/XPU/Metal wheels are CPython-3.12-only while `bench-env` follows whatever Python the user has. The right shape is a **separate `vllm-env/` venv** (and on macOS, whatever `vllm-metal`'s `install.sh` created), with `VllmEngine` spawning `vllm serve` from that interpreter as a subprocess — exactly the pattern `Shared.find_comfyui_python` already establishes for ComfyUI.

**Memory footprint.** vLLM preallocates KV cache to `--gpu-memory-utilization` (0.9 default) and will happily take the whole GPU. On the 8GB-GPU end of this project's target range, most catalog models simply will not load. Setup's memory-ceiling model filtering needs a vLLM-specific, stricter ceiling.

**The two `llama-bench` workloads stay llama.cpp-only.** They already `isinstance`-check `LlamaCppEngine` and skip otherwise, so nothing changes there.

## Setup script plan

All of this lands in `scripts/setup/`. Per the repo's safety rules, none of it gets verified by actually running `setup.sh` — the decision logic is extracted into pure functions and unit tested, and the installer itself is left to the user to exercise.

**1. Detection (read-only), in a new `# ── 4b. vLLM ──` section of `setup_check.py`.** *(done)*
- `find_vllm_python()` / `find_vllm_binary()` — mirror `_find_llamacpp_exe`: check `config.VLLM_VENV/bin/vllm`, then `~/.venv-vllm-metal/bin/vllm` on macOS, then system `PATH`.
- Report found/not-found the same way the llama.cpp section does.

**2. `vllm_platform_support(...)` — the extractable business logic.** *(done)* A pure module-level function (best home: a new `scripts/setup/vllm_install.py`, importable without side effects) taking the values `setup_check.py` has already detected — `os_name`, `machine` (arch), `nvidia_ok`, `nvidia_compute_cap`, `nvidia_max_cuda_version`, `rocm_ok` + ROCm version, `intel_gpu_ok`, `is_dgx_spark`, `python_version` — and returning a small record:

```
VllmSupport(status, method, reason, python_requirement)
  status:  "supported" | "experimental" | "unsupported"
  method:  "cuda_wheel" | "rocm_wheel" | "cpu_wheel" | "nightly_cu130" | "metal_plugin" | None
```

Every row of the matrix table above becomes at least one test case in `tests/test_vllm_install.py`. This is the piece worth getting right; the installer around it is glue.

**3. `install_vllm(support)` — dispatch, one branch per method.** *(done, minus the out-of-scope CPU/XPU branches)*
- Creates `vllm-env/` with the correct interpreter (must fail loudly, not silently downgrade, when the method requires 3.12 and no 3.12 is present).
- `cuda_wheel` → `uv pip install vllm --torch-backend=auto` (fall back to `pip` + the matching `download.pytorch.org` extra index when `uv` is absent).
- `rocm_wheel` → `--extra-index-url https://wheels.vllm.ai/rocm/`.
- `cpu_wheel` → plain wheel; record the `LD_PRELOAD` requirement for the engine to apply at spawn time.
- `nightly_cu130` (DGX Spark) → gated behind an explicit opt-in prompt, since it installs a nightly build.
- `metal_plugin` → run `vllm-metal`'s `install.sh`; it manages its own venv, so `vllm-env/` is not created on this path.
- `unsupported` → print the reason and the realistic alternative (WSL2 on Windows, Docker for XPU) and continue setup without vLLM. Never auto-install the third-party Windows fork.

**4. Setup plan / selection UI.** *(done)* vLLM is opt-in, not automatic — it is a multi-GB install on top of an existing llama.cpp install. Add it to both frontends:
- CLI (`setup_selection.py`): a yes/no in the prerequisites approval block, only offered when `status != "unsupported"`, with the status and download cost stated.
- GUI (`setup_gui.py`): matching checkbox, plus the plan record it returns.
- The "This will:" summary block gains a `• Install vLLM (…)` line.

**5. Model downloads.** *(done)* vLLM weights are whole repos, not single files.
- Extend the catalog (see below) and add a `snapshot_download`-based path alongside `hf_download` (`huggingface_hub` is already a dependency), with allow-patterns to skip `.bin` duplicates of `.safetensors`.
- Reuse `resumable_download.py`'s behavior where possible; whole-repo snapshots need their own completeness check.
- Store under `config.MODELS_DIR / "vllm" / <slug>`, matching `LlamaCppEngine._models_dir()`'s per-engine namespacing.
- `model_inventory.py` and `find_non_catalog_model_dirs` need to become engine-aware rather than assuming `models/llamacpp`.
- Model picker sizes and the memory ceiling must use the vLLM download sizes and the stricter VRAM ceiling when vLLM is selected.

## Engine plan

`scripts/runtime/engines/vllm.py`, registered in `engines/__init__.py`. Lifecycle-wise it is much closer to `LlamaCppEngine` than to a multi-model daemon: `vllm serve` hosts exactly one model, so the same spawn-per-`(tag, num_ctx, n_parallel)` pattern applies, including `ensure_running()` being a preflight and `available()` being `False` between models.

- **API**: OpenAI-compatible server on its own port (`8000`, kept distinct from llama-server's `8080`). `/v1/completions` for `generate`, `/v1/chat/completions` for `chat`/`chat_tools` (tools work natively), `/v1/embeddings` for `embed` — though embedding models need a separate `--task embed` server and may not be worth supporting in the first pass.
- **Token accounting**: request `stream_options={"include_usage": true}` so the final SSE chunk carries `usage.completion_tokens`, matching the llama.cpp engine's rule that SSE fragments are never counted as tokens.
- **Server-side timing**: vLLM does not report a per-request prompt duration the way llama-server's `timings` block does. `server_prompt_sec` should stay `None` (the interface already allows it) rather than being synthesized; alternatively scrape `/metrics`, which is a second pass at best.
- **`prepare_concurrency`**: `--max-num-seqs n_parallel` and `--max-model-len per_slot_ctx`; unlike llama-server's `-c`, `--max-model-len` is genuinely per-sequence, so the `n_parallel` multiplication `LlamaCppEngine` does must **not** be copied here.
- **`start(gpu_visible=False)`**: `VLLM_TARGET_DEVICE=cpu` / device selection at spawn, plus the CPU backend's `LD_PRELOAD`.
- **`is_connection_crash` / `tail_log`**: vLLM's engine-core crashes kill the whole server process, so detection is a dead-socket check plus a stderr tail — simpler than llama.cpp's.
- **`max_context_length`**: read `max_position_embeddings` from the downloaded repo's `config.json`.
- **`runtime_backend`**: report `cuda`/`rocm`/`xpu`/`metal`/`cpu` from the platform vLLM actually initialized (parse the startup banner, or query `/v1/models`-adjacent metadata), not from hardware detection.
- **`resume_artifact_paths` / `resume_runtime_paths`**: hash the snapshot's safetensors files and the `vllm` executable.

## Config, catalog, and CLI changes

- `scripts/runtime/config.py`: `VLLM_PORT = 8000`, `VLLM_URL`, `VLLM_VENV = SCRIPT_DIR / "vllm-env"`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_DTYPE`.
- `scripts/workloads/models.py`: per-tag `vllm_repo` (+ `vllm_download_size`, and per-backend quant variants if that is the direction chosen). Tags themselves stay unchanged — they are opaque catalog identifiers.
- `.gitignore`: `vllm-env/`.
- `--engine all` starts genuinely expanding to two passes. The filename suffixing and internal `"engine"` tagging already exist; what needs checking is that a partially-installed second engine produces a clean skip rather than a failed pass.
- Results should record the vLLM weight identity (repo + revision + quant) alongside the engine name, so a cross-engine comparison is auditable rather than misleading.

## Tests

- `tests/test_vllm_install.py` — `vllm_platform_support()` across every row of the matrix, plus the adversarial cases: aarch64 + CUDA that is *not* DGX Spark, ROCm 6.2 (below minimum), Python 3.13 with a ROCm GPU (wheels are 3.12-only), macOS x86_64, NVIDIA with compute capability 7.0, Intel XPU, Windows + NVIDIA.
- `tests/test_vllm_engine.py` — mirrors `test_llamacpp_engine.py`: HTTP mocked at the `requests`/`urllib` seam, real subprocess spawns excluded. Cover streamed-usage token accounting, `server_prompt_sec` staying `None`, `prepare_concurrency` argument construction (specifically that `per_slot_ctx` is *not* multiplied), crash detection, and `max_context_length` parsing from `config.json`.
- Orchestration coverage comes free — `tests/test_run_accuracy_benchmark.py` uses a fake engine and names no concrete class.
- Any model-inventory/download function that becomes engine-aware needs its existing tests extended for the `vllm` namespace.

## Docs to update

- [`engines.md`](engines.md) — the "only engine" framing in the intro is wrong the moment this lands; add a `VllmEngine` section and update "Selecting an engine".
- [`setup.md`](setup.md) — platform table, what setup installs, the vLLM support matrix, and the DGX Spark note.
- [`cli-reference.md`](cli-reference.md) — `--engine vllm`, plus any new setup flags.
- [`workloads.md`](workloads.md) — that `llamabench`/`llamabenchconc` skip under vLLM, and the cross-engine comparability caveat.
- [`limitations.md`](limitations.md) — quantization is not comparable across engines; Windows/XPU unsupported.
- [`project-structure.md`](project-structure.md), [`testing.md`](testing.md), [`platform-tuning.md`](platform-tuning.md), [`result-compatibility-v4.1.md`](result-compatibility-v4.1.md) if the results shape gains fields.
- Dashboard: results files gain an engine dimension that actually varies. `dashboard/src/constants.js` and `utils/*.js` need to key charts by engine and label the weight/quant difference, not silently overlay two different quantizations on one axis.

## Suggested order of work

1. `vllm_platform_support()` + its tests — no installs, no downloads, fully verifiable.
2. `config.py` entries, catalog `vllm_repo` fields, `models/vllm/` namespacing.
3. `VllmEngine` against a manually installed vLLM (user-driven), with mocked unit tests.
4. `install_vllm()` + setup detection and the CLI/GUI opt-in.
5. Snapshot downloads and model-inventory engine awareness.
6. Dashboard engine dimension.
7. Docs.

## Open questions for the user

1. ~~Which quantization for vLLM weights?~~ **Answered: 4-bit AWQ/GPTQ/W4A16**, to match `Q4_K_M`'s bit width. Every catalog entry now carries a verified `vllm_repo`. Two consequences to watch: AWQ/GPTQ Marlin kernels are CUDA-centric, so some of these will not run on ROCm or Metal, and `google/gemma-4-12B-it-qat-w4a16-ct` is a compressed-tensors QAT checkpoint rather than AWQ — the one entry that differs in kind from the rest.
2. ~~Which platforms are in scope?~~ **Answered: Linux CUDA + ROCm supported, DGX Spark and macOS Metal experimental**, Windows/XPU/CPU-only declined with a reason.
3. **Embeddings under vLLM** — the two embedding models now download their upstream fp16 repos, but serving them needs a separate `--task embed` server per model. Still open whether `VllmEngine` supports the embeddings workload in v1.
4. **Which models actually load under vLLM at these quantizations** is unverified — the repos exist and are ungated, but nothing has been served yet. Expect some to need `--max-model-len` capping or to fail on a given backend.

---

[← Engines](engines.md) · [Back to README](../README.md)
