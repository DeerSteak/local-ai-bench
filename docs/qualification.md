[← Back to README](../README.md)

# Platform qualification

Platform qualification is an ordinary Local AI Bench run with an explicit smallest-model selection. The normal result JSON, its journal, and its normal generated-image directory are the evidence; qualification does not create recipes, manifests, bundles, reports, copied artifacts, shadow runtimes, or a second execution state machine.

## Scope

Shared engine coverage includes single-shot and conversational generation, embeddings, all five accuracy banks, both server-concurrency shapes, and the shortened 120-second sustained workload. llama.cpp also runs `llamabench` and `llamabenchconc`; vLLM also runs `vllmbench`. The same llama.cpp qualification invocation runs Stable Diffusion 1.5 through ComfyUI, but image support is graded and published separately from llama.cpp support. Accuracy uses one deterministic question per bank, repeated workloads use one measured run without a warmup, and prompt sweeps stop at 2K. This proves that each recorded component's shipped workloads can complete on the selected platform; it is not full-catalog performance, full-bank accuracy, or production-duration soak evidence.

## Run

List the explicit platform targets:

```bash
./run_qualification.sh --list-targets
```

Run one target on macOS, Linux, or WSL2:

```bash
./run_qualification.sh dgx-spark-vllm-cuda
```

On Windows:

```text
run_qualification.bat geforce-windows-llamacpp-cuda
```

The launcher calls the normal `setup.sh` or `setup.bat` with a qualification preset that selects exactly one engine, the smallest LLM that supports every required engine workload, the smallest embedding model, and, for llama.cpp, the smallest image model. This is Gemma 3 1B for llama.cpp and Granite 4.1 3B for vLLM because vLLM requires a model-specific tool-call parser. It then verifies the requested platform, architecture, accelerator, and non-Vulkan backend against the shared execution profile before calling the normal `run_bench.sh` or `run_bench.bat` with those explicit selections. A missing ROCm or CUDA runtime therefore stops before an accidental CPU run. Existing installations and downloads are reused by the same setup code used by every other user.

The Radeon WSL2 llama.cpp target installs AMD's pinned ROCm 7.2 WSL stack when `rocminfo` is unavailable, then requires `rocminfo` to see the GPU before setup can continue. This automated path supports AMD's qualified Ubuntu 22.04 and 24.04 WSL distributions and requires the compatible AMD Software: Adrenalin Edition 26.1.1 for WSL2 Windows host driver; another distribution or missing host driver stops with an actionable error instead of falling back to CPU. Radeon WSL2 vLLM is not a qualification target because the shipped wheel cannot discover the GPU through AMD SMI under WSL2. Native Linux Radeon targets continue to use the distribution's normal ROCm installation rather than the WSL package.

Intel Arc Windows qualification is not supported while Windows Smart App Control's enforced `VerifiedAndReputableDesktop` policy is active. Real Arc Pro B65 attempts produced Code Integrity event 3077 for the official llama.cpp SYCL, Vulkan, and CPU DLLs under policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. Setup detects that active policy before runtime or model downloads and directs qualification to native Linux instead; it does not weaken or bypass the host security policy.

A platform-provided vLLM launcher is sufficient for fallback serving but not for the native `vllm bench` workload. During vLLM qualification, setup therefore installs the pinned native vLLM CLI when no `vllm` executable is available, preflights that environment, and uses it for both ordinary serving and the native benchmark. The platform launcher remains available only when a managed executable does not exist.

The default result is `qualification-evidence/TARGET/results_qualification_TARGET.json`; the target directory also receives the journal, local context, accuracy sidecars, and normal generated-image directory. `qualification-evidence/` is gitignored, and raw qualification outputs are never tracked; reviewed support records retain their evidence references without committing generated results. Pass a second argument to choose another result path. A zero exit means the ordinary result identifies the engine runtime, reports a complete run, and every required workload contains complete measured evidence. Errors, timeouts, skipped required cases, missing runtime identity or sections, and incomplete requested counts fail qualification. An expected workload-stage failure exits without a Python traceback, preserves partial evidence, and reports whether the engine workloads passed independently of a failed ComfyUI image stage.

## Support records

After reviewing a result, add that result path and only its completed workload/model coverage to the platform entry in `scripts/release/qualification.py`. The entry records the exact platform, accelerator, backend, runtime version, suite version, and qualification date. Runtime support is derived from the engine workloads and LLM/embedding models; ComfyUI image support is derived independently from `img` and the qualification image model. A single invocation may therefore publish llama.cpp as supported while leaving ComfyUI unverified. Installer lifecycle, packaging, offline operation, telemetry observer effects, and long-duration thermal claims remain separate release gates with their own evidence; they are not recreated inside platform workload qualification.
