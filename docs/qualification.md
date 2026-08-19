[← Back to README](../README.md)

# Platform qualification

Platform qualification is an ordinary Local AI Bench run with an explicit smallest-model selection. The normal result JSON, its journal, and its normal generated-image directory are the evidence; qualification does not create recipes, manifests, bundles, reports, copied artifacts, shadow runtimes, or a second execution state machine.

## Scope

Shared coverage includes single-shot and conversational generation, embeddings, all five accuracy banks, both server-concurrency shapes, and the shortened 120-second sustained workload. llama.cpp also runs `llamabench`, `llamabenchconc`, and Stable Diffusion 1.5 image generation; vLLM also runs `vllmbench`. Accuracy uses one deterministic question per bank, repeated workloads use one measured run without a warmup, and prompt sweeps stop at 2K. This proves that every shipped workload can complete on the selected platform; it is not full-catalog performance, full-bank accuracy, or production-duration soak evidence.

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

The launcher calls the normal `setup.sh` or `setup.bat` with a qualification preset that selects exactly one engine, the smallest LLM that supports every required engine workload, the smallest embedding model, and, for llama.cpp, the smallest image model. This is Gemma 3 1B for llama.cpp and Granite 4.1 3B for vLLM because vLLM requires a model-specific tool-call parser. It then calls the normal `run_bench.sh` or `run_bench.bat` with those explicit selections. Existing installations and downloads are reused by the same setup code used by every other user.

A platform-provided vLLM launcher is sufficient for fallback serving but not for the native `vllm bench` workload. During vLLM qualification, setup therefore installs the pinned native vLLM CLI when no `vllm` executable is available, preflights that environment, and uses it for both ordinary serving and the native benchmark. The platform launcher remains available only when a managed executable does not exist.

The default result is `qualification-evidence/TARGET/results_qualification_TARGET.json`; the target directory also receives the journal, local context, accuracy sidecars, and normal generated-image directory. `qualification-evidence/` is gitignored so an active or concurrent qualification run does not make the recorded source tree dirty. Pass a second argument to choose another result path. A zero exit means the ordinary result reports a complete run and every required workload contains complete measured evidence. Errors, timeouts, skipped required cases, missing sections, or incomplete requested counts fail qualification.

## Support records

After reviewing a passing result, add that result path to the platform entry in `scripts/release/qualification.py`. The entry records the exact platform, accelerator, backend, runtime version, suite version, and qualification date. Installer lifecycle, packaging, offline operation, telemetry observer effects, and long-duration thermal claims remain separate release gates with their own evidence; they are not recreated inside platform workload qualification.
