[← Back to README](../README.md)

# Benchmark limitations

Local AI Bench measures the exact recorded software, model artifact, configuration, and machine state; it does not establish universal performance for a model family or hardware product. Results support decisions only within their documented coverage and methodology.

## Representativeness

The included prompts, question banks, image workflows, context checkpoints, and concurrency shapes are controlled probes, not a complete simulation of every application. Accuracy is bank-specific and can be affected by training-data familiarity. Synthetic concurrency does not reproduce every production arrival pattern, scheduler, retrieval pipeline, tool latency, or user think time. Image results cover the recorded workflow, sampler, resolution, and checkpoint rather than all possible pipelines.

## Hardware and environment variance

Pausing can let hardware cool, background activity change, or operating-system caches age while a model remains loaded. Pause transitions are recorded in schema-4 results, but they do not make samples before and after a long pause equivalent to an uninterrupted run. Review that metadata before using a paused run for sensitive thermal or stability comparisons.

Thermal state, power mode, battery state, ambient temperature, background processes, memory pressure, storage activity, firmware, drivers, runtime build flags, and operating-system scheduling can change results. Unified-memory systems share capacity and bandwidth across CPU, GPU, and other clients. Discrete-GPU fit estimates require headroom beyond weight size for KV cache, runtime buffers, and concurrent slots. A model fitting once does not guarantee safe production capacity.

Independent runs should begin from comparable power, thermal, and background-load conditions. A single run is evidence, not a variance study; vendor claims should use repeated independent runs on each physical system and disclose dispersion and anomalies.

## Cross-result comparison

Do not compare client TTFT with server prompt time or cold single-shot TTFT with cached conversation TTFT. Do not combine results across changed prompts, banks, checkpoint definitions, retry rules, runtimes, quantizations, context settings, tuning profiles, or schema/methodology boundaries without an explicit compatibility determination. Missing and invalid data are not zero. A faster partial run is not automatically better than a slower complete run.

Native llama-bench numbers are useful community-comparable cross-checks but bypass the HTTP/client pipeline and therefore do not replace application-observed latency. CPU-only, force-all, custom model, and future platform-tuned runs must be identified and evaluated separately from the neutral default.

## Cross-engine comparison

Running the same model on two engines does not run the same file. llama.cpp uses this project's `Q4_K_M` GGUF; vLLM uses a separate 4-bit AWQ, GPTQ, or W4A16 checkpoint of the same base model — see [Workloads](workloads.md#per-engine-weights) for the exact mapping. They match on bit width and nothing stronger: `Q4_K_M` is a k-quant with per-block mixed precision, while AWQ and GPTQ are different algorithms with their own calibration data, and one catalog entry (Gemma 4 12B) is a quantization-aware-trained checkpoint rather than a post-training conversion.

A cross-engine chart therefore answers "how fast does each runtime serve this model, quantized the way that runtime does 4-bit" — not "how much faster is one runtime than the other on identical weights." Treat an accuracy difference between engines as a property of the weights at least as much as of the runtime, and do not attribute a throughput difference solely to the serving stack without accounting for the quantization and its memory footprint, which also differ.

The 4-bit weights are also not equally portable. AWQ and GPTQ Marlin kernels are CUDA-centric; on ROCm they are narrower, and on an untargeted gfx (such as `gfx1151`/Strix Halo) loading an AWQ checkpoint is known to be unreliable. A model that benchmarks on one backend may therefore be unavailable on another, which is a coverage difference rather than a performance result.

Neither the weights nor the engines have been validated against each other on real hardware yet. Nothing in this project currently runs vLLM.

## Recommendations

Benchmark evidence can identify measured fit, speed, latency, quality, and capacity tradeoffs; it cannot guarantee subjective satisfaction, future software compatibility, model licensing suitability, safety, or workload correctness. Recommendations must expose their evidence, missing coverage, uncertainty, and conflicts of interest. Pre-release hardware results may change before launch and require vendor-approved disclosure.
