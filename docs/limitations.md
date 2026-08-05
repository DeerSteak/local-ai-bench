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

## Recommendations

Benchmark evidence can identify measured fit, speed, latency, quality, and capacity tradeoffs; it cannot guarantee subjective satisfaction, future software compatibility, model licensing suitability, safety, or workload correctness. Recommendations must expose their evidence, missing coverage, uncertainty, and conflicts of interest. Pre-release hardware results may change before launch and require vendor-approved disclosure.
