[← Back to README](../README.md)

# Benchmark limitations

Local AI Bench measures the exact recorded software, model artifact, configuration, and machine state; it does not establish universal performance for a model family or hardware product. Results support decisions only within their documented coverage and methodology.

## Representativeness

The included prompts, question banks, image workflows, context checkpoints, and concurrency shapes are controlled probes, not a complete simulation of every application. Accuracy is bank-specific and can be affected by training-data familiarity. Synthetic concurrency does not reproduce every production arrival pattern, scheduler, retrieval pipeline, tool latency, or user think time. Image results cover the recorded workflow, sampler, resolution, and checkpoint rather than all possible pipelines.

## Hardware and environment variance

Pausing can let hardware cool, background activity change, or operating-system caches age while a model remains loaded. Pause transitions are recorded in schema-4 results, but they do not make samples before and after a long pause equivalent to an uninterrupted run. Review that metadata before using a paused run for sensitive thermal or stability comparisons.

Thermal state, power mode, battery state, ambient temperature, background processes, memory pressure, storage activity, firmware, drivers, runtime build flags, and operating-system scheduling can change results. Unified-memory systems share capacity and bandwidth across CPU, GPU, and other clients. Their model-headroom estimate uses benchmark process-tree RSS against RAM after the configured OS reserve, while total host use is retained separately to expose memory pressure and background activity; shared allocations not attributed to the process tree may therefore be understated. Discrete-GPU fit estimates use accelerator consumption and require headroom beyond weight size for KV cache, runtime buffers, and concurrent slots. A model fitting once does not guarantee safe production capacity.

A run made inside WSL2 reaches the GPU through a virtualization layer rather than a native driver, so it should not be treated as equivalent to a bare-metal Linux run even on identical hardware. WSL2 also caps guest RAM independently of the host, which changes both the memory-fit estimate and the memory pressure a model actually encounters. Such runs record `wsl: true` in the results profile and are tagged in the dashboard; compare them against other WSL2 runs, or read the difference as including the passthrough cost.

Independent runs should begin from comparable power, thermal, and background-load conditions. A single-run comparison now exposes available within-run dispersion and whether a delta clears a provisional practical threshold, but neither establishes reproducibility; vendor claims require compatible repeated independent trials and disclosure of dispersion and anomalies.

The opt-in sustained workload measures the thermal confounder directly as a throughput-retention timeline and records a nearby ambient reading when supplied. Its cause label is correlation, not proof of hardware throttling: temperature ceiling, power decline, memory pressure, background load, firmware control, and polling overhead can overlap. Temperature collection and the combined sampler have an observer effect and remain opt-in pending source-specific paired qualification. A paused soak is indeterminate because cooling breaks timeline continuity. See [Linux small-system qualification](qualification/sustained-linux-small-systems.md).

Apple Silicon temperature sampling uses private macOS HID services whose sensor names and availability may change across chips or operating-system releases. The source accepts only calibrated values from known CPU, GPU, and `PMU tdie*` sensor-name families, rejects zero and out-of-range values, and combines them into one SoC-package channel rather than implying separate CPU and GPU measurements on a unified chip. This source has real-hardware functional evidence on the M5 Pro but has not completed the repeated-trial observer-effect qualification required for a supported thermal-cause claim.

Opt-in power telemetry reports only its named scope. Accelerator, CPU-package, Apple processor-package estimates, and whole-system-at-the-wall measurements must not share an axis or support a broader energy claim. `powermetrics` explicitly describes its subsystem power as estimated and unsuitable for comparison between devices, so Apple readings support within-device optimization evidence until an independently qualified source establishes a broader claim. Windows AMD ADL uses ASIC power when available and otherwise board power, both normalized as accelerator scope but retained under the distinct `amd-adl` source; it has parser/ABI tests but no real-hardware qualification yet. Sampling has an observer effect and remains opt-in until repeated-trial qualification passes for that source, interval, and combined sampler. Idle baseline varies with battery state and background load and is displayed separately, never silently subtracted.

## Cross-result comparison

Do not compare client TTFT with server prompt time or cold single-shot TTFT with cached conversation TTFT. Do not combine results across changed prompts, banks, checkpoint definitions, retry rules, runtimes, quantizations, context settings, tuning profiles, or schema/methodology boundaries without an explicit compatibility determination. Missing and invalid data are not zero. A faster partial run is not automatically better than a slower complete run.

Native llama-bench numbers are useful community-comparable cross-checks but bypass the HTTP/client pipeline and therefore do not replace application-observed latency. CPU-only, force-all, custom model, and future platform-tuned runs must be identified and evaluated separately from the neutral default.

## Cross-engine comparison

Running the same model on two engines does not run the same file. llama.cpp uses this project's `Q4_K_M` GGUF; vLLM uses a separate 4-bit AWQ, GPTQ, or W4A16 checkpoint of the same base model — see [Workloads](workloads.md#per-engine-weights) for the exact mapping. They match on bit width and nothing stronger: `Q4_K_M` is a k-quant with per-block mixed precision, while AWQ and GPTQ are different algorithms with their own calibration data, and one catalog entry (Gemma 4 12B) is a quantization-aware-trained checkpoint rather than a post-training conversion.

A cross-engine chart therefore answers "how fast does each runtime serve this model, quantized the way that runtime does 4-bit" — not "how much faster is one runtime than the other on identical weights." Treat an accuracy difference between engines as a property of the weights at least as much as of the runtime, and do not attribute a throughput difference solely to the serving stack without accounting for the quantization and its memory footprint, which also differ.

The 4-bit weights are also not equally portable. AWQ and GPTQ Marlin kernels remain more mature on CUDA; a current ROCm wheel may include `gfx1151` while a particular quantized model or kernel path still fails. A model that benchmarks on one backend may therefore be unavailable on another, which is a coverage difference rather than a performance result.

Cross-engine qualification is exact to the recorded platform, architecture, runtime version, backend, suite version, and smallest-model workload coverage. Consult the generated [engine qualification matrix](engines.md#qualification-matrix) for current status; an absent or incomplete row is unverified, and vLLM requires an explicit experimental acknowledgment until its exact configuration completes qualification.

## Recommendations

Benchmark evidence can identify measured fit, speed, latency, quality, and capacity tradeoffs; it cannot guarantee subjective satisfaction, future software compatibility, model licensing suitability, safety, or workload correctness. Recommendations must expose their evidence, missing coverage, uncertainty, and conflicts of interest. Pre-release hardware results may change before launch and require vendor-approved disclosure.

The implemented evaluator works only from complete results with a recorded methodology profile. One survivor can be recommended after hard constraints; ordering multiple survivors requires compatible repeated trials with a qualified LLM throughput, latency, or accuracy verdict. Memory, efficiency, and image throughput may be hard constraints, but they cannot yet order multiple survivors because their repeated-trial metric direction and practical thresholds have not been predeclared. Missing data, incomplete runs, drift, incompatible methodology, too few trials, and unsupported objective comparisons produce insufficient evidence rather than a guessed rank. The artifact describes measurements on the recorded machine and case, not a universal model or purchase recommendation.
