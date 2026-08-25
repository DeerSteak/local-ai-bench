[← Documentation index](../README.md) · [Qualification procedure](../telemetry-qualification.md)

# Memory Observer Screen: macOS M4 Pro at 500 Milliseconds

## Configuration

| Field | Value |
|---|---|
| Platform | Mac mini, Apple M4 Pro, 24 GB unified memory |
| OS | Darwin 25.6.0 |
| Engine | llama.cpp build 10375, Metal backend |
| Application | Local AI Bench 5.1.1 |
| Source commit | `b67bb46865c73b28a260961756bd0cb04bdc5f02` |
| Workload | Single-shot LLM, Qwen 3.5 4B Q4, 2K case |
| Repetition | 2 warmups and 3 measured runs per invocation |
| Sampling | Memory telemetry off/on, 0.5-second interval |
| Design | 20 pairs alternating off-on and on-off, with 30-second gaps |

All 40 result files completed with three valid 2K measurements. The worktree identity was recorded as dirty because the local untracked qualification launcher `qual.sh` existed; no tracked source difference was observed across the series. All telemetry-on cases recorded zero source failures, chronological samples, and ordered idle, model-load, and measured windows.

## Observer-effect result

| Metric | Median impact | Median bound | P90 impact | P90 bound | Result |
|---|---:|---:|---:|---:|---|
| TTFT | −0.75% | 2.00% | +3.51% | 4.00% | Pass |
| Throughput | −0.12% | 1.00% | +0.04% | 2.00% | Pass |
| Client wall time | −0.46% | 1.00% | +0.94% | 2.00% | Pass |

Positive impact means higher latency/wall time or lower throughput. The 0.5-second macOS `psutil` memory source passes the predeclared coarse observer-effect screen. This does not approve default-on telemetry or establish cross-method scientific comparability.

Host unified-memory usage and benchmark process-tree RSS used `psutil`; separate accelerator occupancy was unsupported. The local raw evidence, manifest, and machine-readable report are retained under the gitignored `results/qualification/mac-mini-m4-pro/0.5s/` directory because benchmark results can carry local identity.

## Milestone 3 re-qualification

A fixed-prompt re-qualification on Local AI Bench 6.0-pre3 at clean commit `2f5463e15d87531e0482f8c1ebf33b0a5b0d9223` repeated the 20 alternating pairs with llama.cpp build 10375. All 40 invocations completed with three valid measurements of the same 1,779-token prompt and zero telemetry failures. Median TTFT impact was −0.07% (−0.40 ms) and p90 was +0.02% (+0.13 ms); throughput was +0.04% median and +0.53% p90; client wall time was +0.01% median and +0.43% p90. Every metric passed.

Ten non-overlapping telemetry-off pairs produced 95th-percentile absolute noise of 0.28% TTFT, 2.94% throughput, and 2.58% wall time, all within the defaults derived from the RTX 5090 evidence. Three-request averaging reduced between-invocation TTFT dispersion by 32%, and three separate one-request invocations would cost approximately 2.40 times one three-request invocation. The local raw evidence is retained under `results/qualification/m3-precision-m4-pro-llamacpp-500ms/`.
