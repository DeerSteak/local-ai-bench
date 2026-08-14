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
