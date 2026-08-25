[← Documentation index](../README.md) · [Qualification procedure](../telemetry-qualification.md)

# Memory Observer Screen: macOS M4 Pro at 1 Second

## Configuration

| Field | Value |
|---|---|
| Platform | Mac mini, Apple M4 Pro, 24 GB unified memory |
| OS | Darwin 25.6.0 |
| Engine | llama.cpp build 10375, Metal backend |
| Application | Local AI Bench 5.1.1 |
| Source commit | `915433815b611b952cc97b83d4532863af095a7f` |
| Workload | Single-shot LLM, Qwen 3.5 4B Q4, 2K case |
| Repetition | 2 warmups and 3 measured runs per invocation |
| Sampling | Memory telemetry off/on, 1.0-second interval |
| Design | 20 pairs alternating off-on and on-off, with 30-second gaps |

All 40 result files completed with three valid 2K measurements. The worktree identity was recorded as dirty because the local untracked qualification launcher `qual.sh` existed; no tracked source difference was observed across the series.

## Observer-effect result

| Metric | Median impact | Median bound | P90 impact | P90 bound | Result |
|---|---:|---:|---:|---:|---|
| TTFT | +0.98% | 2.00% | +3.57% | 4.00% | Pass |
| Throughput | −0.06% | 1.00% | +0.13% | 2.00% | Pass |
| Client wall time | −0.09% | 1.00% | +0.86% | 2.00% | Pass |

Positive impact means higher latency/wall time or lower throughput. The 1.0-second macOS `psutil` memory source passes the predeclared coarse observer-effect screen. This does not approve default-on telemetry or establish cross-method scientific comparability.

## Telemetry evidence and limitation

Host unified-memory usage and benchmark process-tree RSS used `psutil`; separate accelerator occupancy was correctly unsupported on Apple unified memory. Recorded source failures were zero. One telemetry-on result stored its 2K lifecycle windows out of chronological order because the sampler thread and a boundary capture could append concurrently. Commit `b67bb46` serializes capture and adds a concurrency regression test. The defect did not affect the benchmark TTFT, throughput, or wall measurements used by this observer screen, but this series is not final lifecycle-window ordering evidence.

The local raw evidence is retained under `results/qualification/mac-mini-m4-pro/1.0s/` and remains gitignored because benchmark results can carry local identity. Its manifest and machine-readable report are stored beside the 40 source results.
