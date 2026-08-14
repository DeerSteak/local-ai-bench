[← Documentation index](../README.md) · [Qualification procedure](../telemetry-qualification.md)

# Memory Observer Screen: macOS M4 Pro at 250 Milliseconds

## Configuration

| Field | Value |
|---|---|
| Platform | Mac mini, Apple M4 Pro, 24 GB unified memory |
| OS | Darwin 25.6.0 |
| Engine | llama.cpp build 10375, Metal backend |
| Application | Local AI Bench 5.1.1 |
| Source commit | `dde9fa6043344714623371466252219e9a08fab8` |
| Workload | Single-shot LLM, Qwen 3.5 4B Q4, 2K case |
| Repetition | 2 warmups and 3 measured runs per invocation |
| Sampling | Memory telemetry off/on, 0.25-second interval |
| Design | 20 pairs alternating off-on and on-off, with 30-second gaps |

All 40 result files completed with three valid 2K measurements. The worktree identity was recorded as dirty because the local untracked qualification launcher `qual.sh` existed; no tracked source difference was observed across the series. All telemetry-on cases recorded zero source failures, chronological samples, and ordered idle, model-load, and measured windows.

## Observer-effect result

| Metric | Median impact | Median bound | P90 impact | P90 bound | Result |
|---|---:|---:|---:|---:|---|
| TTFT | −0.37% | 2.00% | +2.56% | 4.00% | Pass |
| Throughput | −0.29% | 1.00% | −0.06% | 2.00% | Pass |
| Client wall time | −0.21% | 1.00% | +0.96% | 2.00% | Pass |

Positive impact means higher latency/wall time or lower throughput. The 0.25-second macOS `psutil` memory source passes the predeclared coarse observer-effect screen. This does not approve default-on telemetry or establish cross-method scientific comparability.

Host unified-memory usage and benchmark process-tree RSS used `psutil`; separate accelerator occupancy was unsupported. The local raw evidence, manifest, and machine-readable report are retained under the gitignored `results/qualification/mac-mini-m4-pro/0.25s/` directory because benchmark results can carry local identity.

## Interval decision

All three predeclared intervals passed. The measured 2K window retained a median of 42 samples at 1 second, 82 at 0.5 seconds, and 157 at 0.25 seconds. Median telemetry-on result size grew from about 60 KB to 94 KB and 160 KB respectively. Moving from 0.5 to 0.25 seconds found about 75 MB more median host-memory peak and 7 MB more median process-RSS peak in this workload while nearly doubling samples and retained JSON size. Version 6 therefore selects 0.5 seconds as the practical memory-sampling default; 0.25 seconds remains an explicitly configurable qualified cadence rather than the default.
