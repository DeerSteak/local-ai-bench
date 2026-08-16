[← Documentation index](../README.md) · [Qualification procedure](../telemetry-qualification.md)

# Power Observer Screen: macOS M5 Pro

## Configuration

| Field | Value |
|---|---|
| Platform | MacBook Pro, Apple M5 Pro, 48 GB unified memory |
| OS | macOS 26.6.1 / Darwin 25.6.0, arm64 |
| Engine | llama.cpp build 10375, Metal backend |
| Application | Local AI Bench 6.0-pre4 |
| Source commit | `1ee44dd94ceac490dcf2d6d1a84f93cd166cb9ca` |
| Workload | Single-shot LLM, Gemma 3 1B Q4, 2K case |
| Repetition | 2 warmups and 3 measured runs per invocation |
| Sampling | Memory-only off side; combined memory plus `powermetrics` power on side |
| Scope | Processor package: combined CPU, GPU, and ANE power |
| Design | 20 pairs per interval, alternating off-on and on-off, with 30-second gaps |

All 120 invocations across the 0.25, 0.5, and 1.0-second screens completed from a clean worktree at the recorded source commit. All 60 qualifying power-on 2K cases recorded measured-case energy in joules, source and scope, separate idle/model-load/measured windows, and zero failed or null power samples. The preceding 0.5K case in each power-on invocation recorded two unavailable idle-startup reads before `powermetrics` emitted its first value; every 0.5K measured region was complete and recorded, and those startup reads were outside measured energy.

## Observer-effect results

Positive impact means higher latency or wall time, or lower throughput.

| Interval | Metric | Median impact | P90 impact | Bound, median / P90 | Result |
|---:|---|---:|---:|---:|---|
| 0.25 s | TTFT | −0.051% / −0.102 ms | +0.485% / +0.965 ms | 2% + 2 ms / 4% + 4 ms | Pass |
| 0.25 s | Throughput | −0.065% | +0.147% | 1% / 2% | Pass |
| 0.25 s | Client wall time | −0.118% | +0.124% | 1% / 2% | Pass |
| 0.5 s | TTFT | −0.089% / −0.178 ms | +0.441% / +0.876 ms | 2% + 2 ms / 4% + 4 ms | Pass |
| 0.5 s | Throughput | −0.205% | +0.059% | 1% / 2% | Pass |
| 0.5 s | Client wall time | −0.227% | +0.017% | 1% / 2% | Pass |
| 1.0 s | TTFT | −0.001% / −0.001 ms | +0.206% / +0.409 ms | 2% + 2 ms / 4% + 4 ms | Pass |
| 1.0 s | Throughput | +0.032% | +0.491% | 1% / 2% | Pass |
| 1.0 s | Client wall time | +0.075% | +0.441% | 1% / 2% | Pass |

Median task energy was 193.86 J at 0.25 seconds, 192.88 J at 0.5 seconds, and 192.14 J at 1.0 second. The corresponding median measured-window sample counts were 35, 19.5, and 10. All candidates pass the predeclared observer-effect screen. The shared 0.5-second default remains unchanged because it provides adequate resolution for this short task without changing the already-qualified memory methodology; these descriptive differences are not evidence that one interval measures energy more accurately.

This record qualifies the macOS `powermetrics` processor-package source and the combined memory-plus-power sampler on this M5 Pro configuration. Power telemetry remains opt-in, other power sources remain unverified on real hardware, and processor-package energy must not be compared as though it were GPU-only or wall-system energy. The local raw results, SQLite journals, manifests, and machine-readable reports are retained under the gitignored `results/qualification/power-m5-pro-pre4-20260815-224449/` directory because raw benchmark artifacts can contain local identity.
