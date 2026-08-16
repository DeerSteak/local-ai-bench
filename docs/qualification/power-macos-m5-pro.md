[← Documentation index](../README.md) · [Qualification procedure](../telemetry-qualification.md)

# Power Observer Screen: macOS M5 Pro

## Configuration

| Field | Value |
|---|---|
| Platform | MacBook Pro, Apple M5 Pro, 48 GB unified memory |
| OS | macOS 26.6.1 / Darwin 25.6.0, arm64 |
| Engine | llama.cpp build 10375, Metal backend |
| Application | Local AI Bench 6.0-pre3 |
| Source commit | `d5e9b945b0249253b528cce0a2260c3c6286db9a` |
| Workload | Single-shot LLM, Gemma 3 1B Q4, 2K case |
| Repetition | 2 warmups and 3 measured runs per invocation |
| Sampling | Memory-only off side; combined memory plus `powermetrics` power on side |
| Scope | Processor package: combined CPU, GPU, and ANE power |
| Design | 20 pairs per interval, alternating off-on and on-off, with 30-second gaps |

All 120 invocations across the 0.25, 0.5, and 1.0-second screens completed from a clean worktree. Every power-on case recorded measured-case energy in joules, source and scope, separate idle/model-load/measured windows, and zero failed power samples. The first five-pair smoke exposed that a new-session supervised worker lost macOS's terminal-scoped sudo authorization; commit `d5e9b945b0249253b528cce0a2260c3c6286db9a` preserves a distinct cancellable process group while retaining the controlling terminal. A subsequent one-pair check recorded 192.87 J and confirmed the fix before qualification began.

## Observer-effect results

Positive impact means higher latency or wall time, or lower throughput.

| Interval | Metric | Median impact | P90 impact | Bound, median / P90 | Result |
|---:|---|---:|---:|---:|---|
| 0.25 s | TTFT | −0.049% / −0.097 ms | +0.248% / +0.494 ms | 2% + 2 ms / 4% + 4 ms | Pass |
| 0.25 s | Throughput | +0.046% | +0.693% | 1% / 2% | Pass |
| 0.25 s | Client wall time | +0.024% | +0.677% | 1% / 2% | Pass |
| 0.5 s | TTFT | +0.034% / +0.069 ms | +0.930% / +1.864 ms | 2% + 2 ms / 4% + 4 ms | Pass |
| 0.5 s | Throughput | +0.284% | +1.564% | 1% / 2% | Pass |
| 0.5 s | Client wall time | +0.328% | +1.755% | 1% / 2% | Pass |
| 1.0 s | TTFT | +0.070% / +0.139 ms | +0.296% / +0.588 ms | 2% + 2 ms / 4% + 4 ms | Pass |
| 1.0 s | Throughput | +0.226% | +0.425% | 1% / 2% | Pass |
| 1.0 s | Client wall time | +0.182% | +0.414% | 1% / 2% | Pass |

Median task energy was 194.29 J at 0.25 seconds, 195.47 J at 0.5 seconds, and 192.28 J at 1.0 second. The corresponding median measured-window sample counts were 36, 20, and 10. All candidates pass the predeclared observer-effect screen. The shared 0.5-second default remains unchanged because it provides adequate resolution for this short task without changing the already-qualified memory methodology; these descriptive differences are not evidence that one interval measures energy more accurately.

This record qualifies the macOS `powermetrics` processor-package source and the combined memory-plus-power sampler on this M5 Pro configuration. Power telemetry remains opt-in, other power sources remain unverified on real hardware, and processor-package energy must not be compared as though it were GPU-only or wall-system energy. The local raw results, SQLite journals, manifests, and machine-readable reports are retained under the gitignored `results/qualification/power-m5-pro-{0.25,0.5,1.0}/` directories because raw benchmark artifacts can contain local identity.
