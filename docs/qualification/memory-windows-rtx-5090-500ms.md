[← Documentation index](../README.md) · [Qualification procedure](../telemetry-qualification.md)

# Memory Observer Screen: Windows RTX 5090 at 500 Milliseconds

## Configuration

| Field | Value |
|---|---|
| Platform | Ryzen 7 9850X3D, GeForce RTX 5090 with 31.84 GB VRAM, 64 GB RAM |
| OS | Microsoft Windows 11 Home 10.0.26200, build 26200 |
| GPU driver | NVIDIA 610.62 |
| Engine | llama.cpp build 10362, CUDA backend |
| Application | Local AI Bench 5.1.1 |
| Source commit | `d3a6405937d9f47f8ccfa23831a5195f57220369` (clean) |
| Workload | Single-shot LLM, Qwen 3.5 4B Q4, 2K case |
| Repetition | 2 warmups and 3 measured runs per invocation |
| Sampling | Memory telemetry off/on, 0.5-second interval |
| Design | 20 pairs alternating off-on and on-off, with 30-second gaps |

All 40 result files completed with three valid 2K measurements. Recorded start times confirm the alternating physical order. Every telemetry-on case retained ordered idle, model-load, and measured windows with chronological samples.

## Observer-effect result

| Metric | Median impact | Median bound | P90 impact | P90 bound | Result |
|---|---:|---:|---:|---:|---|
| TTFT | +0.43% | 2.00% | +1.15% | 4.00% | Pass |
| Throughput | +0.30% | 1.00% | +1.49% | 2.00% | Pass |
| Client wall time | +0.23% | 1.00% | +0.96% | 2.00% | Pass |

Positive impact means higher latency/wall time or lower throughput. The 0.5-second native-Windows `psutil` plus `nvidia-smi` memory source passes the predeclared coarse observer-effect screen. This does not approve default-on telemetry or establish cross-method scientific comparability.

## Telemetry evidence

Host memory and benchmark process-tree RSS used `psutil`; accelerator occupancy and capacity used unprivileged `nvidia-smi` queries scoped to the visible GPU. All channel and sampler failure counts were zero. The measured 2K cases retained a median of 48 samples. Median peak accelerator use was 4.15 GB, median peak process RSS was 3.97 GB, and median accelerator headroom after the configured reserve was 26.69 GB.

The raw results, event journals, manifest inputs, and machine-readable analysis are retained in the local `rtx-5090-windows/0.5s/` qualification archive and remain outside Git because results can carry local identity.
