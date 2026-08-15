[← Documentation index](../README.md) · [Qualification procedure](../telemetry-qualification.md)

# Memory Re-qualification: WSL RTX 5090 at 500 Milliseconds

## Configuration

| Field | llama.cpp series | vLLM series |
|---|---|---|
| Platform | AMD Ryzen 7 9850X3D, GeForce RTX 5090 with 31.84 GB VRAM, 51 GB available to WSL | Same |
| OS | WSL2, Linux 6.18.33.2-microsoft-standard-WSL2 | Same |
| Engine | llama.cpp build 10362, CUDA backend | vLLM 0.27.1, CUDA backend |
| Application | Local AI Bench 6.0-pre3 | Same |
| Source commit | `f6175fe8ac2c8c8a13ddf287177b82cb07fe121f` (clean) | Same |
| Workload | Single-shot LLM, Gemma 3 1B, fixed 1,779-token 2K prompt | Same |
| Repetition | 2 warmups and 3 measured runs per invocation | Same |
| Sampling | Memory telemetry off/on, 0.5-second interval | Same |
| Design | 20 pairs alternating off-on and on-off, with 30-second gaps | Same |

All 80 result files completed with three valid measurements and identical prompt-token counts. Every telemetry-on case reported zero sampler and channel failures. Result source identities were clean and consistent within both series. The host driver version was not retained in the result files, so the local raw archive remains the source for that environmental detail.

## Observer-effect result

| Engine | Metric | Median impact | Median bound | P90 impact | P90 bound | Result |
|---|---|---:|---:|---:|---:|---|
| llama.cpp | TTFT | −0.13% / −0.10 ms | 2.00% / 2 ms | +4.07% / +2.92 ms | 4.00% / 4 ms | Pass |
| llama.cpp | Throughput | +0.40% | 1.00% | +1.22% | 2.00% | Pass |
| llama.cpp | Client wall time | +0.42% | 1.00% | +1.36% | 2.00% | Pass |
| vLLM | TTFT | +0.68% / +0.19 ms | 2.00% / 2 ms | +6.11% / +1.76 ms | 4.00% / 4 ms | Pass |
| vLLM | Throughput | +0.66% | 1.00% | +0.98% | 2.00% | Pass |
| vLLM | Client wall time | +0.56% | 1.00% | +0.83% | 2.00% | Pass |

Positive impact means higher latency/wall time or lower throughput. TTFT fails only when both its relative and duration bounds are exceeded, preventing sub-millisecond changes in these fast responses from becoming false failures. The `psutil` plus `nvidia-smi` memory sampler passes at the selected 0.5-second interval through both supported LLM engines on this WSL discrete-NVIDIA platform.

## Telemetry and noise evidence

Host memory and benchmark process-tree RSS used `psutil`; accelerator occupancy and capacity used unprivileged `nvidia-smi` queries scoped to the visible GPU. Telemetry-on cases retained a median of 18 samples with llama.cpp and 67 with vLLM. Median peak accelerator use was 2.43 GB and 30.72 GB respectively; median peak process RSS was 0.90 GB and 4.70 GB.

Ten non-overlapping telemetry-off pairs per engine supplied an independent-invocation noise estimate. The largest 95th-percentile absolute relative change across the engines was 7.15% for TTFT, 0.87% for throughput, and 0.89% for client wall time. Applying the precommitted larger-of-noise-or-product-floor rule and rounding upward selects practical thresholds of 8% for TTFT, 3% for throughput, and 3% for wall time.

## Measured-runs decision

The telemetry-off series also compared each invocation's first measured request with its mean of three. Averaging three reduced between-invocation coefficient of variation by 19% for TTFT, 10% for throughput, and 18% for wall time on llama.cpp, and by 53%, 28%, and 28% on vLLM. A one-request invocation would still pay model startup and two warmups: three such independent invocations are estimated to cost 2.57 times one three-request llama.cpp invocation and 2.89 times one vLLM invocation. The default therefore remains three measured requests for efficient within-run dispersion, while reproducibility verdicts continue to require separate independent invocations.

The raw results, event journals, manifests, and machine-readable reports are retained in the local `m3-precision-llamacpp-500ms/` and `m3-precision-vllm-500ms/` qualification archives and remain outside Git because results can carry local identity.
