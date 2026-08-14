[← Version 6 Plan](../VERSION_6_PLAN.md) · [Back to README](../README.md) · [Methodology Contract →](methodology-contract.md)

# Version 6 Foundation

This document freezes the shared definitions and predeclared telemetry screening rules used by Version 6. [Tracking issue #19](https://github.com/DeerSteak/local-ai-bench/issues/19) reports status and links to the authoritative checklists in `VERSION_6_PLAN.md`; it does not duplicate them.

## Schema 5 field map

Schema 5 is introduced by the first writer that adds memory telemetry. Every field below is optional on read, and absence in schema 4 or earlier means not recorded rather than zero.

| Field | Unit | Scope | Source | Availability | Owning event | JSON location | Old-file fallback |
|---|---|---|---|---|---|---|---|
| Window label | enum | Sample | Lifecycle seam | Recorded | Case telemetry | Case `memory.windows[].name` | Not recorded |
| Sample timestamp | Monotonic seconds from sampler start | Sample | Monotonic clock | Recorded | Case telemetry | Case `memory.windows[].samples[].timestamp_sec` | Not recorded |
| Host RAM used | GB | Host | `psutil` | Measured, unknown, or unsupported | Case telemetry | Sample `host_ram_used_gb` | Not recorded |
| Process RSS | GB | Benchmark process tree | `psutil` | Measured, unknown, or unsupported | Case telemetry | Sample `process_rss_gb` | Not recorded |
| Accelerator memory used | GB | Named accelerator scope | Platform source | Measured, unknown, or unsupported | Case telemetry | Sample `accelerator_memory_used_gb` | Not recorded |
| Accelerator memory total | GB | Named accelerator scope | Same as used channel | Measured, unknown, or unsupported | Case telemetry | Sample `accelerator_memory_total_gb` | Not recorded |
| Window peak, mean, and final | GB | Channel within lifecycle window | Derived from valid samples | Recorded or unknown | Case telemetry | Case `memory.windows[].summary.<channel>` | Not recorded |
| Window sample count | Count | Lifecycle window | Sampler | Recorded | Case telemetry | Case `memory.windows[].summary.sample_count` | Not recorded |
| Window duration | Seconds | Lifecycle window | Monotonic clock | Recorded | Case telemetry | Case `memory.windows[].summary.duration_sec` | Not recorded |
| Case peak, mean, and final | GB | Channel across retained sub-windows | Derived from window summaries | Recorded or unknown | Case telemetry | Case `memory.summary.<channel>` | Not recorded |
| Absolute headroom | GB | Case model footprint against memory ceiling | Derived | Recorded or unknown | Case telemetry | Case `memory.headroom.absolute_gb` | Not recorded |
| Fractional headroom | Fraction | Case model footprint against memory ceiling | Derived | Recorded or unknown | Case telemetry | Case `memory.headroom.fraction` | Not recorded |
| Headroom state | Enum | Case | Derived thresholds | Comfortable, tight, exceeded, or unknown | Case telemetry | Case `memory.headroom.state` | Not recorded |
| Headroom basis | Channel name | Case | Derived from memory architecture | Accelerator use or process-tree RSS | Case telemetry | Case `memory.headroom.basis_channel` | Not recorded |
| Sampling interval | Seconds | Case | Sampler configuration | Recorded | Case telemetry | Case `memory.provenance.interval_sec` | Not recorded |
| Channel source | Identifier | Channel | Sampler discovery | Recorded or unavailable | Case telemetry | Case `memory.provenance.channels.<channel>.source` | Not recorded |
| Failed samples | Count | Channel | Sampler | Recorded | Case telemetry | Case `memory.provenance.channels.<channel>.failed_samples` | Not recorded |
| Run peak | GB | Run and channel | Derived from cases | Recorded or unknown | Run finalization | `run.memory_summary.channels.<channel>.peak_gb` | Not recorded |
| Tightest headroom | GB and fraction | Run | Derived from cases | Recorded or unknown | Run finalization | `run.memory_summary.tightest_headroom` | Not recorded |
| Tightest-headroom case | Stable case identity | Run | Case event identity | Recorded or unknown | Run finalization | `run.memory_summary.tightest_headroom.case_id` | Not recorded |

Raw sensor output, process arguments, paths, serial numbers, UUIDs, and host identity are excluded. Schema 5 changes the result envelope only; event, plan, policy, project, journal, and bundle schemas change independently when their own serialized forms change.

## Telemetry vocabulary

- **Sample:** one timestamped sampler observation containing an explicit availability outcome for every configured channel.
- **Channel:** one normalized measured quantity with a single unit and scope, such as process RSS or accelerator memory used.
- **Source:** the allowlisted mechanism that produced a channel, such as `psutil` or `nvidia-smi`; it is not raw command output.
- **Scope:** the physical or process boundary represented by a channel. Differing scopes are not interchangeable even when their units match.
- **Measured window:** a named interval whose work contributes to a benchmark case measurement.
- **Idle window:** a named interval before model load used only as baseline evidence; it is never charged to request efficiency.
- **Unknown:** the channel should be measurable in this configuration, but no trustworthy value was obtained.
- **Unsupported:** no implemented and qualified source can provide the channel for this platform or configuration.

Model load is a third explicit lifecycle window. A case may retain workload-defined measured sub-windows, and its summary is derived from those retained summaries rather than from a discarded run-wide peak.

## Comparison vocabulary

- **Within-case variation:** dispersion among repeated measurements of one case inside one loaded run.
- **Within-run variation:** dispersion among comparable case observations during one invocation; it may include ordering and thermal effects.
- **Between-trial variation:** dispersion among independent compatible invocations and the basis for a reproducibility verdict.
- **Practical threshold:** a predeclared minimum change large enough to matter; it is not an uncertainty estimate.
- **Uncertainty interval:** an interval computed from qualified independent trials under a documented method and minimum count.
- **Paired trial:** two compatible alternatives measured with the same case sequence and paired by trial conditions.
- **Inconclusive:** the available compatible evidence cannot support improved, regressed, or unchanged; it is not a pass or a zero effect.

## Foundation observer-effect screen

The coarse screen evaluates sampling intervals of 0.25, 0.5, and 1.0 seconds using at least 20 alternating telemetry-off/on pairs per platform. It reports descriptive distributions for TTFT, throughput, and measured-case wall time. A source fails the coarse screen if median TTFT impact exceeds 2%, median throughput impact exceeds 1%, median wall-time impact exceeds 1%, or the 90th-percentile impact for any metric exceeds twice its median bound.

Passing this screen cannot approve default-on telemetry or establish scientific comparability. After milestone 3, qualified independent trials must evaluate each source, interval, and combined sampler under its own methodology identity.

The Mac mini M4 Pro and native-Windows RTX 5090 coarse screens passed at all three candidate intervals. Version 6 selects 0.5 seconds as the practical memory-only default: it roughly doubles the sample density of 1 second, while 0.25 seconds nearly doubles retained samples and result size again for a small additional peak-capture benefit in the screened workloads. The detailed records are indexed under `docs/qualification/`.

## Practical-threshold derivation

Use qualified telemetry-off repeated trials and calculate the between-trial paired relative-change distribution. A metric threshold is the larger of its 95th-percentile absolute noise and its product-relevance floor, rounded upward to a whole percentage; the provisional floors are 5% for TTFT, 3% for throughput, and 3% for wall time. Evidence below the interval method's eventual minimum trial count remains inconclusive.

## Minimum real-hardware qualification set

| Platform class | Candidate hardware | Required memory channels | Candidate sources |
|---|---|---|---|
| Discrete NVIDIA GPU on Windows | Ryzen 7 9850X3D, RTX 5090, 64 GB | Host RAM, process RSS, GPU used and total | `psutil`, `nvidia-smi` |
| Discrete multi-GPU NVIDIA on Windows | Core Ultra 7 270K, two RTX 5060 Ti, 64 GB | Host RAM, process RSS, per-device GPU used and total | `psutil`, `nvidia-smi` |
| Discrete NVIDIA GPU under WSL | Either Windows NVIDIA system under WSL | Host RAM, process RSS, visible GPU used and total | `psutil`, `nvidia-smi` |
| Discrete AMD GPU on Windows | Ryzen 7 5800XT, Radeon RX 9060 XT 16 GB, 32 GB | Host RAM, process RSS, GPU used and total where readable | `psutil`; native accelerator source not yet implemented |
| Discrete AMD GPU under WSL | Same Radeon system if WSL is installed and the GPU/runtime is exposed | Host RAM, process RSS, visible GPU used and total where readable | `psutil`, qualified AMD source |
| Coherent unified NVIDIA | DGX Spark, 128 GB | Host/process and applicable accelerator or unified-memory channel | `psutil`, qualified NVIDIA source |
| Unified AMD | Ryzen AI Max+ 395, 128 GB | Host RAM, process RSS, applicable unified-memory channel | `psutil`, qualified AMD source |
| Unified Apple | Mac mini M4 Pro, 24 GB; MacBook Pro M5 Pro, 48 GB | Unified host RAM pool and process RSS; separate accelerator occupancy is unsupported | `psutil` |

Parser fixtures prove normalization and redaction only. Qualification also records permissions, timing, process ownership, sensor scope, OS and source versions, sampling interval, failure counts, and the observer-effect result. The minimum Version 6 claim requires at least one qualified discrete-GPU platform and one qualified unified-memory platform; broader rows remain unverified until their own evidence exists.

## Baseline

The pre-implementation baseline on 2026-08-12 passed 2,411 Python tests and 373 dashboard tests. Dashboard lint and strict TypeScript checks also passed. Local command logs are retained as development artifacts and are not qualification evidence.
