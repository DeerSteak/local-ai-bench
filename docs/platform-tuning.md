# Platform Tuning Profiles

Local AI Bench 4.1 ships one comparison profile, `neutral-v1`. It does not silently select a vendor-optimized profile. Platform-specific installation and crash workarounds are kept separate from measurement tuning, while every measurement-affecting runtime setting used by a selected workload is recorded in the immutable run plan, its execution identity, result bundles, and decision reports.

## Active neutral settings

| Setting | Scope | Rationale and compatibility bound | Verification |
|---|---|---|---|
| llama.cpp batch size `512` | Engine-backed workloads | Fixed neutral request-processing setting for all supported llama.cpp backends | `test_llamacpp_engine.py`, `test_methodology_profile.py` |
| llama.cpp KV cache `q8_0` with flash attention on | Engine-backed workloads | Fixed cache representation; flash attention is required by this quantized cache configuration | `test_llamacpp_engine.py`, `test_methodology_profile.py` |
| llama.cpp GPU layers `auto` or `0` in CPU-only mode | Engine-backed workloads | Lets llama.cpp fit supported acceleration memory without an undisclosed vendor layer count; CPU-only is explicit | `test_llamacpp_engine.py`, `test_methodology_profile.py` |
| Native llama.cpp GPU layers `999` or `0` in CPU-only mode | Native llama-bench workloads | Pins full-offload intent because llama-bench's default does not document equivalent semantics | `test_llamabench_benchmark.py`, `test_llamabench_concurrency_benchmark.py`, `test_methodology_profile.py` |
| ComfyUI dynamic VRAM disabled | Image workload | Compatibility workaround for unresolved combined-checkpoint streaming failures; applied uniformly rather than selected by vendor | `test_methodology_profile.py` and the documented runtime launch invariant |

## Compatibility behavior that is not a tuning profile

Setup selects runtime builds compatible with the detected operating system and accelerator: Metal on Apple Silicon, CUDA or Vulkan on Windows, and CUDA, ROCm, Vulkan, or CPU builds on Linux as available. Windows portable ComfyUI receives its required launch shape, and AMD/Windows may set `TRITON_INTERPRET=1` to avoid a known JIT failure. These choices make the runtime operable; they do not change scoring, prompts, sample inclusion, or thresholds, and they do not create a vendor-optimized result label.

## Power source permissions

| Platform source | Recorded scope | Permission requirement |
|---|---|---|
| macOS `powermetrics` | Processor package estimate covering CPU, GPU, and ANE, not wall power | Run `sudo -v` immediately before an opt-in benchmark; the sampler uses `sudo -n` and never prompts mid-run |
| NVIDIA `nvidia-smi` | Accelerator only, summed across readable devices | The benchmark user must be allowed to query `power.draw` |
| AMD Adrenalin ADL on Windows | Accelerator only, summed across readable adapters | The installed AMD driver must expose a readable ASIC or board-power counter; no administrator permission is requested |
| AMD `rocm-smi` on Linux | Accelerator only, summed across readable devices | The benchmark user must be allowed to query package power |
| Intel RAPL sysfs | CPU package only | The selected `/sys/class/powercap/intel-rapl*/energy_uj` counter must be readable |

Availability discovery never elevates permission, and a denied or unsupported source records an unavailable reason without failing the benchmark. Source scope is part of the methodology identity; package, accelerator, CPU-package, and whole-system measurements are not interchangeable.

The Windows AMD source loads the public ADL interface from the installed Adrenalin driver and prefers ASIC power, falling back to board power when necessary. It does not automate or scrape the Adrenalin control panel. The source remains opt-in and unqualified until repeated trials are recorded on representative Windows AMD hardware.

## Profile change rule

A future platform-tuned profile must have a distinct stable identifier, be explicitly selected before execution, list every effective optimization, and produce a different plan identity. It must ship with rationale, supported platform/runtime bounds, tests, report disclosure, and a methodology compatibility decision. Results from different profiles are not numerically compared without an explicit compatibility rule.

[← Methodology Contract](methodology-contract.md) · [Back to README](../README.md) · [Engines →](engines.md)
