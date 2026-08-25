# Sustained-load qualification on Linux small systems

This procedure captures the real-hardware evidence needed to qualify Version 6.0-pre5 temperature sources and the combined memory, power, temperature, and throughput sampler. The stock, open-air DGX Spark and AMD Ryzen AI Halo Developer Platform completed the procedure on August 17, 2026; the reviewed evidence and exact qualification scope are recorded below.

## Qualified evidence

Both systems ran from clean commit `06cf7757c4bc2ecbf9697bed17e7de31720c0161` with Gemma 3 1B Q4_K_M, a recorded ambient temperature of 20 °C, factory-default system controls, and llama.cpp. Each platform completed three post-fix ten-minute soaks plus 20 alternating telemetry-off/on pairs for both the latency-sensitive and sustained screens at 0.25, 0.5, and 1.0 seconds. All twelve observer-effect reports passed their predeclared bounds, all six long soaks completed with only valid requests and 61 aligned windows, and the current analyzer exactly reproduces every stored classification.

| Platform | Runtime and sources | Post-fix ten-minute results | Evidence |
|---|---|---|---|
| AMD Ryzen AI Halo Developer Platform, Linux 6.18.35+rex+2-amd64 | llama.cpp 9413; CPU package via `hwmon`, GPU die and accelerator power via `rocm-smi` | Mild degradation in all three runs: 91.04%, 94.80%, and 94.04% retention, with onset at 400, 530, and 310 seconds; cause classified power-correlated | [`sustained-linux-20260817-212722`](../../results/qualification/sustained-linux-20260817-212722), [`temperature-linux-20260817-092852`](../../results/qualification/temperature-linux-20260817-092852) |
| NVIDIA DGX Spark, Linux 6.17.0-1029-nvidia-aarch64 | llama.cpp 10362; GPU die and accelerator power via `nvidia-smi` | Stable in all three runs: 99.16%, 99.05%, and 99.13% retention, with no throttle onset and cause classified neither | [`sustained-linux-20260817-212933`](../../results/qualification/sustained-linux-20260817-212933), [`temperature-linux-20260817-092911`](../../results/qualification/temperature-linux-20260817-092911) |

Supported temperature and power channels recorded zero failed samples. Unsupported CPU-package and GPU-hotspot channels on DGX Spark and the unsupported GPU-hotspot channel on Ryzen AI Halo remain explicitly unavailable with zero failures. DGX Spark's host-memory channel remained valid, but `nvidia-smi` did not return accelerator-memory used or total values; those channels remain an item 1 unified-memory coverage limitation and are not qualified by this temperature evidence.

## Preconditions

- Place both systems in their normal open-air positions in the same basement room, with system power and fan controls at factory defaults.
- Keep vents unobstructed, stop unrelated compute work, and record the room ambient temperature near the systems immediately before each invocation.
- Use the same installed xsmall or small model on every repeated run for a platform. Do not compare different quantizations or engines.
- Start from a clean worktree at the exact commit being qualified. The wrapper refuses a dirty real run.
- Inspect temperature availability before interpreting cause. DGX Spark should normally expose NVIDIA GPU die temperature through `nvidia-smi`; Ryzen AI Halo may expose CPU package temperature through Linux `hwmon` and GPU die/hotspot through `rocm-smi`. Missing channels remain unavailable and do not invalidate throughput retention.

## Preview and run

Replace the model tag and ambient reading. Previewing is safe and launches no benchmark:

```bash
bash run_sustained_qualification_linux.sh --model MODEL_TAG --ambient-temp-c 20.5 --dry-run
bash run_sustained_qualification_linux.sh --model MODEL_TAG --ambient-temp-c 20.5
```

Run three ten-minute soaks per platform, invoking the default one-run wrapper separately so each command receives the ambient temperature measured immediately beforehand. Wait at least five minutes and confirm the system has returned to a comparable idle temperature before the next invocation. `--repeats 3 --wait 300` is available only when ambient remains stable enough to reuse one reading. Power telemetry is requested so a qualified source can be correlated, but unavailable power is an honest result. Each output retains the full aligned series and its source/availability provenance.

For a short workflow check, use `--duration 120 --repeats 1`; that is long enough for the classifier but is not the ten-minute qualification configuration. Do not pause a qualification run: pause transitions make its sustained classification indeterminate.

## Evidence review

For every result, verify that `run.status` is `complete`, `sustained` contains exactly the intended model, `actual_duration_sec` reaches the target, and the series has contiguous ten-second windows. Record the temperature sources under `preflight.temperature` and the per-case `temperature.provenance`, including failed samples. Review the dashboard's Sustained Load section for throughput, temperature, and power alignment and compare the initial TPS, steady TPS, retention, onset, performance class, and cause across all three runs.

Archive the raw results, exported bundles if used, dashboard screenshots, ambient readings, exact system placement, OS/kernel, firmware, driver/runtime versions, engine version, model digest, and tested commit. A parser test or a visible sensor alone is not qualification.

## Observer-effect status

Temperature collection remains confined to the opt-in sustained workload and is not default-on for unrelated stages. The exact Ryzen AI Halo `hwmon` plus `rocm-smi` combination and DGX Spark `nvidia-smi` combination passed the unattended 20-pair latency-sensitive and sustained observer screens at every supported interval; 0.5 seconds remains the product cadence. This evidence does not qualify different kernels, sensor combinations, engines, or platforms automatically. If a source is absent or repeatedly fails elsewhere, retain the throughput result, report cause as unavailable or based only on the remaining channel, and label that source unsupported on the tested platform rather than substituting a different sensor silently.
