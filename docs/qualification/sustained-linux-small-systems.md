# Sustained-load qualification on Linux small systems

This procedure captures the real-hardware evidence needed to qualify Version 6.0-pre5 temperature sources and the combined memory, power, temperature, and throughput sampler. It is prepared for the stock, open-air DGX Spark and AMD Ryzen AI Halo Developer Platform; neither platform is claimed qualified until its evidence is reviewed and this document is updated with the exact OS, driver, runtime, source, commit, and results.

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

Temperature collection is confined to the opt-in sustained workload and is not default-on. The combined sampler remains opt-in until the shared 20-pair latency-sensitive observer screen is extended and passed for each exact Linux source combination and interval. The three repeated soaks qualify sensor readability and sustained evidence shape; they do not by themselves prove that polling overhead is negligible. If a source is absent or repeatedly fails, retain the throughput result, report cause as unavailable or based only on the remaining channel, and label that source unsupported on the tested platform rather than substituting a different sensor silently.
