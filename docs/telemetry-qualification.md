# Telemetry Qualification

The milestone-1 observer screen is a supervised real-hardware procedure. The analyzer is read-only and never launches a benchmark: `python -m scripts.release.telemetry_qualification MANIFEST.json --output REPORT.json` reads paired result files, emits descriptive impacts, and exits nonzero when a predeclared bound fails.

Use one installed xsmall LLM model and the single-shot `2K` case for every trial on a platform. Keep the model tag, engine/runtime versions, power mode, driver, background workload, warmups, measured runs, and prompt cap fixed. Run at least 20 pairs for each 0.25, 0.5, and 1.0 second candidate interval. Set `LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC` to the candidate value for both runs in a pair; telemetry-off does not sample or retain the interval, but the environment and commands remain otherwise identical. Do not mix files from different intervals in one manifest.

Within pair 1 run telemetry off and then on; within pair 2 run on and then off; continue alternating pair order. Keep the machine otherwise idle and wait a fixed 30 seconds between invocations, including between pairs. Use explicit output paths and the same command except for `--memory-telemetry`. For example, substitute the exact installed model tag and pair number:

```bash
LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC=1.0 bash run_bench.sh --ui none --tests llm --llm-models MODEL_TAG --max-prompt-tokens 2048 --warmup 2 --runs 3 --out qualification/off-01.json
LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC=1.0 bash run_bench.sh --ui none --tests llm --llm-models MODEL_TAG --max-prompt-tokens 2048 --warmup 2 --runs 3 --memory-telemetry --out qualification/on-01.json
```

The manifest records the physical execution order even though `off` and `on` always point to their corresponding modes:

```json
{
  "platform": "mac-mini-m4-pro-24gb-macos",
  "interval_sec": 1.0,
  "section": "llm",
  "model": "MODEL_SHORT_KEY_IN_RESULTS",
  "case": "2K",
  "pairs": [
    {"order": "off-on", "off": "off-01.json", "on": "on-01.json"},
    {"order": "on-off", "off": "off-02.json", "on": "on-02.json"}
  ]
}
```

Continue the array through pair 20. Paths are resolved relative to the manifest. The analyzer uses each run's recorded mean TTFT, mean throughput, and median valid-sample client wall time. Positive impact means telemetry made latency/wall time worse or throughput lower. TTFT fails at the median only when its impact exceeds both 2% and 2 milliseconds, and at the 90th percentile only when it exceeds both 4% and 4 milliseconds; the report includes both units and bounds. Throughput and wall time fail when their median impact exceeds 1% or their 90th-percentile impact exceeds 2%.

Archive the manifest, report, all referenced result files, exact OS/driver/runtime/source versions, sensor permissions, process ownership and scope notes, telemetry failure counts, and the commit tested. A parser fixture is not hardware qualification, and a passing coarse screen does not make telemetry default-on or establish scientific comparability.

## Repeated-trial runner

`run_telemetry_trials.sh` automates the same alternating procedure for the selected model and engine, defaults to 20 pairs at the provisional 0.5-second interval, waits 30 seconds between invocations, writes explicit per-trial results, builds the manifest, and runs the analyzer. Its default `memory` mode compares telemetry disabled with memory sampling enabled. `--telemetry power` compares memory-only sampling with the combined memory-and-power sampler, preserving the shared-sampler design while isolating the incremental observer cost of power collection. On Linux, `--telemetry temperature` compares the same memory-and-power baseline with temperature added through an internal qualification-only override; the ordinary product still collects temperature only during the opt-in sustained workload.

The default output is under the gitignored `results/qualification/` tree, and a real run refuses to start from a dirty worktree so every result records a reproducible source identity. Completed outputs are skipped on restart; an incomplete output stops the script so it cannot silently become a nominally independent trial. Preview every command without launching a benchmark first:

```bash
bash run_telemetry_trials.sh --model MODEL_TAG --engine llamacpp --dry-run
bash run_telemetry_trials.sh --model MODEL_TAG --engine llamacpp \
  --out-dir results/qualification/memory-this-machine
bash run_telemetry_trials.sh --model MODEL_TAG --engine llamacpp \
  --telemetry power --dry-run
```

Use `--pairs 5` only for a workflow smoke test; it does not meet the 20-pair qualification minimum. The script intentionally runs one engine and one installed xsmall model at 2K so model/runtime changes are not mixed into the observer comparison.

Temperature qualification also supports `--workload sustained --sustained-duration 120 --ambient-temp-c C`. It alternates otherwise identical two-minute soaks with temperature disabled and enabled. The sustained analyzer computes duration-weighted overall throughput and retention ratio from the complete series. Median impact may not exceed 1% throughput or one retention percentage point; the 90th-percentile bounds are 2% and two points. The latency-sensitive screen retains the existing TTFT, throughput, and wall-time bounds. Every temperature-on result must record an available source, while both modes must retain the memory-and-power baseline, or the analyzer rejects the evidence identity before calculating impacts.

The unattended Linux wrapper runs the complete matrix: 20 alternating latency pairs and 20 alternating sustained pairs at each of 0.25, 0.5, and 1.0 seconds, for 240 benchmark invocations total. It uses 30-second waits for latency trials, 120-second waits for sustained trials, and five-minute gaps between sustained interval suites. Outputs are resumable. A rejected observer report is retained and the remaining interval suites continue; an actual benchmark failure stops the wrapper. Preview the whole matrix without launching a benchmark:

```bash
bash run_temperature_qualification_linux.sh --model MODEL_TAG --ambient-temp-c 20.0 --dry-run
bash run_temperature_qualification_linux.sh --model MODEL_TAG --ambient-temp-c 20.0
```

Expect an overnight run: the sustained measurements alone require four hours of active soak time, with controlled waits adding roughly four more hours before model-loading and latency-screen time. The ambient value is the room measurement at matrix start, not a claim that ambient remained constant; record start/end ambient separately with the archived evidence.

On macOS, a real power run requests administrator permission once before starting and refreshes that temporary authorization while the trial series runs; the sampler itself still uses non-interactive `sudo -n` and never prompts during a benchmark case. The manifest and report record the discovered source and measurement scope. Canceling or denying the initial permission stops the run before any benchmark starts.

The M5 Pro release screen has a dedicated overnight wrapper that runs all three intervals for 20 pairs each—120 benchmark invocations total—and uses `caffeinate` to prevent system sleep. It fixes the model and methodology to the qualified configuration and groups the three manifests, reports, and raw-result directories beneath one timestamped root:

```bash
bash run_power_qualification_m5_pro.sh --dry-run
bash run_power_qualification_m5_pro.sh
```
