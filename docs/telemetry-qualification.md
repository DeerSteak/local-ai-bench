# Memory Telemetry Qualification

The milestone-1 observer screen is a supervised real-hardware procedure. The analyzer is read-only and never launches a benchmark: `python -m scripts.release.telemetry_qualification MANIFEST.json --output REPORT.json` reads paired result files, emits descriptive impacts, and exits nonzero when a predeclared bound fails.

Use one installed xsmall LLM model and the single-shot `2K` case for every trial on a platform. Keep the model tag, engine/runtime versions, power mode, driver, background workload, warmups, measured runs, and prompt cap fixed. Run at least 20 pairs for each 0.25, 0.5, and 1.0 second candidate interval. Set `LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC` to the candidate value for both runs in a pair; telemetry-off does not sample or retain the interval, but the environment and commands remain otherwise identical. Do not mix files from different intervals in one manifest.

Within pair 1 run telemetry off and then on; within pair 2 run on and then off; continue alternating pair order. Use explicit output paths and the same command except for `--memory-telemetry`. For example, substitute the exact installed model tag and pair number:

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

Continue the array through pair 20. Paths are resolved relative to the manifest. The analyzer uses each run's recorded mean TTFT, mean throughput, and median valid-sample client wall time. Positive impact means telemetry made latency/wall time worse or throughput lower. A source passes only when median TTFT impact is at most 2%, median throughput impact at most 1%, median wall impact at most 1%, and each metric's 90th-percentile impact is at most twice its median bound.

Archive the manifest, report, all referenced result files, exact OS/driver/runtime/source versions, sensor permissions, process ownership and scope notes, telemetry failure counts, and the commit tested. A parser fixture is not hardware qualification, and a passing coarse screen does not make telemetry default-on or establish scientific comparability.
