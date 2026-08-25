# Local Result History

The benchmark GUI's **Result History** tab discovers benchmark JSON files in the local `results/` directory without importing them into a database or sending metadata anywhere. It shows start time, system, run status, engine, methodology profile, and usable model coverage, with free-text, status, and engine filters.

Malformed JSON and unrelated JSON files are ignored and counted in the status message. Refresh rescans the directory, so files copied in or removed outside the application remain authoritative.

## Dashboard comparison

Select one to six results with the platform's normal multi-selection keys, then choose **Open in Dashboard**. The GUI starts the local dashboard and loads those files directly in their table order, where its charts, raw tables, validity inspection, and model filters provide the comparison. Only the selected files are copied temporarily into the dashboard build directory. A normal server stop removes them, and every subsequent dashboard build clears any copies left by a forcibly closed terminal.

The dashboard preserves missing measurements as missing and displays methodology, accuracy-setting, reliability, and validity warnings rather than presenting unlike evidence as silently equivalent.

## Resume, retry, and fork

Select an interrupted or failed result and choose **Inspect Recovery** to verify its journal, current model/runtime/input identities, durable case coverage, and eligible retries without changing any files. **Resume** repeats that gate, asks for confirmation, skips completed cases, and continues remaining journal-owned stages. **Retry Cases** reopens only the selected failed or incomplete contexts, questions, embedding batches, image resolutions, or concurrency levels. **Fork** always creates a distinct output and job while preserving source provenance; the original result is never overwritten.

To exercise image recovery deliberately, start an image-only run and wait until the Run Log reports that the first measured resolution finished and its representative image was saved. Click **Stop Benchmark** once in the progress window, then inspect and resume the interrupted result here. From a terminal, press `Ctrl+C` once in the terminal running `run_bench.sh`; do not kill ComfyUI separately. A correct inspection reports the saved resolution complete and later resolutions remaining, and resumed execution does not generate the completed resolution again.

Image recovery stores the selected ComfyUI and generated-image directories in an owner-only `*.events.sqlite3.local.json` sidecar. It is bound to the journal job, excluded from portable results and bundles, and removed with **Delete**. Missing or changed model/runtime/input identity requires **Fork**; a missing or malformed private sidecar stops recovery instead of guessing local paths.

Backend comparisons retain the raw delta, provisional practical-change threshold, each side's recorded within-run dispersion, and the valid sample counts behind it. Missing dispersion is reported as insufficient rather than zero, and every single-run comparison states that repeated trials are required for a regression verdict.

## Repeated-trial comparison

Build a durable trial-set artifact from two groups of compatible independent runs:

```bash
python -m scripts.results.trial_set_cli \
  --baseline results/baseline-1.json results/baseline-2.json results/baseline-3.json results/baseline-4.json results/baseline-5.json \
  --candidate results/candidate-1.json results/candidate-2.json results/candidate-3.json results/candidate-4.json results/candidate-5.json \
  --out results/runtime-upgrade.trials.json \
  --report results/runtime-upgrade.trials.md
```

Pooling requires the existing methodology compatibility gate, the same hardware identity, distinct source digests, and at least one common metric. List each side's files in trial order; matching positions are paired automatically only when both sides have the same count and case sequence, and that order also drives drift detection. By default the model identity in every metric must match. Supplying both `--baseline-model` and `--candidate-model` explicitly maps those two different identities onto the same comparison cases and records the mapping in the artifact; supplying only one is invalid. Five trials per side are required before an interval or regression verdict is emitted. Identical case sequences use paired relative changes and a 95% Student-t interval; unequal sequences or counts use a 95% Welch interval. Monotonic ordinal drift forces an inconclusive verdict. The artifact records source digests, descriptive statistics, interval method, comparison mode, practical threshold, drift state, and one of `improved`, `regressed`, `unchanged`, or `inconclusive` for every common metric. The optional Markdown report renders the same evidence and explicitly labels undersized sets as requiring repeated trials.

## Acceptance evaluation

Select a result and choose **Evaluate Policy** to apply a versioned acceptance-policy file. The dialog shows the overall decision and every rule's status, actual value, threshold, and evidence count. Missing, insufficient, or incompatible evidence rejects explicitly.

Select exactly two results and choose **Export Diagnostic** to create a separately reviewed vendor-engineer package containing the first divergence and only its relevant raw evidence, identities, invalidity, and reproduction steps. The earlier selected row is the baseline and the later row is the candidate. See [Vendor Diagnostics](vendor-diagnostics.md).

History remains filesystem-owned and intentionally simple: there is no watcher, background indexer, duplicate cache, account, or sync service. Projects may reference a local baseline path, while the history tab always reads the current result file before dashboard launch, diagnostic export, or evaluation.

[← Benchmark Projects](projects.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
