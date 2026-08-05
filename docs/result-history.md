# Local Result History

The benchmark GUI's **Result History** tab discovers benchmark JSON files in the local `results/` directory without importing them into a database or sending metadata anywhere. It shows start time, system, run status, engine, methodology profile, and usable model coverage, with free-text, status, and engine filters.

Malformed JSON and unrelated JSON files are ignored and counted in the status message. Refresh rescans the directory, so files copied in or removed outside the application remain authoritative.

## Dashboard comparison

Select one to six results with the platform's normal multi-selection keys, then choose **Open in Dashboard**. The GUI starts the local dashboard and loads those files directly in their table order, where its charts, raw tables, validity inspection, and model filters provide the comparison. Only the selected files are copied temporarily into the dashboard build directory. A normal server stop removes them, and every subsequent dashboard build clears any copies left by a forcibly closed terminal.

The dashboard preserves missing measurements as missing and displays methodology, accuracy-setting, reliability, and validity warnings rather than presenting unlike evidence as silently equivalent.

## Acceptance evaluation

Select a result and choose **Evaluate Policy** to apply a versioned acceptance-policy file. The dialog shows the overall decision and every rule's status, actual value, threshold, and evidence count. Missing, insufficient, or incompatible evidence rejects explicitly.

Select exactly two results and choose **Export Diagnostic** to create a separately reviewed vendor-engineer package containing the first divergence and only its relevant raw evidence, identities, invalidity, and reproduction steps. The earlier selected row is the baseline and the later row is the candidate. See [Vendor Diagnostics](vendor-diagnostics.md).

History remains filesystem-owned and intentionally simple: there is no watcher, background indexer, duplicate cache, account, or sync service. Projects may reference a local baseline path, while the history tab always reads the current result file before dashboard launch, diagnostic export, or evaluation.

[← Benchmark Projects](projects.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
