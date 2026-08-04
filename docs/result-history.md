# Local Result History

The benchmark GUI's **Result History** tab discovers benchmark JSON files in the local `results/` directory without importing them into a database or sending metadata anywhere. It shows start time, system, run status, engine, methodology profile, and usable model coverage, with free-text, status, and engine filters.

Malformed JSON and unrelated JSON files are ignored and counted in the status message. Refresh rescans the directory, so files copied in or removed outside the application remain authoritative.

## Baseline comparison

Select a result and choose **Set Baseline**, then select another result and choose **Compare to Baseline**. The comparison lists the union of named supported measurements, preserves missing values as missing, and computes absolute and percentage changes only when both values exist.

Numerical comparison is marked compatible only when application version, engine, methodology profile, and the complete effective configuration match. An unrecorded methodology always blocks compatibility. The measurements remain visible for diagnosis, but the UI labels the comparison blocked rather than presenting unlike runs as equivalent.

## Acceptance evaluation

Select a result and choose **Evaluate Policy** to apply a versioned acceptance-policy file. The dialog shows the overall decision and every rule's status, actual value, threshold, and evidence count. Missing, insufficient, or incompatible evidence rejects explicitly.

With a baseline set, **Export Diagnostic** creates a separately reviewed vendor-engineer package containing the first divergence and only its relevant raw evidence, identities, invalidity, and reproduction steps. See [Vendor Diagnostics](vendor-diagnostics.md).

History remains filesystem-owned and intentionally simple: there is no watcher, background indexer, duplicate cache, account, or sync service. Projects may reference a local baseline path, while the history tab always reads the current result file before comparison or evaluation.

[← Benchmark Projects](projects.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
