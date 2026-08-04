# Decision Reports

Local AI Bench can turn a result JSON file into deterministic, self-contained HTML and PDF decision reports. Reports summarize run identity, stage coverage, measured performance, accuracy, and sample validity without inventing a composite score or treating missing evidence as zero.

Current results also identify the active methodology profile and enumerate the effective runtime optimizations resolved for the selected workload paths. Older results label the profile as unrecorded instead of guessing.

## Generate a report

Use the report CLI after a benchmark has written its result JSON:

```bash
bench-env/bin/python scripts/decision_report_cli.py results/results_system_20260804.json \
  --html results/decision_report_system.html \
  --pdf results/decision_report_system.pdf
```

At least one output flag is required. Both formats are generated locally with no external assets, scripts, telemetry, or network requests. The HTML is a single portable file, and identical validated input produces identical bytes in each format.

The graphical benchmark launcher also provides **Create Report** on the Run Log screen. Choose a result JSON and an HTML destination; the matching PDF is written beside it with the same filename stem.

The GUI then offers an optional acceptance-policy selection. On the CLI, pass `--policy POLICY.json`. When supplied, both report formats show the overall accepted/rejected decision and every named rule's status, actual value, threshold, and evidence count; evidence readiness remains a separate statement so a complete run cannot be mistaken for a passing policy.

Example outputs are available as [HTML](../samples/decision_report_example.html) and [PDF](../samples/decision_report_example.pdf).

## Readiness and evidence

`COMPLETE EVIDENCE` means the source run is complete and every represented performance sample was aggregate-eligible. `REVIEW REQUIRED` means the run is partial, interrupted, failed, legacy, or contains excluded performance samples. Invalid-only cases remain visible in the sample-validity table even when they cannot produce an aggregate row.

The report is an evidence summary, not an acceptance decision. Review the verified result bundle and raw sample exclusion reasons before making purchase, launch, or capacity decisions. Compare only results with compatible methodology, model artifacts, runtimes, cache behavior, and effective settings; see the [Methodology Contract](methodology-contract.md) and [Limitations](limitations.md).

## Reproducibility and safety

Input is rejected when it is not a JSON object or contains non-finite numbers. User-supplied text is escaped in HTML and PDF rendering. The PDF generator uses invariant metadata so timestamps and document identifiers do not make repeated output differ.

[← Dashboard](dashboard.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
