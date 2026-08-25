# Decision Reports

Milestone 11 workspace reports can also be generated from a digest-bound `workspace_selection` artifact with `python -m scripts.results.workspace_export_cli`. The recorded baseline selects the report's primary result, the embedded acceptance policy and authoritative recommendation are applied without a second selection step, and the HTML/PDF report lists every selected result identity. A changed or missing source result fails the export rather than producing a report whose evidence differs from the workspace view.

Local AI Bench can turn a result JSON file into deterministic, self-contained HTML and PDF decision reports. Reports summarize run identity, stage coverage, measured performance, accuracy, and sample validity without inventing a composite score or treating missing evidence as zero.

Current results also identify the active methodology profile and enumerate the effective runtime optimizations resolved for the selected workload paths. Older results label the profile as unrecorded instead of guessing.

## Generate a report

Use the report CLI after a benchmark has written its result JSON:

```bash
python -m scripts.results.decision_report_cli results/results_system_20260804.json \
  --html results/decision_report_system.html \
  --pdf results/decision_report_system.pdf \
  --reviewed-metadata
```

At least one output flag is required. Both formats are generated locally with no external assets, scripts, telemetry, or network requests. The HTML is a single portable file, and identical validated input produces identical bytes in each format.

The graphical benchmark launcher also provides **Create Report** on the Run Log screen. Choose a result JSON, review every outbound identity field, optionally assign private system/hardware aliases, and choose an HTML destination; the matching PDF is written beside it with the same filename stem. CLI generation prints the same metadata and writes nothing until `--reviewed-metadata` is supplied.

The GUI then offers an optional acceptance-policy selection. On the CLI, pass `--policy POLICY.json`. When supplied, both report formats show the overall accepted/rejected decision and every named rule's status, actual value, threshold, and evidence count; evidence readiness remains a separate statement so a complete run cannot be mistaken for a passing policy.

Pass `--recommendation RECOMMENDATION.json` to include a recommendation produced by `scripts.results.recommendation_cli`. The report verifies that the artifact cites the private source result before outbound aliases are applied, then renders its constraints, recommended/tied/insufficient-evidence verdict, candidate groups, reasons, and evidence paths without recomputing policy. A mismatched artifact is rejected rather than attached to the wrong result.

Example outputs are available as [HTML](../samples/decision_report_example.html) and [PDF](../samples/decision_report_example.pdf).

## Readiness and evidence

`COMPLETE EVIDENCE` means the source run is complete and every represented performance sample was aggregate-eligible. `REVIEW REQUIRED` means the run is partial, interrupted, failed, legacy, or contains excluded performance samples. Invalid-only cases remain visible in the sample-validity table even when they cannot produce an aggregate row.

The report keeps evidence readiness separate from an optional acceptance-policy decision. Schema-2 policies can report accepted, rejected, or inconclusive, and a repeated-trials evidence requirement remains inconclusive when the report contains only one run. Review the verified result bundle and raw sample exclusion reasons before making purchase, launch, or capacity decisions. Compare only results with compatible methodology, model artifacts, runtimes, cache behavior, and effective settings; see the [Methodology Contract](methodology-contract.md) and [Limitations](limitations.md).

## Reproducibility and safety

Input is rejected when it is not a JSON object or contains non-finite numbers. User-supplied text is escaped in HTML and PDF rendering. The PDF generator uses invariant metadata so timestamps and document identifiers do not make repeated output differ.

[← Dashboard](dashboard.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
