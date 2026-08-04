# Acceptance Policies

An acceptance policy turns named benchmark evidence into an explicit accepted/rejected decision. Each rule identifies one workload section, model, case, metric, direction, threshold, and minimum evidence count. Policies never average unrelated measurements into a composite score.

## Evaluate a result

Start from [the example policy](../samples/acceptance_policy_example.json), update its exact model and thresholds, then run:

```bash
python -m scripts.results.acceptance_policy_cli \
  results/results_system_20260804.json \
  samples/acceptance_policy_example.json
```

Exit code `0` means every rule passed, `2` means the result was evaluated and rejected, and `1` means the policy or result could not be evaluated. The command writes deterministic JSON containing the overall decision and every rule's status, actual value, threshold, and evidence count.

Policies can also be applied when creating a decision report through the benchmark GUI or with `python -m scripts.results.decision_report_cli RESULT --html REPORT.html --pdf REPORT.pdf --policy POLICY.json --reviewed-metadata`. The report keeps evidence completeness and policy acceptance separate.

## Rule behavior

Supported operators are `at_least` and `at_most`. Performance evidence counts `valid_runs` when present and otherwise the workload's completed `n_runs`; accuracy evidence counts scored questions. Missing metrics, missing cases, non-finite values, and evidence below `minimum_evidence` reject rather than becoming zero or passing silently.

The policy names its required methodology profile. A result with a different or unrecorded profile is rejected as `incompatible_methodology`, so a legacy or tuned result cannot accidentally satisfy a neutral threshold. Current supported metrics cover LLM, conversation, embeddings, images, accuracy, and HTTP concurrency; native diagnostic-tool policies are deliberately deferred until their case vocabulary is stabilized.

## Schema

Acceptance-policy schema 1 has exactly four top-level fields: `schema_version`, `name`, `methodology_profile`, and `rules`. Rule IDs must be unique. Unknown fields, sections, metrics, operators, malformed case shapes, non-finite thresholds, and non-positive evidence requirements are rejected before evaluation.

[← Methodology Contract](methodology-contract.md) · [Back to README](../README.md) · [Decision Reports →](reports.md)
