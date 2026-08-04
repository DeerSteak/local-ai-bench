# Vendor Discrepancy Diagnostics

A vendor diagnostic is a deterministic, explicitly reviewed `.labdiag` JSON artifact for investigating why two systems or runs disagree. It is separate from the redacted support bundle: diagnostics intentionally contain sensitive environment, plan, and raw case evidence and should be shared only with authorized engineers.

## Contents

Schema 1 records full-source SHA-256 digests, reviewed outbound metadata, both system profiles, both immutable run plans, the first divergence, the relevant raw case evidence, invalid-sample/timeout/skip/crash markers for that case, and fixed reproduction steps. It does not copy unrelated result sections, logs, credentials, private filesystem paths, prompts, responses, or arbitrary sidecars.

Identity incompatibility takes precedence over a numerical difference. When identities are compatible, the first named measurement whose baseline and candidate values differ is selected deterministically. Missing evidence remains missing, and raw invalidity is carried beside aggregates rather than normalized away.

## Create and verify

In **Result History**, set a baseline, select the candidate, and choose **Export Diagnostic**. Both systems' outbound metadata must be reviewed before the destination is selected.

```bash
python scripts/vendor_diagnostic_cli.py create BASELINE CANDIDATE OUTPUT.labdiag
# Review the printed metadata, then:
python scripts/vendor_diagnostic_cli.py create BASELINE CANDIDATE OUTPUT.labdiag --reviewed-metadata

python scripts/vendor_diagnostic_cli.py verify OUTPUT.labdiag BASELINE CANDIDATE
```

Verification recomputes both complete source-result digests. Any source mutation, substitution, or accidental pairing fails verification. A correction or rerun produces a new diagnostic rather than changing an already delivered artifact.

## Engineer workflow

1. Verify the diagnostic against the retained source results.
2. Resolve identity incompatibility before interpreting numerical deltas.
3. Reproduce the exact run plans without changing tuning, cache, retry, or validity rules.
4. Inspect the first divergent case's raw valid and invalid samples before comparing aggregates.
5. Record any externally supplied optimization as a separately identified methodology/tuning profile and produce new evidence.

[← Local Result History](result-history.md) · [Back to README](../README.md) · [Outbound Review →](outbound-review.md)
