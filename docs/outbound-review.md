# Outbound Metadata Review

Result bundles and decision reports can disclose embargoed system names, hardware names, operating-system details, memory, backend, model identities, application version, and plan identity. GUI export paths therefore show the exact outbound identity fields and require approval before choosing the final destination.

The review dialog accepts optional system and hardware aliases. Aliases modify only the exported derivative; the durable source result remains unchanged. Every derivative records a SHA-256 digest of the original identity fields and the alias classes applied, allowing an authorized user with the original result to verify which private source produced an aliased report or bundle.

CLI export is deliberately two-step. A command without `--reviewed-metadata` prints the outbound fields, writes nothing, and exits with an error. After review, repeat it with `--reviewed-metadata`; optional `--system-alias NAME` and `--hardware-alias NAME` apply private labels.

```bash
python scripts/result_bundle_cli.py export RESULT BUNDLE --reviewed-metadata \
  --system-alias "System A" --hardware-alias "Prototype A"

python scripts/decision_report_cli.py RESULT --html REPORT.html --pdf REPORT.pdf \
  --reviewed-metadata --system-alias "System A" --hardware-alias "Prototype A"
```

Aliases do not redact model names, measurements, prompts, responses, artifacts, or arbitrary sidecars. The preview lists model identities, and users must inspect attached artifacts separately. Use the allowlisted support-bundle workflow when raw results are unnecessary.

To verify that a bundle came from a retained private source result, run `python scripts/result_bundle_cli.py verify BUNDLE --source-result ORIGINAL_RESULT`. Integrity, aggregates, methodology availability, and the source-identity digest must all match.

[← Security and Privacy](security-and-privacy.md) · [Back to README](../README.md) · [Local Data Lifecycle →](data-lifecycle.md)
