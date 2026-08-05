# Reference evidence

The reference-result corpus is an allowlisted collection of immutable result identities, not a folder into which every submitted JSON is copied. Each eligible record identifies the exact methodology, hardware configuration, model artifact and quantization, runtime, and captured environment. A verified result bundle and its digest remain the underlying evidence.

No current sample result is promoted into the supported reference corpus merely to populate it. Existing dashboard samples and test fixtures are synthetic or lack the complete current identity required for recommendations. The corpus therefore starts empty and gains records only after representative hardware qualification; this is safer than labeling old or illustrative measurements as commercially verified.

## Evidence tiers

| Tier | Meaning | Supported recommendation use |
|---|---|---|
| Verified | Bundle identity, methodology compatibility, required coverage, environment identity, and review status are verified and unexpired | Eligible for the exact represented configuration |
| Vendor | Supplied by a hardware or model vendor but not independently verified under the complete policy | Context only |
| Community | Supplied by a community member and pending or unable to complete verification | Context only |
| Rejected | Tampered, incompatible, withdrawn, superseded, policy-violating, or otherwise unsuitable | Never |

Vendor identity does not imply verification. Community and vendor evidence must remain visibly labeled and cannot affect a supported recommendation's ranking. It may suggest a configuration for a user's own verification run, provided the source and uncertainty are displayed.

## Submission lifecycle

1. Accept a portable result bundle, source type, submitter attestation, permission to retain/display it, and optional embargo date.
2. Verify digests, result and plan schemas, methodology, workload/bank versions, finite data, required identities, and absence of prohibited outbound metadata.
3. Run automated policy checks, then record a named reviewer and timestamp for any promotion to `verified`.
4. Assign an expiration date based on methodology/runtime relevance. Expired evidence stays inspectable but loses recommendation eligibility until reverified.
5. Handle a correction by creating a new immutable result identity and marking the old record as superseded. Never overwrite delivered evidence in place.
6. Record rejection, withdrawal, expiration, and removal reasons. Removal from public display does not erase the audit record unless retention law or the submitter's rights require deletion.

Moderation rejects fabricated identity, altered measurements, undisclosed tuning that changes comparison fairness, prohibited/confidential content, license violations, and abusive submissions. Appeals receive a separate reviewer when possible. Opening submissions to third parties also requires abuse handling, privacy terms, and a staffed contact path; the local policy code alone is not that operating function.
