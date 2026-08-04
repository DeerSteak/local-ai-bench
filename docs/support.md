# Support operations

Support is an evidence-preserving diagnostic process. It does not ask users to destroy partial results, disclose secrets, or hand over unrestricted machine access as the first response.

## Intake

Record request ID, customer/project and authorized contacts, entitlement/support tier when one exists, affected version/platform/configuration, impact and urgency, first/last occurrence, reproducibility, recent changes, exact safe error, preserved result/job identity, attempted actions, embargo/data classification, permitted communication channel, and supplied artifact inventory. Ask the user to preview and approve a `.labsupport` bundle; request any additional file separately with purpose, fields, size, retention, audience, and redaction instructions.

Never request access tokens, private keys, passwords, full environment dumps, unrestricted home-directory archives, proprietary prompts/responses, or embargoed identity when a minimized reproduction or alias suffices. Do not copy customer data into public issues, personal storage, or an unapproved AI/service.

## Severity and response

| Severity | Definition | Initial action |
|---|---|---|
| S0 security | Suspected active compromise, credential exposure, embargo disclosure, unsafe update/artifact, or cross-customer exposure | Invoke the security incident process immediately; pause affected transfer/release/publication |
| S1 critical | Data corruption/loss, destructive install behavior, broad inability to run on a supported configuration, or stranded unsafe process with no workaround | Preserve evidence, provide containment, assign engineering owner and next update time |
| S2 high | Major supported workflow blocked or decision evidence unreliable, with a costly workaround | Validate scope/identity, provide safe workaround when available, schedule correction |
| S3 normal | Limited defect, confusing behavior, performance discrepancy, or partial degradation with usable evidence/workaround | Triage against support matrix and collect minimized diagnostics |
| S4 guidance | Configuration, methodology, interpretation, or unsupported/preview question | Answer with documentation, limitations, and support-status boundary |

Contracted response targets and coverage hours belong in the commercial terms; do not promise them until pricing, staffing, holidays, escalation coverage, and customer communication channels are defined.

## Investigation and escalation

Confirm affected identity and supported status, reproduce with the smallest safe case, distinguish product defect from methodology incompatibility/environment/runtime/model behavior, and state what prior data remains usable. Preserve originals; derived/redacted copies receive separate identities. Escalate benchmark discrepancies with the vendor diagnostic package, code defects with focused tests and source revision, security issues through the security process, and licensing/data-rights questions to the named legal/licensing owner.

Every handoff records owner, evidence, hypothesis, action, result, customer-safe update, next step, and deadline. Remote commands are never executed through a coordinator shell API; if interactive access is exceptionally approved, use the customer's controlled mechanism and document scope and revocation.

## Redaction, retention, and resolution

Classify every artifact on receipt, store it only in the authorized project, restrict access, record expiry/deletion, and avoid duplicating it into chat or ticket systems. Redaction removes secrets and unnecessary content while retaining hashes/aliases needed to correlate evidence. If redaction would make diagnosis impossible, obtain explicit approval for the minimum additional field rather than silently broadening collection.

A resolution states root cause or current best-supported explanation, affected versions/configurations, preserved/invalid evidence, workaround, fix and verification, compatibility or migration impact, release availability, and remaining limitations. Close only after the customer receives the agreed artifact or answer and disposition of retained data is recorded. Reopen recurring issues under the same problem record; create a security incident, product decision, or methodology record when the finding changes a durable contract.
