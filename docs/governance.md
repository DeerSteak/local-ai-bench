# Product and methodology governance

This is the public change-approval contract. It defines evidence and authority for shipping changes without exposing the private commercial roadmap, customer pipeline, pricing, or embargoed work.

## Change classes

| Change class | Required evidence | Required approval before stable release |
|---|---|---|
| Scope or supported-platform claim | User problem, explicit inclusion/exclusion, support impact, qualification evidence | Product owner and release owner |
| Measurement or methodology | Hypothesis, timing/validity boundary, fixtures, expected result impact, compatibility decision | Methodology owner and product owner |
| Result, plan, event, bundle, report, or API schema | Schema diff, backward/forward compatibility, migration/fallback, dashboard/export coverage, golden tests | Engineering and methodology owners |
| Security or data handling | Threat/data-classification impact, abuse cases, tests, residual risk, rollback | Security owner; product owner for accepted residual high risk |
| Dependency, model, dataset, runtime, font, or distributed artifact | Provenance, version/digest, license/notice, redistribution mode, security review | Legal/licensing and engineering owners |
| UI behavior or default | User impact, option inventory, accessibility impact, screenshots/manual review when visual, tests | Design/product owner and engineering owner |
| Stable release | All stable gates, supported matrix, signed artifacts, rollback evidence, known issues | Product, methodology, engineering, security, legal/licensing, design, and release owners for their domains |

One person may hold several roles during the founder-led phase, but approvals still record the role and evidence separately. No author may satisfy an independent security assessment or counsel-review gate by reviewing their own work under another role name.

## Decision process

1. Classify the change and list affected contracts before implementation.
2. Identify the authoritative source of truth and the smallest implementation that resolves the current need.
3. Record compatibility, security/privacy, licensing, user-facing, support, migration, rollback, and removal consequences; mark non-applicable areas explicitly.
4. Produce the required tests, qualification artifacts, documentation, and reviewer evidence.
5. Record approval, rejection, or time-bounded risk acceptance with date, role, rationale, evidence links, conditions, and expiry.
6. Reopen the decision when its assumptions, supported matrix, methodology, license, or customer evidence changes.

Methodology-affecting changes require a new immutable workload/methodology identity when comparisons would otherwise mix different semantics. Additive implementation detail does not earn a new abstraction or version unless it changes a public contract or solves a demonstrated ownership/reliability problem.

## Decision records

Material architecture and data-ownership decisions are recorded in [Architecture Decisions](architecture-decisions.md). Methodology baselines, compatibility decisions, release approvals, license reviews, security exceptions, and supported-platform promotions use the same durable pattern: identifier, status, date, owner role, context, decision, alternatives, consequences, evidence, follow-up, and superseding record. Records are append-only in meaning; corrections supersede rather than silently rewrite delivered decisions.

The private prioritized backlog may reference these public records, but public documents never link to or name the private commercial roadmap. Embargoed/customer decisions stay in the customer's restricted project and expose only an approved public summary when one is needed.
