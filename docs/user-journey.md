# Project-to-Report User Journey

This journey is the authoritative user-facing sequence for a local decision project. Every path ends with preserved evidence, a clear next action, or an explicit user cancellation; the application never converts an incomplete path into a successful run.

## 1. Launch and discover

The launcher selects the GUI on a usable local desktop and the terminal frontend over SSH/headless sessions, with explicit overrides. Discovery reads hardware, storage, runtimes, installed models, ComfyUI, and blockers without downloading, installing, starting servers, or modifying configuration. A blocked prerequisite links to Setup; cancellation closes the launcher without offering to run.

## 2. Create or open a project

The user selects Custom, creates a decision project for hardware comparison, model selection, acceptance validation, capacity planning, or regression, or opens an existing `.labproject`. The project restores portable configuration while retaining machine-local output and ComfyUI paths. Unavailable engines, tests, or models stop application of the project and name the missing items.

## 3. Configure and review

Default mode locks a supported standard configuration. Custom mode exposes guided and advanced controls with units, ranges, consequences, reset actions, presets, installed-model selection, prompt/generation caps, run counts, timeouts, CPU-only mode, force-all, and offline mode. The final preview resolves tests, exact models, cases, process/load count, estimated duration, storage/network implications, paths, and every measurement-affecting setting. Invalid or unsupported combinations are corrected before launch, not during a long run.

## 4. Execute

The configuration GUI starts a supervised benchmark subprocess and presents a separate always-on-top progress view over the terminal. Stage/model queues, current work, durable completion, valid coverage, retries, invalid drops, resource use, remaining-time estimate, and detailed log stay visible. Results are checkpointed at stage/model/case boundaries appropriate to each workload, including each streamed native llama-bench case.

## 5. Cancel or fail

Cancel requests interruption, escalates bounded process termination if necessary, flushes the active workload's latest durable evidence, marks active stage/run state interrupted, tears down owned runtimes, and returns the user to a reviewable result. An unhandled failure classifies preparation, execution, and cleanup separately; cleanup runs even when persistence fails. The result and UI state what failed, what data survived, whether it remains usable, what automatic action occurred, and the next step.

## 6. Resume, retry, or fork

Resume is allowed only when plan, model artifact, runtime, methodology, and case-boundary identity match. Completed valid cases are not rerun. A changed identity requires a fork with a new job/run lineage; a user may also intentionally start a fresh run. Retry targets explicitly failed/incomplete work and never overwrites the prior source evidence. The current GUI records safe resume/fork foundations, while complete moderated pause/resume/retry interaction remains a release gate.

## 7. Review evidence

The user opens Result History, filters and multi-selects local files, then opens them directly in the dashboard for visual comparison and raw-sample inspection of valid, excluded, and legacy evidence. An acceptance policy evaluates exact workload/model/case metrics with required evidence counts; missing, incompatible, insufficient, and invalid data reject rather than becoming zero. Evidence completeness and policy acceptance remain separate.

## 8. Export and report

Before a result bundle or report is written, the user reviews outbound system/hardware/model identity, may assign private aliases, and approves the export. Bundles verify schema, file digests, methodology availability, reproducible aggregates, and optionally the retained private source identity. Reports are deterministic, self-contained, disclose methodology/offline/optimization state, show coverage and exclusions, list acceptance evidence, and state limitations without a hidden composite score.

## 9. Correct or escalate

If systems disagree or a vendor engineer requests investigation, the source result, event history, run plan, runtime/environment identity, raw/invalid samples, support bundle, and first divergent case form the reproduction basis. Corrections create a new verified artifact or fork; they do not mutate an already delivered bundle without a new identity and explanation.

## Completion states

| State | Meaning | User action |
|---|---|---|
| Complete and accepted | Required evidence is valid and every named policy rule passes | Review/export the decision package |
| Complete and rejected | Execution completed, but one or more policy rules failed | Review measurements and discrepancy evidence |
| Complete without policy | Evidence is complete; no approval threshold was supplied | Compare/review or attach a policy |
| Partial/interrupted | Durable evidence exists but requested work remains | Review usable evidence, then safely resume/fork or accept declared partial scope |
| Failed | A terminal failure prevented requested completion | Follow the recorded next step; prior durable evidence remains inspectable |
| Incompatible | Methodology/runtime/configuration identity prevents comparison | Select compatible evidence or create a separately labeled analysis |
| Cancelled before execution | No benchmark work began | Return to configuration or exit |

[← Product Requirements](product-requirements.md) · [Back to README](../README.md) · [Reports →](reports.md)
