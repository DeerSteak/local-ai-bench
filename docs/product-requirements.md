# Product Requirements: Pre-Launch Hardware Validation

## Decision and user

The primary 6.0-pre3 product workflow helps a hardware-vendor performance or product team decide whether measured local-AI behavior on an upcoming small-system launch is credible, reproducible, explainable, and ready for internal or external use. The product supplies an independent validation source; it does not promise favorable results or replace the vendor's own engineering benchmarks.

The accountable user owns the launch evidence or advises the decision maker. They may have pre-release hardware, established competing benchmarks, a vendor engineer available for discrepancies, and strict embargo/offline requirements. One engagement normally runs the agreed suite once across multiple systems; independent repeated runs must produce consistent evidence within the declared methodology and acceptance policy.

## Required outcome

A successful project produces a validated immutable run plan, durable measurements that survive individual case/model failures and interruption, explicit valid/invalid coverage, a verified portable result bundle, named acceptance-policy outcomes when supplied, a discrepancy record when evidence diverges, and a self-contained reviewed report with limitations. Delivery is the agreed evidence and analysis, not a guaranteed performance ranking.

## In-scope workflow

1. Discover the local hardware, runtimes, models, storage, and blockers without installing or changing the system.
2. Create or open a hardware-comparison or acceptance-validation project and resolve every measurement-affecting choice.
3. Review the run plan, methodology profile, load count, time/disk estimates, paths, offline requirement, and embargo handling before execution.
4. Execute locally with durable case/model checkpoints, supervised cleanup, explicit retry/timeout/invalidity behavior, and visible progress.
5. Resume or fork only at safe case boundaries when plan, artifact, runtime, and methodology identity permit it.
6. Inspect raw samples and exclusions, compare compatible systems against a selected baseline, and evaluate named acceptance rules.
7. Review outbound metadata, apply private aliases when required, export a verified bundle, and generate a deterministic HTML/PDF report.
8. Preserve enough local identity and diagnostics to reproduce or escalate the first discrepancy without disclosing unrelated private data.

## Required product qualities

- The GUI must be understandable without benchmark terminology, expose every sensible measurement lever, provide safe defaults, and keep advanced controls configurable without hiding resolved values.
- No error, timeout, cancellation, persistence problem, or implausible measurement may erase previously durable successful evidence.
- Missing, partial, invalid, incompatible, and legacy evidence must remain visibly distinct from zero and from accepted evidence.
- The application must not silently apply vendor tuning, compare incompatible methodology, export embargoed identity, or make unpreviewed system-wide changes.
- Common work stays local and direct: focused functions, small immutable records, and filesystem-owned artifacts are preferred over speculative services or generic frameworks.

## Supported 6.0-pre3 scope

The supported default evidence set is single-shot LLM, conversation, embeddings, image generation, and MCQ/math/reasoning/code/tool accuracy. HTTP and native llama.cpp concurrency/throughput workloads are opt-in diagnostics. The supported runtime is llama.cpp plus ComfyUI for images; multi-node transport, a second inference engine, hosted collaboration, automatic updates, commercial entitlement, and automated consumer purchasing recommendations are not stable 4.1 capabilities.

## Acceptance of this product slice

The workflow is product-complete only when a clean supported machine can reach a first valid result without developer intervention, an interrupted run preserves completed evidence, incompatible comparisons are blocked, report/bundle output is reviewable and verifiable, and the full supported hardware qualification matrix passes its release criteria. User research, signed distribution, security assessment, legal terms, pricing, and paid-launch evidence remain separate release gates.

[← Methodology Contract](methodology-contract.md) · [Back to README](../README.md) · [User Journey →](user-journey.md)
