# Consumer Recommendation Policy

Local AI Bench may help a consumer choose a model for hardware they already own or choose a Mac configuration for intended local-AI work. A recommendation is an explanation over inspectable compatible evidence, never a hidden score, parameter-count shortcut, sponsorship placement, or guarantee that an untested workload will behave the same way.

## Supported goals

| Goal | Required primary evidence | Secondary evidence |
|---|---|---|
| General chat | Conversation client TTFT, decode throughput, valid context coverage | MCQ/reasoning quality and model fit |
| Coding | Code accuracy plus interactive TTFT/throughput | Long-context coverage and tool accuracy |
| Long context | Valid evidence at the requested context depth with cache state disclosed | Throughput, TTFT, memory headroom, and early-exit reason |
| Tool use | Tool accuracy and tool-style concurrency when multi-request use is intended | Conversation responsiveness |
| Image generation | Image latency at the requested resolution/model family | Memory fit and supported workflow/checkpoint identity |
| Quality priority | Goal-specific accuracy evidence from the exact model/runtime | Responsiveness as a separately displayed tradeoff |
| Interactive-speed priority | Client TTFT first, then decode throughput, at the requested context | Goal-specific accuracy as a hard floor or explicit caveat |

Unsupported goals, missing required evidence, incompatible methodology, unknown model license, or unknown memory fit produce **Insufficient evidence**, not a guessed recommendation.

## Evidence eligibility

Supported recommendations require a verified result bundle or a locally retained source result whose methodology profile, application/result schema, runtime, model artifact, effective configuration, bank versions, and relevant cache state are known. Required cases need the policy's minimum valid samples; invalid and missing samples never count as zero. The recommendation cites the exact result, model, case, metric, valid/excluded counts, run status, and limitation that affected each conclusion.

The [reference-evidence policy](reference-evidence.md) additionally requires an unexpired allowlisted record with exact methodology, hardware, model, runtime, and environment identities. Vendor and community submissions remain contextual evidence until independently promoted through that process.

Evidence confidence is labeled:

- **Verified** — supported methodology, complete required coverage, verified source/bundle identity, and required valid evidence.
- **Limited** — compatible first-party evidence with declared partial coverage, low repetition count, or an unqualified environment; useful for narrowing choices but not a supported final recommendation.
- **Community preview** — externally submitted evidence that has not passed the supported verification/moderation policy; visibly separated and disabled from supported rankings by default.
- **Insufficient** — missing, invalid, incompatible, expired, license-unknown where commercial use matters, or outside the supported catalog.

## Memory fit and headroom

The current local preflight estimates weight/runtime memory as the downloaded model size multiplied by `1.2`. Discrete GPUs reserve `1 GB` of VRAM; unified-memory/CPU systems reserve `8 GB` of system RAM. A model is a fit candidate only when the estimated requirement is no greater than the remaining ceiling.

This estimate does not fully model context-dependent KV cache, parallel slots, runtime workspace, display use, other applications, or OS memory pressure. Therefore it can exclude an obvious non-fit but cannot by itself support “will fit” language. A supported recommendation requires a verification run at the intended context/concurrency, and reports the observed maximum valid depth plus any OOM, auto-offload, or slow-exit behavior. Unknown VRAM/RAM produces unknown fit rather than passing.

## Ranking and ties

Ranking is constraint-first and lexicographic, not additive. Candidates are removed when required evidence, fit, compatibility, license, or user constraints fail. The chosen goal then defines an ordered list of evidence dimensions; for example interactive speed orders by accepted TTFT, then throughput, while coding orders by code accuracy floor/priority before responsiveness. Every dimension remains displayed in its original unit.

Values within the declared tolerance remain tied. A tie may be broken only by a user-selected secondary priority such as memory requirement, context coverage, or price; otherwise the product presents tied choices. Missing values never act as a tiebreaker. Parameter count may explain model size but cannot determine ranking by itself.

## Existing-GPU workflow

1. Detect or confirm GPU/unified memory, runtime/backend, available memory, and installed models.
2. Ask for the supported goal, intended context, quality/speed priority, concurrency, and license/use constraints.
3. Exclude known non-fits and incompatible/unsupported artifacts; label unknown fit.
4. Match eligible verified evidence to the exact hardware/runtime/methodology and rank lexicographically for the selected goal.
5. Present tied ranked candidates with expected fit, TTFT, throughput, quality, context coverage, uncertainty, and cited evidence.
6. Run the selected model locally at the intended context/concurrency to verify fit and responsiveness before calling it supported.

## Mac-purchase workflow

1. Ask for supported workloads, models/quality target, context, concurrency, image use, budget, portability/form factor, and expected useful life.
2. Match only currently cataloged retail Mac configurations with verified compatible reference results; do not extrapolate an untested chip/memory tier as if measured.
3. Present **Minimum** (meets hard evidence/fit floors), **Recommended** (meets floors with declared headroom), and **Upgrade benefit** (specific measured capability gained by the next configuration).
4. Show unified-memory reserve, estimated and verified fit, maximum valid context, responsiveness, quality evidence, thermal/run conditions, uncertainty, and current price/source date separately.
5. If no configuration has eligible evidence, say so and offer a verification plan rather than a purchase recommendation.

Prices and retail configurations change and must be refreshed from authoritative sources before display. Affiliate relationships, vendor payments, loaned/pre-release hardware, sponsorship, and engineering assistance are disclosed beside affected evidence but cannot alter eligibility, rank order, thresholds, uncertainty, or wording. Commercial partners receive no unpublished ranking control.

## Allowed recommendation language

The product may say that a candidate met named measured thresholds on a named configuration, ranked ahead on specified evidence, fit during an identified verification run, or offers a measured upgrade benefit. It must include uncertainty and caveats. It may not say “best” without scope, claim universal fit/performance, infer quality from size alone, conceal missing evidence, mix tuned and neutral profiles, or convert a paid relationship into a performance claim.

Every recommendation view and report links to its supporting measurements, raw valid/excluded evidence, methodology profile, result/bundle identity, catalog versions, and [general limitations](limitations.md). Corrections preserve prior published identity and issue a new evidence/recommendation version with the reason.

[← Product Requirements](product-requirements.md) · [Back to README](../README.md) · [Methodology Contract →](methodology-contract.md)
