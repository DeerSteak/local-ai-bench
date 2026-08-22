[← Documentation index](README.md) · [Model catalog](catalogs.md) · [Workloads](workloads.md)

# Version 6 model catalog audit

This is the precommitted decision contract and working report for Milestone 9. It lands before candidate benchmark results are interpreted and before any incumbent is removed, moved, or replaced. Candidate inclusion is not the default outcome: the audit prefers the smallest lineup that preserves measurable capability and architecture coverage.

## Decision order

Every model first passes the hard gates below. Models that pass are compared only against incumbents serving the same measurable role. The audit does not combine unrelated properties into a single weighted score; each keep, replace, move, legacy-only, defer, or reject decision states which gate or role evidence determined it.

The home-lab LLM baseline is 4-bit weights on both engines: prefer `Q4_K_M` for llama.cpp and a vLLM-supported W4 format such as AWQ, GPTQ, compressed-tensors W4A16, or BitsAndBytes NF4. A different Q4 variant is acceptable when no maintained `Q4_K_M` exists, but BF16 upstream weights and MLX checkpoints do not satisfy the vLLM artifact requirement. Quantization-format support still varies by accelerator, so source readiness must be followed by the managed-runtime compatibility screen. Apache-2.0 and the reviewed [OpenMDW 1.1](https://openmdw.ai/license/1-1/) license are accepted by this audit for local evaluation and redistribution by reference; OpenMDW redistribution of model materials requires retaining its agreement and applicable notices.

1. **Stable identity and provenance:** an exact upstream repository, pinned revision, artifact filename, architecture/configuration identity, maintained Q4 GGUF source, and provenance-linked 4-bit vLLM source must be available for an LLM. Community conversions require traceable source-model provenance.
2. **License and access:** the model card and authoritative license must permit redistribution by reference and the intended local evaluation. Gated access, acceptable-use terms, and commercial-use limitations are recorded rather than inferred.
3. **Shipped-runtime compatibility:** the exact artifact must work through current project-managed llama.cpp and, for an LLM or embedding advertised for vLLM, the current project-managed vLLM environment. An unreleased runtime, local engine patch, or undocumented custom code blocks default-catalog admission.
4. **Lifecycle correctness:** setup discovery, download identity, artifact completeness, load, unload, deterministic completion, chat formatting, cancellation, recovery, and applicable tool calling must work without manual repair.
5. **Context validity:** 2K and the role-relevant deeper context must complete without exceeding the model's declared context or silently changing the benchmark methodology.
6. **Resource coverage:** the artifact must fit a declared tier and preserve a useful path across the suite's supported hardware range. Total model and dependency size, measured peak memory, and representative run time are counted.
7. **Measurable role:** an addition must fill a capability or architecture role exercised by the current workload suite, or replace an incumbent in that role. Novelty, popularity, or vendor benchmark claims do not establish a role.
8. **Evidence quality:** close performance claims require compatible repeated-trial verdicts. A one-off speed, quality, or memory delta may motivate more testing but cannot decide replacement.

## Retirement and replacement rules

- **Keep** an incumbent that passes the hard gates and retains a distinct measurable role without a clearly better qualified replacement.
- **Replace** only within the same role when the candidate passes every hard gate and the recorded evidence justifies the change without a material regression in another required dimension.
- **Move tier** only when verified total parameter count crosses the documented cumulative-tier boundary; active MoE parameters do not determine tier.
- **Legacy-only** preserves labels, colors, tier metadata, and result lookup for old evidence after a model is removed from the active catalog.
- **Defer** a candidate whose identity is sound but whose runtime, artifact, license, quality measure, or real-hardware evidence is not ready.
- **Reject** a candidate that fails a hard gate or merely duplicates an incumbent without a measurable role.

## Incumbent inventory

All decisions begin as pending. Architecture, license, context, and source claims will be filled from pinned authoritative sources before compatibility testing.

The tracked register at `scripts/release/model_catalog_incumbents.json` assigns every active entry an exact upstream identity and measurable role. `bench-env/bin/python -m scripts.release.model_catalog_inventory` joins that register to the live catalog and emits a machine-readable inventory containing the exact llama.cpp, vLLM, or ComfyUI artifacts, tier, parameter count, declared download size, gating state, and `pending_evidence` decision. It hard-fails if either side gains, loses, or duplicates an entry, so the table below cannot silently drift away from what setup ships.

| Family/tier | Incumbent | Current measurable role | Initial decision |
| --- | --- | --- | --- |
| LLM/xsmall | Gemma 3 1B | Smallest low-memory llama.cpp baseline and qualification model | Pending audit |
| LLM/xsmall | Granite 4.1 3B | Smallest tool-capable cross-engine qualification baseline | Pending audit |
| LLM/xsmall | Qwen3.5 4B | Compact Qwen-family general instruction model | Pending audit |
| LLM/small | Granite 4.1 8B | IBM-family small model with explicit vLLM tool parsing | Pending audit |
| LLM/small | Qwen3.5 9B | Mid-small Qwen-family general instruction model | Pending audit |
| LLM/small | Gemma 4 12B | Upper-small Gemma-family model with explicit vLLM tool parsing | Pending audit |
| LLM/medium | Gemma 3 27B | Dense medium-tier baseline | Pending audit |
| LLM/medium | Nemotron Cascade 2 30B-A3B | NVIDIA hybrid sparse medium-tier coverage | Pending audit |
| LLM/medium | Qwen3.6 35B-A3B | Qwen sparse medium-tier coverage | Pending audit |
| LLM/large | Llama 3.3 70B | Dense large-tier baseline | Pending audit |
| LLM/large | Qwen3-Coder-Next 80B-A3B | Code-focused sparse large-tier coverage | Pending audit |
| LLM/large | Nemotron 3 Super 120B-A12B | Highest-capacity hybrid sparse coverage | Pending audit |
| Embedding | Nomic Embed Text v1.5 | Compact embedding throughput and retrieval baseline | Pending audit |
| Embedding | MixedBread Embed Large v1 | Larger embedding-capacity comparison | Pending audit |
| Image/xsmall | Stable Diffusion 1.5 | Smallest image and qualification pipeline | Pending audit |
| Image/small | SDXL 1.0 | Widely supported higher-resolution baseline | Pending audit |
| Image/medium | Stable Diffusion 3.5 Large | Medium-tier modern diffusion coverage | Pending audit |
| Image/large | FLUX.1-dev | Large-tier FLUX pipeline baseline | Pending audit |
| Image/large | FLUX.2-dev | Highest-memory current FLUX pipeline | Pending audit |

The metadata-only incumbent source audit was refreshed on August 21, 2026 with `bench-env/bin/python -m scripts.release.model_catalog_incumbent_audit --output docs/model-catalog-incumbent-source-audit-v6.json`; its complete pinned output is [model-catalog-incumbent-source-audit-v6.json](model-catalog-incumbent-source-audit-v6.json). Eight entries are source-ready under the audit's strict public Apache-2.0 gate. Eleven require review rather than automatic retirement: Gemma and Llama entries require accepted gated terms, NVIDIA entries carry custom licenses, current image baselines carry OpenRAIL or other gated terms, Qwen3-Coder-Next's selected GGUF card does not name the exact upstream repository, and the gated image repositories could not be configuration-inspected anonymously. These are explicit decision inputs; a familiar incumbent is not grandfathered past the same access, license, provenance, and artifact checks applied to candidates.

## Candidate register

Repository names are hypotheses from the Version 6 plan until the source audit records an exact revision, artifact, license, and compatibility result.

| Family | Candidate | Role under evaluation | Overlapping incumbent |
| --- | --- | --- | --- |
| LLM | Qwen 3.8 27B | Contemporary dense medium-tier Qwen coverage | Gemma 3 27B |
| LLM | Muse Glimmer 30B | Alternative medium-tier architecture and instruction behavior | Medium lineup |
| LLM | Nemotron 3.5 Lightning 30B-A3B | Newer NVIDIA sparse medium-tier coverage | Nemotron Cascade 2 |
| LLM | Gemma 4 26B-A4B | Newer sparse Gemma medium-tier coverage | Gemma 3 27B |
| Embedding | EmbeddingGemma 300M | Very small multilingual embedding baseline | Nomic Embed Text |
| Embedding | Qwen3 Embedding 0.6B | Compact instruction-aware embedding coverage | Nomic Embed Text |
| Embedding | Qwen3 Embedding 4B | Larger instruction-aware embedding coverage | MixedBread Embed Large |
| Image | FLUX.2 Klein 4B | Smaller FLUX.2 pipeline | FLUX.1-dev and FLUX.2-dev |
| Image | Z-Image Turbo | Fast image-generation pipeline | SDXL and current large image models |

## Source audit snapshot

The metadata-only source audit was refreshed on August 22, 2026 with `bench-env/bin/python -m scripts.release.model_catalog_audit --output docs/model-catalog-source-audit-v6.json`. It resolves the current repository commit, access state, declared license, architecture, context ceiling, chat-template source, publisher generation config, exact indexed weights, support files, preferred standalone GGUF, quantization method and bit width, source-model provenance, download/like signals, and exact ComfyUI pipeline file digests without downloading weights. The command exits nonzero while any candidate has an unresolved hard gate; the complete machine-readable snapshot is [model-catalog-source-audit-v6.json](model-catalog-source-audit-v6.json).

| Candidate | Upstream identity/configuration | llama.cpp or ComfyUI artifact | vLLM artifact | Source-audit status |
| --- | --- | --- | --- | --- |
| Qwen 3.8 27B | `1d4bf0f2ff60`; `qwen3_5`, 262,144 context | `UD-Q4_K_M` (15.33 GiB) | cyankiwi compressed-tensors W4A16 (19.57 GiB; 384,748 downloads) | Source ready |
| Muse Glimmer 30B | `a4e59da52a7b`; `muse_glimmer`, 131,072 context | `Q4_K_M` (15.61 GiB) | Unsloth BitsAndBytes NF4 (20.68 GiB; 16,407 downloads) | Source ready; actual vLLM platform support remains a screen result because BitsAndBytes is NVIDIA-only in vLLM's current hardware table |
| NVIDIA Nemotron 3.5 Lightning 30B-A3B | `d468880b6ad3`; `nemotron_h`, 262,144 context | Unsloth `UD-Q4_K_M` (23.53 GiB) | Local Axiom AWQ 4-bit (16.83 GiB; 826 downloads) | Source ready; the AWQ repository's exact quantization manifest supplies its instruction-model source revision |
| Gemma 4 26B-A4B | `4d7ae4984b7d`; `gemma4`, 262,144 context | Unsloth `UD-Q4_K_M` (15.78 GiB) | cyankiwi compressed-tensors W4A16 (16.01 GiB; 2,671,723 downloads) | Source ready |
| EmbeddingGemma 300M | `57c266a740f5`; configuration inaccessible without approved credentials | `IQ4_XS` (0.28 GiB) | Upstream safetensors, inaccessible without approval | Blocked: manual upstream access and Gemma license review |
| Qwen3 Embedding 0.6B | `97b0c614be4d`; `qwen3`, 32,768 context | `Q8_0` (0.60 GiB) | Upstream safetensors | Source ready: exact same-publisher GGUF variant explicitly identifies the embedding series and selected 0.6B model |
| Qwen3 Embedding 4B | `5cf2132abc99`; `qwen3`, 40,960 context | `Q4_K_M` (2.33 GiB) | Upstream safetensors | Source ready: exact same-publisher GGUF variant explicitly identifies the embedding series and selected 4B model |
| FLUX.2 Klein 4B | `e7b7dc27f91d`; `Flux2KleinPipeline` | Pinned 15.02 GiB ComfyUI diffusion, Qwen text encoder, and VAE set | — | Blocked: the authoritative VAE dependency carries a custom license requiring review |
| Z-Image Turbo | `f332072aa78b`; `ZImagePipeline` | Pinned 19.27 GiB ComfyUI diffusion, Qwen text encoder, and VAE set | — | Source ready |

This is source readiness only, not engine compatibility or catalog acceptance. The `lmstudio-community/Qwen3.8-27B-MLX-4bit` and Kagandi Nemotron MLX repositories are valid MLX artifacts but are not vLLM checkpoints; Qwen therefore uses a compressed-tensors W4A16 source and Lightning uses the provenance-pinned Local Axiom AWQ source. Nemotron Nano 9B v2 was removed from the candidate register because Lightning is its newer replacement, so no Nano hardware screen will be requested. The audit also excludes MTP speculative-head GGUFs from standalone model selection; they are optional dependencies rather than baseline weights.

## Compatibility screen

The screen uses the shipped setup and runtime paths and records engine, runtime version, artifact digest, model revision, hardware profile, and every outcome. Preview and validation tooling may be automated, but a real setup or benchmark is launched only by the maintainer on the selected hardware.

`bench-env/bin/python -m scripts.release.model_catalog_screen --list` reports which pinned candidates may enter the screen and the exact hard gates blocking the rest. `--candidate ID --engine llamacpp|vllm` prints the side-effect-free import and normal-benchmark plan. Adding `--execute` performs the exact-revision import, runs the real workload with one warmup and one measured pass, pauses and interrupts after the first durable case, resumes through the normal recovery executor, and writes the result, journal, initial/resume logs, and `screen-report.json` under `results/catalog-audit/`. The report uses portable relative paths and records the byte size and SHA-256 of each required evidence file, so a copied directory can be verified without its original absolute path. LLM screens require passing runtime formatting preflight, exact sampler identity, and valid 2K/deep evidence on both request paths. Z-Image Turbo uses [Comfy-Org's core-node workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_z_image_turbo.json) with pinned diffusion, Qwen text-encoder, and VAE revisions; its gate also requires every measured resolution's retained PNG with size and SHA-256 in the report. Missing evidence is a failure. FLUX.2 Klein remains blocked pending license review and a fixed accepted workflow.

`bash run_model_catalog_screens.sh` executes the complete source-ready LLM matrix unattended: Qwen 3.8, Muse Glimmer, Nemotron 3.5 Lightning, and Gemma 4 across llama.cpp and vLLM. It continues after a failed cell, prints a final pass/fail summary, and exits nonzero if any cell failed; rerunning reuses exact imports and completed valid results through the screen runner. Use `--engine llamacpp` or `--engine vllm` when a host provides only one managed engine, `--list` for a side-effect-free matrix preview, and `--output-root DIR` to place evidence elsewhere.

1. Resolve the pinned source revision and verify every expected file before loading.
2. Discover the model through normal setup inventory and confirm its catalog identity is unchanged across restart.
3. Load and unload once, then repeat after cancellation to prove cleanup.
4. Run deterministic completion and chat-format probes at 2K and the role-relevant deeper context.
5. Verify the embedded or source-pinned chat template and reject raw template markup or empty output.
6. Exercise applicable tool calling, embeddings, or the fixed ComfyUI workflow.
7. Interrupt one measured case, inspect recovery, resume, and confirm completed cases are not repeated.
8. Record peak memory, setup/download size, elapsed screen time, errors, warnings, and unsupported states.

## Comparison evidence

LLM decisions use accuracy by category, single-shot prefill and decode, conversation growth, tool and chat concurrency, memory headroom, sustained behavior, supported context, artifact size, and representative run time. Close performance differences require the repeated-trial evaluator; an inconclusive verdict remains inconclusive.

Embedding decisions use runtime coverage, dimensions, context, multilingual and instruction-aware behavior, throughput, memory, artifact size, license/access friction, and retrieval quality. The current embedding workload does not establish retrieval quality, so no embedding replacement can be justified solely from its throughput result.

Image decisions use complete dependency size, peak memory, supported resolution, fixed workflow settings, latency, lifecycle reliability, license/access friction, and prompt/image quality. Speed alone cannot justify replacement, and quality review must use the same prompts, resolutions, seeds, and workflow identity.

## Sampling profiles

The comparable `deterministic-baseline-v1` sampler is implemented under methodology profile `neutral-v2`. It pins temperature zero; neutral top-k/top-p/min-p, presence penalty, frequency penalty, repetition penalty, seed, and logit bias across llama.cpp and vLLM; neutralizes llama.cpp-specific active samplers; and launches managed vLLM with repository generation defaults disabled. The semantic and engine-resolved controls are stored in run-plan schema 5 methodology identity before candidate performance runs begin.

Publisher-recommended sampling is optional through a schema-1 `--publisher-sampling-profile` file, source-pinned to an exact repository revision, fully resolved without engine defaults, and assigned `publisher-v1` methodology. The compatibility screen can generate and use this file with `--publisher-sampling`; publisher-profile results never pool with or replace deterministic-baseline evidence.

## Required report outputs

For every incumbent and candidate, the completed audit will record exact source and artifact identities, license, architecture, dense/total/active parameters where applicable, context, engine compatibility, lifecycle screen, role decision, evidence references, and unresolved gaps. It will also total the active catalog's download size and estimate representative run time before any accepted catalog change is implemented.

`bench-env/bin/python -m scripts.release.model_catalog_readiness` emits the current machine-readable readiness report without launching setup or a workload; add `--check` to fail until every source-ready LLM and embedding has one valid llama.cpp and one valid vLLM screen and every source-ready image candidate has one valid ComfyUI screen. The gate rehashes every portable evidence artifact, revalidates the result's lifecycle contract, requires exact source, runtime, and hardware identity, rejects duplicate or orphaned reports, lists unresolved incumbent source reviews, and totals exact active-catalog artifact bytes separately for llama.cpp, vLLM, and ComfyUI. A pre-rubric run or an unbound result file is not accepted as decision evidence.

The pre-hardware snapshot at [model-catalog-readiness-v6.json](model-catalog-readiness-v6.json) records thirteen required candidate screens still outstanding and eleven incumbent source/license reviews. The exact selected weight totals for the current catalog are 244.86 GiB for llama.cpp, 245.69 GiB for vLLM, and 107.95 GiB for ComfyUI; these are separate complete-catalog stores, not a claim that setup downloads every engine or every model by default. Proposed-lineup cost and run-time deltas remain unavailable until evidence supports actual keep/replace decisions.
