[← Documentation index](README.md) · [Model catalog](catalogs.md) · [Workloads](workloads.md)

# Version 6 model catalog audit

This is the precommitted decision contract and working report for Milestone 9. It lands before candidate benchmark results are interpreted and before any incumbent is removed, moved, or replaced. Candidate inclusion is not the default outcome: the audit prefers the smallest lineup that preserves measurable capability and architecture coverage.

## Decision order

Every model first passes the hard gates below. Models that pass are compared only against incumbents serving the same measurable role. The audit does not combine unrelated properties into a single weighted score; each keep, replace, move, legacy-only, defer, or reject decision states which gate or role evidence determined it.

1. **Stable identity and provenance:** an exact upstream repository, pinned revision, artifact filename, architecture/configuration identity, and maintained GGUF source must be available. Community conversions require traceable source-model provenance.
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

## Candidate register

Repository names are hypotheses from the Version 6 plan until the source audit records an exact revision, artifact, license, and compatibility result.

| Family | Candidate | Role under evaluation | Overlapping incumbent |
| --- | --- | --- | --- |
| LLM | Qwen 3.8 27B | Contemporary dense medium-tier Qwen coverage | Gemma 3 27B |
| LLM | Muse Glimmer 30B | Alternative medium-tier architecture and instruction behavior | Medium lineup |
| LLM | Nemotron 3.5 Lightning 30B-A3B | Newer NVIDIA sparse medium-tier coverage | Nemotron Cascade 2 |
| LLM | Gemma 4 26B-A4B | Newer sparse Gemma medium-tier coverage | Gemma 3 27B |
| LLM | Nemotron Nano 9B v2 | NVIDIA small-tier coverage | Granite 4.1 8B and Qwen3.5 9B |
| Embedding | EmbeddingGemma 300M | Very small multilingual embedding baseline | Nomic Embed Text |
| Embedding | Qwen3 Embedding 0.6B | Compact instruction-aware embedding coverage | Nomic Embed Text |
| Embedding | Qwen3 Embedding 4B | Larger instruction-aware embedding coverage | MixedBread Embed Large |
| Image | FLUX.2 Klein 4B | Smaller FLUX.2 pipeline | FLUX.1-dev and FLUX.2-dev |
| Image | Z-Image Turbo | Fast image-generation pipeline | SDXL and current large image models |

## Source audit snapshot

The metadata-only source audit was refreshed on August 21, 2026 with `bench-env/bin/python -m scripts.release.model_catalog_audit --output docs/model-catalog-source-audit-v6.json`. It resolves the current repository commit, access state, declared license, architecture, context ceiling, chat-template source, publisher generation config, exact indexed safetensors set, support files, preferred standalone GGUF, size, GGUF base-model provenance, and exact ComfyUI pipeline file digests without downloading weights. The command exits nonzero while any candidate has an unresolved hard gate; the complete machine-readable snapshot is [model-catalog-source-audit-v6.json](model-catalog-source-audit-v6.json).

| Candidate | Upstream identity | Configuration | Selected local artifact | Source-audit status |
| --- | --- | --- | --- | --- |
| Qwen 3.8 27B | `1d4bf0f2ff60` | `qwen3_5`, 262,144 context, standalone chat template; publisher `temp=1`, `top-k=20`, `top-p=.95` | `Qwen3.8-27B-UD-Q4_K_M.gguf` (15.33 GiB) | Source ready |
| Muse Glimmer 30B | `a4e59da52a7b` | `muse_glimmer`, 131,072 context, standalone chat template; publisher `temp=1`, `top-k=64`, `top-p=.95` | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (15.61 GiB) | Source ready |
| NVIDIA Nemotron 3.5 Lightning 30B-A3B | `434456c9a675` | `nemotron_h`, 262,144 context, no chat template or publisher sampler | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (17.60 GiB) | Blocked: base repository has no chat template, custom license review, and base/instruct provenance mismatch |
| Gemma 4 26B-A4B | `4d7ae4984b7d` | `gemma4`, 262,144 context, standalone chat template; publisher `temp=1`, `top-k=64`, `top-p=.95` | `gemma-4-26B-A4B-it-Q4_0.gguf` (13.61 GiB) | Source ready |
| NVIDIA Nemotron Nano 9B v2 | `6533e8de2c68` | `nemotron_h`, 131,072 context, tokenizer chat template; custom code | `nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf` (6.08 GiB) | Blocked: custom code and license review plus undeclared GGUF license |
| EmbeddingGemma 300M | `57c266a740f5` | Configuration inaccessible without approved credentials | `embeddinggemma-300m-iq4_xs.gguf` (0.28 GiB) | Blocked: manual upstream access and Gemma license review |
| Qwen3 Embedding 0.6B | `97b0c614be4d` | `qwen3`, 32,768 context | `Qwen3-Embedding-0.6B-Q8_0.gguf` (0.60 GiB) | Blocked: GGUF card identifies the generic base model, not the selected embedding upstream |
| Qwen3 Embedding 4B | `5cf2132abc99` | `qwen3`, 40,960 context | `Qwen3-Embedding-4B-Q4_K_M.gguf` (2.33 GiB) | Blocked: GGUF card identifies the generic base model, not the selected embedding upstream |
| FLUX.2 Klein 4B | `e7b7dc27f91d` | `Flux2KleinPipeline` | Pinned 15.02 GiB ComfyUI diffusion, Qwen text encoder, and VAE set | Blocked: the authoritative VAE dependency carries a custom license requiring review |
| Z-Image Turbo | `f332072aa78b` | `ZImagePipeline` | Pinned 19.27 GiB ComfyUI diffusion, Qwen text encoder, and VAE set | Source ready |

This is source readiness only, not engine compatibility or catalog acceptance. In particular, the audit excludes MTP speculative-head GGUFs from standalone model selection; they are optional dependencies rather than baseline weights. The source audit exposed and fixed the shared importer incorrectly treating `mtp-*.gguf` as runnable model variants.

## Compatibility screen

The screen uses the shipped setup and runtime paths and records engine, runtime version, artifact digest, model revision, hardware profile, and every outcome. Preview and validation tooling may be automated, but a real setup or benchmark is launched only by the maintainer on the selected hardware.

`bench-env/bin/python -m scripts.release.model_catalog_screen --list` reports which pinned candidates may enter the screen and the exact hard gates blocking the rest. `--candidate ID --engine llamacpp|vllm` prints the side-effect-free import and normal-benchmark plan. Adding `--execute` performs the pinned custom-model import, runs the real `llm` and conversation workloads through 32K with one warmup and one measured pass, pauses and interrupts after the first durable case, resumes through the normal recovery executor, and writes the result, journal, initial/resume logs, and `screen-report.json` under `results/catalog-audit/`. The final gate requires a complete recovered run, passing runtime formatting preflight, exact `neutral-v2` sampler identity, and valid 2K/deep evidence on both request paths; missing evidence is a failure. Image candidates remain blocked from this launcher until their fixed ComfyUI workflows are implemented in the normal image workload.

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
