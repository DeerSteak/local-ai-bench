[← Back to README](../README.md)

# Workloads

Seven workload families are available: LLM generation (two test modes), image generation, embeddings, accuracy (MCQ, math, reasoning, code, and tool use), HTTP concurrency, llama-bench, and llama-bench concurrency. The last three are opt-in. Setup leaves models estimated not to fit unchecked by default; at runtime, missing models are skipped and a model-specific timeout or failure preserves completed measurements instead of aborting the entire results file.

Every "Size" figure below is the model's actual on-disk download size, rounded **up** to the next 0.1 GB (not nearest) — the same convention `setup_check.py` uses for its own disk-space check, so an estimate never undersells how much room a model actually needs.

**Contents**
- [LLM](#llm)
  - [Extra-small tier (<6B params)](#extra-small-tier-6b-params)
  - [Small tier (≤20B params)](#small-tier-20b-params)
  - [Medium tier (26–35B params)](#medium-tier-2635b-params)
  - [Large tier (70B+ params)](#large-tier-70b-params)
  - [Per-engine weights](#per-engine-weights)
  - [Dense vs. Mixture-of-Experts (MoE)](#dense-vs-mixture-of-experts-moe)
- [Image Generation](#image-generation)
- [Embeddings](#embeddings)
- [Accuracy](#accuracy)
  - [Timeouts and loop detection](#timeouts-and-loop-detection)
  - [Math](#math)
  - [Code](#code)
  - [Tool Use](#tool-use)
  - [Bank versioning](#bank-versioning)
- [Concurrency](#concurrency)
- [llama-bench](#llama-bench)
- [llama-bench Concurrency](#llama-bench-concurrency)

## LLM

Twelve models across four tiers (three per tier) are in the default catalog. Models that were not downloaded are skipped; a timeout or repeatable runner crash stops that model's current workload and preserves any measurements already collected.

The suite provides **two separate LLM tests**. When both are selected, it completes the single-shot stage for all models before starting the conversation stage:

- **Single-shot** — a large deterministic prompt, padded to the target size and sent on every measured run at up to five context lengths (512 / 2K / 8K / 32K / 64K, whichever the model's own context window reaches), so comparable invocations receive identical content. llama.cpp disables prompt caching on these requests, while vLLM gives each request a fresh cache salt, forcing genuine cold prefill without random prompt-content differences. This simulates dropping a large document, codebase, or transcript into a single prompt and asking one question about it.

The padded body itself (`Shared.build_prompt_for_context`) is a deterministic slice of `scripts/workloads/data/long_document.txt`, a public-domain novel, rather than one short paragraph repeated to length — real prose avoids the degenerate immediate-EOS responses some models produce when a prompt reads as an obviously repeated loop. Context and variant select a stable offset; ordinary single-shot runs use variant zero, while concurrent requests use distinct stable variants. The source document comfortably covers the largest checkpoint (65536 tokens) in one slice; wrapping only kicks in for a target beyond the document's own length.
- **Conversation** — a real multi-turn chat, measured at up to ten depths (0 / 2K / 4K / 8K / 16K / 32K / 48K / 64K / 80K / 96K, subject to the model and `--max-prompt-tokens` limits below): the model explains Plato's Allegory of the Cave, then each following turn asks for more detail on a section, growing the conversation from a blank slate. This test is expensive, so it always runs one conversation regardless of `--runs`.

Conversation commits each sampled checkpoint to the event journal before growing toward the next one. A timeout, engine crash, interruption, or JSON-export failure therefore retains every shallower completed checkpoint for that model; the child runner rebuilds the public `llm_conversation` section from those events.

By default, every completed workload case retains named idle, model-load/warmup, and measured memory windows at the qualified 0.5-second interval; `--no-memory-telemetry` disables collection. Accuracy uses one measured sub-window per question, images use one per resolution, and embedding uses one for its repeated document calls. Native tools that reveal a case only when its result arrives switch from load or idle to measured at delivery, so their measured window is a boundary sample rather than the tool's hidden internal interval; the source measurements remain authoritative for timing. The sampler records normalized host RAM, benchmark process-tree RSS, and accelerator memory where readable; missing channels remain unknown or unsupported rather than zero. ROCm accelerator counters are used only for confirmed discrete GPUs because an integrated GPU's small firmware VRAM aperture is not its usable unified-memory pool. Peak, mean, final, sample count, interval, failed-sample count, source, and headroom are persisted with the case, while load-window samples never enter request timing metrics. Headroom compares accelerator use with the usable accelerator ceiling on discrete-memory systems, and benchmark process-tree RSS with the usable model ceiling on unified-memory systems; total host RAM remains separate environmental evidence so background OS activity is not counted as model usage. Qualified telemetry-on and telemetry-off performance is comparable when every other methodology setting matches.

Before any LLM measurement, preflight verifies local weight presence/readability, chat-template metadata, the plan's maximum requested context against model and engine limits, and per-model tool-call support. Missing template or excess-context findings are recorded quality warnings; `--force-all` acknowledges those warnings but never bypasses unreadable/incomplete weights, a failed runtime load/format round-trip, or failure to restore clean model state. A hard failure removes only that model from the immutable execution plan, while unsupported tool calls retain the model for other workloads and produce the existing `tool_calls_unsupported` skip in tool workloads. Passing checks are retained alongside failures under top-level `preflight`, and probe text/timing never enters a measured section.

Native llama-bench, llama-batched-bench, and vLLM bench hide model loading and expose a case only after its work completes. Their `measured:*:includes-load` window therefore starts before subprocess execution and conservatively includes the hidden load for the first case; streamed native rows begin the next measured segment immediately after the prior row is committed. This captures actual subprocess memory without inventing a load boundary, but these native peaks must not be interpreted as request-only occupancy.

Both tests cap their context lengths to each model's real maximum, read from the downloaded GGUF metadata, and `--max-prompt-tokens` further caps their deepest measured prompt size. The conversation plan otherwise targets at most 128K and samples through 96K, leaving up to 4K of KV-cache headroom where the model's native ceiling allows it. A model whose native ceiling or configured prompt cap is lower gets a correspondingly shorter plan; the `0K` opening checkpoint remains when a cap is below 2K. llama.cpp's auto-fit (`-ngl auto`, see [Engines](engines.md)) spills weights (and the KV-cache for CPU-resident layers) to system memory instead of OOMing, and the within-conversation slow-TPS early exit (below) stops a run that's too slow to be worth continuing — so a small model isn't given a shorter plan just for being small.

Single-shot's server-side `num_ctx` at each checkpoint is the padded prompt size plus the 512-token generation budget (`LLMPrefillBenchmark.prefill_server_ctx`), clamped to the model's real max — the same "headroom beyond the measured size" principle as the conversation plan above, and as [Concurrency](#concurrency)'s per-slot context. Without that headroom, a checkpoint at or near the model's native ceiling leaves no room for the model to actually generate a response: the server hits the context limit almost immediately, and TPS at that depth reads as ~0. This can still happen at the single checkpoint that lands exactly on a model's real max — a warning is logged when it does, since there's no more room to give.

The conversation test grows toward each checkpoint in bounded steps rather than one giant turn or many small fixed ones: it takes large steps (up to 4096 tokens) while more than 8K tokens from the target, then switches to fine steps (up to 1024 tokens) once close, so the turn that actually lands on a checkpoint doesn't overshoot it by much. Growth also stops at 99.5% of each checkpoint's target rather than the exact value — TTFT at a given depth isn't sensitive to a sub-1% difference, and it's far smaller than this test's own run-to-run noise. This roughly halves the turns needed to reach 96K versus growing in small fixed steps throughout.

Each growth turn's step size is computed from an *effective* depth — the last known ground-truth prompt size (`prompt_eval_count`) plus that turn's own response length — not the ground-truth number alone. A turn's response only shows up in `prompt_eval_count` once the *next* turn's prompt is evaluated, so using the ground-truth number by itself understates how much context is actually occupied by the time the next turn's request goes out. That gap is usually harmless when there's slack between the checkpoint and `num_ctx`, but at a checkpoint sitting exactly at a model's real ceiling (`num_ctx == model_max`, zero headroom) it's enough to push a request over the hard context limit.

Prompt padding is approximate — a character count at roughly 4 chars per token — so the real prompt can land slightly either side of the target. llama.cpp absorbs an overshoot by generating a few tokens fewer; vLLM rejects the request, so its server allocation carries a small tolerance (see [Engines](engines.md#vllmengine)). The measured depth is unchanged either way.

These two tests measure genuinely different things, and their TTFT numbers are **not** comparable at face value — see [What the charts mean](dashboard.md#what-the-charts-mean) for why the conversation test's TTFT is typically far lower than the single-shot test's at the same nominal context length.

The single-shot slow-model check applies at its first checkpoint (512 tokens): below 15 tok/s, deeper single-shot contexts are skipped unless `--force-all` is set. When single-shot and conversation run together, the conversation pre-flight also excludes a model with no usable single-shot data, a repeatable runner crash, a first-checkpoint timeout, or that first-checkpoint slow marker. A timeout only at a deeper single-shot context does not by itself exclude conversation. Running `--tests conv` alone has no single-shot pre-flight data, so it attempts every selected model.

When an explicitly validated journal recovery is prepared, single-shot skips contexts already complete, records a new numbered attempt for an interrupted/failed/invalid/timed-out context, and retains every prior attempt as raw evidence. Compatible aggregates use only the latest attempt so a retry does not silently mix old and new samples. Result History exposes this through **Inspect Recovery**, **Resume**, and **Retry Cases**.

Conversation recovery must rebuild a model's cache from a fresh conversation because server KV state is not durable evidence. Reached checkpoints already complete are used only to reconstruct that cache and are not recorded again; the first incomplete checkpoint receives the new attempt. A reconstructed checkpoint cannot newly trigger the slow-model cutoff and prevent recovery from reaching the pending work.

Separately, *within* the conversation test itself: if the decode speed at any history depth drops below the slow-model cutoff, the conversation exits early and records results to that point. Pass `--force-all` to ignore this cutoff and always run every context length (see [CLI Reference](cli-reference.md)).

### Extra-small tier (<6B params)

| Model | llama.cpp Tag | vLLM Tag | llama.cpp Size | Architecture |
|---|---|---|---|---|
| Gemma 3 1B | `gemma3:1b-it-q4_K_M` | `gaunernst/gemma-3-1b-it-int4-awq` | ~0.8 GB | Dense |
| Granite 4.1 3B 4-Bit Quantization | `granite4.1:3b-q4_K_M` | `cyankiwi/granite-4.1-3b-AWQ-INT4` | ~2.1 GB | Dense |
| Qwen3.5 4B 4-Bit Quantization | `qwen3.5:4b-q4_K_M` | `cyankiwi/Qwen3.5-4B-AWQ-4bit` | ~3.1 GB | Dense |

The extra-small tier deliberately spans three roles. Gemma 3 1B is the ultra-light speed floor, showing what the suite costs on the smallest practical general model. Granite 4.1 3B is the compact structured-execution and tool-calling specialist. Qwen3.5 4B is the more capable general executor, trading some speed for stronger instruction following, reasoning, coding, and tool use. This makes the tier useful for evaluating fast worker models rather than filling it with three interchangeable general chat baselines.

### Small tier (≤20B params)

| Model | llama.cpp Tag | vLLM Tag | llama.cpp Size | Architecture |
|---|---|---|---|---|
| Granite 4.1 8B 4-Bit Quantization | `granite4.1:8b-q4_K_M` | `cyankiwi/granite-4.1-8b-AWQ-INT4` | ~5.4 GB | Dense |
| Qwen3.5 9B 4-Bit Quantization | `qwen3.5:9b-q4_K_M` | `QuantTrio/Qwen3.5-9B-AWQ` | ~6.2 GB | Dense |
| Gemma 4 12B 4-Bit Quantization | `gemma4:12b-it-q4_K_M` | `mattbucci/gemma-4-12B-AWQ` | ~7.7 GB | Dense |

The small tier scales the same worker-model experiment upward. Granite 4.1 8B measures how the Granite tool/structured-execution specialization improves with more capacity, while Qwen3.5 9B is the corresponding stronger general executor. Gemma 4 12B anchors the top of the tier and pairs with Gemma 4 26B-A4B in the medium tier, while Qwen3.5 9B pairs with Qwen 3.8 27B. Together, these pairs expose whether extra capacity materially improves execution reliability, while Gemma 1B and Gemma 4 12B bracket the two lower tiers with a speed floor and capability ceiling.

### Medium tier (26–35B params)

| Model | llama.cpp Tag | vLLM Tag | llama.cpp Size | Architecture |
|---|---|---|---|---|
| Gemma 4 26B-A4B 4-Bit Quantization | `gemma4:26b-a4b-it-ud-q4_K_M` | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | ~16.9 GB | MoE — 4B active of 26B total |
| Qwen 3.8 27B 4-Bit Quantization | `qwen3.8:27b-ud-q4_K_M` | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | ~16.5 GB | Dense |
| Nemotron 3.5 Lightning 30B-A3B | `nemotron3.5-lightning:30b-a3b-ud-q4_K_M` | `Local-Axiom-AI/Nemotron-3.5-Lightning-awq` | ~25.3 GB | Hybrid Mamba MoE — 3B active of 30B total |

The medium tier contrasts three current architectures with similar total parameter counts but different execution costs and vendor roles. Qwen 3.8 27B is the dense general-purpose baseline. Gemma 4 26B-A4B supplies sparse Gemma-family coverage with 4B active parameters, while Nemotron 3.5 Lightning supplies NVIDIA's hybrid long-context architecture with 3B active parameters. These replace Gemma 3 27B, Qwen3.6 35B-A3B, and Nemotron Cascade 2 without increasing the active catalog.

### Large tier (70B+ params)

| Model | llama.cpp Tag | vLLM Tag | llama.cpp Size | Architecture |
|---|---|---|---|---|
| Llama 3.3 70B 4-Bit Quantization | `llama3.3:70b-instruct-q4_K_M` | `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` | ~39.7 GB | Dense |
| Qwen3-Coder-Next 80B-A3B 4-Bit Quantization | `qwen3-coder-next:80b-a3b-q4_K_M` | `bullpoint/Qwen3-Coder-Next-AWQ-4bit` | ~48.4 GB | Hybrid-attention MoE — 3B active of 80B total |
| Nemotron 3 Super 120B | `nemotron-3-super:120b` | `cyankiwi/NVIDIA-Nemotron-3-Super-120B-A12B-AWQ-4bit` | ~87.0 GB | Hybrid Mamba-Transformer MoE — 12B active of 120B total |

The large tier assigns each slot a distinct role. Llama 3.3 70B is the dense general-purpose baseline. Qwen3-Coder-Next is the long-horizon execution specialist, trained for coding agents, complex tool use, and recovery after failed actions. Nemotron 3 Super is the broader agentic reasoning model and represents the planner/verifier role for long-context, multi-step workflows. This combination measures a dense generalist, a fast sparse tool specialist, and a more capable sparse planner instead of using two large Llama-family models with overlapping general-purpose roles.

The tier is intentionally limited to one model per role and avoids spending multiple slots on overlapping general-purpose models. Its capabilities also align with what the suite can measure: the code and tool accuracy tests exercise structured execution, while conversation and concurrency exercise sustained context and multi-request behavior. Qwen3-Coder-Next therefore represents carrying out long sequences of tool-heavy steps, while Nemotron 3 Super represents planning and verifying the broader workflow. Their different sparse architectures add an inference comparison that model size alone would not expose.

### Tool calling across engines

llama.cpp parses tool calls from the model's own chat template, so every catalog model can be measured. vLLM cannot: it returns **no** `tool_calls` at all unless the server is started with `--enable-auto-tool-choice --tool-call-parser <name>`, and the parser is model-specific. A model with no parser configured is therefore **skipped** with `skip_reason: "tool_calls_unsupported"` rather than measured — scoring unparsed output as wrong answers would publish 0% for a model that was never actually tested.

Preflight records this tool capability before execution as a workload-scoped check. It does not turn missing tool support into a whole-model failure; non-tool measurements remain eligible, and the tool bank owns the durable per-model skip reason.

`models.py` carries `vllm_tool_parser` per entry, set only where vLLM documents a parser for that family: `granite4` (Granite 4.1), `gemma4` (Gemma 4), `llama3_json` (Llama 3.3), and `qwen3_coder` (Qwen3-Coder-Next). vLLM documents none for Nemotron, and the correct choice for Qwen3.5/3.8 is unconfirmed — those are left unset until a real run settles them. The value is not a guess to be filled in casually: a *valid but wrong* parser fails silently, producing unparsed calls that score as wrong answers, which is exactly the outcome the skip exists to prevent.

### Per-engine weights

The tier tables above give each model's identifier on both engines. The llama.cpp tag names a `Q4_K_M` GGUF, and that file is llama.cpp's alone — vLLM cannot use it, so each catalog entry carries a second set of weights (`vllm_repo`/`vllm_download_size` in `models.py`) that setup downloads separately into vLLM's own HuggingFace cache when vLLM is a selected engine (see [Setup](setup.md#choosing-engines)). Same model, same tier, different identifier and different file. This table adds each vLLM repo's quantization format and download size:

| Tag | vLLM weights | Format | Size |
|---|---|---|---|
| `gemma3:1b-it-q4_K_M` | `gaunernst/gemma-3-1b-it-int4-awq` | AWQ INT4 | ~1.1 GB |
| `granite4.1:3b-q4_K_M` | `cyankiwi/granite-4.1-3b-AWQ-INT4` | AWQ INT4 | ~2.4 GB |
| `qwen3.5:4b-q4_K_M` | `cyankiwi/Qwen3.5-4B-AWQ-4bit` | AWQ INT4 | ~4.1 GB |
| `granite4.1:8b-q4_K_M` | `cyankiwi/granite-4.1-8b-AWQ-INT4` | AWQ INT4 | ~5.5 GB |
| `qwen3.5:9b-q4_K_M` | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | AWQ INT4 | ~9.1 GB |
| `gemma4:12b-it-q4_K_M` | `mattbucci/gemma-4-12B-AWQ` | AWQ INT4 | ~7.8 GB |
| `gemma4:26b-a4b-it-ud-q4_K_M` | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | compressed-tensors W4A16 | ~17.2 GB |
| `qwen3.8:27b-ud-q4_K_M` | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | compressed-tensors W4A16 | ~21.0 GB |
| `nemotron3.5-lightning:30b-a3b-ud-q4_K_M` | `Local-Axiom-AI/Nemotron-3.5-Lightning-awq` | AWQ INT4 | ~18.1 GB |
| `llama3.3:70b-instruct-q4_K_M` | `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` | AWQ INT4 | ~39.8 GB |
| `qwen3-coder-next:80b-a3b-q4_K_M` | `bullpoint/Qwen3-Coder-Next-AWQ-4bit` | AWQ INT4 | ~48.3 GB |
| `nemotron-3-super:120b` | `cyankiwi/NVIDIA-Nemotron-3-Super-120B-A12B-AWQ-4bit` | AWQ INT4 | ~80.7 GB |
| `nomic-embed-text` | `nomic-ai/nomic-embed-text-v1.5` | fp16 | ~0.6 GB |
| `mxbai-embed-large` | `mixedbread-ai/mxbai-embed-large-v1` | fp16 | ~0.7 GB |

These are selected to match `Q4_K_M`'s **bit width**, which is as close as the two runtimes get — see [Limitations](limitations.md#cross-engine-comparison) for what that does and doesn't license you to conclude. Where several builds of the same model exist, the one whose footprint sits closest to the GGUF is preferred over the most popular or the most official, because a size gap is a precision gap (see below). The two embedding models are unquantized upstream fp16 repos on both engines.

Sizes here are the sum of the repo's safetensors and config files. A vLLM snapshot is generally *not* the same size as the corresponding GGUF — compare the two size columns before selecting both engines, and see [Setup](setup.md#choosing-engines).

#### Where the two builds diverge

Both columns say "4-bit", and that label hides real differences. `Q4_K_M` is mixed precision by design: llama.cpp promotes selected tensors to `Q6_K`, so its effective rate lands nearer 5 bits per weight than 4. AWQ and GPTQ recipes make their own choices about what to leave in higher precision — embeddings, `lm_head`, the MoE router, Mamba state layers — and different quantizers disagree. The result is that two files both described as 4-bit can differ by a factor of two in size.

Dividing each build's size by the model's parameter count gives an effective bits-per-weight, and comparing that across a pair shows how far apart they really are. Most of the catalog is within ±0.25 bpw. Three entries are not, and all three lean the same way:

| Model | GGUF | vLLM | Δ bits/weight |
|---|---|---|---|
| Qwen3.5 9B | ~5.5 | ~8.1 | **+2.6** |
| Gemma 3 1B | ~6.4 | ~8.8 | **+2.4** |
| Qwen3.5 4B | ~6.2 | ~8.2 | **+2.0** |

On those three, the vLLM build carries meaningfully more precision than the GGUF. Expect it to look slightly *better* on accuracy — fewer bits discarded means less quantization error — and slightly *worse* on tokens/sec, because decode is memory-bandwidth-bound and a larger file moves more bytes per token. Neither difference is attributable to the runtime, which is the whole point of flagging them.

This matters only when reading one engine against the other. Comparing llama.cpp results across machines, or vLLM results across machines, is unaffected: both sides of that comparison use the same weights.

The effect concentrates in small models because a vocabulary embedding is a large fraction of a 1–4B parameter model, and most quantizers keep it at higher precision. That inflates both sides, unevenly. It is not a sign that a better build exists — for each of these three, the entry in use is already the closest match available.

### Dense vs. Mixture-of-Experts (MoE)

A **dense** model runs every one of its parameters for every token it generates. A **Mixture-of-Experts (MoE)** model instead routes each token through only a small subset of specialized "expert" sub-networks, out of many more it holds in total — so most of its parameters sit idle on any given token. Catalog tags spell this out for MoE variants with an `-aN` suffix (e.g. `gemma4:26b-a4b-it-ud-q4_K_M`): the number after `a` is how many parameters actually activate per token ("active"), versus the number before it (total parameters, which is what drives memory/VRAM use).

Because decode speed tracks active parameters far more closely than total size or VRAM footprint, an MoE model can generate noticeably faster than a dense model of similar total size. That gap is exactly why the medium and large tiers each pair their two MoE entries with one dense model (Qwen 3.8 27B and Llama 3.3 70B): total download size alone would put an MoE model like Nemotron 3.5 Lightning (3B active of 30B total) in the same tier as models many times slower to run, so a dense representative keeps each tier honest about what it actually costs in generation time, not just disk space. Nemotron 3.5 Lightning and Nemotron 3 Super use a hybrid Mamba architecture, while Qwen3-Coder-Next combines gated delta networks with sparse and full attention; both approaches reduce long-context cost relative to a conventional dense transformer, but exercise different inference paths.

**Reasoning models** (Nemotron 3.5 Lightning here, a unified model for both reasoning and non-reasoning tasks) generate internal thinking tokens before their answer, via llama-server's separate `reasoning_content` field rather than mixing them into the answer text. Tokens/sec uses llama-server's generated-token count, including thinking output; streamed text fragments are never treated as tokens. Both single-shot and conversation TTFT are client-observed from immediately before the HTTP request opens until the first content, reasoning, or tool fragment arrives. Conversation requests still reuse the existing KV cache, while llama-server's separate prompt-evaluation duration is retained only in explicitly named server timing fields.

Measured generation results record requested, completed, and valid sample counts. Invalid non-finite, negative, internally inconsistent, or TTFT-after-wall measurements remain visible as diagnostic run entries but are excluded from means and raw valid-sample arrays; legacy `n_runs` keeps its historical completed-call meaning. With at least two valid samples, results also include medians and coefficients of variation without dropping outliers or assigning an instability verdict.

**Llama representation:** Llama 3.3 70B supplies the large tier's dense general-purpose baseline, complementing the specialized sparse models without duplicating their roles.

`--maxtier` caps LLM models (and image models, see below) at a given tier and below; `--llm-models` narrows further to specific tags or wildcards (e.g. `--llm-models "qwen*"`) within whatever tier is selected. The old `--models` spelling remains an identical alias — see [CLI Reference](cli-reference.md).

## Image Generation

Five models are tested at 1024×1024 and 1536×1536 — except Stable Diffusion 1.5, which uses 512×512 and 768×768 instead (see below). Any model whose checkpoint is absent from `models/comfyui/checkpoints/` is skipped automatically; setup downloads selected checkpoints there and configures the resolved ComfyUI installation to search that managed path.

Each measured run (`--runs`, default 3) uses a different seed, starting at 42 — an identical seed and workflow would let ComfyUI cache every node and return a cached result almost instantly instead of actually re-running generation. Every image model also gets exactly one warmup at its first resolution with seed 41; image generation does not use `--warmup`. Each generation gets twice `--timeout` (600 seconds by default).

Between models, ComfyUI is asked to unload whatever checkpoint it has resident (`/free` with `unload_models`/`free_memory`) so each model starts from a clean VRAM state — its automatic model-swap-on-load is the only thing that would otherwise free the previous checkpoint, and on the MPS backend its free-VRAM detection is unreliable, so a model can stay resident far longer than it would on CUDA. After a timed-out generation, the benchmark also interrupts the running job and clears ComfyUI's queue before continuing — `/interrupt` and the queue-clear both return before the job actually unwinds, so a dead job would otherwise occupy ComfyUI's single execution slot and every later submission would queue silently behind it.

If a model's warmup crashes the ComfyUI process outright (e.g. a native segfault while loading text-encoder weights), the benchmark detects the dead server and restarts it before moving to the next model, so a crash on one checkpoint doesn't silently doom every remaining image model to an instant connection-refused failure. If ComfyUI can't be restarted at all, the run stops there and preserves whatever image results were already collected.

| Model | Checkpoint | Steps | Size | Tier | HuggingFace login |
|---|---|---|---|---|---|
| Stable Diffusion 1.5 | `v1-5-pruned-emaonly.safetensors` | 20 | ~4.3 GB | xsmall | No |
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~7.0 GB | small | No |
| SD3.5 Large | `sd3.5_large.safetensors` | 28 | ~16.5 GB | medium | Yes (free) |
| Flux.1-dev | `flux1-dev.safetensors` | 20 | ~23.9 GB | large | Yes (free) |
| Flux.2-dev | `flux2-dev.safetensors` | 28 | ~64.5 GB | large | Yes (free) |

**Stable Diffusion 1.5** was trained at 512×512; testing it at the other models' 1024/1536 resolutions produces visibly degraded (duplicated-subject) output, so it gets its own native-range pair — 512×512 and 768×768 (the same 1.5x step used for everything else) — instead of the shared resolution list.

`--maxtier` caps image models the same way it caps LLMs. `--image-models` then narrows that tier-capped list using the stable short IDs in the table (`sd15`, `sdxl`, `sd35-large`, `flux-dev`, `flux2-dev`) or case-sensitive wildcards such as `"sd*"` — see [CLI Reference](cli-reference.md).

SD3.5 Large, Flux.1-dev, and Flux.2-dev require a free HuggingFace account and license acceptance — see [HuggingFace token](setup.md#huggingface-token) in the setup guide.

Generated sample images are saved under `results/images_<hostname>_<timestamp>/` — see [Project Structure](project-structure.md). If `--out` puts the main JSON elsewhere, the image folder remains under `results/` and is named from that output stem.

Each model/resolution pair is a journal-owned recovery case. All requested repetitions and the representative PNG save attempt complete before the case commits; a compatible resume skips committed resolutions and reruns only the interrupted, failed, or timed-out resolution. Generated PNGs keep their visible legacy names while a content-addressed copy and digest make the saved artifact durable; artifact metadata remains outside the compatible results JSON. Resume identity covers every existing selected checkpoint/workflow asset plus the selected ComfyUI entrypoint and interpreter bytes.

## Embeddings

Two models — Nomic Embed Text and MixedBread Embed Large — measured on a single real-world task: chunking a real multi-chapter document (`scripts/workloads/data/sample_document.txt`, ~27 chapters) into paragraph-sized pieces (capped at 150 words each) and embedding every chunk from it in one call, the way a RAG ingestion pipeline actually embeds a document — rather than sweeping arbitrary batch sizes that don't correspond to real client behavior. The chunk cap also keeps every chunk safely under any embedding model's context length, regardless of the source document's formatting.

Each model gets `--warmup` discarded calls first — the very first embed call against a freshly-loaded model pays a one-time model-load cost that has nothing to do with steady-state throughput, so it is absorbed before the `--runs` measured calls (default 3) rather than skewing them. Embedding calls use the engine interface's fixed 120-second request timeout.

Each model's complete document batch is one journal-owned recovery case keyed by the model and full corpus-content hash. Its per-call wall/model-load timings, validity, aggregate throughput, and telemetry commit without retaining embedding vectors; a compatible resume skips completed models and reruns only an unfinished model batch. Changing the corpus requires a fork rather than mixing measurements from different input text.

Under llama.cpp, `--llamacpp-no-repack` applies to embedding models because embeddings use the same `llama-server` model-loading path as generation, conversation, accuracy, and HTTP-concurrency workloads. It also applies to llama-bench concurrency through `llama-batched-bench`; the standard native workload is unchanged because `llama-bench` does not support the option. The setting does not affect vLLM or ComfyUI. It changes the runtime's internal weight layout, not the embedding calculation or result quality, and can trade reduced startup time and peak loading memory for lower CPU or partially offloaded throughput.

If you see repeated connection errors or crashes during the embedding tests (some GPU backends are unstable or immature under batched embedding workloads), try `--cpu-only`. This restarts the active engine with GPU devices hidden for every engine-backed test in the run (`llm`/`conv`/`mcq`/`math`/`reasoning`/`code`/`tool`/`emb`/`conc_tool`/`conc_chat`), then restores normal GPU mode afterward. See [CLI Reference](cli-reference.md).

| Model | Tag | Size |
|---|---|---|
| Nomic Embed Text | `nomic-embed-text` | ~0.3 GB |
| MixedBread Embed Large | `mxbai-embed-large` | ~0.7 GB |

`--embedding-models` narrows this workload to exact catalog tags or case-sensitive wildcards, such as `--embedding-models nomic-embed-text`.

## Accuracy

All five accuracy workloads use the same `--llm-models` selection as single-shot and conversation; `--models` remains its backward-compatible alias. Since decoding uses the identity-bearing `deterministic-baseline-v1` profile—temperature 0 plus explicit neutral shared and engine-specific logit controls—each workload makes one measured pass and ignores `--runs`.

Question banks are validated when loaded, before any model inference begins: every question needs an ID and category, IDs must be unique, and each workload requires the fields it consumes during inference and scoring. Scoring repeats the common validation defensively for programmatically supplied questions but omits a missing optional breakdown value instead of discarding completed measurements.

Each accuracy question is a journal-owned recovery case keyed by the model, full question-bank content hash, and question ID. Its graded value, raw response, timeout/token-budget/loop diagnostics, and attempt number commit before the next question starts; the aggregate result and `answers_<test>_*.json` sidecar are projections of those events. A compatible resume skips completed questions, while any bank-content change fails identity validation and requires a fork.

### MCQ

Every LLM model (all four tiers, same models as the LLM test above) answers a fixed bank of 150 multiple-choice questions once each, via a real chat turn (`/v1/chat/completions`) asking for just the letter of the correct answer.

The question bank (`scripts/workloads/data/mcq_questions.json`) covers eight categories — science, history, geography, logic, literature, arithmetic, commonsense, and language — with introductory items retained for score continuity and a substantially harder second half. Correct-answer positions are balanced across A–D (38/38/37/37) *and* randomly ordered (seeded, so the file is reproducible) — balance alone doesn't rule out an exploitable fixed-cycle ordering (e.g. "guess A, then B, then C, then D, repeat"), so both properties matter. A model's free-form reply is parsed conservatively: a bare answer letter wins first; otherwise the last boxed or explicitly marked answer wins, including negated and resultative corrections such as `making C the correct answer`; then comes a leading answer marker or leading letter, and finally a single unambiguous uppercase choice mentioned anywhere. Repeated mentions of the same choice are accepted, but competing unmarked choices count as unanswered (wrong).

Results report overall accuracy plus a per-category breakdown, so a model that's strong on arithmetic but weak on commonsense reasoning (or vice versa) is visible rather than averaged away.

Run just this test with `--tests mcq`.

### Timeouts and loop detection

Each accuracy question gets 8192 completion tokens by default, configurable with `--acc-token-budget`. The first request receives 60% (4915 tokens at the default). Only a literal length stop sends a second request: the original history, the first response as assistant context, and a concise instruction to return the complete final answer. That pass receives the remaining 40% (3277 tokens), and only its replacement response is graded. A normal, malformed, or wrong first response is graded immediately without retrying.

`--acc-timeout` (default 60 seconds) is one wall-clock deadline covering model loading and both passes; the final-answer pass never receives fresh time. If time expires before it starts, pass 1 is scored and the timeout is recorded. Accuracy warmups and questions use the same explicit 32K server context allocation.

A timeout or exhausted second-pass token budget ends only that question; the bank continues. Whatever the graded pass streamed before the cutoff is captured and passed through that workload's normal parser/scorer, so a partial response can still be correct, wrong-but-parseable, or empty.

Results record `budget_nudged_count`/`ids` when the second pass was sent and `budget_exceeded_count`/`ids` when it also reached its token limit. These are independent of correctness and of `timed_out_count`/`ids`. Streaming responses are also checked periodically for a loop heuristic: a 12+ word chunk repeated three or more times, or recurring self-correction/hedging phrases such as "wait," and "let me reconsider." That can stop a likely loop before the full timeout, so `likely_loop_ids` is separate. Completed responses are never loop-checked, and a flagged partial response is retained in `likely_loop_ids` only if its final score is wrong.

### Math

Every LLM model answers a fixed bank of 150 math problems once each (temperature 0, same deterministic-decoding reasoning as MCQ, so this workload also ignores `--runs`), asked to respond with only the final numeric answer. The question bank (`scripts/workloads/data/math_questions.json`) spans 30 categories, from arithmetic and word problems through combinatorics, number theory, calculus, linear algebra, statistics, complex numbers, and conditional probability.

A model's free-form reply is parsed in a confidence-ordered cascade: a bare numeric response wins first; otherwise the last boxed or explicitly marked answer wins. The next tier combines a numeric-only first line corroborated by either the response's final numeric value or its final completed `= N` result with safe conclusions introduced by “therefore,” “thus,” or sentence-leading “so.” A direct scalar conclusion is accepted as written; when it begins an expression, the parser uses a stated result after the final `=` in that same clause rather than its first operand or a number from a later sentence. The last number remains the compatibility fallback. Parsed values must be finite, so an integer too large for a Python float is treated as unanswered instead of emitting browser-incompatible `Infinity`. Each answer is checked against the question's known numeric answer within its own per-question tolerance (most are exact); a reply with no finite number counts as unanswered (wrong).

Results report overall accuracy plus a per-category breakdown, same as MCQ.

Run just this test with `--tests math`.

### Reasoning

Every LLM model answers 60 original, knowledge-light A–D questions from `scripts/workloads/data/reasoning_questions.json`. The bank has six questions in each of ten categories: formal deduction, constraint satisfaction, relational, temporal, spatial, causal/counterfactual, argument analysis, state tracking, discrete passage reasoning, and rule induction. Questions 41–60 form a deliberately difficult tail: every category contributes two `very_hard` items, and answer positions remain balanced across the full bank.

The bank is validated before use rather than trusted as arbitrary JSON. Its versioned top-level shape, research-source metadata, categories, exact question fields, unique IDs, A–D choices, category references, difficulty values, rationales, skill tags, and original provenance must all be valid; malformed content aborts with a clear error instead of silently changing what gets scored.

Reasoning uses MCQ's tested explicit-answer patterns—bare choices, leading markers, boxed/Markdown/tagged answers, and explicit corrections—but disables its final unstructured-letter fallback. This matters for long reasoning: merely mentioning `A` or `C` in an explanation is not a final selection. If the model ignores the request to return only a letter, it must still state the answer structurally; otherwise the response counts as unanswered rather than risking a false positive.

Results report overall accuracy, per-category accuracy, and per-difficulty accuracy. The dashboard shows both category and difficulty charts, making performance on the `very_hard` tail visible rather than hiding it in the aggregate.

Run just this test with `--tests reasoning`.

### Code

Every LLM model answers a fixed bank of 60 coding problems once each (temperature 0, same deterministic-decoding reasoning as MCQ/math, so this workload also ignores `--runs`). The question bank (`scripts/workloads/data/code_problems.json`) covers 13 categories — algorithms, arithmetic, divide-and-conquer, dynamic programming, graph, intervals, list, matrix, number theory, search, stack, stateful, and string — with visible and hidden expected-output cases for each problem.

Problems come in two shapes:
- **Function problems** (most of the bank): the model writes one function matching a given name and signature. Each test case is an `args`/`expected` pair.
- **Stateful problems** (category `stateful` — including caches, tries, disjoint sets, and streaming median structures): the model writes a class instead, and each test case is a scenario: construct a fresh instance, call a sequence of methods in order, and compare every return value against an expected sequence. A fresh instance is used per test case, so one scenario's state can never leak into another.

The model's reply is parsed for a fenced Python code block (falling back to the whole reply if it wrote bare code without fencing), then that code is run against every one of the problem's visible *and* hidden test cases in a restricted isolated Python child. Before launch, syntax is parsed and direct imports, file access, dynamic execution/introspection names, and dunder-attribute access are rejected. The child uses isolated/no-site Python mode, a minimal environment and temporary working directory, restricted builtins, an audit hook denying imports/filesystem/process/network operations, a parent-observed memory ceiling, bounded stdout/stderr, and one total execution deadline; POSIX also applies CPU, output-file, and child-process resource limits. The harness flushes a private framed result after each completed test, allowing diagnostics from earlier tests to survive when a later test hangs or exceeds a resource limit. Candidate stdout is ignored unless it matches the private result protocol. A problem counts as correct only if every test case passes; a reply with no extractable code, or code that fails even one test case, counts as wrong.

These controls materially reduce ordinary generated-code risk but are not a kernel or hypervisor security boundary. Python-level restrictions can have interpreter/runtime escapes, and Windows lacks the additional POSIX resource limits until a Job Object or packaged sandbox owns the child. The code workload therefore remains preview for hostile-input or multi-tenant use; stable commercial qualification requires supported-platform OS containment, adversarial escape review, and a decision about disabling it where that containment is unavailable.

Results report overall accuracy plus a per-category breakdown, same as MCQ/math.

Run just this test with `--tests code`.

### Tool Use

Every LLM model answers a fixed bank of 100 tool-calling questions once each (temperature 0, same deterministic-decoding reasoning as MCQ/math/code, so this workload also ignores `--runs`). Each question offers the model an OpenAI-style `tools` array (function name, description, and JSON-schema parameters) via `/v1/chat/completions` with `tool_choice: "auto"`, and is scored on whether the model called the right tool with the right arguments — or correctly declined to call anything when none of the offered tools genuinely fit. The question bank (`scripts/workloads/data/tool_questions.json`) spans 20 five-question categories. The first half covers straightforward calls, basic selection and extraction, enum/numeric/boolean arguments, optional parameters, multi-argument calls, and obvious declines. The harder half covers close tool distinctions, semantic conversions, nested arrays/objects, omitting unspecified optional arguments, semantic enum mapping, missing-information and near-miss declines, large distractor sets, instruction-like content that must remain literal data, and corrections or negations.

The decline cases matter as much as the call cases: a model that fires a tool for a request none of the tools can serve, lacks required information, or would violate an explicit "do not" instruction is as wrong as one that calls the wrong tool. Correct behavior there is calling nothing. A positive case requires exactly one tool call, so emitting the expected call alongside an unintended second action fails.

Argument comparison is recursive. Numeric strings are accepted for numeric values (`"20"` matches `20`), but booleans never match numbers. Baseline questions use subset matching, allowing extra keys for continuity; advanced questions marked `strict_arguments` require the same keys at every nested object level. Arrays are positional by default, while scenarios can mark genuinely set-like fields such as labels or recipients with `unordered_keys`, which compares those arrays as multisets while preserving duplicates. Questions can opt specific free-text fields into whitespace-, case-, and terminal-punctuation-insensitive comparison with `normalized_string_keys`; identifiers and other undeclared strings remain exact. New free-text arguments such as titles, messages, notes, and bodies must be declared there when that tolerance is intended. Because the question-bank hash covers the JSON file, adding or changing this metadata automatically distinguishes the revised tool bank from earlier results.

Results report overall accuracy plus a per-category breakdown, same as MCQ/math/code.

Run just this test with `--tests tool`.

Run every accuracy-style test at once with `--tests acc` — expands to MCQ, math, reasoning, code, and tool, and de-duplicates against any of them also listed explicitly. See [CLI Reference](cli-reference.md).

### Bank versioning

Question banks grow and change over time (the MCQ and math banks each doubled in size in one revision, for example), so a raw correct count from one results file is never safely comparable to another without knowing which version of the bank produced it — 40/50 and 40/150 both look like "40 correct" but mean very different things. To make that comparison safe:

- Every results JSON records a `bank_versions` object — a short hash of each accuracy bank's file contents (`mcq`, `math`, `reasoning`, `code`, `tool`) at the time of that run, computed from the raw bank bytes (not just parsed field values, so even a whitespace-only or key-reordering change is caught). Two results files only used the exact same question set if their `bank_versions` entries match.
- The crash cache each accuracy test keeps (`.mcq_crash_cache.json`, `.math_crash_cache.json`, `.reasoning_crash_cache.json`, `.code_crash_cache.json`, `.tool_crash_cache.json`) records the bank version a model crashed against, so a model that crashed repeatedly on an old, smaller bank isn't silently skipped forever once the bank has since changed — the stale entry is ignored and the model is retried. `--retry-crashed-models` bypasses current crash-cache entries across workload families for one execution without deleting them; a repeated crash is still handled and recorded normally. Every crash cache is also scoped per engine — a catalog tag shares one identifier across engines, but a crash under llama.cpp's GGUF says nothing about vLLM's separate weights and runtime for that same tag, so a crash recorded under one engine never skips that model under another.
- Percentages normalize for bank size, but a changed bank can also change difficulty and composition. Use matching `bank_versions` hashes for direct model/system comparisons; treat cross-version percentages as contextual rather than apples-to-apples.
- Every workload's per-model loop (LLM, conversation, embeddings, concurrency, accuracy, llama-bench, llama-bench concurrency, vllm bench) has a top-level `except Exception` around each model's iteration, on top of the specific crash/timeout handling described above — an unexpected error (a bug in the runner itself, not just an OOM or a hung load) still records that model as `crashed` via `Shared.unexpected_model_failure` and moves on, rather than aborting the whole stage. A single model failing this way must never take the rest of the run down with it.

`--sample N` (see [CLI Reference](cli-reference.md)) is a separate, dev-only mode for fast local iteration. It uses a deterministic round-robin across categories; every category is represented when `N` is at least that bank's category count, while smaller samples cover as many categories as their size permits. The exact sampled IDs are recorded under `sample_ids`. Sampled runs are reproducible, but are not comparable with full-bank or differently sampled runs.

## Concurrency

The interactive launcher uses its one visible LLM checklist for concurrency as well as single-shot, conversation, and accuracy tests. It passes that selection through explicit `--llm-models`, so unchecked models—including large models unchecked by default—do not enter a frontend-launched concurrency run. Direct CLI behavior is unchanged: concurrency without an LLM selector uses every downloaded LLM and ignores `--maxtier`; an explicit `--llm-models`/`--models` narrows that downloaded scope.

Every other LLM test in this suite is strictly one request at a time — these two measure how per-request latency and aggregate throughput scale as multiple simultaneous requests hit the same loaded model, which matters far more than single-stream numbers for anyone thinking about serving more than one user (or one agent's parallel tool-calls) at once. They're two separate tests because "agentic tool-calling fan-out" and "many simultaneous chat users" are genuinely different workload shapes — different concurrency ceilings, different per-request context, and different early-exit tradeoffs — not one sweep with one shape. Both are opt-in via `--tests conc_tool`/`--tests conc_chat` (`--tests conc` runs both) — not part of the default set, since each takes noticeably longer per model than a single request.

Both tests scope to **every LLM model actually downloaded locally**, ignoring `--maxtier` — a machine that only downloaded xsmall/small models tests those; one that downloaded medium/large too tests those as well. This is deliberate: unlike the fixed-tier restriction these tests used to have, download presence is itself a decent proxy for "this machine has the memory to try." An explicit `--llm-models` selection (or its `--models` alias) still narrows further within whatever is downloaded.

Every level respawns `llama-server` (the concurrency level is part of `LlamaCppEngine._ensure_model`'s want/have check, so a level change always forces a fresh process), which means each level's first-ever inference is on a brand-new process at that specific concurrent shape. `--warmup` (default 2) throwaway concurrent batches are fired and discarded at each level before the real measured one, for exactly the same reason every other test in this suite warms up before measuring — a fresh process's first real decode can carry one-time overhead (kernel autotuning, CUDA graph capture, and similar) that has nothing to do with steady-state throughput.

At each level, N concurrent requests are fired at once using distinct deterministic padded single-shot prompt variants and up to 512 generated tokens per request (`config.GENERATE_MAX_TOKENS`). Request-level cache controls prevent a prior batch from serving any prompt as a cache hit. After the configured warmup batches, one measured concurrent batch is recorded; `--runs` does not repeat concurrency batches. Results include mean/stdev per-request TTFT and decode tokens/sec plus aggregate tokens/sec: authoritative native token-ID counts divided by the measured batch's wall-clock duration. When the engine reports an attributable prompt token count and server-side duration, results also include genuine per-request prefill tokens/sec; this is never inferred from TTFT, and concurrent vLLM requests omit it because its shared metrics histogram cannot isolate one request. Per-request TTFT includes request dispatch, slot queueing, prompt evaluation, first-token sampling, and delivery of the first streamed output.

Each slot's real ctx budget (`ConcurrencyBenchmark.slot_ctx_for`) adds that 512-token generation headroom on top of the padded prompt sizes below — not the bare prompt size.

### Tool (`conc_tool`)

Simulates the short-context fan-out common in agentic workflows. It uses ordinary completion requests rather than the tool-calling API—the "tool" label describes the serving shape, not a function-call accuracy test. The sweep covers 1, 2, 4, 6, 8, 12, and 16 simultaneous requests, each with a 4,096-token padded context.

This test **never soft-exits on slow tok/s** — every level always runs and gets recorded, since the whole ceiling here (16-way) is cheap enough that a real data point at every level is worth more than an inferred one. Only a hard stop (see below) ends its sweep early.

### Chat (`conc_chat`)

Simulates a chat server under load — many simultaneous long-conversation users. Swept through concurrency levels 1, 2, 4, 8, 16, 24, and 32, each request given 16,384 tokens of its own context (a long conversation history, at scale).

Unlike the tool test, this one **does** soft-exit (see below) — at up to 32-way concurrency, a model that's already cratered to a few tokens/sec per request costs an enormous amount of wall-clock time to keep climbing for a foregone conclusion.

### Escalation stopping

Escalation to the next level stops for one of two reasons:
- **Hard stop** (both tests) — the model fails to even load at that level (out of memory, hung load) or the engine's runner crashes repeatedly during a batch. A connection crash is retried up to the shared crash-retry limit before the sweep stops or writes its crash cache (`.concurrency_tool_crash_cache.json` / `.concurrency_chat_crash_cache.json`, one per test so a crash on one doesn't affect retry state on the other). A load failure is not cached, since a lower level on the next run is cheap to retry and will very likely still succeed.
- **Soft stop** (chat only) — once concurrency level 8 has actually been reached, per-request tokens/sec dropping below the usual slow-model cutoff means climbing further would only confirm what's already obvious. Levels 1, 2, 4, and 8 always run and get recorded regardless of how slow they already look. `--force-all` disables this the same way it does elsewhere.

### Memory snapshots

Each level also records a memory snapshot, taken right after that level finishes loading (model + full KV cache allocated) and before the batch fires — the steadiest point to read how much headroom is actually left, rather than numbers that fluctuate mid-batch. It always includes system RAM used/total; GPU VRAM used/total is added too when `nvidia-smi` or `rocm-smi` answers (a `rocm-smi` reading is only trusted for a confirmed-discrete AMD card — an APU's reported VRAM is often just a small BIOS-fixed carve-out, not the real usable pool). On a unified-memory machine — Apple Silicon, or an NVIDIA/AMD box like a DGX Spark or Strix Halo, where the model competes with the OS for the same physical pool — system RAM is the number that actually reflects total headroom; GPU VRAM there is supplementary, not a substitute. A load failure also records a `memory_at_failure` snapshot, so it's clear what memory state actually triggered the ceiling.

### Power and energy telemetry

`--power-telemetry` extends the existing memory sampler rather than starting a second timeline. Every sample shares one monotonic timestamp and lifecycle window. Power is recorded from `powermetrics` on macOS, `nvidia-smi` on NVIDIA, the installed Adrenalin driver's ADL counters on Windows AMD, `rocm-smi` on Linux AMD, or readable Intel RAPL counters; discovery records unavailable with a normalized reason and never turns failure into zero. Idle baseline is retained separately, while case energy integrates adjacent valid readings across each contiguous measured region using their actual uneven timestamps. A measured label change does not break an active region, but a pause, idle, or model-load sample does, so energy never bridges an unmeasured gap. Request-window workloads do not charge model load or idle energy to the request; the native subprocess exceptions are described below.

Every figure carries scope: Apple processor package, accelerator, CPU package, or whole system. The dashboard refuses mixed-scope efficiency series. Generation cases export valid completed tokens per joule, images export completed images per joule, and embeddings export successful chunk embeddings per joule, with raw work count and energy retained beside the ratio. Native prefill/decode cases use their completed prompt or generated token count, but llama-bench, llama-batched-bench, and vLLM bench hide subprocess model loading and therefore conservatively include that load in their `measured:*:includes-load` energy. Their efficiency ratios are full native-case values, not request-only values, and must not be compared as though the boundary matched LLM, accuracy, embedding, or image request efficiency. Accuracy workloads retain power evidence but do not invent an efficiency unit that the methodology has not defined.

### A note on tok/s outliers

Under heavy concurrent-slot contention, llama-server's streamed timing data can occasionally report a decode duration implausibly small for its token count. Single-shot and conversation retry such a request once; concurrency discards and retries the whole batch once so the retry still measures the requested contention level. The wall-clock correction is diagnostic only and never contributes to aggregates. If the retry is also implausible, that request remains in `invalid_runs` with reason `implausible_server_tps`; other valid concurrent requests can still score the level, while a batch with no valid requests stops that model's escalation as invalid. See `LlamaCppEngine._sanitize_tps` in [Engines](engines.md#llamacppengine).

Prepared concurrency recovery skips levels already complete and runs pending levels in their original escalation order with a new attempt number. Each retried level still starts a fresh server configuration and measures one whole concurrent batch, so recovery never combines requests from different contention attempts into one aggregate.

Both HTTP concurrency stages run in supervised children and commit only after the final whole-batch outcome is known. The immutable plan carries each ladder, per-request context size, and chat soft-exit floor. A first implausible batch is never mixed with its retry: only the second batch's individual samples, batch elapsed time, aggregate throughput, memory snapshot, and validity diagnostics enter the durable case.

## Sustained load

`sustained` is an opt-in continuous-generation soak at a fixed 2K prompt depth. It runs one selected LLM at a time for at least 600 seconds by default, dividing the real wall-clock interval into aligned ten-second windows. Each window records generated-token throughput plus available memory, power, SoC-package temperature, CPU-package temperature, GPU-die temperature, and GPU-hotspot temperature from the shared sampler. Apple Silicon reads calibrated Celsius values directly from macOS's private HID temperature services without sudo: its known CPU, GPU, and M5 `PMU tdie*` sensor families are combined into one unified SoC-package channel because the CPU and GPU share the same package. Platform sources that expose genuinely separate CPU or GPU measurements retain their distinct channels. Use `--tests sustained --llm-models MODEL_TAG`; `--sustained-duration` changes the minimum duration and `--ambient-temp-c` records a nearby room measurement taken immediately before the run. The workload requires memory telemetry so every channel uses the same timeline.

The first three windows define initial throughput and the final six define steady-state throughput. Retention is steady-state TPS divided by initial TPS: at least 95% is stable, 85% through less than 95% is mild degradation, and less than 85% is significant degradation. Classification requires at least 120 seconds, enough complete windows, and a nonempty series in which every request passed measurement validation; shorter, pause-interrupted, or validation-rejected evidence is indeterminate. Throttle onset is the first of three consecutive post-baseline windows more than 5% below initial throughput, so neither a baseline window nor one transient dip becomes a throttling claim.

Performance and suspected cause are separate. Temperature correlation requires the decline to meet a measured temperature ceiling pattern, power correlation requires a coincident sustained power decline, and both or neither may be reported. A hot but stable system remains stable; missing sensors make cause unavailable without erasing valid throughput retention. The full normalized series, analysis, sensor availability, ambient reading, and actual duration remain in the result for audit. See [Linux small-system qualification](qualification/sustained-linux-small-systems.md).

## llama-bench

Opt-in (`--tests llamabench`, not part of the default set) — runs llama.cpp's own `llama-bench` tool directly against every model in the same `--maxtier`/`--llm-models` scope as `llm`/`conv`/accuracy, instead of going through this project's HTTP/SSE pipeline. It overlaps substantially with the `llm` test's own prefill/decode measurements, and that overlap is intentional: `llama-bench` is the tool most community-published throughput numbers use, so a run here is directly comparable to other people's numbers for the same hardware/model/quantization, and — since it bypasses this project's own timing/HTTP code entirely — a divergence between its numbers and `llm`'s is a useful signal about where a difference comes from. `config.LLAMABENCH_PP` deliberately matches every non-zero prefill (`CONTEXT_LENGTHS`) and conversation (`CONV_CHECKPOINTS`) size, since `llama-bench` is meant to eventually stand in for both. It still doesn't measure the same thing as the conversation test: `llama-bench` always benchmarks an isolated prompt at a fixed size, never a KV-cache-reused, naturally growing multi-turn conversation — matching the checkpoint sizes only means the two are comparable at each depth, not that one reproduces the other's cache-reuse behavior.

The supervised native runner preserves llama-bench's efficient lifecycle: one prefill sweep command and one decode sweep command load each model, with all requested cases passed in each command. Current llama-bench releases derive context capacity from each prompt and generation case rather than accepting the older explicit `-c` option. Every JSONL row is committed to the journal as it arrives, including its completed internal repetitions. If case 18 times out, cases 1–17 remain exportable without rerunning or reloading them merely to checkpoint.

On prepared recovery, completed native case IDs are removed from the new command matrix. Remaining prefill sizes stay in one sweep, while decode depths with the same remaining generation sizes share one sweep; a fresh run therefore keeps its original two-command shape, and a partial recovery adds a process only when the remaining matrix is no longer representable as one llama-bench cross product. The original model plan and completed rows remain immutable.

Unlike every other workload, this one is inherently llama.cpp-specific rather than engine-agnostic — `llama-bench` is llama.cpp's own tool with no cross-engine equivalent — so it's skipped with a warning under any future non-llama.cpp engine rather than attempting a translation that doesn't exist.

For each model, prefill covers `config.LLAMABENCH_PP` as standalone `-p` tests, while decode covers the cross product of those depths via `-d` and `config.LLAMABENCH_TG` via `-n`. The suite starts only two subprocesses per model—one complete prefill sweep and one complete decode sweep—and passes the configured repetition count directly to llama-bench, restoring model reuse across cases and repetitions. JSONL output flushes after each completed case; the suite checkpoints that row immediately and retains llama-bench's individual throughput samples, mean, and standard deviation without dropping outliers. If case 18 times out, cases 1–17 remain saved; the unfinished case and later cases in that sweep are unavailable. The manifest records this streamed internal-repetition mode so the dashboard warns when it is compared with the briefly used per-case-process files or older non-streaming files. GPU offload passes `-ngl 999` for full offload, or `0` under `--cpu-only`, because `llama-bench` accepts only numeric layer counts. A model can therefore run successfully through llama-server's automatic partial offload yet fail this native workload when its weights and requested context do not fit in accelerator memory; that failure means the full-offload native configuration did not fit, not that llama.cpp cannot serve the model. The selected `--gpu-split-mode` is forwarded to both native llama.cpp tools. The optional `--llamacpp-no-repack` setting is forwarded only to `llama-batched-bench`, because `llama-bench` does not support it.

`config.LLAMABENCH_TIMEOUT` (1800s) is an idle timeout, not a ceiling on the whole sweep — `run_one` watches the stderr progress lines already streamed via `--progress` and kills the subprocess only if none arrive for that long, so a legitimately long multi-depth sweep (which can run for hours on the wider `LLAMABENCH_PP` range) doesn't get killed mid-run just for taking a while.

`llama-bench` is invoked with its own `--progress` flag, and its stderr progress lines are streamed to the console live as they arrive rather than buffered until the subprocess exits. Each completed JSONL case is also logged and saved, so long sweeps expose both in-case activity and durable case completion.

Each model's `llamabench` result contains `prefill_entries` and `decode_entries`; each entry adds raw `ts_runs` and requested/completed repetition counts to llama-bench's fields. Both native llama.cpp tools explicitly enable Flash Attention when using the suite's quantized KV cache instead of relying on backend-dependent automatic selection. Partial model results retain their entries and add timeout/error diagnostics instead of replacing measurements. Older shapes remain loadable in the dashboard.

Requires `llama-bench` to be installed — `setup.sh`/`setup.bat` install it alongside `llama-server` (see [Setup](setup.md)); if it's missing, the test prints where to get it and records nothing rather than failing the whole run.

## llama-bench Concurrency

Opt-in (`--tests llamabenchconc`, not part of the default set) — runs llama.cpp's own `llama-batched-bench` tool (a different binary from `llama-bench` above) against every model in the same `--maxtier`/`--llm-models` scope as `llm`/`conv`, measuring how decode throughput scales as more sequences are generated in parallel. It overlaps with [Concurrency](#concurrency)'s `conc_tool`/`conc_chat` on purpose, the same way `llamabench` overlaps with `llm`: those two sweep concurrency through this project's own llama-server and HTTP/SSE client, while this one bypasses all of it and lets llama.cpp batch the sequences itself, so a divergence between them isolates whether a concurrency ceiling comes from the serving layer or from the model/hardware. `conc_tool`/`conc_chat` report both per-request and aggregate tokens/sec; `llamabenchconc` reports only the aggregate — `speed_tg` in each output row is the combined decode throughput across all parallel sequences at that level, not a per-request rate.

Like `llamabench`, this workload is inherently llama.cpp-specific and is skipped with a warning under any future non-llama.cpp engine.

For each model, one `llama-batched-bench` subprocess sweeps the configured prompt, generation, and parallel-sequence matrix. Every valid JSONL row is checkpointed as it arrives, so an idle timeout or later failure preserves earlier matrix cases with requested/completed counts and diagnostics. GPU runs pass `-ngl auto`, allowing the native tool to fit layers to available accelerator memory; CPU-only runs pass `-ngl 0`. The resolved methodology records that policy alongside context fitting and GPU-split settings.

Resume preserves the one-process-per-model shape for a fresh run and partitions only the unfinished `(tg, pl)` cells into the fewest Cartesian native sweeps that do not include a committed row. It never duplicates completed evidence or presents selected-row retry that the native command cannot honor.

This build of `llama-batched-bench` prints nothing to stderr and has no `--progress` flag, so progress comes from stdout instead: results are requested as `--output-format jsonl`, one JSON object per (pp, tg, pl) combination, and each line is parsed and logged as it arrives. Blank or non-JSON lines are ignored. The same idle timeout as `llamabench` applies (`config.LLAMABENCH_TIMEOUT`) — the sweep is killed only if no output arrives for that long, not after a fixed total duration.

Each model's `llamabenchconc` result is either `{"entries": [...], "pp": <effective prompt depth>, "ctx_size": <-c value used>}` — where `entries` are the parsed JSONL rows verbatim, with `llama-batched-bench`'s own field names (`pp`, `tg`, `pl`, `n_kv`, `t_pp`, `speed_pp`, `t_tg`, `speed_tg`, `t`, `speed`, ...) — or `{"error": "..."}`.

Requires `llama-batched-bench` to be installed — `setup.sh`/`setup.bat` install it alongside `llama-server` (see [Setup](setup.md)); if it's missing, the test prints where to get it and records nothing rather than failing the whole run.

## vllm bench

Opt-in (`--tests vllmbench`, not part of the default set) — runs vLLM's own `vllm bench` tool against every model in the same `--maxtier`/`--llm-models` scope as `llm`/`conv`, and is the vLLM counterpart to [llama-bench](#llama-bench): the tool the engine's own community publishes numbers with, run outside this project's HTTP/SSE pipeline so a divergence from the `llm` test isolates where a difference comes from. It is skipped with a warning under any non-vLLM engine, mirroring how `llamabench` is skipped under any non-llama.cpp engine.

Two subcommands run per size, both offline — they load the weights themselves rather than talking to a server, so the stage stops this project's vLLM server first and nothing else may hold the GPU:

- **`vllm bench latency`** measures one batch end to end and reports `avg_latency` in seconds, plus every iteration and a percentile map. vLLM's own defaults (30 iterations, 10 warmups) are far more than this suite needs, so `config.VLLMBENCH_ITERS`/`VLLMBENCH_WARMUP_ITERS` pin them down.
- **`vllm bench throughput`** runs `config.VLLMBENCH_NUM_PROMPTS` prompts and reports `elapsed_time`, `num_requests`, `total_num_tokens`, and rates.

vLLM Bench uses `config.LLAMABENCH_PP` for input sizes and `VLLMBENCH_OUTPUT` for output sizes, so both native engines share the same prompt shapes and `--max-prompt-tokens` cap. A pair is skipped when input plus output exceeds the model's context: vLLM rejects such a request outright, where llama-server merely generates fewer tokens.

**These numbers are not comparable to `llamabench`, and the dashboard never puts them on the same chart.** Two independent reasons, either of which alone would be enough. The weights differ — llama.cpp runs Q4_K_M GGUFs while vLLM runs 4-bit AWQ/GPTQ safetensors of the same base model (see [Per-engine weights](#per-engine-weights)). And the metrics differ: `llama-bench` reports separate prefill and decode token rates, while `vllm bench latency` reports whole-batch seconds and `throughput` reports a combined rate over prompt *and* output tokens. This suite derives an output-only rate (`requests × output_len / elapsed`) rather than reporting vLLM's `tokens_per_second`, which counts prompt tokens too.

KV-cache precision stays consistent within each engine so the server and native cross-checks do not silently test different cache formats. `llama-bench` and `llama-batched-bench` receive the same `q8_0` cache used by llama-server, falling back to `f16` for tensor split; `vllm bench latency` and `throughput` receive the same supported `fp8` or fallback `auto` selected for the managed vLLM server.

Native vLLM bench caps `--gpu-memory-utilization` at 0.85, uses 0.10 on unified-memory GB10 systems, and runs with `--enforce-eager`. The GB10 limit leaves most host memory outside vLLM's executor reservation, while eager execution avoids CUDA-graph compilation consuming memory outside that reservation; the same explicit methodology applies to every platform and to both regular and qualification runs.

Each model's `vllmbench` result contains `latency_entries` and `throughput_entries`, each entry carrying its `input_len`/`output_len` alongside the parsed measurements. A model that times out or fails keeps the entries it completed and adds `timed_out`/`timed_out_at`/`error` diagnostics rather than discarding them.

Each completed latency or throughput size is committed to the event journal before the next subprocess starts. Resume skips committed sizes and gives an interrupted, failed, or timed-out size a new numbered attempt; selected-case retry can target one eligible size without rerunning other completed evidence.

Requires the benchmark extra, which the base vLLM package does not include — setup installs `vllm[bench]` (see [Setup](setup.md)). If `vllm bench` is unavailable the test prints the `pip install 'vllm[bench]'` hint and records nothing rather than failing the run.

Concurrency through vLLM's own tooling (`vllm bench serve`) is not implemented yet. Unlike these two subcommands it requires a *running* server, so it cannot reuse this stage's shape; it would sit alongside `conc_tool`/`conc_chat` as a further cross-check rather than replacing them.

---

[← Setup](setup.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
