[← Back to README](../README.md)

# Workloads

Five workload types are benchmarked: LLM generation (two test modes), image generation, embeddings, accuracy (multiple-choice question answering, math word problems, coding problems, and tool calling), and concurrency (opt-in — see below). Every workload skips models automatically when they don't fit in available memory — no configuration needed on smaller hardware.

Every "Size" figure below is the model's actual on-disk download size, rounded **up** to the next 0.1 GB (not nearest) — the same convention `setup_check.py` uses for its own disk-space check, so an estimate never undersells how much room a model actually needs.

**Contents**
- [LLM](#llm)
  - [Extra-small tier (<6B params)](#extra-small-tier-6b-params)
  - [Small tier (≤20B params)](#small-tier-20b-params)
  - [Medium tier (26–35B params)](#medium-tier-2635b-params)
  - [Large tier (70B+ params)](#large-tier-70b-params)
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

## LLM

Twelve models across four tiers (three per tier) are in the default catalog. Models that were not downloaded are skipped; a timeout or repeatable runner crash stops that model's current workload and preserves any measurements already collected.

The suite provides **two separate LLM tests**. When both are selected, it completes the single-shot stage for all models before starting the conversation stage:

- **Single-shot** — a large prompt, padded to the target size and sent fresh (with unique content) on every run, measured at up to five context lengths (512 / 2K / 8K / 32K / 64K, whichever the model's own context window reaches), so it's always a genuine cold prefill of that many tokens with nothing cached. This simulates dropping a large document, codebase, or transcript into a single prompt and asking one question about it.
- **Conversation** — a real multi-turn chat, measured at up to ten depths (0 / 2K / 4K / 8K / 16K / 32K / 48K / 64K / 80K / 96K, subject to the model and tier limits below): the model explains Plato's Allegory of the Cave, then each following turn asks for more detail on a section, growing the conversation from a blank slate. This test is expensive, so it always runs one conversation regardless of `--runs`.

Both tests cap their context lengths to each model's real maximum, read from the downloaded GGUF metadata. For medium/large catalog models and custom models, the conversation plan targets at most 128K and samples through 96K, leaving up to 4K of KV-cache headroom where the model's native ceiling allows it. xsmall/small catalog models use a smaller plan: a 64K growth target, up to roughly 68K of allocated context including headroom, and a top sampled checkpoint of 48K. A model whose native ceiling is lower gets a correspondingly shorter plan.

These two tests measure genuinely different things, and their TTFT numbers are **not** comparable at face value — see [What the charts mean](dashboard.md#what-the-charts-mean) for why the conversation test's TTFT is typically far lower than the single-shot test's at the same nominal context length.

The single-shot slow-model check applies at its first checkpoint (512 tokens): below 15 tok/s, deeper single-shot contexts are skipped unless `--force-all` is set. When single-shot and conversation run together, the conversation pre-flight also excludes a model with no usable single-shot data, a repeatable runner crash, a first-checkpoint timeout, or that first-checkpoint slow marker. A timeout only at a deeper single-shot context does not by itself exclude conversation. Running `--tests conv` alone has no single-shot pre-flight data, so it attempts every selected model.

Separately, *within* the conversation test itself: if the decode speed at any history depth drops below the slow-model cutoff, the conversation exits early and records results to that point. Pass `--force-all` to ignore this cutoff and always run every context length (see [CLI Reference](cli-reference.md)).

### Extra-small tier (<6B params)

| Model | Tag | Size | Architecture |
|---|---|---|---|
| Gemma 3 1B | `gemma3:1b-it-q4_K_M` | ~0.8 GB | Dense |
| Llama 3.2 3B Q4_K_M | `llama3.2:3b-instruct-q4_K_M` | ~2.1 GB | Dense |
| Phi 4 Mini | `phi4-mini` | ~2.5 GB | Dense |

### Small tier (≤20B params)

| Model | Tag | Size | Architecture |
|---|---|---|---|
| Mistral 7B v0.3 Q4_K_M | `mistral:7b-instruct-v0.3-q4_K_M` | ~4.4 GB | Dense |
| Llama 3.1 8B Q4_K_M | `llama3.1:8b-instruct-q4_K_M` | ~5.0 GB | Dense |
| Phi 4 14B | `phi4:14b-q4_K_M` | ~8.3 GB | Dense |

### Medium tier (26–35B params)

| Model | Tag | Size | Architecture |
|---|---|---|---|
| Qwen3.6 27B Q4_K_M | `qwen3.6:27b-q4_K_M` | ~16.8 GB | Dense |
| Nemotron 3 Nano 30B-A3B | `nemotron-3-nano:30b-a3b-q4_K_M` | ~24.0 GB | Hybrid Mamba-Transformer MoE — 3B active of 30B total |
| Qwen3.6 35B-A3B | `qwen3.6:35b-a3b` | ~24.0 GB | MoE — 3B active of 35B total |

### Large tier (70B+ params)

| Model | Tag | Size | Architecture |
|---|---|---|---|
| Llama 3.3 70B Q4_K_M | `llama3.3:70b-instruct-q4_K_M` | ~39.7 GB | Dense |
| Llama 4 Scout 16x17B | `llama4:16x17b` | ~67.0 GB | MoE — 17B active of ~109B total |
| Nemotron 3 Super 120B | `nemotron-3-super:120b` | ~87.0 GB | Hybrid Mamba-Transformer MoE — 12B active of 120B total |

### Dense vs. Mixture-of-Experts (MoE)

A **dense** model runs every one of its parameters for every token it generates. A **Mixture-of-Experts (MoE)** model instead routes each token through only a small subset of specialized "expert" sub-networks, out of many more it holds in total — so most of its parameters sit idle on any given token. Catalog tags spell this out for MoE variants with an `-aN` suffix (e.g. `qwen3.6:35b-a3b`): the number after `a` is how many parameters actually activate per token ("active"), versus the number before it (total parameters, which is what drives memory/VRAM use).

Because decode speed tracks active parameters far more closely than total size or VRAM footprint, an MoE model can generate noticeably faster than a dense model of similar total size. That gap is exactly why the medium and large tiers each pair their two MoE entries with one dense model (Qwen3.6 27B and Llama 3.3 70B): total download size alone would put an MoE model like Nemotron 3 Nano (3B active of 30B total) in the same tier as models many times slower to run, so a dense representative keeps each tier honest about what it actually costs in generation time, not just disk space. Nemotron 3 Nano and Nemotron 3 Super are a different kind of MoE than the others: a hybrid Mamba-Transformer architecture, where Mamba's state-space layers handle most sequence processing (linear-time in sequence length) and only a subset of experts activate per token on top of that — distinct from Qwen3.6 and Llama 4's more conventional transformer-based MoE, despite similar active/total parameter ratios.

**Reasoning models** (Nemotron 3 Nano here, a unified model for both reasoning and non-reasoning tasks) generate internal thinking tokens before their answer, via llama-server's separate `reasoning_content` field rather than mixing them into the answer text. Tokens/sec uses llama-server's generated-token count, including thinking output; streamed text fragments are never treated as tokens. Single-shot TTFT is measured from request start until the first output reaches the client, while conversation TTFT retains llama-server's prompt-evaluation duration so it represents only the newly-prefilled turn against the reused conversation cache.

**Llama versions:** Llama 3.2 tops out at 3B parameters; the 8B slot uses Llama 3.1; the large tier's dense entry uses Llama 3.3 70B. Llama 4 Scout is the large tier's MoE entry, at 16 experts (17B active of ~109B total).

`--maxtier` caps LLM models (and image models, see below) at a given tier and below; `--models` narrows further to specific tags or wildcards (e.g. `--models "llama*"`) within whatever tier is selected — see [CLI Reference](cli-reference.md).

## Image Generation

Five models are tested at 1024×1024 and 1536×1536 — except Stable Diffusion 1.5, which uses 512×512 and 768×768 instead (see below). Any model whose checkpoint is absent from `ComfyUI/models/checkpoints/` is skipped automatically; `setup_check.py` downloads them on first run.

Each measured run (`--runs`, default 3) uses a different seed, starting at 42 — an identical seed and workflow would let ComfyUI cache every node and return a cached result almost instantly instead of actually re-running generation. Every image model also gets exactly one warmup at its first resolution with seed 41; image generation does not use `--warmup`. Each generation gets twice `--timeout` (600 seconds by default).

| Model | Checkpoint | Steps | Size | Tier | HuggingFace login |
|---|---|---|---|---|---|
| Stable Diffusion 1.5 | `v1-5-pruned-emaonly.safetensors` | 20 | ~4.3 GB | xsmall | No |
| SDXL | `sd_xl_base_1.0.safetensors` | 20 | ~7.0 GB | small | No |
| SD3.5 Large | `sd3.5_large.safetensors` | 28 | ~16.5 GB | medium | Yes (free) |
| Flux.1-dev | `flux1-dev.safetensors` | 20 | ~23.9 GB | large | Yes (free) |
| Flux.2-dev | `flux2-dev.safetensors` | 28 | ~64.5 GB | large | Yes (free) |

**Stable Diffusion 1.5** was trained at 512×512; testing it at the other models' 1024/1536 resolutions produces visibly degraded (duplicated-subject) output, so it gets its own native-range pair — 512×512 and 768×768 (the same 1.5x step used for everything else) — instead of the shared resolution list.

`--maxtier` caps image models the same way it caps LLMs — see [CLI Reference](cli-reference.md).

SD3.5 Large, Flux.1-dev, and Flux.2-dev require a free HuggingFace account and license acceptance — see [HuggingFace token](setup.md#huggingface-token) in the setup guide.

Generated sample images are saved under `results/images_<hostname>_<timestamp>/` — see [Project Structure](project-structure.md). If `--out` puts the main JSON elsewhere, the image folder remains under `results/` and is named from that output stem.

## Embeddings

Two models — Nomic Embed Text and MixedBread Embed Large — measured on a single real-world task: chunking a real multi-chapter document (`sample_document.txt`, ~27 chapters) into paragraph-sized pieces (capped at 150 words each) and embedding every chunk from it in one call, the way a RAG ingestion pipeline actually embeds a document — rather than sweeping arbitrary batch sizes that don't correspond to real client behavior. The chunk cap also keeps every chunk safely under any embedding model's context length, regardless of the source document's formatting.

Each model gets `--warmup` discarded calls first — the very first embed call against a freshly-loaded model pays a one-time model-load cost that has nothing to do with steady-state throughput, so it is absorbed before the `--runs` measured calls (default 3) rather than skewing them. Embedding calls use the engine interface's fixed 120-second request timeout.

If you see repeated connection errors or crashes during the embedding tests (some GPU backends are unstable or immature under batched embedding workloads), try `--cpu-only`. This restarts the active engine with GPU devices hidden for every engine-backed test in the run (`llm`/`conv`/`mcq`/`math`/`code`/`tool`/`emb`/`conc_tool`/`conc_chat`), then restores normal GPU mode afterward. See [CLI Reference](cli-reference.md).

| Model | Tag | Size |
|---|---|---|
| Nomic Embed Text | `nomic-embed-text` | ~0.3 GB |
| MixedBread Embed Large | `mxbai-embed-large` | ~0.7 GB |

## Accuracy

Every LLM model (all four tiers, same models as the LLM test above) answers a fixed bank of 150 multiple-choice questions once each, via a real chat turn (`/v1/chat/completions`) asking for just the letter of the correct answer. Since decoding is deterministic (temperature 0), a single pass through the question bank is representative — repeating it wouldn't change the answers, unlike the performance tests, so this workload ignores `--runs`.

The question bank (`scripts/data/mcq_questions.json`) covers eight categories — science, history, geography, logic, literature, arithmetic, commonsense, and language — with introductory items retained for score continuity and a substantially harder second half. Correct-answer positions are balanced across A–D (38/38/37/37) *and* randomly ordered (seeded, so the file is reproducible) — balance alone doesn't rule out an exploitable fixed-cycle ordering (e.g. "guess A, then B, then C, then D, repeat"), so both properties matter. A model's free-form reply is parsed conservatively: a bare answer letter wins first; otherwise the last boxed or explicitly marked answer wins, then a leading answer marker or leading letter, and finally a single unambiguous uppercase choice mentioned anywhere. Repeated mentions of the same choice are accepted, but competing unmarked choices count as unanswered (wrong).

Results report overall accuracy plus a per-category breakdown, so a model that's strong on arithmetic but weak on commonsense reasoning (or vice versa) is visible rather than averaged away.

Run just this test with `--tests mcq`.

### Timeouts and loop detection

Each accuracy question gets `--acc-timeout` seconds (default 60) to answer — much shorter than the 300-second `--timeout` used elsewhere, since these tests generate one question at a time with an unbounded token budget (a fixed cap risks truncating a reasoning model's answer, so the wall-clock timeout is the real bound). A model that gets stuck reasoning in circles on a single question would otherwise burn the full 300s before anyone found out; at 10% of a 150-question bank that's 15 questions × 300s = 75 minutes lost to one model.

Accuracy warmups and questions use the same explicit 32K server context allocation. This keeps the warmup on the exact llama-server configuration the first scored question uses while leaving enough room for unbounded reasoning within the per-question wall-clock deadline.

A timeout ends only that question; the bank continues. Whatever the model streamed before the cutoff is captured and passed through that workload's normal parser/scorer, so a partial response can still be correct, wrong-but-parseable, or empty. The timeout is recorded regardless of the resulting score.

Each model's results record `timed_out_count` and `timed_out_ids` whenever at least one question reaches the wall-clock limit. Streaming responses are also checked periodically for a loop heuristic: a 12+ word chunk repeated three or more times, or recurring self-correction/hedging phrases such as "wait," and "let me reconsider." That can stop a likely loop before the full timeout, so `likely_loop_ids` is a separate diagnostic rather than a subset of `timed_out_ids`. Completed responses are never loop-checked, and a flagged partial response is retained in `likely_loop_ids` only if its final score is wrong.

### Math

Every LLM model answers a fixed bank of 150 math problems once each (temperature 0, same deterministic-decoding reasoning as MCQ, so this workload also ignores `--runs`), asked to respond with only the final numeric answer. The question bank (`scripts/data/math_questions.json`) spans 30 categories, from arithmetic and word problems through combinatorics, number theory, calculus, linear algebra, statistics, complex numbers, and conditional probability.

A model's free-form reply is parsed in a confidence-ordered cascade: a bare numeric response wins first; otherwise the last boxed or explicitly marked answer wins, then a concluding statement introduced by words such as “therefore,” “thus,” or “so,” and finally the last number as a compatibility fallback. Boxed and explicitly marked answers share one tier so a later self-correction wins. Each answer is checked against the question's known numeric answer within its own per-question tolerance (most are exact); a reply with no number counts as unanswered (wrong).

Results report overall accuracy plus a per-category breakdown, same as MCQ.

Run just this test with `--tests math`.

### Code

Every LLM model answers a fixed bank of 60 coding problems once each (temperature 0, same deterministic-decoding reasoning as MCQ/math, so this workload also ignores `--runs`). The question bank (`scripts/data/code_problems.json`) covers 13 categories — algorithms, arithmetic, divide-and-conquer, dynamic programming, graph, intervals, list, matrix, number theory, search, stack, stateful, and string — with visible and hidden expected-output cases for each problem.

Problems come in two shapes:
- **Function problems** (most of the bank): the model writes one function matching a given name and signature. Each test case is an `args`/`expected` pair.
- **Stateful problems** (category `stateful` — including caches, tries, disjoint sets, and streaming median structures): the model writes a class instead, and each test case is a scenario: construct a fresh instance, call a sequence of methods in order, and compare every return value against an expected sequence. A fresh instance is used per test case, so one scenario's state can never leak into another.

The model's reply is parsed for a fenced Python code block (falling back to the whole reply if it wrote bare code without fencing), then that code is run against every one of the problem's visible *and* hidden test cases in one isolated subprocess under one problem-level deadline — so a model's bad output (infinite loop, crash, syntax error) can't hang or corrupt the benchmark itself, using the same process-isolation-plus-timeout approach as HumanEval-style code-eval harnesses rather than a hardened security sandbox. The harness flushes a private framed result after each completed test, allowing diagnostics from earlier tests to survive when a later test hangs; missing tests are then marked as timed out. Candidate stdout is ignored unless it matches the private result protocol. A problem counts as correct only if every test case passes; a reply with no extractable code, or code that fails even one test case, counts as wrong.

Results report overall accuracy plus a per-category breakdown, same as MCQ/math.

Run just this test with `--tests code`.

### Tool Use

Every LLM model answers a fixed bank of 100 tool-calling questions once each (temperature 0, same deterministic-decoding reasoning as MCQ/math/code, so this workload also ignores `--runs`). Each question offers the model an OpenAI-style `tools` array (function name, description, and JSON-schema parameters) via `/v1/chat/completions` with `tool_choice: "auto"`, and is scored on whether the model called the right tool with the right arguments — or correctly declined to call anything when none of the offered tools genuinely fit. The question bank (`scripts/data/tool_questions.json`) spans 20 five-question categories. The first half covers straightforward calls, basic selection and extraction, enum/numeric/boolean arguments, optional parameters, multi-argument calls, and obvious declines. The harder half covers close tool distinctions, semantic conversions, nested arrays/objects, omitting unspecified optional arguments, semantic enum mapping, missing-information and near-miss declines, large distractor sets, instruction-like content that must remain literal data, and corrections or negations.

The decline cases matter as much as the call cases: a model that fires a tool for a request none of the tools can serve, lacks required information, or would violate an explicit "do not" instruction is as wrong as one that calls the wrong tool. Correct behavior there is calling nothing. A positive case requires exactly one tool call, so emitting the expected call alongside an unintended second action fails.

Argument comparison is recursive. Numeric strings are accepted for numeric values (`"20"` matches `20`), but booleans never match numbers. Baseline questions use subset matching, allowing extra keys for continuity; advanced questions marked `strict_arguments` require the same keys at every nested object level. Arrays are positional by default, while scenarios can mark genuinely set-like fields such as labels or recipients with `unordered_keys`, which compares those arrays as multisets while preserving duplicates. Questions can opt specific free-text fields into whitespace-, case-, and terminal-punctuation-insensitive comparison with `normalized_string_keys`; identifiers and other undeclared strings remain exact. New free-text arguments such as titles, messages, notes, and bodies must be declared there when that tolerance is intended. Because the question-bank hash covers the JSON file, adding or changing this metadata automatically distinguishes the revised tool bank from earlier results.

Results report overall accuracy plus a per-category breakdown, same as MCQ/math/code.

Run just this test with `--tests tool`.

Run every accuracy-style test at once with `--tests acc` — expands to MCQ, math, code, and tool, and de-duplicates against any of them also listed explicitly, without changing how `--tests acc` itself is invoked as more benchmarks join this group in the future. See [CLI Reference](cli-reference.md).

### Bank versioning

Question banks grow and change over time (the MCQ and math banks each doubled in size in one revision, for example), so a raw correct count from one results file is never safely comparable to another without knowing which version of the bank produced it — 40/50 and 40/150 both look like "40 correct" but mean very different things. To make that comparison safe:

- Every results JSON records a `bank_versions` object — a short hash of each accuracy bank's file contents (`mcq`, `math`, `code`, `tool`) at the time of that run, computed from the raw bytes of `scripts/data/*_questions.json` / `code_problems.json` (not just parsed field values, so even a whitespace-only or key-reordering change is caught). Two results files only used the exact same question set if their `bank_versions` entries match.
- The crash cache each accuracy test keeps (`.mcq_crash_cache.json`, `.math_crash_cache.json`, `.code_crash_cache.json`, `.tool_crash_cache.json`) records the bank version a model crashed against, so a model that crashed repeatedly on an old, smaller bank isn't silently skipped forever once the bank has since changed — the stale entry is ignored and the model is retried.
- Percentages normalize for bank size, but a changed bank can also change difficulty and composition. Use matching `bank_versions` hashes for direct model/system comparisons; treat cross-version percentages as contextual rather than apples-to-apples.

`--sample N` (see [CLI Reference](cli-reference.md)) is a separate, dev-only mode for fast local iteration. It uses a deterministic round-robin across categories; every category is represented when `N` is at least that bank's category count, while smaller samples cover as many categories as their size permits. The exact sampled IDs are recorded under `sample_ids`. Sampled runs are reproducible, but are not comparable with full-bank or differently sampled runs.

## Concurrency

Every other LLM test in this suite is strictly one request at a time — these two measure how per-request latency and aggregate throughput scale as multiple simultaneous requests hit the same loaded model, which matters far more than single-stream numbers for anyone thinking about serving more than one user (or one agent's parallel tool-calls) at once. They're two separate tests because "agentic tool-calling fan-out" and "many simultaneous chat users" are genuinely different workload shapes — different concurrency ceilings, different per-request context, and different early-exit tradeoffs — not one sweep with one shape. Both are opt-in via `--tests conc_tool`/`--tests conc_chat` (`--tests conc` runs both) — not part of the default set, since each takes noticeably longer per model than a single request.

Both tests scope to **every LLM model actually downloaded locally**, ignoring `--maxtier` — a machine that only downloaded xsmall/small models tests those; one that downloaded medium/large too tests those as well. This is deliberate: unlike the fixed-tier restriction these tests used to have, download presence is itself a decent proxy for "this machine has the memory to try." `--models` still narrows further within whatever's downloaded.

Every level respawns `llama-server` (the concurrency level is part of `LlamaCppEngine._ensure_model`'s want/have check, so a level change always forces a fresh process), which means each level's first-ever inference is on a brand-new process at that specific concurrent shape. `--warmup` (default 2) throwaway concurrent batches are fired and discarded at each level before the real measured one, for exactly the same reason every other test in this suite warms up before measuring — a fresh process's first real decode can carry one-time overhead (kernel autotuning, CUDA graph capture, and similar) that has nothing to do with steady-state throughput.

At each level, N concurrent requests are fired at once using independent padded single-shot prompts and up to 512 generated tokens per request. Each gets a nonce prefix so no two requests share a cached prompt prefix. After the configured warmup batches, one measured concurrent batch is recorded; `--runs` does not repeat concurrency batches. Results include mean/stdev per-request TTFT and tokens/sec plus aggregate tokens/sec: authoritative native token-ID counts divided by the measured batch's wall-clock duration. Per-request TTFT includes request dispatch, slot queueing, prompt evaluation, first-token sampling, and delivery of the first streamed output.

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

### A note on tok/s outliers

Under heavy concurrent-slot contention, llama-server's own streamed timing data can occasionally misreport a request's decode time as implausibly tiny relative to its token count, which would otherwise show up as a wildly inflated (sometimes six-figure) tok/s reading for that one request. `LlamaCppEngine` sanity-checks every self-reported tok/s value against `config.MAX_PLAUSIBLE_TPS` and substitutes a wall-clock-measured rate whenever the server's number is physically implausible, so this shouldn't show up in results — see `LlamaCppEngine._sanitize_tps` in [Engines](engines.md#llamacppengine).

---

[← Setup](setup.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
