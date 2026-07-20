[← Back to README](../README.md)

# Workloads

Five workload types are benchmarked: LLM generation (two test modes), image generation, embeddings, accuracy (multiple-choice question answering, math word problems, and coding problems), and concurrency (opt-in — see below). Every workload skips models automatically when they don't fit in available memory — no configuration needed on smaller hardware.

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
  - [Bank versioning](#bank-versioning)
- [Concurrency](#concurrency)

## LLM

Eight models across four tiers (two per tier) are benchmarked by default. If any warmup or measured run exceeds the 300-second timeout, the model is skipped and the benchmark moves on — small GPUs naturally skip the large models without any flags.

Every model is run through **two separate LLM tests**, back to back:

- **Single-shot** — a large prompt, padded to the target size and sent fresh (with unique content) on every run, measured at four context lengths (2K / 8K / 32K / 64K), so it's always a genuine cold prefill of that many tokens with nothing cached. This simulates dropping a large document, codebase, or transcript into a single prompt and asking one question about it.
- **Conversation** — a real multi-turn chat, measured at up to eight depths (0 / 2K / 4K / 8K / 16K / 32K / 64K / 96K, whichever the model's own context window reaches): the model explains Plato's Allegory of the Cave, then each following turn asks for more detail on a section, growing the conversation from a blank slate toward 96K. This test is expensive (many turns growing all the way to the sampling ceiling), so it always runs a single conversation regardless of `--runs` — that flag only repeats the other tests (see [CLI Reference](cli-reference.md)).

The context window handed to a model for the conversation test is still the full 128K if it supports at least that much, otherwise its own real maximum — read live from the downloaded GGUF's own metadata for the exact model that's actually installed, not assumed or hardcoded. Sampling deliberately stops at 96K rather than 128K so the conversation always has real headroom left against that context window instead of scraping the ceiling — most models' real maximum is exactly 128K, with no slack to spare for the growth loop's final turns.

These two tests measure genuinely different things, and their TTFT numbers are **not** comparable at face value — see [What the charts mean](dashboard.md#what-the-charts-mean) for why the conversation test's TTFT is typically far lower than the single-shot test's at the same nominal context length.

If a model's single-shot decode speed drops below 15 tok/s at some context length, the single-shot test stops there — deeper context lengths are skipped for that model, since a slower run would just be a longer wait for a data point nobody needs. A model is excluded from the conversation test *entirely* if it timed out or was already marked too slow in the single-shot test (e.g. too large to fit in memory) — that pre-flight check only runs when the single-shot test ran earlier in the same session; running `--tests conv` on its own has no single-shot data to check against, so every model is tested there.

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

**Reasoning models** (Nemotron 3 Nano here, a unified model for both reasoning and non-reasoning tasks) generate internal thinking tokens before their answer, via llama-server's own separate `reasoning_content` field rather than mixing them into the answer text. Tokens/sec includes this thinking output — the engine's reported generation count and duration cover the whole response, thinking included, with no separate accounting. TTFT reflects prompt-processing time only (the engine's reported prompt-eval duration), which happens before generation starts, so it is not affected by how much the model reasons afterward.

**Llama versions:** Llama 3.2 tops out at 3B parameters; the 8B slot uses Llama 3.1; the large tier's dense entry uses Llama 3.3 70B. Llama 4 Scout is the large tier's MoE entry, at 16 experts (17B active of ~109B total).

`--maxtier` caps LLM models (and image models, see below) at a given tier and below; `--models` narrows further to specific tags or wildcards (e.g. `--models "llama*"`) within whatever tier is selected — see [CLI Reference](cli-reference.md).

## Image Generation

Five models are tested at 1024×1024 and 1536×1536 — except Stable Diffusion 1.5, which uses 512×512 and 768×768 instead (see below). Any model whose checkpoint is absent from `ComfyUI/models/checkpoints/` is skipped automatically; `setup_check.py` downloads them on first run.

Each measured run (`--runs`, default 3) uses a different seed (`seed + run index`) — an identical seed and workflow would let ComfyUI cache every node and return a cached result almost instantly instead of actually re-running generation. The warmup run before those also uses its own distinct seed, for the same reason.

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

Generated sample images are saved to `results/images_<hostname>_<timestamp>/`, alongside the matching results JSON — see [Project Structure](project-structure.md).

## Embeddings

Two models — Nomic Embed Text and MixedBread Embed Large — measured on a single real-world task: chunking a real multi-chapter document (`sample_document.txt`, ~27 chapters) into paragraph-sized pieces (capped at 150 words each) and embedding every chunk from it in one call, the way a RAG ingestion pipeline actually embeds a document — rather than sweeping arbitrary batch sizes that don't correspond to real client behavior. The chunk cap also keeps every chunk safely under any embedding model's context length, regardless of the source document's formatting.

Like the other test types, each model gets `--warmup` discarded runs first — the very first embed call against a freshly-loaded model pays a one-time model-load cost that has nothing to do with steady-state throughput, so it's absorbed before the `--runs` measured runs (default 3) rather than skewing them. The active engine uses the GPU on all supported platforms (Metal, CUDA, ROCm), so results are directly comparable across machines.

If you see repeated connection errors or crashes during the embedding tests (some GPU backends are unstable or immature under batched embedding workloads), try `--cpu-only` to force CPU-only inference instead — in some cases this is also faster or just more stable than a flaky GPU path. This restarts the active engine with GPU devices hidden for every test in the run that goes through it (`llm`/`conv`/`mcq`/`emb`, not just embeddings), then restores normal GPU mode afterward. See [CLI Reference](cli-reference.md).

| Model | Tag | Size |
|---|---|---|
| Nomic Embed Text | `nomic-embed-text` | ~0.3 GB |
| MixedBread Embed Large | `mxbai-embed-large` | ~0.7 GB |

## Accuracy

Every LLM model (all four tiers, same models as the LLM test above) answers a fixed bank of 150 multiple-choice questions once each, via a real chat turn (`/v1/chat/completions`) asking for just the letter of the correct answer. Since decoding is deterministic (temperature 0), a single pass through the question bank is representative — repeating it wouldn't change the answers, unlike the performance tests, so this workload ignores `--runs`.

The question bank (`scripts/data/mcq_questions.json`) covers eight categories — science, history, geography, logic, literature, arithmetic, commonsense, and language — with introductory items retained for score continuity and a substantially harder second half. Correct-answer positions are balanced across A–D (38/38/37/37) *and* randomly ordered (seeded, so the file is reproducible) — balance alone doesn't rule out an exploitable fixed-cycle ordering (e.g. "guess A, then B, then C, then D, repeat"), so both properties matter. A model's free-form reply is scanned for the first standalone letter (A–D) that's actually one of that question's choices, so a model that reasons out loud before answering ("...so the answer is B") is still scored correctly; a reply with no matching letter counts as unanswered (wrong).

Results report overall accuracy plus a per-category breakdown, so a model that's strong on arithmetic but weak on commonsense reasoning (or vice versa) is visible rather than averaged away.

Run just this test with `--tests mcq`.

### Timeouts and loop detection

Each accuracy question gets `--acc-timeout` seconds (default 60) to answer — much shorter than the 300-second `--timeout` used elsewhere, since these tests generate one question at a time with an unbounded token budget (a fixed cap risks truncating a reasoning model's answer, so the wall-clock timeout is the real bound). A model that gets stuck reasoning in circles on a single question would otherwise burn the full 300s before anyone found out; at 10% of a 150-question bank that's 15 questions × 300s = 75 minutes lost to one model.

A timed-out question is scored wrong and the run moves on to the next question — a single timeout only affects that one question rather than the whole bank, since zeroing out every remaining question would unfairly penalize a model that's merely slow on one hard question rather than actually stuck. Whatever text the model had already streamed before the cutoff is captured and scored the same as a completed answer (parsed for a valid letter/number/code block), rather than treated as a blank — a cut-off response can still be right, wrong-but-parseable, or genuinely empty, and those are different outcomes worth telling apart.

Each model's results record `timed_out_count` and `timed_out_ids` (which questions hit the timeout) whenever at least one did. Every timed-out response is additionally checked against a loop heuristic — flagging a response if a 12+ word chunk repeats verbatim three or more times, or if a self-correction/hedging phrase ("wait,", "let me reconsider", "there seems to have been...", "apolog-", etc.) recurs three or more times — since a model that gets cut off mid-answer is either genuinely still working through a hard question or stuck restating the same reasoning (or the same code, one indentation level deeper each time) without ever converging. This check only ever runs on a timed-out question's partial text — a completed, submitted answer is never checked for looping, no matter how wrong or repetitive-looking it is, since a wrong answer that was actually submitted isn't a loop. Models with at least one flagged response get `likely_loop_count` and `likely_loop_ids` in their results, alongside `timed_out_count`/`timed_out_ids`.

### Math

Every LLM model answers a fixed bank of 150 math problems once each (temperature 0, same deterministic-decoding reasoning as MCQ, so this workload also ignores `--runs`), asked to respond with only the final numeric answer. The question bank (`scripts/data/math_questions.json`) spans 30 categories, from arithmetic and word problems through combinatorics, number theory, calculus, linear algebra, statistics, complex numbers, and conditional probability.

A model's free-form reply is scanned for the *last* number it states, not the first — a model that reasons out loud before answering ("347 + 589 = 936, so the answer is 936") states its final answer last, and intermediate numbers earlier in the reasoning shouldn't be mistaken for it. Each answer is checked against the question's known numeric answer within its own per-question tolerance (most are exact); a reply with no number counts as unanswered (wrong).

Results report overall accuracy plus a per-category breakdown, same as MCQ.

Run just this test with `--tests math`.

### Code

Every LLM model answers a fixed bank of 60 coding problems once each (temperature 0, same deterministic-decoding reasoning as MCQ/math, so this workload also ignores `--runs`). The question bank (`scripts/data/code_problems.json`) covers 13 categories — algorithms, arithmetic, divide-and-conquer, dynamic programming, graph, intervals, list, matrix, number theory, search, stack, stateful, and string — with visible and hidden expected-output cases for each problem.

Problems come in two shapes:
- **Function problems** (most of the bank): the model writes one function matching a given name and signature. Each test case is an `args`/`expected` pair.
- **Stateful problems** (category `stateful` — including caches, tries, disjoint sets, and streaming median structures): the model writes a class instead, and each test case is a scenario: construct a fresh instance, call a sequence of methods in order, and compare every return value against an expected sequence. A fresh instance is used per test case, so one scenario's state can never leak into another.

The model's reply is parsed for a fenced Python code block (falling back to the whole reply if it wrote bare code without fencing), then that code is run against every one of the problem's visible *and* hidden test cases in an isolated subprocess — so a model's bad output (infinite loop, crash, syntax error) can't hang or corrupt the benchmark itself, using the same process-isolation-plus-timeout approach as HumanEval-style code-eval harnesses rather than a hardened security sandbox. A problem counts as correct only if every test case passes; a reply with no extractable code, or code that fails even one test case, counts as wrong.

Results report overall accuracy plus a per-category breakdown, same as MCQ/math.

Run just this test with `--tests code`.

Run every accuracy-style test at once with `--tests acc` — expands to MCQ, math, and code, and de-duplicates against any of them also listed explicitly, without changing how `--tests acc` itself is invoked as more benchmarks join this group in the future. See [CLI Reference](cli-reference.md).

### Bank versioning

Question banks grow and change over time (the MCQ and math banks each doubled in size in one revision, for example), so a raw correct count from one results file is never safely comparable to another without knowing which version of the bank produced it — 40/50 and 40/150 both look like "40 correct" but mean very different things. To make that comparison safe:

- Every results JSON records a `bank_versions` object — a short hash of each accuracy bank's file contents (`mcq`, `math`, `code`) at the time of that run, computed from the raw bytes of `scripts/data/*_questions.json` / `code_problems.json` (not just parsed field values, so even a whitespace-only or key-reordering change is caught). Two results files only used the exact same question set if their `bank_versions` entries match.
- The crash cache each accuracy test keeps (`.mcq_crash_cache.json`, `.math_crash_cache.json`, `.code_crash_cache.json`) records the bank version a model crashed against, so a model that crashed repeatedly on an old, smaller bank isn't silently skipped forever once the bank has since changed — the stale entry is ignored and the model is retried.
- Accuracy percentages (as opposed to raw correct counts) stay meaningfully comparable across bank versions, since they're already normalized by the bank's size at that time.

`--sample N` (see [CLI Reference](cli-reference.md)) is a separate, dev-only mode for fast local iteration: instead of the full bank, it runs a deterministic, stratified N-question subset — every category still represented, and the same N always draws the same questions for a given bank version. The exact sampled question IDs are recorded in the results JSON under `sample_ids`, so a sampled run is reproducible and auditable, but it's never meant to be compared against a full-bank run or published as a real score.

## Concurrency

Every other LLM test in this suite is strictly one request at a time — these two measure how per-request latency and aggregate throughput scale as multiple simultaneous requests hit the same loaded model, which matters far more than single-stream numbers for anyone thinking about serving more than one user (or one agent's parallel tool-calls) at once. They're two separate tests because "agentic tool-calling fan-out" and "many simultaneous chat users" are genuinely different workload shapes — different concurrency ceilings, different per-request context, and different early-exit tradeoffs — not one sweep with one shape. Both are opt-in via `--tests conc_tool`/`--tests conc_chat` (`--tests conc` runs both) — not part of the default set, since each takes noticeably longer per model than a single request.

Both tests scope to **every LLM model actually downloaded locally**, ignoring `--maxtier` — a machine that only downloaded xsmall/small models tests those; one that downloaded medium/large too tests those as well. This is deliberate: unlike the fixed-tier restriction these tests used to have, download presence is itself a decent proxy for "this machine has the memory to try." `--models` still narrows further within whatever's downloaded.

Every level respawns `llama-server` (the concurrency level is part of `LlamaCppEngine._ensure_model`'s want/have check, so a level change always forces a fresh process), which means each level's first-ever inference is on a brand-new process at that specific concurrent shape. `--warmup` (default 2) throwaway concurrent batches are fired and discarded at each level before the real measured one, for exactly the same reason every other test in this suite warms up before measuring — a fresh process's first real decode can carry one-time overhead (kernel autotuning, CUDA graph capture, and similar) that has nothing to do with steady-state throughput.

At each level, N concurrent requests are fired at once (a real prompt padded to the test's per-request context size, not a multi-turn conversation — much cheaper to hit an exact context size in one shot than growing there turn by turn, see the LLM conversation test above). Each one gets its own nonce prefix so none of them share a cached prompt prefix with each other — without that, an engine's prefix cache would serve some requests near-instantly regardless of real concurrency, understating exactly the contention this test exists to measure. Two numbers are recorded per level: the mean/stdev of each individual request's own TTFT and tokens/sec (the number that should visibly degrade as concurrency climbs), and the aggregate tokens/sec — total tokens generated across every concurrent stream, divided by that batch's real wall-clock duration (the number that shows overall system capacity, which typically climbs, plateaus, then can decline past a saturation point).

### Tool (`conc_tool`)

Simulates agentic/tool-calling fan-out — a handful of concurrent requests, each a short tool-call-shaped turn. Swept through concurrency levels 1, 2, 4, 6, 8, 12, and 16, each request given 4,096 tokens of its own context (short, matching a real tool-call turn's shape — system prompt, schema, a short result — not a long document).

This test **never soft-exits on slow tok/s** — every level always runs and gets recorded, since the whole ceiling here (16-way) is cheap enough that a real data point at every level is worth more than an inferred one. Only a hard stop (see below) ends its sweep early.

### Chat (`conc_chat`)

Simulates a chat server under load — many simultaneous long-conversation users. Swept through concurrency levels 1, 2, 4, 8, 16, 24, and 32, each request given 16,384 tokens of its own context (a long conversation history, at scale).

Unlike the tool test, this one **does** soft-exit (see below) — at up to 32-way concurrency, a model that's already cratered to a few tokens/sec per request costs an enormous amount of wall-clock time to keep climbing for a foregone conclusion.

### Escalation stopping

Escalation to the next level stops for one of two reasons:
- **Hard stop** (both tests) — the model fails to even load at that level (out of memory, hung load) or the engine's runner crashes mid-batch. This is the real ceiling for that model on this hardware, not a bug — repeated engine crashes are the only case recorded to a crash cache (`.concurrency_tool_crash_cache.json` / `.concurrency_chat_crash_cache.json`, one per test so a crash on one doesn't affect retry state on the other), since a load failure at a given level isn't something worth remembering to skip on the next run (a lower level is cheap to retry and will very likely still succeed).
- **Soft stop** (chat only) — once concurrency level 8 has actually been reached, per-request tokens/sec dropping below the usual slow-model cutoff means climbing further would only confirm what's already obvious. Levels 1, 2, 4, and 8 always run and get recorded regardless of how slow they already look. `--force-all` disables this the same way it does elsewhere.

### Memory snapshots

Each level also records a memory snapshot, taken right after that level finishes loading (model + full KV cache allocated) and before the batch fires — the steadiest point to read how much headroom is actually left, rather than numbers that fluctuate mid-batch. It always includes system RAM used/total; GPU VRAM used/total is added too when `nvidia-smi` or `rocm-smi` answers (a `rocm-smi` reading is only trusted for a confirmed-discrete AMD card — an APU's reported VRAM is often just a small BIOS-fixed carve-out, not the real usable pool). On a unified-memory machine — Apple Silicon, or an NVIDIA/AMD box like a DGX Spark or Strix Halo, where the model competes with the OS for the same physical pool — system RAM is the number that actually reflects total headroom; GPU VRAM there is supplementary, not a substitute. A load failure also records a `memory_at_failure` snapshot, so it's clear what memory state actually triggered the ceiling.

### A note on tok/s outliers

Under heavy concurrent-slot contention, llama-server's own streamed timing data can occasionally misreport a request's decode time as implausibly tiny relative to its token count, which would otherwise show up as a wildly inflated (sometimes six-figure) tok/s reading for that one request. `LlamaCppEngine` sanity-checks every self-reported tok/s value against `config.MAX_PLAUSIBLE_TPS` and substitutes a wall-clock-measured rate whenever the server's number is physically implausible, so this shouldn't show up in results — see `LlamaCppEngine._sanitize_tps` in [Engines](engines.md#llamacppengine).

---

[← Setup](setup.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
