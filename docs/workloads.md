[← Back to README](../README.md)

# Workloads

Four workload types are benchmarked: LLM generation (two test modes), image generation, embeddings, and accuracy (multiple-choice question answering, math word problems, and coding problems). Every workload skips models automatically when they don't fit in available memory — no configuration needed on smaller hardware.

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

## LLM

Twelve models across four tiers are benchmarked by default. If any warmup or measured run exceeds the 300-second timeout, the model is skipped and the benchmark moves on — small GPUs naturally skip the large models without any flags.

Every model is run through **two separate LLM tests**, back to back:

- **Single-shot** — a large prompt, padded to the target size and sent fresh (with unique content) on every run, measured at four context lengths (2K / 8K / 32K / 64K), so it's always a genuine cold prefill of that many tokens with nothing cached. This simulates dropping a large document, codebase, or transcript into a single prompt and asking one question about it.
- **Conversation** — a real multi-turn chat, measured at up to eight depths (0 / 2K / 4K / 8K / 16K / 32K / 64K / 96K, whichever the model's own context window reaches): the model explains Plato's Allegory of the Cave, then each following turn asks for more detail on a section, growing the conversation from a blank slate toward 96K. This test is expensive (many turns growing all the way to the sampling ceiling), so it always runs a single conversation regardless of `--runs` — that flag only repeats the other tests (see [CLI Reference](cli-reference.md)).

The context window handed to a model for the conversation test is still the full 128K if it supports at least that much, otherwise its own real maximum — looked up live from Ollama for the exact model that's actually pulled, not assumed or hardcoded. Sampling deliberately stops at 96K rather than 128K so the conversation always has real headroom left against that context window instead of scraping the ceiling — most models' real maximum is exactly 128K, with no slack to spare for the growth loop's final turns.

These two tests measure genuinely different things, and their TTFT numbers are **not** comparable at face value — see [What the charts mean](dashboard.md#what-the-charts-mean) for why the conversation test's TTFT is typically far lower than the single-shot test's at the same nominal context length.

If a model's single-shot decode speed drops below 15 tok/s at some context length, the single-shot test stops there — deeper context lengths are skipped for that model, since a slower run would just be a longer wait for a data point nobody needs. A model is excluded from the conversation test *entirely* if it timed out or was already marked too slow in the single-shot test (e.g. too large to fit in memory) — that pre-flight check only runs when the single-shot test ran earlier in the same session; running `--tests conv` on its own has no single-shot data to check against, so every model is tested there.

Separately, *within* the conversation test itself: if the decode speed at any history depth drops below the slow-model cutoff, the conversation exits early and records results to that point. Pass `--force-all` to ignore this cutoff and always run every context length (see [CLI Reference](cli-reference.md)).

### Extra-small tier (<6B params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| Llama 3.2 3B Q4_K_M | `llama3.2:3b-instruct-q4_K_M` | ~2.1 GB | Dense |
| Phi 4 Mini | `phi4-mini` | ~2.5 GB | Dense |
| Qwen3.5 4B | `qwen3.5:4b` | ~3.4 GB | Dense |

### Small tier (≤20B params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| Llama 3.1 8B Q4_K_M | `llama3.1:8b-instruct-q4_K_M` | ~5.0 GB | Dense |
| Gemma 4 E4B | `gemma4:e4b` | ~9.7 GB | Dense (Per-Layer Embeddings — see below) |
| GPT-OSS 20B (MXFP4) | `gpt-oss:20b` | ~13.8 GB | MoE — 3.6B active of ~20.9B total |

### Medium tier (26–35B params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| Gemma 4 26B | `gemma4:26b` | ~18.0 GB | MoE — 4B active of ~26B total |
| DeepSeek-R1 32B | `deepseek-r1:32b` | ~19.9 GB | Dense |
| Qwen3.6 35B-A3B | `qwen3.6:35b-a3b` | ~24.0 GB | MoE — 3B active of 35B total |

### Large tier (70B+ params)

| Model | Ollama tag | Size | Architecture |
|---|---|---|---|
| DeepSeek-R1 70B | `deepseek-r1:70b` | ~42.6 GB | Dense |
| Llama 4 Scout 16x17B | `llama4:16x17b` | ~67.0 GB | MoE — 17B active of ~109B total |
| GPT-OSS 120B (MXFP4) | `gpt-oss:120b` | ~65.4 GB | MoE — 5.1B active of ~116.8B total |

### Dense vs. Mixture-of-Experts (MoE)

A **dense** model runs every one of its parameters for every token it generates. A **Mixture-of-Experts (MoE)** model instead routes each token through only a small subset of specialized "expert" sub-networks, out of many more it holds in total — so most of its parameters sit idle on any given token. Ollama tags spell this out for MoE variants with an `-aN` suffix (e.g. `qwen3.6:35b-a3b`) or in the model's own naming (e.g. Gemma's `26B-A4B`): the number after `a` is how many parameters actually activate per token ("active"), versus the number before it (total parameters, which is what drives memory/VRAM use).

Because decode speed tracks active parameters far more closely than total size or VRAM footprint, an MoE model can generate noticeably faster than a dense model of similar total size — in the medium tier here, Qwen3.6 35B-A3B activates only ~3B parameters per token versus DeepSeek-R1 32B's dense 32B, despite both landing in a similar VRAM footprint (~20–22 GB). Gemma 4 E4B is a dense model that uses a different technique, Per-Layer Embeddings (PLE), to shrink its loaded memory footprint (~8B total parameters, ~4.5B "effective") without changing how much compute each token costs — every layer's core weights still run for every token, unlike MoE's routing.

**DeepSeek-R1** is a reasoning model that generates internal thinking tokens before its answer, via the engine's own separate reasoning field (Ollama's `thinking`, llama-server's `reasoning_content`) rather than mixing them into the answer text. Tokens/sec includes this thinking output — the engine's reported generation count and duration cover the whole response, thinking included, with no separate accounting. TTFT reflects prompt-processing time only (the engine's reported prompt-eval duration), which happens before generation starts, so it is not affected by how much the model reasons afterward.

**Llama versions:** Llama 3.2 tops out at 3B parameters. The 8B slot uses Llama 3.1; the 70B slot uses Llama 3.3, the most recent 70B instruction model.

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

Two models via Ollama — Nomic Embed Text and MixedBread Embed Large — measured on a single real-world task: chunking a real multi-chapter document (`sample_document.txt`, ~27 chapters) into paragraph-sized pieces (capped at 150 words each) and embedding every chunk from it in one call, the way a RAG ingestion pipeline actually embeds a document — rather than sweeping arbitrary batch sizes that don't correspond to real client behavior. The chunk cap also keeps every chunk safely under any embedding model's context length, regardless of the source document's formatting.

Like the other test types, each model gets `--warmup` discarded runs first — the very first embed call against a freshly-loaded model pays a one-time model-load cost that has nothing to do with steady-state throughput, so it's absorbed before the `--runs` measured runs (default 3) rather than skewing them. The active engine uses the GPU on all supported platforms (Metal, CUDA, ROCm), so results are directly comparable across machines.

If you see repeated connection errors or crashes during the embedding tests (some GPU backends are unstable or immature under batched embedding workloads), try `--cpu-only` to force CPU-only inference instead — in some cases this is also faster or just more stable than a flaky GPU path. This restarts the active engine with GPU devices hidden for every test in the run that goes through it (`llm`/`conv`/`mcq`/`emb`, not just embeddings), then restores normal GPU mode afterward. See [CLI Reference](cli-reference.md).

| Model | Ollama tag | Size |
|---|---|---|
| Nomic Embed Text | `nomic-embed-text` | ~0.3 GB |
| MixedBread Embed Large | `mxbai-embed-large` | ~0.7 GB |

## Accuracy

Every LLM model (all four tiers, same models as the LLM test above) answers a fixed bank of 150 multiple-choice questions once each, via a real chat turn (`/api/chat`) asking for just the letter of the correct answer. Since decoding is deterministic (temperature 0), a single pass through the question bank is representative — repeating it wouldn't change the answers, unlike the performance tests, so this workload ignores `--runs`.

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

---

[← Setup](setup.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
