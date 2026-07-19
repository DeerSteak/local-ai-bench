[← Back to README](../README.md)

# CLI Reference

**Contents**
- [Flag details](#flag-details)
- [Examples](#examples)
- [Comparing results](#comparing-results)

```
run_bench.sh [options]  # Linux / macOS
run_bench.bat [options]   # Windows

--tests TESTS           Tests to run: any of llm conv, emb, img, mcq, math,
                        code, acc as shorthand for every accuracy-style test
                        (currently mcq, math, and code), or conc (default: all
                        seven — llm conv emb img mcq math code; conc is
                        opt-in — see Workloads)
--engine ENGINE         Inference engine to benchmark against, or 'all' to run
                        the full --tests suite once per registered engine (default:
                        llamacpp). 'all' writes a separate results file per
                        engine (engine name appended to the filename). Only
                        llama.cpp is registered today, so 'all' is currently a
                        no-op — kept so scripts/docs referencing --engine don't
                        need to change once a second engine (e.g. MLX) is added.
                        See Engines
--cpu-only              Force CPU-only inference for every test that goes
                        through the active engine (llm, conv, mcq, math, code,
                        emb, conc) by restarting it with GPU devices hidden,
                        then restores normal GPU mode afterward
--warmup N              Warmup runs before measuring (default: 2)
--runs N                Measured runs per checkpoint, averaged (default: 3,
                        range: 1-10). Applies separately to every model and
                        context length in the single-shot LLM test, so total
                        measured time scales roughly in proportion — e.g.
                        going from 3 to 6 runs roughly doubles it. Ignored by
                        the LLM conversation test, which always runs a single
                        conversation regardless of this flag (it's expensive
                        enough — many turns growing all the way to the
                        sampling ceiling — that repeating it isn't worth the
                        time). Warmup time is unaffected (see --warmup).
--timeout N             Seconds per run before skipping model — applies to
                        warmup and to the llm/conv/emb/img tests (default: 300)
--acc-timeout N         Seconds per question before giving up on it, for the
                        accuracy tests (mcq/math/code) — a timed-out question
                        is scored wrong (using whatever partial text the
                        model had streamed so far, if any) and the run moves
                        on to the next question (default: 60)
--maxtier TIER          Cap LLM models (single-shot + conversation) AND image
                        models at this tier and below: xsmall (<6B / +SD1.5),
                        small (≤20B / +SDXL), medium (26–35B / +SD3.5 Large),
                        large (70B+ / +Flux.1-dev, Flux.2-dev — default, no cap)
--models TAGS           Only test these LLM models (llm/conv/mcq/math/code tests) —
                        exact catalog tags or wildcards, e.g. 'llama*' matches
                        every tag starting with 'llama'. Applied after
                        --maxtier, narrowing the catalog's models further; a
                        pattern matching nothing in the catalog falls back to
                        matching tags actually downloaded locally, so a model
                        outside the curated catalog can still be tested (see
                        --list-models). Quote wildcards so your shell doesn't
                        glob-expand them first (default: every catalog model
                        in the selected tier)
--list-models           List every model actually downloaded locally,
                        marking which are in the curated catalog (models.py)
                        vs custom/extra, then exit without running anything.
                        Useful for finding the exact tag to pass to --models
--sample N              Dev-only: run the accuracy tests (mcq/math/code)
                        against a deterministic N-question subset of each
                        bank instead of the full thing, stratified so every
                        category is represented. The same N always yields the
                        same questions for a given bank version, and the
                        sampled IDs are recorded in the output JSON under
                        'sample_ids'. Never use for a result meant to be
                        compared against a full-bank run, or published
                        (default: full bank)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results/results_<hostname>_<timestamp>.json)
--force-all             Ignore the 15 tok/s slow-model cutoff: run every context
                        length in the LLM single-shot test and always run the
                        conversation test, even for models that would otherwise
                        be marked slow and skipped. Doesn't override real
                        failures (timeouts, missing data). Rarely needed —
                        default: false
```

Every test except the LLM conversation test and the accuracy-style tests (MCQ, math, code) measures `--runs` runs (default 3) per checkpoint and averages them; the conversation test always runs a single conversation, and the accuracy-style tests always answer their question bank once (deterministic decoding makes repeats pointless).

## Flag details

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--tests` | any of `llm conv emb img mcq math code`, plus `acc` and `conc` | all seven (`llm conv emb img mcq math code`) | Space-separated list; order doesn't matter. `acc` expands to every accuracy-style test (currently `mcq`, `math`, and `code`) and de-duplicates against any of them also listed explicitly. `conc` (the concurrency test — see [Concurrency](workloads.md#concurrency)) is opt-in, not part of the default set |
| `--engine` | any registered engine name, or `all` | `llamacpp` | Which inference engine to benchmark against. `all` runs the full `--tests` suite once per registered engine (sorted order) and writes a separate results file for each (engine name appended to the filename). Only llama.cpp is registered today, so `all` behaves identically to the default until a second engine (e.g. MLX) is added. See [Engines](engines.md) |
| `--cpu-only` | (flag) | off | Restarts the engine with GPU devices hidden for every test that goes through it (`llm`/`conv`/`mcq`/`math`/`code`/`emb`/`conc`), then restores normal GPU mode afterward — useful on GPU backends unstable under one of those workloads |
| `--warmup` | integer | `2` | Discarded runs before measured runs, per model/checkpoint |
| `--runs` | integer, `1`–`10` | `3` | Measured runs per checkpoint, averaged. Applies separately to every model and context length in the single-shot LLM test, so total measured time scales roughly in proportion — e.g. 6 runs roughly doubles measured time versus the default. Ignored by the LLM conversation test, which always runs a single conversation. Warmup time is unaffected |
| `--timeout` | integer (seconds) | `300` | Per run (warmup or measured) for `llm`/`conv`/`emb`/`img`, and for every test's warmup; exceeding it skips the rest of that model |
| `--acc-timeout` | integer (seconds) | `60` | Per question for the accuracy tests (`mcq`/`math`/`code`) only; exceeding it scores that one question wrong and moves on to the next — see [Accuracy](workloads.md#accuracy) |
| `--maxtier` | `xsmall` / `small` / `medium` / `large` | `large` (no cap) | Cumulative — each tier includes every tier below it. The concurrency test (`conc`) ignores this — it always restricts to xsmall+small, since each concurrent request needs its own KV cache (see [Concurrency](workloads.md#concurrency)) |
| `--models` | space-separated tags and/or wildcards (e.g. `llama*`) | none (every catalog model in the selected tier) | Affects `llm`/`conv`/`mcq`/`math`/`code`/`conc` tests. Matching is case-sensitive and exact-or-wildcard (`fnmatch`-style: `*`/`?`/`[...]`), not substring. Applied after `--maxtier` (or, for `conc`, after its fixed xsmall+small restriction), narrowing the catalog's models further — but a pattern that matches nothing in the catalog falls back to matching against tags actually downloaded locally, so a model outside the curated catalog (`models.py`) can still be tested. Quote wildcards (`"llama*"`) so your shell doesn't expand them first |
| `--list-models` | (flag) | off | Lists every model actually downloaded locally, tagging each as `catalog` or `custom`, then exits without running anything — the quickest way to find the exact tag to pass to `--models` |
| `--sample` | integer `N` | none (full bank) | Dev-only. Runs `mcq`/`math`/`code` against a deterministic, stratified N-question subset of each bank instead of the full one — every category still represented, same N always picks the same questions for a given bank version. The sampled question IDs are recorded in the results JSON under `sample_ids`. Don't use it for a result meant to be compared against a full-bank run or published — see [bank versioning](workloads.md#bank-versioning) |
| `--comfyui` | path | `./ComfyUI` | Only needed if ComfyUI lives somewhere else |
| `--out` | filename | `results/results_<hostname>_<timestamp>.json` | Overrides the auto-generated path entirely — an explicit path is used as-is, not placed under `results/`. Generated images (`--tests img`) still land under `results/`, in an `images_<name>` folder alongside it (see [Project Structure](project-structure.md)) |
| `--force-all` | (flag) | off | See [LLM workload](workloads.md#llm) for what the slow-model cutoff normally skips |

## Examples

```bash
# Full run — large models skipped automatically if they don't fit
bash run_bench.sh

# LLM only, quick check
bash run_bench.sh --tests llm

# Skip image generation
bash run_bench.sh --tests llm conv emb

# Conversation benchmark only
bash run_bench.sh --tests conv

# Accuracy tests only — MCQ, math, and code (also: --tests acc)
bash run_bench.sh --tests mcq math code

# Cap at small-tier models and below — skips medium/large LLMs and
# medium/large-tier image models (SD3.5 Large, Flux.1-dev, Flux.2-dev),
# leaving SD1.5 and SDXL for the image test
bash run_bench.sh --maxtier small

# Only the Llama models, every tier — wildcard matches every Llama tag
bash run_bench.sh --tests llm --models "llama*"

# One specific model plus a wildcard group
bash run_bench.sh --tests llm --models phi4-mini "nemotron-3*"

# Find the exact tag for a model you've downloaded but isn't in the catalog
bash run_bench.sh --list-models

# Run every registered engine, one pass each — writes a results file per
# engine. Currently a no-op (only llama.cpp is registered)
bash run_bench.sh --engine all --tests llm mcq

# Quick dev iteration on the accuracy tests — 10 questions per bank instead
# of the full thing; never compare this against a full-bank result
bash run_bench.sh --tests acc --sample 10

# Give slow hardware more time per run
bash run_bench.sh --timeout 600

# Give a slower model more time per accuracy question before it's marked wrong
bash run_bench.sh --tests acc --acc-timeout 120

# Concurrency test only — 1/2/4/8/16/32/64-way sweep on xsmall+small models
bash run_bench.sh --tests conc
```

A full run takes several hours, depending on your hardware and which options you select.

## Comparing results

Copy result files from all machines to one machine, then load them into the [dashboard](dashboard.md).

---

[← Workloads](workloads.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
