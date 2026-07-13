[← Back to README](../README.md)

# CLI Reference

**Contents**
- [Flag details](#flag-details)
- [Examples](#examples)
- [Comparing results](#comparing-results)

```
run_bench.sh [options]  # Linux / macOS
run_bench.bat [options]   # Windows

--tests TESTS           Tests to run: any of llm conv, emb, img, mcq, or acc as
                        shorthand for every accuracy-style test (currently
                        just mcq) (default: all five — llm conv emb img mcq)
--cpu-only              Force CPU-only inference for every Ollama-backed test
                        (llm, conv, mcq, emb) by restarting Ollama with GPU
                        devices hidden, then restores normal GPU mode
                        afterward
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
--timeout N             Seconds per run before skipping model (default: 300)
--maxtier TIER          Cap LLM models (single-shot + conversation) AND image
                        models at this tier and below: xsmall (<6B / +SD1.5),
                        small (≤20B / +SDXL), medium (26–35B / +SD3.5 Large),
                        large (70B+ / +Flux.1-dev, Flux.2-dev — default, no cap)
--models TAGS           Only test these LLM models (llm/conv/mcq tests) —
                        exact Ollama tags or wildcards, e.g. 'llama*' matches
                        every tag starting with 'llama'. Applied after
                        --maxtier, so it can only narrow that tier's models
                        further (default: every model in the selected tier)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results/results_<hostname>_<timestamp>.json)
--force-all             Ignore the 15 tok/s slow-model cutoff: run every context
                        length in the LLM single-shot test and always run the
                        conversation test, even for models that would otherwise
                        be marked slow and skipped. Doesn't override real
                        failures (timeouts, missing data). Rarely needed —
                        default: false
```

Every test except the LLM conversation test and the MCQ accuracy test measures `--runs` runs (default 3) per checkpoint and averages them; the conversation test always runs a single conversation, and the MCQ test always answers its question bank once (deterministic decoding makes repeats pointless).

## Flag details

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--tests` | any of `llm conv emb img mcq`, plus `acc` | all five (`llm conv emb img mcq`) | Space-separated list; order doesn't matter. `acc` expands to every accuracy-style test (currently just `mcq`) and de-duplicates against any of them also listed explicitly |
| `--cpu-only` | (flag) | off | Restarts Ollama with GPU devices hidden for every Ollama-backed test that's running (`llm`/`conv`/`mcq`/`emb`), then restores normal GPU mode afterward — useful on GPU backends unstable under one of those workloads |
| `--warmup` | integer | `2` | Discarded runs before measured runs, per model/checkpoint |
| `--runs` | integer, `1`–`10` | `3` | Measured runs per checkpoint, averaged. Applies separately to every model and context length in the single-shot LLM test, so total measured time scales roughly in proportion — e.g. 6 runs roughly doubles measured time versus the default. Ignored by the LLM conversation test, which always runs a single conversation. Warmup time is unaffected |
| `--timeout` | integer (seconds) | `300` | Per run (warmup or measured); exceeding it skips the rest of that model |
| `--maxtier` | `xsmall` / `small` / `medium` / `large` | `large` (no cap) | Cumulative — each tier includes every tier below it |
| `--models` | space-separated Ollama tags and/or wildcards (e.g. `llama*`) | none (every model in the selected tier) | Only affects `llm`/`conv`/`mcq` tests. Matching is case-sensitive and exact-or-wildcard (`fnmatch`-style: `*`/`?`/`[...]`), not substring. Applied after `--maxtier`, so it narrows that tier's models rather than adding models outside it |
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

# MCQ accuracy only (also: --tests acc)
bash run_bench.sh --tests mcq

# Cap at small-tier models and below — skips medium/large LLMs and
# medium/large-tier image models (SD3.5 Large, Flux.1-dev, Flux.2-dev),
# leaving SD1.5 and SDXL for the image test
bash run_bench.sh --maxtier small

# Only the Llama models, every tier — wildcard matches every Llama tag
bash run_bench.sh --tests llm --models "llama*"

# One specific model plus a wildcard group
bash run_bench.sh --tests llm --models gpt-oss:20b "deepseek-r1*"

# Give slow hardware more time per run
bash run_bench.sh --timeout 600
```

A full run takes several hours, depending on your hardware and which options you select.

## Comparing results

Copy result files from all machines to one machine, then load them into the [dashboard](dashboard.md).

---

[← Workloads](workloads.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
