[← Back to README](../README.md)

# CLI Reference

**Contents**
- [Launch modes](#launch-modes)
- [Flag details](#flag-details)
- [Examples](#examples)
- [Comparing results](#comparing-results)

```
run_bench.sh [options]  # Linux / macOS
run_bench.bat [options]   # Windows

--tests TESTS           Tests to run: any of llm conv, emb, img, mcq, math,
                        reasoning, code, tool, acc as shorthand for every accuracy-style
                        test (mcq, math, reasoning, code, and tool), or
                        conc_tool/conc_chat (conc is shorthand for both)
                        (default: all nine — llm conv emb img mcq math reasoning code
                        tool; conc_tool/conc_chat are opt-in — see Workloads)
--engine ENGINE         Inference engine to benchmark against, or 'all' to run
                        the full --tests suite once per registered engine (default:
                        llamacpp). 'all' writes a separate results file per
                        engine (engine name appended to the filename). Only
                        llama.cpp is registered today, so 'all' is currently a
                        no-op — kept so scripts/docs referencing --engine don't
                        need to change once a second engine (e.g. MLX) is added.
                        See Engines
--cpu-only              Force CPU-only inference for every test that goes
                        through the active engine (llm, conv, mcq, math, reasoning, code,
                        tool, emb, conc_tool, conc_chat) by restarting it with
                        GPU devices hidden, then restores normal GPU mode afterward
--warmup N              Engine-backed warmups before measuring (default: 2).
                        Images always use one warmup and ignore this flag
--runs N                Measured runs averaged for single-shot LLM, embeddings,
                        and images (default: 3, range: 1-10). Conversation and
                        accuracy use one pass; concurrency uses one measured
                        batch per level. Warmup time is unaffected
--timeout N             Seconds per generation/chat call and engine warmup
                        (default: 300). Images use 2x this value, embeddings
                        use a fixed 120s, and accuracy uses --acc-timeout
--acc-timeout N         Seconds per question before giving up on it, for the
                        accuracy tests (mcq/math/reasoning/code/tool). Any partial response
                        is scored normally, the timeout is recorded, and the
                        run moves on to the next question (default: 60)
--maxtier TIER          Cap LLM models (single-shot + conversation) AND image
                        models at this tier and below: xsmall (<6B / +SD1.5),
                        small (≤20B / +SDXL), medium (26–35B / +SD3.5 Large),
                        large (70B+ / +Flux.1-dev, Flux.2-dev — default, no cap)
--llm-models TAGS       Only test these LLM models (llm/conv/mcq/math/reasoning/code/tool/
                        conc_tool/conc_chat tests) —
                        exact catalog tags or wildcards, e.g. 'llama*' matches
                        every tag starting with 'llama'. Applied after
                        --maxtier, narrowing the catalog's models further while
                        also matching installed custom-model tags, so one wildcard
                        can select both catalog and custom models (see --list-models).
                        Quote wildcards so your shell doesn't
                        glob-expand them first (default: every catalog model
                        in the selected tier). --models is a backward-compatible
                        alias with identical behavior
--embedding-models TAGS Only test these embedding catalog tags or wildcards
                        (default: every embedding model)
--image-models SHORTS   Only test these image-model short IDs or wildcards,
                        applied after --maxtier (default: every image model in
                        the selected tier)
--list-models           List installed catalog LLMs, embeddings, custom LLMs,
                        and catalog image checkpoints, then exit. Image discovery
                        uses the effective --comfyui path
--sample N              Dev-only: run the accuracy tests (mcq/math/reasoning/code/tool)
                        against a deterministic N-question subset of each
                        bank instead of the full thing, selected round-robin
                        across categories. Every category is represented when
                        N reaches that bank's category count. The same N always
                        yields the same questions for a given bank version, and the
                        sampled IDs are recorded in the output JSON under
                        'sample_ids'. Never use for a result meant to be
                        compared against a full-bank run, or published
                        (default: full bank)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results/results_<hostname>_<timestamp>.json)
--force-all             Ignore the 15 tok/s slow-model cutoff: run every context
                        length in the LLM single-shot and conversation tests,
                        and disable the chat-concurrency soft exit. Doesn't
                        override real failures (timeouts, missing data). Rarely needed —
                        default: false
```

## Launch modes

Running `run_bench.sh` or `run_bench.bat` with no arguments opens the interactive launcher. It reads the selected engine and the setup-managed `./ComfyUI` installation, then immediately shows the test checklist. Only installed models are shown. The confirmed selection is translated into the public flags below and launches the non-interactive `scripts/benchmark.py` CLI. The launcher selects one engine at a time; `--engine all` remains available through the direct CLI because inventories can differ between engines.

Passing even one argument bypasses the launcher and forwards every argument directly to `benchmark.py`, preserving existing automation and direct CLI defaults. This includes `--help` and `--list-models`. Calling `python scripts/benchmark.py ...` directly is also always non-interactive.

After confirmation, the launcher saves the selected engine, tests, and model IDs to the gitignored `.benchmark_frontend_state.json` in the project root. The next no-argument launch restores entries that are still installed and available and labels the menus as restored. Delete `.benchmark_frontend_state.json` to reset the launcher to current defaults. A missing, malformed, or incompatible file uses the defaults below. If an entire remembered test or model family is no longer applicable, that family also falls back to the current defaults; stale entries never make an installed model appear or block the menu. Cancelling does not save, and a state-write failure warns but does not prevent the benchmark from starting.

On Windows, double-clicking `run_bench.bat` uses a best-effort Explorer-launch check to pause after completion so the final status remains visible. Launches from an existing command prompt exit normally. The pause affects presentation only; the batch file saves and returns the benchmark's original exit code.

On first use or when no saved selection applies, the interactive launcher's state is:

| Area | Initial state |
|---|---|
| Engine | The CLI's default registered engine (currently `llamacpp`); selectable first if multiple engines exist |
| ComfyUI directory | The setup-managed `./ComfyUI` directory; there is no extra path prompt |
| Single-shot LLM and conversation | Checked when an installed catalog or custom LLM is available |
| Embeddings | Checked when an installed embedding model is available |
| Image generation | Checked when an installed catalog image checkpoint is available |
| MCQ, math, reasoning, code, and tool accuracy | Unchecked |
| Tool and chat concurrency | Unchecked |
| Installed xsmall/small/medium catalog LLMs | Checked |
| Installed large catalog LLMs | Unchecked |
| Installed custom LLMs | Unchecked; displayed by folder name and excluded from tier toggles |
| Installed embedding models | Checked; individually toggleable or grouped with `emb` |
| Installed xsmall/small/medium image models | Checked |
| Installed large-tier image models | Unchecked, including Flux.1-dev and Flux.2-dev |
| Uninstalled models | Not displayed |

The model screen uses one LLM selection for single-shot, conversation, accuracy, and concurrency tests. Number/range controls toggle individual models. `xs`, `s`, `m`, and `l` toggle installed catalog LLM and image models in that tier together: an all-selected tier becomes unselected, while a partially selected or unselected tier becomes fully selected. `custom` and `emb` independently toggle those groups. If catalog models are missing, a read-only hint reports counts by family and suggests `bash setup.sh` or `setup.bat`; it never runs setup.

After the selection summary, `Start this benchmark? [Y/n]` defaults to yes; press Enter to launch or enter `n` to cancel.

The interactive launcher clears the terminal before its initial display, between menu screens, and before subsequent redraws while preserving the welcome banner through the first single-engine test screen and the final model choices through confirmation. It uses the native `cls` command on Windows and ANSI terminal clearing elsewhere. Launcher prompts remain untimestamped. Once execution starts, benchmark status and progress messages are prefixed with local time as `[HH:MM:SS]`. Model responses, results data, answer sidecars, and generated artifacts are unchanged.

`--runs` applies only to single-shot LLM, embeddings, and image generation. Conversation and each accuracy test make one measured pass, while concurrency records one measured batch per level.

## Flag details

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--tests` | any of `llm conv emb img mcq math reasoning code tool`, plus `acc`, `conc_tool`, `conc_chat`, and `conc` | all nine (`llm conv emb img mcq math reasoning code tool`) | Space-separated list; order doesn't matter. `acc` expands to every accuracy-style test (`mcq`, `math`, `reasoning`, `code`, and `tool`) and de-duplicates against any of them also listed explicitly; `conc` expands the same way to `conc_tool conc_chat`. `conc_tool` (agentic/tool-calling fan-out, 1–16-way) and `conc_chat` (many simultaneous chat users, 1–32-way) — see [Concurrency](workloads.md#concurrency) — are opt-in, not part of the default set |
| `--engine` | any registered engine name, or `all` | `llamacpp` | Which inference engine to benchmark against. `all` runs the full `--tests` suite once per registered engine (sorted order) and writes a separate results file for each (engine name appended to the filename). Only llama.cpp is registered today, so `all` behaves identically to the default until a second engine (e.g. MLX) is added. See [Engines](engines.md) |
| `--cpu-only` | (flag) | off | Restarts the engine with GPU devices hidden for every test that goes through it (`llm`/`conv`/`mcq`/`math`/`reasoning`/`code`/`tool`/`emb`/`conc_tool`/`conc_chat`), then restores normal GPU mode afterward — useful on GPU backends unstable under one of those workloads |
| `--warmup` | integer | `2` | Discarded warmups before measurement for every engine-backed workload: once per loaded model for LLM/conversation/accuracy, per model call for embeddings, and per concurrency level. Image generation always performs one warmup at the model's first resolution and does not use this flag |
| `--runs` | integer, `1`–`10` | `3` | Measured runs averaged for single-shot LLM at each context, embeddings, and images at each resolution. Ignored by conversation, accuracy, and concurrency. Warmup count is unaffected |
| `--timeout` | integer (seconds) | `300` | Per generation/chat call for single-shot, conversation, concurrency, and their engine warmups. Images use twice this value (600s by default). Embedding calls retain the engine's fixed 120s timeout; accuracy questions use `--acc-timeout` |
| `--acc-timeout` | integer (seconds) | `60` | Per question for `mcq`/`math`/`reasoning`/`code`/`tool`; the partial response is scored normally, the timeout is recorded, and the bank continues — see [Accuracy](workloads.md#accuracy) |
| `--maxtier` | `xsmall` / `small` / `medium` / `large` | `large` (no cap) | Cumulative — each tier includes every tier below it. `conc_tool`/`conc_chat` ignore this — they scope to every LLM model actually downloaded locally instead, since download presence is itself a decent proxy for "this machine can try it" (see [Concurrency](workloads.md#concurrency)) |
| `--llm-models` (`--models` alias) | space-separated tags and/or wildcards (e.g. `llama*`) | none (every catalog model in the selected tier) | Affects `llm`/`conv`/`mcq`/`math`/`reasoning`/`code`/`tool`/`conc_tool`/`conc_chat` tests. Matching is case-sensitive and exact-or-wildcard (`fnmatch`-style: `*`/`?`/`[...]`), not substring. Applied after `--maxtier` (or, for concurrency, after downloaded-model scoping), narrowing catalog entries while also unioning any matching installed custom tags. `--llm-models` is canonical; `--models` remains fully backward compatible. Quote wildcards (`"llama*"`) so the shell does not expand them |
| `--embedding-models` | space-separated catalog tags and/or wildcards | none (every catalog embedding model) | Affects `emb` only. Matching is case-sensitive and exact-or-wildcard on the model's `tag` |
| `--image-models` | space-separated catalog short IDs and/or wildcards (e.g. `sd*`) | none (every image model allowed by `--maxtier`) | Affects `img` only. Matching is case-sensitive and exact-or-wildcard on the stable `short` values in `models.py`; it narrows the image list after `--maxtier` |
| `--list-models` | (flag) | off | Read-only inventory of installed catalog LLMs, embeddings, custom LLM folders, and catalog image checkpoints, then exit. It does not require or start an inference server. `--engine` selects the inventory (`all` lists each engine), and `--comfyui` selects the image checkpoint directory |
| `--sample` | integer `N` | none (full bank) | Dev-only. Runs `mcq`/`math`/`reasoning`/`code`/`tool` against a deterministic N-question subset of each bank, selected round-robin across categories. Every category is represented only when N is at least that bank's category count. IDs are recorded under `sample_ids`; sampled and full-bank results are not comparable — see [bank versioning](workloads.md#bank-versioning) |
| `--comfyui` | path | `./ComfyUI` | Only needed if ComfyUI lives somewhere else |
| `--out` | filename | `results/results_<hostname>_<timestamp>.json` | Overrides the main JSON path entirely. Accuracy answer sidecars and generated-image folders still go under the repository's `results/` directory, named from the main output's stem — see [Project Structure](project-structure.md) |
| `--force-all` | (flag) | off | Disables the slow-TPS exits for single-shot LLM, conversation, and chat concurrency. It does not bypass timeouts, crashes, missing data, or load failures |

An explicitly supplied selector that resolves to no models for a selected workload is a command error and exits before hardware profiling, result-file creation, or server orchestration. This applies consistently to `--llm-models`/`--models`, `--embedding-models`, and `--image-models`. Selectors for workload families absent from `--tests` are ignored for this validation. Omitted selectors retain the defaults above; selecting a catalog model that is not downloaded still reaches the workload's existing missing-model handling.

## Examples

```bash
# Open the interactive launcher (only installed models are shown)
bash run_bench.sh

# LLM only, quick check
bash run_bench.sh --tests llm

# Skip image generation
bash run_bench.sh --tests llm conv emb

# Conversation benchmark only
bash run_bench.sh --tests conv

# Accuracy tests only — MCQ, math, reasoning, code, and tool (also: --tests acc)
bash run_bench.sh --tests mcq math reasoning code tool

# Cap at small-tier models and below — skips medium/large LLMs and
# medium/large-tier image models (SD3.5 Large, Flux.1-dev, Flux.2-dev),
# leaving SD1.5 and SDXL for the image test
bash run_bench.sh --maxtier small

# Only the Llama models, every tier — wildcard matches every Llama tag
bash run_bench.sh --tests llm --llm-models "llama*"

# One specific model plus a wildcard group
bash run_bench.sh --tests llm --llm-models phi4-mini "nemotron-3*"

# One embedding model only
bash run_bench.sh --tests emb --embedding-models nomic-embed-text

# SD-family image checkpoints only, still subject to --maxtier
bash run_bench.sh --tests img --image-models "sd*"

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

# Give a slower model more time per accuracy question before its partial answer is scored
bash run_bench.sh --tests acc --acc-timeout 120

# Both concurrency tests — 1-16-way tool-style + 1-32-way chat-server sweeps
# on every downloaded model
bash run_bench.sh --tests conc

# Chat-server concurrency test only
bash run_bench.sh --tests conc_chat
```

A full run takes several hours, depending on your hardware and which options you select.

## Comparing results

Copy result files from all machines to one machine, then load them into the [dashboard](dashboard.md).

---

[← Workloads](workloads.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
