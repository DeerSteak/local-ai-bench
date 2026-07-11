[← Back to README](../README.md)

# CLI Reference

**Contents**
- [Flag details](#flag-details)
- [Examples](#examples)
- [Comparing results](#comparing-results)

```
run_linux_mac.sh [options]  # Linux / macOS
run_windows.bat [options]   # Windows

--tests llm conv emb img  Tests to run (default: all four)
--emb-cpu-only          Force CPU-only inference for the embedding tests
                        (restarts Ollama with GPU devices hidden, then
                        restores normal GPU mode afterward)
--warmup N              Warmup runs before measuring (default: 2)
--runs N                Measured runs per checkpoint, averaged (default: 3,
                        range: 1-10). Applies separately to every model,
                        context length, and test mode that's enabled, so
                        total measured time scales roughly in proportion —
                        e.g. going from 3 to 6 runs roughly doubles it.
                        Warmup time is unaffected (see --warmup).
--timeout N             Seconds per run before skipping model (default: 300)
--maxtier TIER          Cap LLM models (single-shot + conversation) AND image
                        models at this tier and below: xsmall (<6B / +SD1.5),
                        small (≤20B / +SDXL), medium (26–35B / +SD3.5 Large),
                        large (70B+ / +Flux.1-dev, Flux.2-dev — default, no cap)
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results/results_<hostname>_<timestamp>.json)
--force-all             Ignore the 15 tok/s slow-model cutoff: run every context
                        length in the LLM single-shot test and always run the
                        conversation test, even for models that would otherwise
                        be marked slow and skipped. Doesn't override real
                        failures (timeouts, missing data). Rarely needed —
                        default: false
```

Every test measures `--runs` runs (default 3) per checkpoint and averages them.

## Flag details

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--tests` | any of `llm conv emb img` | all four | Space-separated list; order doesn't matter |
| `--emb-cpu-only` | (flag) | off | Only affects the embeddings test |
| `--warmup` | integer | `2` | Discarded runs before measured runs, per model/checkpoint |
| `--runs` | integer, `1`–`10` | `3` | Measured runs per checkpoint, averaged. Applies separately to every model, context length, and test mode that's enabled, so total measured time scales roughly in proportion — e.g. 6 runs roughly doubles measured time versus the default. Warmup time is unaffected |
| `--timeout` | integer (seconds) | `300` | Per run (warmup or measured); exceeding it skips the rest of that model |
| `--maxtier` | `xsmall` / `small` / `medium` / `large` | `large` (no cap) | Cumulative — each tier includes every tier below it |
| `--comfyui` | path | `./ComfyUI` | Only needed if ComfyUI lives somewhere else |
| `--out` | filename | `results/results_<hostname>_<timestamp>.json` | Overrides the auto-generated path entirely — an explicit path is used as-is, not placed under `results/`. Generated images (`--tests img`) still land under `results/`, in an `images_<name>` folder alongside it (see [Project Structure](project-structure.md)) |
| `--force-all` | (flag) | off | See [LLM workload](workloads.md#llm) for what the slow-model cutoff normally skips |

## Examples

```bash
# Full run — large models skipped automatically if they don't fit
bash run_linux_mac.sh

# LLM only, quick check
bash run_linux_mac.sh --tests llm

# Skip image generation
bash run_linux_mac.sh --tests llm conv emb

# Conversation benchmark only
bash run_linux_mac.sh --tests conv

# Cap at small-tier models and below — skips medium/large LLMs and
# medium/large-tier image models (SD3.5 Large, Flux.1-dev, Flux.2-dev),
# leaving SD1.5 and SDXL for the image test
bash run_linux_mac.sh --maxtier small

# Give slow hardware more time per run
bash run_linux_mac.sh --timeout 600
```

A full run takes several hours, depending on your hardware and which options you select.

## Comparing results

Copy result files from all machines to one machine, then load them into the [dashboard](dashboard.md).

---

[← Workloads](workloads.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
