[← Back to README](../README.md)

# Project Structure

**Contents**
- [`scripts/` in detail](#scripts-in-detail)
- [`results/` in detail](#results-in-detail)

| File / Folder | Purpose |
|---|---|
| `setup.sh` | One-shot setup for macOS and Linux |
| `setup.bat` | One-shot setup for Windows |
| `run_linux_mac.sh` | Activates the venv and runs `scripts/benchmark.py` on Linux / macOS |
| `run_windows.bat` | Activates the venv and runs `scripts/benchmark.py` on Windows |
| `launch_dashboard.py` | Builds and serves the dashboard, opens browser automatically |
| `scripts/` | Benchmark implementation — see [How It Works](how-it-works.md#code-organization) for what each module does |
| `results/` | Benchmark output — `results_*.json` plus a matching `images_*/` folder per run with the generated sample images |
| `dashboard/` | The results-explorer web app (React + Vite) |
| `samples/` | Sample `results_*.json` files for trying the dashboard without running a benchmark |
| `models.py` (in `scripts/`) | Single source of truth for every model definition — imported by `benchmark.py`, `setup_check.py`, and `shared.py` |
| `requirements.txt` | Python dependencies, installed by the setup scripts |
| `sample_document.txt` | The corpus chunked and embedded by the embeddings test |
| `hf.txt` | Optional saved HuggingFace token (see [Setup](setup.md#huggingface-token)) — not tracked in git |
| `.embed_crash_cache.json` | Records model/document combos that crashed Ollama's runner repeatedly, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry |

## `scripts/` in detail

| Module | Purpose |
|---|---|
| `benchmark.py` | CLI entry point — argument parsing and test orchestration |
| `config.py` | Shared constants (URLs, paths, timeouts, run counts) |
| `shared.py` | Cross-cutting helpers: logging, server lifecycle, machine profiling, Ollama/ComfyUI HTTP clients |
| `llm_prefill_benchmark.py` | Single-shot LLM test |
| `llm_conversation_benchmark.py` | Multi-turn conversation LLM test |
| `embedding_benchmark.py` | Embeddings test |
| `image_benchmark.py` | Image generation test (ComfyUI workflow builders + submission) |
| `models.py` | Model definitions (tags, checkpoints, tiers, sizes) |
| `setup_check.py` | Hardware detection, model picker, unattended install |

## `results/` in detail

Each benchmark run produces one results file and, if the image test ran, one matching images folder — a sibling of the JSON file, not nested inside a shared `images/` folder, so both sort and select together in a file browser (Finder, Explorer, Nautilus, ...):

```
results/
  results_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  images_Mac_Studio_M4_Max_64_GB_20260711_090000/
    sdxl_1024x1024.png
    sdxl_1536x1536.png
    flux-dev_1024x1024.png
    ...
```

The images folder's name is always the results filename's stem with `results_` swapped for `images_` — so the hostname and timestamp suffix is identical between the two, letter for letter. This holds even when `--out` overrides the default naming (falling back to `images_<name>` if the given filename doesn't start with `results_`). See [CLI Reference](cli-reference.md) for the `--out` flag.

`results/` is gitignored — nothing under it is tracked. Load its contents into the [dashboard](dashboard.md) to compare across machines.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md)
