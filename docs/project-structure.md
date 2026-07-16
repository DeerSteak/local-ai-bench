[← Back to README](../README.md)

# Project Structure

**Contents**
- [`scripts/` in detail](#scripts-in-detail)
- [`results/` in detail](#results-in-detail)

| File / Folder | Purpose |
|---|---|
| `setup.sh` | One-shot setup for macOS and Linux |
| `setup.bat` | One-shot setup for Windows |
| `run_bench.sh` | Activates the venv and runs `scripts/benchmark.py` on Linux / macOS |
| `run_bench.bat` | Activates the venv and runs `scripts/benchmark.py` on Windows |
| `launch_dashboard.sh` | Builds and serves the dashboard on Linux / macOS, opens browser automatically |
| `launch_dashboard.bat` | Builds and serves the dashboard on Windows, opens browser automatically |
| `tests.sh` | Activates the venv and runs unit/integration tests on Linux / macOS — see [Testing](testing.md) |
| `tests.bat` | Activates the venv and runs unit/integration tests on Windows — see [Testing](testing.md) |
| `scripts/` | Benchmark implementation — see [How It Works](how-it-works.md#code-organization) for what each module does |
| `results/` | Benchmark output — `results_*.json` plus a matching `images_*/` folder (generated sample images) and `answers_mcq_*.json` / `answers_math_*.json` / `answers_code_*.json` sidecar files (wrong answers' raw model output) per run |
| `dashboard/` | The results-explorer web app (React + Vite) |
| `tests/` | The unit and integration test suite — see [Testing](testing.md) |
| `samples/` | Sample `results_*.json` files for trying the dashboard without running a benchmark |
| `models.py` (in `scripts/`) | Single source of truth for every model definition — imported by `benchmark.py`, `setup_check.py`, and `shared.py` |
| `requirements.txt` | Python dependencies, installed by the setup scripts |
| `sample_document.txt` | The corpus chunked and embedded by the embeddings test |
| `scripts/data/` | Question banks used by accuracy tests — `mcq_questions.json` (150 questions), `math_questions.json` (150 questions), `code_problems.json` (60 problems), plus [`TEST_BANK_NOTES.md`](../scripts/data/TEST_BANK_NOTES.md) authoring and harness notes |
| `hf.txt` | Optional saved HuggingFace token (see [Setup](setup.md#huggingface-token)) — not tracked in git |
| `.coveragerc` | Coverage config for the test suite — omits `setup_check.py` (unsafe to import) and excludes live-server/subprocess code marked `# pragma: no cover`, so `pytest --cov` reports coverage of the unit-testable code only |
| `.llm_crash_cache.json` | Records LLM models that crashed Ollama's runner repeatedly during the single-shot test, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry |
| `.conv_crash_cache.json` | Same as above, for the conversation test |
| `.embed_crash_cache.json` | Records model/document combos that crashed Ollama's runner repeatedly, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry |
| `.mcq_crash_cache.json` | Same as above, for the MCQ accuracy test. Also records which question-bank version (a short content hash) the crash happened against, so a crash recorded on an old/smaller bank doesn't skip a model forever once the bank changes — see [bank versioning](workloads.md#bank-versioning) |
| `.math_crash_cache.json` | Same as above, for the math accuracy test |
| `.code_crash_cache.json` | Same as above, for the code accuracy test |

The old `compare.py` CLI tool has been dropped — it's been replaced by the [dashboard](dashboard.md).

## `scripts/` in detail

| Module | Purpose |
|---|---|
| `benchmark.py` | CLI entry point — argument parsing and test orchestration |
| `config.py` | Shared constants (URLs, paths, timeouts, run counts) |
| `shared.py` | Cross-cutting helpers: logging, machine profiling, engine-agnostic run/crash orchestration, ComfyUI server lifecycle/HTTP client |
| `engines/base.py`, `engines/ollama.py` | `InferenceEngine` interface and its `OllamaEngine` implementation (Ollama server lifecycle + HTTP client) — the seam a future llama.cpp/MLX engine would implement, see [Engines](engines.md) |
| `llm_prefill_benchmark.py` | Single-shot LLM test |
| `llm_conversation_benchmark.py` | Multi-turn conversation LLM test |
| `embedding_benchmark.py` | Embeddings test |
| `image_benchmark.py` | Image generation test (ComfyUI workflow builders + submission) |
| `mcq_benchmark.py` | MCQ accuracy test |
| `models.py` | Model definitions (tags, checkpoints, tiers, sizes) |
| `setup_check.py` | Hardware detection, model picker, unattended install |
| `data/` | Question banks used by accuracy tests (see above) |

## `results/` in detail

Each benchmark run produces one results file, plus one sibling file or folder per test that has its own bulky output — not nested inside a shared folder, so everything from one run sorts and selects together in a file browser (Finder, Explorer, Nautilus, ...):

```
results/
  results_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  images_Mac_Studio_M4_Max_64_GB_20260711_090000/
    sdxl_1024x1024.png
    sdxl_1536x1536.png
    flux-dev_1024x1024.png
    ...
  answers_mcq_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  answers_math_Mac_Studio_M4_Max_64_GB_20260711_090000.json
  answers_code_Mac_Studio_M4_Max_64_GB_20260711_090000.json
```

Each sibling name is always the results filename's stem with `results_` swapped for the sibling's own prefix (`images_`, `answers_mcq_`, `answers_math_`, `answers_code_`) — so the hostname and timestamp suffix is identical across all of them, letter for letter. This holds even when `--out` overrides the default naming (falling back to `<prefix><name>` if the given filename doesn't start with `results_`). See [CLI Reference](cli-reference.md) for the `--out` flag.

The `answers_*.json` sidecars hold each accuracy test's wrong answers, keyed by model, with the model's full raw response text — kept out of the main results JSON since raw model output (unbounded generation, see `docs/workloads.md`) is large relative to everything else in there and would otherwise bloat it substantially.

`results/` is gitignored — nothing under it is tracked. Load its contents into the [dashboard](dashboard.md) to compare across machines.

---

[← Engines](engines.md) · [Back to README](../README.md) · [Testing →](testing.md)
