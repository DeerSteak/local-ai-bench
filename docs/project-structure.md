[← Back to README](../README.md)

# Project Structure

**Contents**
- [`scripts/` in detail](#scripts-in-detail)
- [`results/` in detail](#results-in-detail)
  - [Main results JSON](#main-results-json)

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
| `results/` | Default benchmark output — `results_*.json`, generated-image folders, and `answers_mcq_*` / `answers_math_*` / `answers_code_*` / `answers_tool_*` JSON sidecars containing every accuracy-test response |
| `dashboard/` | The results-explorer web app (React + Vite) |
| `tests/` | The unit and integration test suite — see [Testing](testing.md) |
| `samples/` | Sample `results_*.json` files for trying the dashboard without running a benchmark |
| `models/` | Downloaded LLM/embedding GGUF files, namespaced per engine (`models/llamacpp/<tag-slug>/`) — created by `setup_check.py`, gitignored |
| `models.py` (in `scripts/`) | Single source of truth for every model definition — imported by `benchmark.py`, `setup_check.py`, and `shared.py` |
| `requirements.txt` | Python dependencies, installed by the setup scripts |
| `sample_document.txt` | The corpus chunked and embedded by the embeddings test |
| `scripts/data/` | Question banks used by accuracy tests — `mcq_questions.json` (150 questions), `math_questions.json` (150 questions), `code_problems.json` (60 problems), `tool_questions.json` (100 tool-calling questions) |
| `hf.txt` | Optional saved HuggingFace token (see [Setup](setup.md#huggingface-token)) — not tracked in git |
| `.coveragerc` | Coverage config for the test suite — omits `setup_check.py` (unsafe to import) and excludes live-server/subprocess code marked `# pragma: no cover`, so `pytest --cov` reports coverage of the unit-testable code only |
| `.llm_crash_cache.json` | Records LLM models that crashed the active engine's runner repeatedly during the single-shot test, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry |
| `.conv_crash_cache.json` | Same as above, for the conversation test |
| `.embed_crash_cache.json` | Records model/document combos that crashed the active engine's runner repeatedly, so future runs skip retrying a deterministic crash — created automatically, safe to delete to retry |
| `.mcq_crash_cache.json` | Same as above, for the MCQ accuracy test. Also records which question-bank version (a short content hash) the crash happened against, so a crash recorded on an old/smaller bank doesn't skip a model forever once the bank changes — see [bank versioning](workloads.md#bank-versioning) |
| `.math_crash_cache.json` | Same as above, for the math accuracy test |
| `.code_crash_cache.json` | Same as above, for the code accuracy test |
| `.tool_crash_cache.json` | Same as above, for the tool-calling accuracy test |
| `.concurrency_tool_crash_cache.json` | Records repeatable engine crashes from the tool-style concurrency sweep; safe to delete to retry |
| `.concurrency_chat_crash_cache.json` | Same as above, for the chat concurrency sweep |

The old `compare.py` CLI tool has been dropped — it's been replaced by the [dashboard](dashboard.md).

## `scripts/` in detail

| Module | Purpose |
|---|---|
| `benchmark.py` | CLI entry point — argument parsing and test orchestration |
| `config.py` | Shared constants (URLs, paths, timeouts, run counts) |
| `shared.py` | Cross-cutting helpers: logging, machine profiling, engine-agnostic run/crash orchestration, ComfyUI server lifecycle/HTTP client |
| `hardware.py` | GPU/system-memory detection, shared-memory classification, and model-fit estimates |
| `engines/base.py`, `engines/llamacpp.py` | `InferenceEngine` interface and `LlamaCppEngine` — server lifecycle + HTTP/process client, see [Engines](engines.md) |
| `llm_prefill_benchmark.py` | Single-shot LLM test |
| `llm_conversation_benchmark.py` | Multi-turn conversation LLM test |
| `embedding_benchmark.py` | Embeddings test |
| `image_benchmark.py` | Image generation test (ComfyUI workflow builders + submission) |
| `concurrency_benchmark.py` | Shared implementation for the tool-style and chat concurrency sweeps |
| `mcq_benchmark.py` | MCQ accuracy test |
| `math_benchmark.py` | Numeric-answer math accuracy test |
| `code_benchmark.py` | Isolated Python code-generation accuracy test |
| `tool_benchmark.py` | Tool-calling accuracy test |
| `models.py` | Model definitions (tags, checkpoints, tiers, sizes) |
| `setup_check.py` | Hardware detection, model picker, unattended install |
| `setup_policy.py` | Pure, unit-tested setup decisions used by the otherwise interactive installer |
| `data/` | Question banks used by accuracy tests (see above) |

## `results/` in detail

By default, each benchmark run produces one main results file plus one separately named file or folder for workloads with bulky output:

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
  answers_tool_Mac_Studio_M4_Max_64_GB_20260711_090000.json
```

Each auxiliary name is derived from the main results filename's stem by swapping `results_` for `images_`, `answers_mcq_`, `answers_math_`, `answers_code_`, or `answers_tool_`. If the stem does not begin with `results_`, the auxiliary prefix is prepended instead. With the default output, this preserves the hostname and timestamp across the set. If `--out` places the main JSON elsewhere, only that main file follows the custom directory; images and answer sidecars still go under the repository's `results/` directory. See [CLI Reference](cli-reference.md).

`--engine all` (see [Engines](engines.md)) appends the engine name to the results filename's stem for each pass, so a run of the example above would produce `results_..._090000_llamacpp.json` (and one more per additional engine, once a second one is registered) side by side, each tagged internally with `"engine"`.

The `answers_*.json` sidecars hold every question's answer for that accuracy test, keyed by model, each with the model's full raw response text and a `correct` flag — kept out of the main results JSON since raw model output (unbounded generation, see `docs/workloads.md`) is large relative to everything else in there and would otherwise bloat it substantially. The main results JSON's own `incorrect` list (per model, per test) is unaffected and still covers only wrong answers.

### Main results JSON

The main file is checkpointed throughout a run, so completed stages and models survive an interruption. Its top level contains:

| Key | Contents |
|---|---|
| `version`, `engine` | Benchmark schema/version label and inference-engine name |
| `profile` | Host description, OS/release, architecture, Python version, RAM, UTC timestamp, effective inference backend (`cuda`, `rocm`, `metal`, `xpu`, `vulkan`, or `cpu`), and separately detected `hardware_backend` |
| `bank_versions` | Content hashes for the MCQ, math, code, and tool banks |
| `sample_ids` | Exact per-bank IDs only when `--sample` was used |
| `llm`, `llm_conversation` | Per-model context/checkpoint measurements and any timeout, crash, slow-TPS, or skip markers |
| `mcq`, `math`, `code`, `tool` | Per-model overall/category scores plus timeout and likely-loop diagnostics when present |
| `embeddings`, `images` | Per-model throughput or per-resolution generation-time measurements |
| `concurrency_tool`, `concurrency_chat` | Per-model/per-level TTFT, per-request and aggregate throughput, token/batch timing, memory snapshots, and stop markers |

Performance workloads retain means, standard deviations, run counts, and—where applicable—the individual measured values. Concurrency snapshots include system RAM and add GPU VRAM when a trustworthy discrete-GPU reading is available; a failed load records `memory_at_failure`. Missing keys and empty sections are valid because the dashboard supports partial runs and older schema versions.

`results/` is gitignored — nothing under it is tracked. Load its contents into the [dashboard](dashboard.md) to compare across machines.

---

[← Engines](engines.md) · [Back to README](../README.md) · [Testing →](testing.md)
