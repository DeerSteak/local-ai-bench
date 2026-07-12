[← Back to README](../README.md)

# Testing

This document provides a guide to the test suite of `local-ai-bench`, explaining how to run the tests and describing what each test module validates.

**Contents**
- [Running Tests](#running-tests)
- [Prerequisites](#prerequisites)
- [Test Infrastructure Configuration](#test-infrastructure-configuration)
- [Test Suite Breakdown](#test-suite-breakdown)

---

## Running Tests

Tests are written using [pytest](https://docs.pytest.org/) and can be run using the platform-specific wrapper scripts:

**Linux / macOS**
```bash
bash tests.sh
```

**Windows**
```cmd
tests.bat
```

Both wrapper scripts accept arbitrary arguments and forward them directly to `pytest`. For example, to run a specific test file or filter by test name:

```bash
# Run a specific test module
bash tests.sh tests/test_config.py

# Run only tests matching a pattern
bash tests.sh -k "select_tier"

# Show verbose output
bash tests.sh -v
```

---

## Prerequisites

The test runner scripts require that the project's virtual environment (`bench-env/`) has been created first. Run the corresponding setup script if you haven't already:

- **Linux / macOS:** `bash setup.sh`
- **Windows:** `setup.bat`

The test wrappers automatically load this virtual environment and silently install/update test dependencies from [tests/requirements.txt](tests/requirements.txt) before launching pytest.

---

## Test Infrastructure Configuration

- **[conftest.py](tests/conftest.py)**
  Sets up the import path for the test suite. It injects the `scripts/` directory into `sys.path` so that modules can be imported directly as top-level namespaces (e.g., `config`, `shared`, `models`, `benchmark`) matching how they import each other at runtime.

---

## Test Suite Breakdown

The test suite consists of **14 test modules** validating different components of the application, from configuration structure to low-level client streaming, memory-tier pruning, and task scheduling.

### Benchmark Logic & CLI Orchestration

- **[test_benchmark_conv_skip.py](tests/test_benchmark_conv_skip.py)**
  Validates logic in `benchmark.py` for deciding when to skip models during the LLM conversation test. It asserts that:
  - Models with missing or empty LLM data are skipped.
  - A model that timed out or crashed in the single-shot test skips the conversation test, propagating the failure details.
  - If decode speeds (Tokens/sec) drop below the threshold (determined by `config.SLOW_MODEL_MIN_TPS`) during the first context check, the model is skipped.
  - The `--force-all` flag bypasses the slow model cutoff, but does not override actual timeouts or crashes.

- **[test_benchmark_select_tier.py](tests/test_benchmark_select_tier.py)**
  Tests the tier selection logic (`select_tier` in `benchmark.py`) for filtering model workloads. It verifies:
  - Passing `None` or `large` runs all tiers.
  - Smaller tier caps function cumulatively (e.g., `medium` includes `xsmall`, `small`, and `medium` workloads).
  - Tiers correctly filter both LLM models and ComfyUI image models.
  - Human-readable tier label descriptions are returned and distinct.

- **[test_config.py](tests/test_config.py)**
  Performs structural sanity checks on the constants in `config.py`. It verifies that:
  - Context lengths are strictly sorted in ascending order and unique.
  - Important directories (`RESULTS_DIR`, `COMFYUI_DIR`) resolve correctly relative to the project root.
  - Execution run count (`N_RUNS`) is positive.
  - Endpoint URLs (Ollama and ComfyUI) have proper HTTP schemas.

- **[test_models.py](tests/test_models.py)**
  Validates model configuration records in `models.py`. It checks:
  - LLM models list matches the concatenated list of individual size tiers.
  - Within each tier, LLM models are ordered by parameter counts (`params_b`).
  - Model tags and shortcodes are globally unique.
  - Required fields (e.g., download size, model tags, parameters, samplers, schedulers) exist in model definitions.

---

### Workload Implementations

- **[test_embedding_benchmark.py](tests/test_embedding_benchmark.py)**
  Tests the custom document chunking mechanism in `EmbeddingBenchmark`. It verifies:
  - Clean paragraph-level division.
  - Filtering out paragraphs shorter than the `min_words` boundary.
  - Enforcement of the `max_words` limit by splitting large chunks on sentence boundaries.
  - Implementation of hard word-boundary splits (without loss or reordering of words) when punctuation is absent (e.g., code snippets, raw data logs).
  - Normalization of irregular whitespace.

- **[test_image_benchmark.py](tests/test_image_benchmark.py)**
  Tests ComfyUI image generation workloads, API triggers, and state management. It covers:
  - Proper routing of workflow builders for model classes (SDXL, SD3, Flux, and custom formats).
  - Graph syntax validation for ComfyUI workflow JSON structures (e.g., verifying that the specified checkpoint files, seeds, prompts, and output dimensions are properly wired, and that all node references exist).
  - Execution controls, verifying that `comfyui_free_models` and `comfyui_interrupt_and_clear` post correctly to the server API, handle connection failures gracefully, and poll status correctly until the queue drains.

- **[test_llm_conversation_benchmark.py](tests/test_llm_conversation_benchmark.py)**
  Tests parameters and algorithms within `LLMConversationBenchmark` for multi-turn testing. It verifies:
  - Follow-up prompts cycle sequentially through sections of the conversation prompt text, wrapping around cleanly.
  - Growth checkpoints (`CONV_CHECKPOINTS`) are sorted and fit within the target ceiling.
  - The step-size calculator (`compute_growth_step`) correctly scales token additions, clamps to safe bounds (`CONV_STEP_MIN`/`CONV_STEP_MAX`), enforces context safety margins (`CONV_SAFETY_MARGIN`) for non-final checks, consumes the full context room on the final step, and signals when the context is full.

---

### Shared Helpers & APIs

- **[test_shared_crash_cache.py](tests/test_shared_crash_cache.py)**
  Tests the model crash-tracking database (which prevents repeated attempts of deterministic crashes). It verifies:
  - Reading from a missing or corrupted file falls back to an empty cache dict.
  - Successful serialization roundtripping.
  - Unwritable disk locations do not crash the runner (write failures are swallowed).
  - The cache matches keys correctly to output skip markers.
  - Exception analyzer (`is_connection_crash`) properly identifies connection errors and server crashes (e.g. BrokenPipe, ConnectionReset, and actively refused sockets) while letting normal runtime errors bubble up.

- **[test_shared_find_comfyui_python.py](tests/test_shared_find_comfyui_python.py)**
  Tests the search hierarchy in `Shared.find_comfyui_python` for locating the correct python executable to start ComfyUI. It validates the priority order:
  1. Windows portable embedded Python (`python_embeded/python.exe`).
  2. The local `venv/bin/python` under ComfyUI.
  3. The local `.venv/bin/python` under ComfyUI.
  4. The current external active virtual environment (`VIRTUAL_ENV`).
  5. The current system interpreter (`sys.executable`).

- **[test_shared_ollama_maintenance.py](tests/test_shared_ollama_maintenance.py)**
  Tests Ollama lifecycle hooks and server state controls:
  - `ollama_reachable_or_abort` detects whether Ollama is running.
  - `model_pulled` checks for exact or implicit matches (like tags missing `:latest`) in the local image list.
  - `ollama_model_max_ctx` parses architectural options to identify a model's true context limit, falling back to configuration defaults on failure.
  - `unload_model` issues a keep-alive termination request and handles network errors.
  - `unload_all_models` queries loaded models and terminates them.
  - `wait_until_unloaded` polls until a model is fully evicted.

- **[test_shared_ollama_streaming.py](tests/test_shared_ollama_streaming.py)**
  Validates NDJSON response stream parsing for Ollama completion endpoints. It tests:
  - Derivation of TTFT and Tokens/sec from server performance fields.
  - Fallback calculation of TTFT using local system time if fields are missing.
  - Resilience against empty, blank, or malformed JSON stream lines.
  - Response extraction preferring the standard content payload over reasoning (`thinking`) fields, but falling back to reasoning text if needed.
  - Intercepting HTTP 500 error payloads to extract clean diagnostic messages (e.g. "model requires more system memory") rather than raising generic HTTP error statuses.

- **[test_shared_run_measured_calls.py](tests/test_shared_run_measured_calls.py)**
  Tests the execution loop for benchmark runs (`run_measured_calls`). It checks:
  - Correct execution count under normal operations.
  - Instant stoppage and exit status marking on timeout.
  - Skipping a single run's metrics if a standard execution error occurs, while proceeding to the next run.
  - If a connection crash occurs, attempting Ollama recovery. If recovery succeeds, the loop retries the failed run. If recovery fails, the benchmark halts and records the model as crashed.
  - `slow_tps_early_exit` early termination logic based on performance speeds.

- **[test_shared_stats.py](tests/test_shared_stats.py)**
  Validates general helpers in `Shared`:
  - `mean` and `stdev` mathematical routines (including handling empty lists or single-element inputs).
  - Context prompt text builder, assuring that generated prompts meet the target length in characters, do not crash on tiny inputs, and use a varying nonce prefix to bypass model prompt cache hits.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
