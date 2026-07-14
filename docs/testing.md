[← Back to README](../README.md)

# Testing

This document provides a guide to the test suite of `local-ai-bench` — both the Python benchmark suite (`pytest`, in `tests/`) and the dashboard (`vitest`, in `dashboard/src/`) — explaining how to run each and describing what each test module validates.

**Contents**
- [Running Tests](#running-tests)
- [Prerequisites](#prerequisites)
- [Test Infrastructure Configuration](#test-infrastructure-configuration)
- [Coverage](#coverage)
- [Test Suite Breakdown](#test-suite-breakdown)
- [Dashboard Tests](#dashboard-tests)

---

## Running Tests

This section and the next four cover the Python benchmark suite's tests (`tests/`, pytest). See [Dashboard Tests](#dashboard-tests) at the end for the separate `dashboard/` test suite (vitest).

Python tests are written using [pytest](https://docs.pytest.org/) and can be run using the platform-specific wrapper scripts:

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

The test wrappers automatically load this virtual environment and silently install/update test dependencies from [tests/requirements.txt](../tests/requirements.txt) before launching pytest.

---

## Test Infrastructure Configuration

- **[conftest.py](../tests/conftest.py)**
  Sets up the import path for the test suite. It injects the `scripts/` directory into `sys.path` so that modules can be imported directly as top-level namespaces (e.g., `config`, `shared`, `models`, `benchmark`) matching how they import each other at runtime.

---

## Coverage

`pytest-cov` (not installed by default — `pip install pytest-cov` into `bench-env`) reports line coverage:

```bash
bash tests.sh --cov=scripts --cov-report=term-missing
```

[`.coveragerc`](../.coveragerc) at the repo root shapes that report so it reflects the code this suite is actually meant to exercise, rather than being diluted by code that can't safely be unit tested:

- `scripts/setup_check.py` is omitted entirely — it has no `__main__` guard, so importing it would run the whole interactive install flow (prompts, downloads).
- Individual functions that spawn real subprocesses, poll a live Ollama/ComfyUI server, or orchestrate a full test run (`main()`, each workload class's `run()`, `Shared.start_ollama`, `Shared.ensure_comfyui`, etc.) are marked `# pragma: no cover` at their `def` line — coverage.py excludes the whole function body when the pragma sits on the line that opens it.

With that config in place, coverage sits around 95% for the code the suite targets — the remaining gaps are a handful of fine-grained exception branches inside otherwise-tested functions, not whole untested subsystems.

---

## Test Suite Breakdown

The test suite consists of **22 test modules** validating different components of the application, from configuration structure and model definitions to low-level Ollama/ComfyUI HTTP client streaming.

### Benchmark Logic & CLI Orchestration

- **[test_benchmark_conv_skip.py](../tests/test_benchmark_conv_skip.py)**
  Validates logic in `benchmark.py` for deciding when to skip models during the LLM conversation test. It asserts that:
  - Models with missing or empty LLM data are skipped.
  - A model that timed out or crashed in the single-shot test skips the conversation test, propagating the failure details.
  - If decode speeds (Tokens/sec) drop below the threshold (determined by `config.SLOW_MODEL_MIN_TPS`) during the first context check, the model is skipped.
  - The `--force-all` flag bypasses the slow model cutoff, but does not override actual timeouts or crashes.

- **[test_benchmark_select_tier.py](../tests/test_benchmark_select_tier.py)**
  Tests the tier selection logic (`select_tier` in `benchmark.py`) for filtering model workloads. It verifies:
  - Passing `None` or `large` runs all tiers.
  - Smaller tier caps function cumulatively (e.g., `medium` includes `xsmall`, `small`, and `medium` workloads).
  - Tiers correctly filter both LLM models and ComfyUI image models.
  - Human-readable tier label descriptions are returned and distinct.

- **[test_benchmark_expand_tests.py](../tests/test_benchmark_expand_tests.py)**
  Tests the `--tests` shorthand-group expansion logic (`expand_tests` in `benchmark.py`). It verifies:
  - `acc` expands to `ACCURACY_TESTS` (currently `mcq`, `math`, and `code`).
  - Ordinary test names pass through unchanged.
  - Order is preserved when `acc` is mixed with other test names.
  - No duplicates result from combining `acc` with one of its own expanded members, or from repeating a plain test name.

- **[test_benchmark_filter_models.py](../tests/test_benchmark_filter_models.py)**
  Tests the `--models` filtering logic (`filter_models_by_pattern` in `benchmark.py`). It verifies:
  - No patterns (`None` or `[]`) returns the model list unchanged.
  - An exact tag matches only that model.
  - A wildcard (`llama*`) matches every tag sharing that prefix.
  - Matching is case-sensitive — an uppercase pattern against lowercase tags matches nothing.
  - Multiple overlapping patterns union their matches without duplicating a model that satisfies more than one.
  - A pattern matching nothing returns an empty list rather than erroring.
  - Filtering preserves the original model order.

- **[test_benchmark_resolve_custom_models.py](../tests/test_benchmark_resolve_custom_models.py)**
  Tests the `--models` catalog-fallback logic (`resolve_custom_models` in `benchmark.py`) that lets a pattern matching nothing in the curated catalog still resolve against a model actually pulled in Ollama. It verifies:
  - A pattern that matches the catalog behaves exactly like `filter_models_by_pattern` and does not also pull in unrelated installed tags.
  - A pattern matching nothing in the catalog falls back to matching installed tags, producing a `"(custom)"`-labeled entry.
  - A pattern matching neither the catalog nor anything installed resolves to nothing (not an error).
  - A wildcard can resolve to multiple installed tags at once.
  - Catalog and custom patterns can be mixed in the same `--models` invocation.
  - `sanitize_tag_to_short` turns a raw Ollama tag's `:`/`/` characters into `-`, matching the style of curated `short` identifiers in `models.py`.

- **[test_benchmark_sidecar_path.py](../tests/test_benchmark_sidecar_path.py)**
  Tests `sidecar_path` in `benchmark.py`, which builds a sibling filename (an `answers_mcq_*.json`, `images_*/`, etc.) alongside the main results JSON. It verifies:
  - A `results_`-prefixed output path has that prefix swapped for the sidecar's own prefix.
  - A custom `--out` filename that doesn't start with `results_` falls back to prepending the sidecar prefix instead.
  - Every sidecar prefix (`answers_mcq_`, `answers_math_`, `answers_code_`, `images_`) produces the same hostname/timestamp suffix from the same results path, so sidecars from one run are always identifiable as a set.

- **[test_config.py](../tests/test_config.py)**
  Performs structural sanity checks on the constants in `config.py`. It verifies that:
  - Context lengths are strictly sorted in ascending order and unique.
  - Important directories (`RESULTS_DIR`, `COMFYUI_DIR`) resolve correctly relative to the project root.
  - Execution run count (`N_RUNS`) is positive.
  - Endpoint URLs (Ollama and ComfyUI) have proper HTTP schemas.

- **[test_models.py](../tests/test_models.py)**
  Validates model configuration records in `models.py`. It checks:
  - LLM models list matches the concatenated list of individual size tiers.
  - Within each tier, LLM models are ordered by parameter counts (`params_b`).
  - Model tags and shortcodes are globally unique.
  - Required fields (e.g., download size, model tags, parameters, samplers, schedulers) exist in model definitions.

---

### Workload Implementations

- **[test_embedding_benchmark.py](../tests/test_embedding_benchmark.py)**
  Tests the custom document chunking mechanism in `EmbeddingBenchmark`. It verifies:
  - Clean paragraph-level division.
  - Filtering out paragraphs shorter than the `min_words` boundary.
  - Enforcement of the `max_words` limit by splitting large chunks on sentence boundaries.
  - Implementation of hard word-boundary splits (without loss or reordering of words) when punctuation is absent (e.g., code snippets, raw data logs).
  - Normalization of irregular whitespace.

- **[test_image_benchmark.py](../tests/test_image_benchmark.py)**
  Tests ComfyUI image generation workloads, API triggers, and state management. It covers:
  - Proper routing of workflow builders for model classes (SDXL, SD3, Flux, and Flux2), including the unrecognized-type fallback to SDXL.
  - Graph syntax validation for ComfyUI workflow JSON structures (e.g., verifying that the specified checkpoint files, seeds, prompts, and output dimensions are properly wired, and that all node references exist).
  - Execution controls, verifying that `comfyui_free_models` and `comfyui_interrupt_and_clear` post correctly to the server API, handle connection failures gracefully, and poll status correctly until the queue drains.

- **[test_llm_conversation_benchmark.py](../tests/test_llm_conversation_benchmark.py)**
  Tests parameters and algorithms within `LLMConversationBenchmark` for multi-turn testing. It verifies:
  - Follow-up prompts cycle sequentially through sections of the conversation prompt text, wrapping around cleanly.
  - Growth checkpoints (`CONV_CHECKPOINTS`) are sorted and fit within the target ceiling.
  - The step-size calculator (`compute_growth_step`) takes larger steps (`CONV_STEP_MAX_FAR`) when far from the target and smaller ones (`CONV_STEP_MAX`) once within 8K tokens of it, clamps to `CONV_STEP_MIN`, enforces context safety margins (`CONV_SAFETY_MARGIN`) for non-final checks, consumes the full context room on the final step, and signals when the context is full.

- **[test_mcq_benchmark.py](../tests/test_mcq_benchmark.py)**
  Tests the pure logic in `MCQBenchmark`. It verifies:
  - `build_prompt` includes the question text and every answer choice.
  - `parse_answer` extracts a model's chosen letter from free-form text — bare letters, punctuated letters (`"B."`, `"(B)"`), and letters embedded in a reasoning sentence ("...so the answer is B") — while rejecting letters that aren't among the question's valid choices and not false-positiving on ordinary words/contractions that happen to contain a letter (e.g. the "d" in "I'd").
  - `score` tallies correct/total and per-category accuracy correctly, including unanswered (`None`) responses counting as incorrect, and produces a matching `incorrect` list.
  - `load_questions` returns a well-formed dataset from the real `scripts/data/mcq_questions.json` file — unique IDs, and every question's answer is one of its own choices.

- **[test_math_benchmark.py](../tests/test_math_benchmark.py)**
  Tests the pure logic in `MathBenchmark`. It verifies:
  - `build_prompt` includes the question text and asks for a numeric-only answer.
  - `parse_answer` extracts a model's final numeric answer from free-form text — bare integers, decimals, negative numbers, thousands-comma-separated numbers, and numbers with a trailing `%` — taking the *last* number stated rather than the first, so a model that reasons out loud before answering is still scored on its final answer, and returning `None` when no number (or only a bare `-`) is found.
  - `score` tallies correct/total and per-category accuracy correctly, treating an answer as correct when it falls within the question's own tolerance (defaulting to `0` — exact match — when absent), counting unanswered (`None`) responses as incorrect, and producing a matching `incorrect` list.
  - `load_questions` returns a well-formed dataset from the real `scripts/data/math_questions.json` file — unique IDs, and every question has a numeric `answer` and non-negative numeric `tolerance`.

- **[test_code_benchmark.py](../tests/test_code_benchmark.py)**
  Tests the pure logic in `CodeBenchmark`, including real (not mocked) subprocess execution of generated code — no Ollama server needed. It verifies:
  - `build_prompt` includes the question text and the target function/class name, renders `visible_tests` as worked examples (for both function and stateful problems, including constructor `init` args), omits the examples block entirely when there are no visible tests, and never leaks `hidden_tests` values into the prompt — proven not just by substring-scanning the rendered text but by deleting the `hidden_tests` key from every real question in the bank and confirming `build_prompt` doesn't raise (i.e. never even looks it up).
  - `extract_code` pulls the body out of a fenced code block (`` ```python `` or bare `` ``` ``) when present, and falls back to the whole reply when a model ignores the fencing instruction.
  - `execute_tests` runs a candidate function against a list of test cases in an isolated subprocess: correct/incorrect results score independently per test case, a runtime error in one test case doesn't abort the others, a syntax error or reference to an undefined function name fails every test case with an error message, and an infinite loop is killed and reported as a `"timeout"` rather than hanging the test run.
  - `evaluate_question` requires every visible *and* hidden test case to pass for a problem to count as correct, and short-circuits to a `"no code found"` failure without spawning a subprocess when no code could be extracted.
  - `score` tallies correct/total and per-category accuracy correctly, including unanswered (`None`) responses counting as incorrect, and produces a matching `incorrect` list.
  - `load_questions` returns a well-formed dataset from the real `scripts/data/code_problems.json` file — unique IDs, and every problem has a function name and at least one visible and one hidden test case, each with `args` and `expected`.

---

### Shared Helpers & APIs

- **[test_shared_bank_versioning.py](../tests/test_shared_bank_versioning.py)**
  Tests the question-bank-versioning helpers in `Shared` that back `--sample` and the accuracy tests' bank-aware crash cache. It verifies:
  - `file_hash` is stable for identical file content, differs when content differs, and returns a short (12-character) hex digest.
  - `stratified_sample` returns the bank unchanged (not even reordered) once `n` meets or exceeds its size, otherwise returns exactly `n` questions, touches every category represented within the requested `n`, is deterministic across repeated calls for the same `(bank, n)`, and never produces duplicate IDs.
  - `check_crash_cache`'s `expected_bank_hash` parameter ignores a cached crash recorded against a different bank version (so a stale crash doesn't skip a model forever after the bank changes) while still honoring one that matches, and non-bank-aware callers that omit `expected_bank_hash` still honor a `bank_hash`-tagged entry as before.
  - `record_crash`'s `extra` parameter is merged into the stored cache record (e.g. `bank_hash`), and is simply omitted when not passed.

- **[test_shared_crash_cache.py](../tests/test_shared_crash_cache.py)**
  Tests the model crash-tracking database (which prevents repeated attempts of deterministic crashes). It verifies:
  - Reading from a missing or corrupted file falls back to an empty cache dict.
  - Successful serialization roundtripping.
  - Unwritable disk locations do not crash the runner (write failures are swallowed).
  - The cache matches keys correctly to output skip markers.
  - Exception analyzer (`is_connection_crash`) properly identifies connection errors and server crashes (e.g. BrokenPipe, ConnectionReset, and actively refused sockets) while letting normal runtime errors bubble up.

- **[test_shared_find_comfyui_python.py](../tests/test_shared_find_comfyui_python.py)**
  Tests the search hierarchy in `Shared.find_comfyui_python` for locating the correct python executable to start ComfyUI. It validates the priority order:
  1. Windows portable embedded Python (`python_embeded/python.exe`).
  2. The local `venv/bin/python` under ComfyUI.
  3. The local `.venv/bin/python` under ComfyUI.
  4. The current external active virtual environment (`VIRTUAL_ENV`).
  5. The current system interpreter (`sys.executable`).

- **[test_shared_ollama_maintenance.py](../tests/test_shared_ollama_maintenance.py)**
  Tests Ollama lifecycle hooks and server state controls:
  - `ollama_reachable_or_abort` detects whether Ollama is running.
  - `model_pulled` checks for exact or implicit matches (like tags missing `:latest`) in the local image list.
  - `ollama_model_max_ctx` parses architectural options to identify a model's true context limit, falling back to configuration defaults on failure.
  - `unload_model` issues a keep-alive termination request and handles network errors.
  - `unload_all_models` queries loaded models and terminates them.
  - `wait_until_unloaded` polls until a model is fully evicted.

- **[test_shared_ollama_streaming.py](../tests/test_shared_ollama_streaming.py)**
  Validates NDJSON response stream parsing for Ollama completion endpoints. It tests:
  - Derivation of TTFT and Tokens/sec from server performance fields.
  - Fallback calculation of TTFT using local system time if fields are missing.
  - Resilience against empty, blank, or malformed JSON stream lines.
  - Response extraction preferring the standard content payload over reasoning (`thinking`) fields, but falling back to reasoning text if needed.
  - Intercepting HTTP 500 error payloads to extract clean diagnostic messages (e.g. "model requires more system memory") rather than raising generic HTTP error statuses.

- **[test_shared_run_measured_calls.py](../tests/test_shared_run_measured_calls.py)**
  Tests the execution loop for benchmark runs (`run_measured_calls`). It checks:
  - Correct execution count under normal operations.
  - Instant stoppage and exit status marking on timeout.
  - Skipping a single run's metrics if a standard execution error occurs, while proceeding to the next run.
  - If a connection crash occurs, attempting Ollama recovery. If recovery succeeds, the loop retries the failed run. If recovery fails, the benchmark halts and records the model as crashed.
  - `slow_tps_early_exit` early termination logic based on performance speeds.

- **[test_shared_looks_like_loop.py](../tests/test_shared_looks_like_loop.py)**
  Tests the degenerate-generation-loop heuristic used on timed-out accuracy-test responses (`Shared.looks_like_loop`):
  - Detects a 12+ word run repeated verbatim 3+ times; false on normal prose, short text, and below-threshold repetition; respects custom `ngram_words`/`min_repeats`.
  - Detects a paraphrased loop via repeated hedging/self-correction phrases (e.g. "let me reconsider", "there seems to have been") even with no verbatim repeated run; false when a hedge phrase appears only once.

- **[test_shared_stats.py](../tests/test_shared_stats.py)**
  Validates general helpers in `Shared`:
  - `mean` and `stdev` mathematical routines (including handling empty lists or single-element inputs).
  - Context prompt text builder, assuring that generated prompts meet the target length in characters, do not crash on tiny inputs, and use a varying nonce prefix to bypass model prompt cache hits.

---

## Dashboard Tests

The dashboard (`dashboard/`) has its own, separate test suite using [Vitest](https://vitest.dev/) — it doesn't share `bench-env` or pytest, since it's a JavaScript/React project with its own `node_modules`.

**Running:**

```bash
cd dashboard
npm test              # runs the suite once
npx vitest            # watch mode, reruns on file changes
npx vitest -t "getBarStatusLabel"   # filter by test name
```

`npm run lint` (ESLint) should also pass after any change to `dashboard/src`.

**Scope:** this suite covers `dashboard/src/utils.js` (chart data builders, status-label logic, formatting, model-registry lookups) and `dashboard/src/constants.js` (model-registry consistency). It deliberately does **not** cover React component rendering — no React Testing Library, no jsdom component mounting. The risk this suite guards against is bad data logic silently producing wrong or blank charts, not broken rendering; component-level testing would be a separate, heavier addition if ever needed.

- **[utils.test.js](../dashboard/src/utils.test.js)**
  Tests the pure data-transformation and formatting functions in `utils.js`. Notably:
  - `getBarStatusLabel` — the crashed/timed-out/slow-tps precedence and "which checkpoints get relabeled Skipped vs. show real data" logic (the slow checkpoint's own value is always shown, never hidden behind a status label; only checkpoints *after* it get relabeled).
  - `getImageBarStatusLabel` — the equivalent for image generation timeouts.
  - `buildLLMBarData`/`buildLLMBarConfigs` — that per-checkpoint values and status overlays are correctly assembled, and that a file which stopped early still gets chart columns for depths another compared file reached.
  - `sortBarData`/`findMostStrenuousKey` — ranking rows by the deepest metric with real data, in both directions (higher/lower is better), with missing values always sorted last regardless of direction.
  - `getModelSizeTier` — the known-model lookup and the param-count-parsed fallback for unrecognized models, including its tier boundaries.
  - `fmt` — unit-specific formatting (ms/sec/tps/sps thresholds, K-notation cutoffs, null handling).
  - `sanitizeForFilename` — collapsing special characters for PNG export filenames.
  - `flattenLLMData` — the single-row whole-model-skip case vs. one row per real checkpoint.

- **[constants.test.js](../dashboard/src/constants.test.js)**
  Cross-checks the model registries in `constants.js` against each other — every model in `LLM_MODEL_ORDER`/`IMAGE_MODEL_ORDER`/`EMBED_MODEL_ORDER` has a corresponding label and color (and, for LLM models, a valid size tier), and none of the order lists contain duplicates. This catches the most common maintenance mistake here: adding a model to one registry without adding it to the others it's expected to appear in.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md) · [CLI Reference →](cli-reference.md)
