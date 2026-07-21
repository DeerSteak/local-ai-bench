[← Back to README](../README.md)

# Engines

**Contents**
- [Why an interface](#why-an-interface)
- [The interface](#the-interface)
- [`LlamaCppEngine`](#llamacppengine)
- [Selecting an engine](#selecting-an-engine)
- [Adding a new engine](#adding-a-new-engine)
- [Testing](#testing)

llama.cpp is this project's only inference engine — Ollama support was
removed after a head-to-head performance comparison on this project's
hardware found llama.cpp marginally faster with no measurable downside. The
`InferenceEngine` interface below stays engine-agnostic on purpose, and
`--engine` still takes a name (or `all`, see [Selecting an
engine](#selecting-an-engine) below): it's not speculative abstraction left
over from that comparison, it's what lets a future engine (e.g. MLX) plug in
without touching any workload module.

## Why an interface

Every benchmark test needs to talk to a running inference server: start/stop
it, check which models are installed, warm one up, run generate/chat/embed
calls, and unload it when done. `scripts/engines/` is the seam for that: a
workload module calls `engine.chat(...)`, never anything server-specific, so
`benchmark.py` can swap in a different `InferenceEngine` implementation
without any workload module knowing or caring which. The interface is sized
to exactly what `LlamaCppEngine` needs, not speculatively designed for
engines that don't exist yet.

## The interface

`InferenceEngine` ([`scripts/engines/base.py`](../scripts/engines/base.py))
is an ABC with three groups of methods:

| Group | Methods |
|---|---|
| Server/process lifecycle | `ensure_running`, `start(gpu_visible=...)`, `stop`, `available`, `reachable_or_abort`, `wait_for_recovery`, `is_connection_crash`, `tail_log`, `runtime_backend` |
| Model lifecycle | `model_pulled`, `list_installed_models`, `max_context_length`, `warmup`, `unload`, `unload_all`, `wait_until_unloaded`, `prepare_concurrency` |
| Inference | `generate` (single-shot), `chat` (multi-turn), `chat_tools` (tool-calling), `embed` |

A few design choices worth knowing if you're reading or extending this:

- **`start(gpu_visible=False)` is the CPU-only knob.** The caller just says
  "no GPU"; `LlamaCppEngine` remembers it and passes `-ngl 0` on the next
  per-model spawn (llama-server has no standalone process to restart until a
  model is actually requested).
- **`is_connection_crash` / `wait_for_recovery` / `tail_log`** exist because
  crash handling isn't generic: `Shared.run_measured_calls` needs to tell a
  model-runner crash apart from an ordinary failure, wait for the server to
  respawn, and surface real log output — and what counts as "the runner
  crashed" and how recovery works is engine-specific. These three methods are
  the seam that keeps that crash-retry *logic* (in `Shared`) generic while the
  crash *detection* (in the engine) isn't.
- **`generate`/`chat`/`embed` return plain tuples**, not engine-specific
  response objects, so a caller scoring a benchmark run never touches
  anything engine-shaped.
- **`chat_tools` is a separate method, not an argument to `chat`**, so the
  tool-calling accuracy test can offer an OpenAI-style tools array and read
  back a parsed `tool_calls` list (`[{"name", "arguments"}]`) without changing
  `chat`'s signature or return arity at every existing call site. It returns
  `chat`'s five values plus that list (empty when the model called nothing).
- **`prepare_concurrency(tag, n_parallel, per_slot_ctx, warmup_runs, timeout)`** (used only by the
  concurrency test — see [Workloads](workloads.md#concurrency)) scales
  `per_slot_ctx` up by `n_parallel` before it becomes llama-server's `-c`,
  because `-c` is a *total* KV-cache budget split across `--parallel` slots,
  not a per-slot size.

## `LlamaCppEngine`

[`scripts/engines/llamacpp.py`](../scripts/engines/llamacpp.py) drives
llama.cpp's `llama-server` directly. It resolves each catalog tag to GGUF
file(s) downloaded ahead of time by `setup_check.py` into
`config.MODELS_DIR/llamacpp/<slug>/` (`LlamaCppEngine._models_dir()` returns
`config.MODELS_DIR / self.name`, namespacing each engine's own model
directory so a future engine, e.g. MLX, gets its own `models/mlx/` subtree
instead of colliding with llama.cpp's GGUF layout) — one subdirectory per
tag under that, named from the tag with `:`/`/` replaced by `_`
(`LlamaCppEngine._slug`) — rather than reading any inference server's own
model store. `models.py`'s catalog entries carry
`hf_repo` (a HuggingFace repo id) and `hf_file` (a filename, or a list of
filenames for a model split across multiple GGUF parts — the two large-tier
models, Llama 4 Scout and Nemotron 3 Super, are split 2-way and 3-way
respectively) so `setup_check.py` and `LlamaCppEngine` agree on exactly which
file(s) a tag resolves to; `model_pulled`/`list_installed_models` check that
every listed file exists under that tag's subdirectory, and
`max_context_length` reads the real context length straight from the first
file's GGUF metadata. The existing `tag` field values (e.g.
`"llama3.2:3b-instruct-q4_K_M"`) are unchanged in shape — they're now opaque
catalog identifiers rather than literal server tags, but every other file
that already keyed off them (results JSON, crash caches, `--models`) doesn't
need to change. A non-catalog directory can contain either one GGUF or one
complete, consistently named multipart GGUF set; its directory name becomes
the custom tag advertised by `list_installed_models` and resolved by the same
model-loading path.

Shape differences from an always-on multi-model daemon, both consequences of
llama-server being a process-per-model server:

- **No standalone "up but idle" state.** `ensure_running()` is a preflight —
  confirm the binary exists and `_models_dir()` is reachable — not an
  actual server start; the real `llama-server` subprocess spawns lazily per
  tag in `_ensure_model`, and restarts whenever the requested
  `(tag, num_ctx, embedding-mode, n_parallel)` combination changes. This is also why
  `available()` is `False` between models, not just before the first one —
  a workload's `run()` must gate its top-of-run preflight on
  `ensure_running()`, never `available()`/`reachable_or_abort()`, or every
  model looks unreachable before `_ensure_model` ever gets a chance to spawn
  one. `reachable_or_abort()`/`wait_for_recovery()` are both unconditionally
  `True` on this engine for the same reason: there's no shared always-on
  server to check between models or wait to self-heal — a failed spawn
  raises on that model's own call instead, caught by the normal
  crash-handling loop in `Shared.run_measured_calls`.
- **Binary resolution** (`_binary_path`) checks, in order: `config.LLAMACPP_DIR`
  (where `setup_check.py`'s Linux source build and Windows prebuilt-zip
  install both vendor it), `PATH`, then — macOS only — the two well-known
  Homebrew prefixes (`/opt/homebrew/bin`, `/usr/local/bin`) directly, since a
  `brew install`-created symlink isn't guaranteed to be on `PATH` in whatever
  shell runs the benchmark. No terminal restart or re-sourced rc file is ever
  required.
- **`-jinja`** renders the model's own embedded chat template
  (`tokenizer.chat_template` GGUF metadata) rather than llama.cpp's built-in
  template-guessing heuristics. Not every GGUF has that metadata embedded
  (older conversions, mainly) — there's no setup-time check yet that warns
  when it's missing, so a silent template mismatch on those models is still
  a possible, if narrow, source of a confusing quality difference.
- **Generated-token accounting** uses native streamed token IDs for
  `/completion` and the trailing `usage.completion_tokens` value for
  OpenAI-compatible chat, including reasoning tokens. SSE text fragments are
  transport units and are never counted as tokens.
- **`_sanitize_tps`** guards `generate()`/`chat()`'s tok/s calculation
  against a real observed llama-server quirk: under heavy concurrent-slot
  contention (see [Concurrency](workloads.md#concurrency)), a streamed
  chunk's `timings.predicted_ms` can be implausibly tiny relative to
  `predicted_n`, producing a tps ratio with no physical basis (six-figure
  values observed on real hardware). Any self-reported tps above
  `config.MAX_PLAUSIBLE_TPS` is replaced with a wall-clock estimate
  (authoritatively counted tokens over measured decode time) instead — a sanity
  tripwire, not a tuned threshold, since no real single-request stream gets
  remotely close to it on current hardware. Whenever this fires,
  `_warn_tps_sanitized` logs the raw `predicted_n`/`predicted_ms` values
  that produced the bad reading (not just the corrected number), so a run
  that hits this is still diagnosable — worth keeping an eye on the console
  output if you're trying to track down why llama-server reported it.
- **`runtime_backend`** runs llama-server's read-only `--list-devices` query
  and reports the build/runtime family it can actually use (`cuda`, `rocm`,
  `metal`, `xpu`, `vulkan`, or `cpu`). Results retain physical GPU detection
  separately as `profile.hardware_backend`, so a Vulkan Windows build or a
  CPU-only Linux build is not mislabeled from hardware presence alone.

## Selecting an engine

`benchmark.py` takes `--engine <name>|all` (default: `llamacpp`; `all`
expands to every name in `scripts/engines/__init__.py`'s registry, sorted,
and runs the full `--tests` suite once per engine, back to back, writing a
separate results file for each — engine name appended to the filename, and
each file tagged internally with `"engine"` so it's self-identifying even if
renamed):

```
python scripts/benchmark.py --engine llamacpp --tests llm
python scripts/benchmark.py --engine all
```

Only `llamacpp` is registered today, so there's nothing to actually select
between yet — `--engine all` runs the same single pass `--engine llamacpp`
does. The flag and the `all` expansion logic (`resolve_engine_names` in
`benchmark.py`) exist now so a second engine (e.g. MLX) slots in later
without any CLI or docs changes. Image generation doesn't depend on
`--engine` (a separate ComfyUI call), so a multi-engine `all` run captures it
once, on the first pass, rather than once per engine.

`main()` constructs the engine once per pass via `get_engine(engine_name)`
([`scripts/engines/__init__.py`](../scripts/engines/__init__.py)) and passes
the same instance into every workload's `run()`. Nothing downstream imports
`LlamaCppEngine` directly — everything goes through the `InferenceEngine`
methods, so a second engine wouldn't require touching `mcq_benchmark.py`,
`embedding_benchmark.py`, or any other workload module.

## Adding a new engine

1. Create `scripts/engines/<name>.py` with a class implementing every
   `InferenceEngine` method.
2. Register it in `scripts/engines/__init__.py`'s registry dict.
3. Nothing else changes. `Shared.run_measured_calls`, `Shared.run_accuracy_benchmark`,
   and every workload module's `run()` already take an `engine` parameter and
   only call `InferenceEngine` methods.

A process-per-model server (`LlamaCppEngine`, or MLX's `mlx_lm.server`) needs
a few things an always-on multi-model daemon doesn't:

- `warmup`/`unload` actually start/stop the underlying process per model,
  rather than issuing a keep-alive request to an already-running server —
  see `LlamaCppEngine._ensure_model`/`_stop_process`.
- `model_pulled`/`list_installed_models` need their own notion of
  "installed" — `LlamaCppEngine` checks its GGUF files under
  `config.MODELS_DIR` (see above); an engine with its own weight format (MLX,
  via `mlx_lm.convert`) needs its own model-directory convention instead.
- `max_context_length` reads the model's own metadata (GGUF header / MLX
  config) directly, rather than asking a server for it.

## Testing

[`tests/test_llamacpp_engine.py`](../tests/test_llamacpp_engine.py) tests
`LlamaCppEngine` directly (HTTP mocked at the `requests`/`urllib` seam; real
subprocess spawns, like real HTTP calls, stay outside the test suite — see
[Testing](testing.md)). Orchestration logic that consumes an engine
(`Shared.run_measured_calls`, `Shared.run_accuracy_benchmark`) is tested
against a fake `InferenceEngine` double with canned in-memory responses, no
network involved — see
[`tests/test_run_accuracy_benchmark.py`](../tests/test_run_accuracy_benchmark.py).
A new engine gets orchestration-test coverage for free — nothing in that
test file references `LlamaCppEngine` by name.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md) · [Project Structure →](project-structure.md)
