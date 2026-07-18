[← Back to README](../README.md)

# Engines

**Contents**
- [Why an interface](#why-an-interface)
- [The interface](#the-interface)
- [`OllamaEngine`](#ollamaengine)
- [`LlamaCppEngine`](#llamacppengine)
- [Selecting an engine](#selecting-an-engine)
- [Adding a new engine](#adding-a-new-engine)
- [Testing](#testing)

## Why an interface

Every benchmark test needs to talk to a running inference server: start/stop
it, check which models are installed, warm one up, run generate/chat/embed
calls, and unload it when done. `scripts/engines/` is the seam for that: a
workload module calls `engine.chat(...)`, never anything server-specific, so
`benchmark.py` can point the same test suite at `OllamaEngine` or
`LlamaCppEngine` (MLX is a plausible third) without any workload module
knowing or caring which. The interface is sized to exactly what those two
implementations need, not speculatively designed for engines that don't
exist yet.

## The interface

`InferenceEngine` ([`scripts/engines/base.py`](../scripts/engines/base.py))
is an ABC with three groups of methods:

| Group | Methods |
|---|---|
| Server/process lifecycle | `ensure_running`, `start(gpu_visible=...)`, `stop`, `available`, `reachable_or_abort`, `wait_for_recovery`, `is_connection_crash`, `tail_log` |
| Model lifecycle | `model_pulled`, `list_installed_models`, `max_context_length`, `warmup`, `unload`, `unload_all`, `wait_until_unloaded` |
| Inference | `generate` (single-shot), `chat` (multi-turn), `embed` |

A few design choices worth knowing if you're reading or extending this:

- **`start(gpu_visible=False)` is the CPU-only knob.** The caller just says
  "no GPU"; each engine picks its own mechanism — `OllamaEngine` sets
  `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` empty
  and restarts, `LlamaCppEngine` remembers it and passes `-ngl 0` on the next
  per-model spawn (llama-server has no standalone process to restart until a
  model is actually requested) — `benchmark.py` doesn't need to know which.
- **`is_connection_crash` / `wait_for_recovery` / `tail_log`** exist because
  crash handling isn't generic: `Shared.run_measured_calls` needs to tell a
  model-runner crash apart from an ordinary failure, wait for the server to
  respawn, and surface real log output — and what counts as "the runner
  crashed" and how recovery works is engine-specific. These three methods are
  the seam that keeps that crash-retry *logic* (in `Shared`) generic while the
  crash *detection* (in the engine) isn't.
- **`generate`/`chat`/`embed` return plain tuples**, not engine-specific
  response objects, so a caller scoring a benchmark run never touches
  anything Ollama-shaped.

## `OllamaEngine`

[`scripts/engines/ollama.py`](../scripts/engines/ollama.py) is the Ollama
REST/process client: server lifecycle (`start`/`stop`/`ensure_running`),
model lifecycle (`warmup`/`unload`/`list_installed_models`), and
`generate`/`chat`/`embed` over Ollama's HTTP API. Its docstrings cover the
*why* behind specific choices — e.g. why `config.OLLAMA_ENV_DEFAULTS` is
pinned on `start()`, why `num_ctx` matters for avoiding a full model reload.

Two things live on `Shared` rather than the engine:

- **`OllamaTimeout`/`OllamaLoopDetected`** stay defined in `shared.py` —
  they're generic timeout/loop-signaling exception types referenced by name
  in several files, not Ollama-specific despite the name (`LlamaCppEngine`
  raises the same types from its own `chat`/`generate`).
- **Process cleanup on crash** (`Shared._managed_procs`, drained by
  `Shared.shutdown_managed()`) stays a single shared list on `Shared`, since
  ComfyUI's server process shares that same shutdown path. `OllamaEngine`
  just registers its subprocess into it; it doesn't own cleanup itself.
  `OllamaEngine._cpu_only_active` tracks whether *this* engine is running
  GPU-hidden, so `shutdown_managed()` knows to kill rather than leave a
  GPU-hidden server running silently in the background.

## `LlamaCppEngine`

[`scripts/engines/llamacpp.py`](../scripts/engines/llamacpp.py) drives
llama.cpp's `llama-server` directly, reusing models already pulled via
`ollama pull` instead of downloading its own copy — the point is a clean,
Ollama-free read on the same weights, not a separate model catalog. Shape
differences from `OllamaEngine`, all consequences of llama-server being a
process-per-model server rather than an always-on multi-model daemon:

- **No standalone "up but idle" state.** `ensure_running()` is a preflight —
  confirm the binary exists and Ollama's model store is reachable — not an
  actual server start; the real `llama-server` subprocess spawns lazily per
  tag in `_ensure_model`, and restarts whenever the requested
  `(tag, num_ctx, embedding-mode)` combination changes (mirrors Ollama's own
  num_ctx-mismatch-triggers-reload behavior, so both engines pay the same
  *kind* of cold-swap cost via different mechanisms). This is also why
  `available()` is `False` between models, not just before the first one —
  a workload's `run()` must gate its top-of-run preflight on
  `ensure_running()`, never `available()`/`reachable_or_abort()`, or every
  model looks unreachable before `_ensure_model` ever gets a chance to spawn
  one. `reachable_or_abort()`/`wait_for_recovery()` are both unconditionally
  `True` on this engine for the same reason: there's no shared always-on
  server to check between models or wait to self-heal — a failed spawn
  raises on that model's own call instead, caught by the normal
  crash-handling loop in `Shared.run_measured_calls`.
  *kind* of cold-swap cost via different mechanisms).
- **Models resolve straight from Ollama's on-disk blob store**
  (`_resolve_blob_path`), no HTTP call to Ollama involved: Ollama stores a
  pulled model as an OCI-like manifest (`manifests/registry.ollama.ai/library/
  <model>/<tag>`) pointing at a content-addressed GGUF blob
  (`blobs/sha256-<hash>`) — llama.cpp identifies GGUF files by magic bytes,
  not extension, so that blob is a valid `-m` target as-is.
- **Binary resolution** (`_binary_path`) checks, in order: `config.LLAMACPP_DIR`
  (where `setup_check.py`'s Linux source build and Windows prebuilt-zip
  install both vendor it), `PATH`, then — macOS only — the two well-known
  Homebrew prefixes (`/opt/homebrew/bin`, `/usr/local/bin`) directly, since a
  `brew install`-created symlink isn't guaranteed to be on `PATH` in whatever
  shell runs the benchmark. No terminal restart or re-sourced rc file is ever
  required.
- **Ollama's model-store location** (`_ollama_models_dir`) is the current
  user's `~/.ollama/models` (checked first), or — on Linux, for a
  service-managed install — `/var/snap/ollama/common/models` (Ubuntu's
  `snap install ollama`) or `/usr/share/ollama/.ollama/models` (the official
  `curl | sh` installer's systemd service, which runs as a dedicated `ollama`
  system user). `ollama list` works from any shell regardless, since it just
  talks to the running server over HTTP; resolving the real on-disk path is
  only needed for direct blob/manifest reads.
- **`-jinja`** renders the model's own embedded chat template
  (`tokenizer.chat_template` GGUF metadata) rather than llama.cpp's built-in
  template-guessing heuristics, for closer parity with what Ollama renders
  from the same weights. Not every GGUF has that metadata embedded (older
  conversions, mainly) — there's no setup-time check yet that warns when it's
  missing, so a silent template divergence between engines is still a
  possible, if narrow, source of a confusing quality difference. Worth
  building if it ever actually shows up in a comparison.

## Selecting an engine

`benchmark.py` takes `--engine ollama|llamacpp|both` (default: `llamacpp` —
marginally faster than Ollama and a closer read on raw model capability,
without Ollama's scheduling/wrapper overhead; `both` runs the whole suite
once per engine, back to back, and writes a separate results file for each,
tagged internally with `"engine"` so the file is self-identifying even if
renamed):

```
python scripts/benchmark.py --engine llamacpp --tests llm
python scripts/benchmark.py --engine both
```

Whichever engine is about to start, the other is stopped first — even a
stray instance this process didn't launch itself — so the two never compete
for GPU memory at once, whether that's a single-engine run against a
background Ollama service or the pass switch inside `--engine both`.

`main()` constructs the engine once per pass via `get_engine(engine_name)`
([`scripts/engines/__init__.py`](../scripts/engines/__init__.py)) and passes
the same instance into every workload's `run()`. Nothing downstream imports
`OllamaEngine`/`LlamaCppEngine` directly — everything goes through the
`InferenceEngine` methods, so a third engine (e.g. MLX) wouldn't require
touching `mcq_benchmark.py`, `embedding_benchmark.py`, or any other workload
module.

## Adding a new engine

1. Create `scripts/engines/<name>.py` with a class implementing every
   `InferenceEngine` method.
2. Register it in `scripts/engines/__init__.py`'s registry dict and add it to
   `benchmark.py`'s `--engine` `choices`.
3. Nothing else changes. `Shared.run_measured_calls`, `Shared.run_accuracy_benchmark`,
   and every workload module's `run()` already take an `engine` parameter and
   only call `InferenceEngine` methods.

A process-per-model server (`LlamaCppEngine`, or MLX's `mlx_lm.server`) needs
a few things Ollama's always-on multi-model daemon doesn't:

- `warmup`/`unload` actually start/stop the underlying process per model,
  rather than issuing a keep-alive request to an already-running server —
  see `LlamaCppEngine._ensure_model`/`_stop_process`.
- `model_pulled`/`list_installed_models` need their own notion of
  "installed" — `LlamaCppEngine` walks Ollama's manifest tree directly (see
  above); an engine with its own weight format (MLX, via `mlx_lm.convert`)
  needs its own model-directory convention instead.
- `max_context_length` reads the model's own metadata (GGUF header / MLX
  config) instead of Ollama's `/api/show`.

For any engine that reuses Ollama's downloaded weights rather than
converting its own: quantization is whatever Ollama pulled for that tag
(e.g. `q4_K_M`) — worth being explicit about when a results table calls two
engines' runs "the same model."

## Testing

[`tests/test_ollama_engine.py`](../tests/test_ollama_engine.py) and
[`tests/test_llamacpp_engine.py`](../tests/test_llamacpp_engine.py) test
`OllamaEngine`/`LlamaCppEngine` directly (HTTP mocked at the
`requests`/`urllib` seam; real subprocess spawns, like real HTTP calls, stay
outside the test suite — see [Testing](testing.md)). Orchestration logic
that consumes an engine (`Shared.run_measured_calls`,
`Shared.run_accuracy_benchmark`) is tested against a fake `InferenceEngine`
double with canned in-memory responses, no network involved — see
[`tests/test_run_accuracy_benchmark.py`](../tests/test_run_accuracy_benchmark.py).
A new engine gets orchestration-test coverage for free — nothing in that
test file references `OllamaEngine` or `LlamaCppEngine` by name.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md) · [Project Structure →](project-structure.md)
