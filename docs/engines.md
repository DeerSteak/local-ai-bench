[← Back to README](../README.md)

# Engines

**Contents**
- [Why an interface](#why-an-interface)
- [The interface](#the-interface)
- [`OllamaEngine`](#ollamaengine)
- [Selecting an engine](#selecting-an-engine)
- [Adding a new engine](#adding-a-new-engine)
- [Testing](#testing)

## Why an interface

Every benchmark test needs to talk to a running inference server: start/stop
it, check which models are installed, warm one up, run generate/chat/embed
calls, and unload it when done. Until this interface existed, all of that was
Ollama-specific code called directly from `Shared` and each workload module.
That was fine while Ollama was the only server this project talked to, but it
meant adding a second one (llama.cpp with CUDA/ROCm/Vulkan, MLX on Apple
Silicon) would have meant threading `if engine == "ollama"` branches through
every benchmark.

`scripts/engines/` exists so a workload module never knows which server it's
talking to — it calls `engine.chat(...)`, not `Shared.ollama_chat(...)`. Only
`OllamaEngine` exists today; the interface is sized to exactly what the
current call sites need, not speculatively designed for engines that don't
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

- **`start(gpu_visible=False)` is the CPU-only knob.** It used to be an
  Ollama-specific env-var dict (`HIP_VISIBLE_DEVICES=""` etc.) passed in by
  the caller. Now the caller just says "no GPU" and the engine picks
  whichever mechanism applies (env vars for Ollama; a future llama.cpp engine
  would use `--n-gpu-layers 0`, MLX would pick a CPU device) — `benchmark.py`
  doesn't need to know which.
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

[`scripts/engines/ollama.py`](../scripts/engines/ollama.py) is the only
implementation today. It's a near-verbatim move of what used to be
`Shared.ollama_*` — same behavior, same docstrings explaining the *why*
(e.g. why `config.OLLAMA_ENV_DEFAULTS` is pinned on `start()`, why `num_ctx`
matters for avoiding a full model reload). Nothing about how the benchmark
suite behaves against Ollama changed in the move to this interface.

Two things stayed put rather than moving onto the engine:

- **`OllamaTimeout`/`OllamaLoopDetected`** stay defined in `shared.py` —
  they're generic timeout/loop-signaling exception types referenced by name
  in several files, not Ollama-specific despite the name (a future engine
  would raise the same types from its own `chat`/`generate`).
- **Process cleanup on crash** (`Shared._managed_procs`, drained by
  `Shared.shutdown_managed()`) stays a single shared list on `Shared`, since
  ComfyUI's server process shares that same shutdown path. `OllamaEngine`
  just registers its subprocess into it; it doesn't own cleanup itself.
  `OllamaEngine._cpu_only_active` tracks whether *this* engine is running
  GPU-hidden, so `shutdown_managed()` knows to kill rather than leave a
  GPU-hidden server running silently in the background.

## Selecting an engine

`benchmark.py` takes `--engine` (default and, today, only choice: `ollama`):

```
python scripts/benchmark.py --engine ollama --tests llm
```

`main()` constructs the engine once via `get_engine(args.engine)`
([`scripts/engines/__init__.py`](../scripts/engines/__init__.py)) and passes
the same instance into every workload's `run()`. Nothing downstream imports
`OllamaEngine` directly — everything goes through the `InferenceEngine`
methods, so `--engine llama-cpp` (once that engine exists) wouldn't require
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

Things to get right for a process-per-model server (llama.cpp's `llama-server`,
MLX's `mlx_lm.server`) that Ollama's always-on multi-model daemon didn't have
to think about:

- Ollama pulls/swaps models on demand from one long-running server; a
  llama.cpp/MLX server is typically launched pointed at one model file.
  `warmup`/`unload` likely need to actually start/stop the underlying process
  per model rather than issuing a keep-alive request to an already-running
  server.
- `model_pulled`/`list_installed_models` will need a different notion of
  "installed" (a GGUF file on disk / an MLX model directory, not an Ollama
  tag).
- `max_context_length` will need to read the model's own metadata (GGUF
  header / MLX config) instead of Ollama's `/api/show`.

## Testing

[`tests/test_ollama_engine.py`](../tests/test_ollama_engine.py) tests
`OllamaEngine` directly (HTTP mocked at the `requests`/`urllib` seam — see
[Testing](testing.md)). Orchestration logic that consumes an engine
(`Shared.run_measured_calls`, `Shared.run_accuracy_benchmark`) is tested
against a fake `InferenceEngine` double with canned in-memory responses, no
network involved — see
[`tests/test_run_accuracy_benchmark.py`](../tests/test_run_accuracy_benchmark.py).
That split is the actual payoff of the interface: a future second engine
gets real coverage from day one, and the orchestration tests never need to
change when it's added.

---

[← How It Works](how-it-works.md) · [Back to README](../README.md) · [Project Structure →](project-structure.md)
