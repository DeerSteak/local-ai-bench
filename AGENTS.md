# AGENTS.md

Instructions for AI coding agents working in this repository. Read this before making changes.

## What this is

`local-ai-bench` is a cross-platform benchmark suite for local LLM generation (Ollama), image generation (ComfyUI), and embeddings. It's designed to run unattended on real hardware (from 8GB GPUs up to unified-memory workstations) and produce comparable results across machines via a React/Vite dashboard.

Full docs live in [`docs/`](docs/) — [Project Structure](docs/project-structure.md), [How It Works](docs/how-it-works.md), [Testing](docs/testing.md), [Workloads](docs/workloads.md), [CLI Reference](docs/cli-reference.md), [Setup](docs/setup.md), [Dashboard](docs/dashboard.md). This file is the entry point and summary — when in doubt, the docs above are authoritative and more detailed.

**`ComfyUI/` is a vendored third-party dependency, not part of this project.** Don't treat it as code to maintain, refactor, or write tests for.

## Repo layout

```
scripts/            Python benchmark implementation (see below)
tests/               pytest suite — one test module roughly per scripts/ module
dashboard/           React + Vite results-explorer web app
docs/                 Detailed docs (see links above)
samples/             Sample results_*.json for trying the dashboard without a real run
bench-env/           Project venv (gitignored) — created by setup.sh/setup.bat
requirements.txt      Runtime deps, installed by setup scripts into bench-env/
tests/requirements.txt  Test-only deps (pytest), installed by tests.sh/.bat into bench-env/
setup.sh / setup.bat        One-shot install + interactive model picker
run_bench.sh / .bat          Activates bench-env, runs scripts/benchmark.py
dashboard.sh / .bat           Builds + serves the dashboard
tests.sh / .bat                Activates bench-env, runs pytest
.coveragerc            Coverage config — see Testing section below
```

`scripts/` modules:
- `benchmark.py` — CLI entry point, argument parsing, orchestration (`main()`)
- `config.py` — shared constants (URLs, paths, timeouts, run counts)
- `shared.py` — cross-cutting helpers: logging, server lifecycle, machine profiling, Ollama/ComfyUI HTTP clients
- `llm_prefill_benchmark.py` — single-shot cold-prefill LLM test
- `llm_conversation_benchmark.py` — multi-turn conversation LLM test
- `embedding_benchmark.py` — embeddings test
- `image_benchmark.py` — image generation test (ComfyUI workflow builders + submission)
- `models.py` — single source of truth for every model definition (tags, checkpoints, tiers, sizes)
- `setup_check.py` — hardware detection, interactive model picker, unattended install (called by `setup.sh`/`setup.bat`)

## Critical safety rules

**Never execute `setup_check.py`, `setup.sh`, or `setup.bat` directly to test a change**, even with piped/non-interactive stdin. These scripts have real, hard-to-reverse side effects: installing Ollama via Homebrew, pulling multi-GB models, cloning ComfyUI, downloading multi-GB checkpoints. This happened once already — running `setup_check.py --all` to "just check the fallback path" actually installed Ollama, 11 Homebrew packages, and started pulling a ~5GB model for real, requiring manual cleanup. To test logic inside these scripts, extract the specific function and test it in isolation (a `pty` harness for terminal-UI logic, a plain unit test for everything else) — never run the real entrypoint. If the full script genuinely needs an end-to-end run, ask the user to run it themselves.

**Never run `benchmark.py` for a real test run** unless the user explicitly asks — it drives real Ollama/ComfyUI servers, loads multi-GB models into memory, and can take hours.

**Use the existing `bench-env/` venv, not scratch venvs.** It's the user's real, persistent environment (created by `setup.sh`/`setup.bat`, already has `requirements.txt` installed) and is what every wrapper script (`run_bench.sh`, `dashboard.sh`, `tests.sh`) activates. Activate it or call `bench-env/bin/python`/`bench-env/bin/pip` directly for any check against this codebase (running tests, computing coverage, syntax-checking, simulating a module) — on Windows it's `bench-env\Scripts\python.exe`/`bench-env\Scripts\pip.exe`. Only fall back to a throwaway venv if `bench-env/` genuinely doesn't exist yet and creating it isn't appropriate.

**`hf.txt` (repo root, gitignored) holds a real HuggingFace access token.** Never print, log, or commit its contents, and never include it in a diff or summary shown to anyone other than the user in their own terminal.

## Environment / running things

```bash
# Run the test suite (safe, no live side effects)
bash tests.sh
bash tests.sh -k "select_tier"        # filter by test name
bash tests.sh --cov=scripts --cov-report=term-missing   # with coverage (needs pytest-cov)

# Everything else below has real side effects — confirm with the user first
bash setup.sh              # installs Ollama, models, ComfyUI checkpoints
bash run_bench.sh           # runs the real benchmark suite
bash dashboard.sh           # builds + serves the results dashboard
```

Windows equivalents are the same commands with `.bat` instead of `.sh` (no `bash` prefix).

## Domain & algorithm notes

The reasoning below isn't fully written down anywhere else — the docs describe *what* these tests do, this is the *why* behind specific implementation choices.

**Two LLM test modes measure genuinely different things — don't compare their TTFT numbers at face value.**
- **Single-shot** (`llm_prefill_benchmark.py`): a fresh, unique-content prompt padded to a target size (2K/8K/32K/64K), sent cold every run. TTFT is a genuine cold prefill — the whole prompt is processed with nothing cached.
- **Conversation** (`llm_conversation_benchmark.py`): one real multi-turn chat, grown from a blank slate toward 96K. TTFT here measures only the *new* turn's marginal cost, relying on the backend's slot/KV-cache reuse — that's why conversation TTFT at, say, 32K is a small fraction of single-shot TTFT at 32K. TPS (decode speed) *is* comparable between the two, since it depends on total context depth in both cases, not on what's cached.

**Model tiers are cumulative.** `xsmall` (<6B), `small` (≤20B), `medium` (26–35B), `large` (70B+) are defined in `models.py`. `--maxtier medium` runs xsmall+small+medium, not just medium — `select_tier()` in `benchmark.py` applies the identical cumulative cap to image models via each image model's own `tier` field. To add a model to a tier, add it to the right list in `models.py`; the cap logic itself shouldn't need to change.

**Conversation growth step sizing** (`LLMConversationBenchmark.compute_growth_step`) balances two failure modes: growing to a checkpoint in one giant turn overshoots it by a wide margin (defeating the point of sampling *at* that depth), while growing in small fixed steps takes far more turns — and wall-clock time — than necessary. The fix: take large steps (`CONV_STEP_MAX_FAR = 4096`) while more than 8K tokens from the target, then switch to fine steps (`CONV_STEP_MAX = 1024`) once close, so the turn that actually lands on a checkpoint doesn't overshoot it much. Growth also stops at 99.5% of each checkpoint's target (`target * 0.995`) rather than the exact value, deliberately — TTFT in this test isn't sensitive to total depth (see above), and TPS varies smoothly enough that a sub-1% depth difference is far smaller than this test's own run-to-run noise (it only runs once per model, `CONV_RUNS = 1`). This roughly halved the turns needed to reach 96K with no measurable loss of precision. Don't "fix" this by growing to the exact target, and don't shrink `CONV_STEP_MAX_FAR` without checking the turn-count cost against a real target/num_ctx trace first.

**The conversation test's slow-model early exit checks TPS only *at* sampled checkpoints (0K/2K/4K/.../96K), never mid-growth between them.** If TPS at a checkpoint is below `config.SLOW_MODEL_MIN_TPS`, that checkpoint's real (slow) measurement is still recorded, then the run stops — deeper checkpoints are skipped. This is separate from the single-shot test's own pre-flight skip in `benchmark.py`'s `conv_skip_entry()`, which decides whether a model enters the conversation test at all, before it starts. `--force-all` bypasses both cutoffs.

**Results JSON is a schema that evolves across versions, and the dashboard has to tolerate that.** People compare results generated by different versions of this benchmark suite across different machines, so a results file is never guaranteed to have every field a newer schema might expect. `dashboard/src/utils.js` leans heavily on optional chaining (`f.data[section]?.[model]?.[ctx]`, not `f.data[section][model][ctx]`) for exactly this reason. When adding a new field to the results JSON or new dashboard code that reads it, preserve this — assume any given key might be missing on an older file, don't assume presence.

**Dashboard checkpoint handling is driven entirely by `CTX_ORDER`** (`dashboard/src/constants.js`) — never hardcode a context-depth label (`"32K"`, `"96K"`, etc.) elsewhere in the dashboard. `getBarStatusLabel()` in `utils.js` looks up any depth's position in `CTX_ORDER` to decide whether to render real data or a "Skipped (X Too Slow)" label — this is why the dashboard already handles the slow-exit above firing at *any* checkpoint depth, not just the first one, with no special-casing. If a new checkpoint is ever added to `LLMConversationBenchmark.CONV_CHECKPOINTS`, add it to `CTX_ORDER` too and the rest follows automatically.

## Testing conventions

This is the part to get right — **write comprehensive, valuable tests for anything you touch in `scripts/`**, not superficial ones.

**Structure:**
- `tests/conftest.py` puts `scripts/` on `sys.path`, so tests import modules the same way the scripts import each other (`import config`, `from shared import Shared`, etc.) — bare top-level imports, not `scripts.foo`.
- One test file roughly per source module; split further when a module has multiple distinct concerns (e.g. `shared.py` → `test_shared_crash_cache.py`, `test_shared_ollama_streaming.py`, `test_shared_ollama_maintenance.py`, `test_shared_run_measured_calls.py`, `test_shared_stats.py`, `test_shared_find_comfyui_python.py`).
- Use `pytest`'s plain `assert`, `monkeypatch`, `unittest.mock.patch`, and `tmp_path` fixtures — no custom test framework.

**What to test — the real boundary:**
- **Do** unit test pure logic and anything mockable at a clean seam: parsing, calculation, decision/skip logic, config selection, request/response shaping. Mock `requests`/`urllib` calls and `Shared.*` seams rather than hitting a real server.
- **Don't** try to unit test code that spawns real subprocesses, polls a live Ollama/ComfyUI server, or orchestrates a full run end-to-end (`benchmark.py`'s `main()`, each workload class's `run()`, `Shared.start_ollama`/`ensure_comfyui`/`get_hostname`/`detect_backend`, etc.). These are marked `# pragma: no cover` at their `def` line (coverage.py excludes the whole function body from that point) rather than skipped silently — the exclusion is deliberate and documented, not a gap to "fix" by adding a live-server test.
- `scripts/setup_check.py` is entirely omitted from coverage via `.coveragerc` (`omit = [scripts/setup_check.py]`) — it has no `__main__` guard, so importing it runs the whole interactive install flow. Don't try to cover it directly; see the safety rules above for how to test logic inside it.

**Extract before testing, when logic is buried in a loop.** Several times in this project's history, business logic embedded in a large orchestration loop turned out to be worth pulling into its own pure, testable function rather than leaving it untested inside a `# pragma: no cover` method:
- `conv_skip_entry()` in `benchmark.py` — the conversation-test skip/reason logic, pulled out of `main()`'s loop
- `select_tier()` in `benchmark.py` — the `--maxtier` cumulative model/image filtering, pulled out of `main()`
- `compute_growth_step()` in `llm_conversation_benchmark.py` — the conversation-growth step-sizing math, pulled out of `run()`
- `ImageBenchmark.build_workflow()` — the flux/flux2/sd3/sdxl dispatch, deduplicated out of two copies inside `run()`

When you find similar buried logic (a decision, a calculation, a dispatch) inside an orchestration method that's otherwise untestable, prefer extracting it to a `@staticmethod`/module-level function and testing *that*, over leaving it untested or writing a heavy integration-style test.

**When a bug or edge case isn't obvious from reading the code, verify empirically before trusting your own trace.** For non-trivial control flow (e.g. does a growth loop actually terminate, does it overshoot a bound, does an early-exit fire at the right point), write a throwaway script that imports the real function and runs it against representative inputs — using `bench-env/bin/python`, not a scratch venv — rather than relying purely on hand-tracing. This caught nothing wrong so far, but it's the standard this project holds review to.

**The dashboard (`dashboard/`) has no automated test suite** — `package.json` only has `dev`/`build`/`lint`/`preview` scripts, no Jest/Vitest/Testing Library. Don't assume one exists or silently skip verifying a dashboard change either — run `npm run lint` from `dashboard/` after touching `dashboard/src`, and manually trace/verify logic changes (e.g. against a sample file in `samples/`) the way this project's growth-loop and dashboard-rendering logic were verified in review: by tracing the actual code path, not just reading it. If a dashboard change is complex enough that a real test would be valuable, say so to the user rather than adding a test framework unasked.

**Coverage:** `pytest-cov` isn't installed by default — `bench-env/bin/pip install pytest-cov` first. Run via `bash tests.sh --cov=scripts --cov-report=term-missing`. `.coveragerc` shapes the report to reflect only the code meant to be unit-tested (see above); with it in place, coverage sits around 95%. Don't chase the exact number — the remaining gaps are fine-grained exception branches, not whole untested subsystems. If you add a new `run()`-style orchestration method or similarly untestable function, mark it `# pragma: no cover` at the `def` line rather than leaving it to silently drag the coverage number down without explanation.

## Code conventions

- **CLI-overridable config uses dotted access, not `from` imports.** `RUN_TIMEOUT` and `N_RUNS` in `config.py` can be overridden by `--timeout`/`--runs` at runtime (`config.RUN_TIMEOUT = args.timeout`). Every reference to them elsewhere must be `config.RUN_TIMEOUT`/`config.N_RUNS` (dotted attribute lookup) — never `from config import RUN_TIMEOUT`, which binds a stale copy at import time and silently ignores the override.
- **Logging goes through `Shared.log/ok/warn/err/section`**, not bare `print()`, for consistent colored CLI output across all workload modules.
- **`VERSION` in `config.py` and the `# Local AI Bench vX.Y` title in `README.md` must be bumped together.** They're two independent strings with no code linking them — nothing will catch a mismatch except noticing it.
- **`N_RUNS` is fixed at 3, averaged directly, no outlier-dropping or early-exit-on-slow logic for the run count itself.** This was a deliberate simplification from an earlier 5-run design with more complex gating — variance across 5 runs was already low enough that 3 plain averaged runs is sufficient. Don't reintroduce run-count configurability or outlier-dropping without the user asking for it again. (Note: `SLOW_MODEL_MIN_TPS` skip/early-exit logic for the *conversation* test is separate and current — see `docs/workloads.md`.)
- **Crash caches** (`.llm_crash_cache.json`, `.conv_crash_cache.json`, `.embed_crash_cache.json`, gitignored) memoize deterministic Ollama-runner crashes per model/test so repeat invocations don't rediscover the same crash. If you touch this pattern, keep it symmetric across the three workload modules that use it.
- **No comments explaining *what* code does** — names should do that. Comments are reserved for non-obvious *why* (a constraint, a workaround, a subtle invariant) — this codebase already leans heavily on that style; match it.
- Don't add features, config flags, or abstractions beyond what's asked. This project has already had flags/gating logic deliberately *removed* for being more complexity than the variance they controlled for was worth (see `N_RUNS` above) — err toward the simpler version.

## Design history worth knowing

- **`compare.py` was intentionally removed**, replaced by the dashboard's multi-file comparison. Don't recreate it or treat its absence as a regression.
- **`run_linux_mac.sh`/`run_windows.bat` → `run_bench.sh`/`run_bench.bat`**, and **`launch_dashboard.py` → `dashboard.sh`/`dashboard.bat`** (a Python `http.server`-based dashboard launcher replaced by shell scripts that shell out to `vite preview`). If you see references to the old names anywhere (docs, comments, scripts), they're stale — fix them.
- **`scripts/` used to be a single flat `benchmark.py`** (2200+ lines) before a "big refactor to make it maintainable" split it into the current module layout. If old context or a stale doc references top-level `config.py`/`models.py`/`setup_check.py` (not under `scripts/`), that's pre-refactor and wrong now.
- **Interactive setup UX is deliberately plain-`input()`, numbered-list, not arrow-key/raw-terminal.** An arrow-key checkbox menu was tried and rejected after repeated bugs (stray keystrokes leaking between prompts, a sub-installer's own confirmation prompt swallowing a keypress). The current pattern: one approval prompt for prerequisites up front, numbered-list toggle selection (`2 4 7-9`, tier keys, `a` for all, `q` to cancel), then fully unattended install with zero further prompts. Preserve this if touching `setup_check.py`.
- **Chart card headers in the dashboard**: the model/system "eyebrow" label above a chart title must be the visually dominant text — larger and bolder than the chart title itself, not the reverse — since it's the first thing needed to identify what a card shows, especially once exported standalone as a PNG.

## Keeping docs and the dashboard in sync

Docs and the dashboard don't update themselves — treat them as required outputs of a behavior change, not optional cleanup. This applies **even if the task didn't ask you to touch docs/dashboard at all**: if your change to `scripts/` alters something a doc describes or something the dashboard consumes, updating it is part of finishing the task, not a separate favor.

**Changed the results JSON shape?** (a new field, a new `skip_reason`/status value, a new checkpoint/context-depth label, a renamed key) → the dashboard doesn't get this for free. Update `dashboard/src/constants.js` and `dashboard/src/utils.js` (see the `CTX_ORDER` pattern above) so the new shape actually renders instead of silently showing blank/wrong cells. This is exactly how the conversation test's slow-exit and `compare.py`'s removal were handled correctly — both were done in the same change as the behavior change, not after.

**Changed a CLI flag, a default, a tier definition, a model list, or test behavior?** → update the doc that describes it:
- Flags/defaults → `docs/cli-reference.md`
- What's tested, model tiers/lists, workload behavior → `docs/workloads.md`
- Execution order, algorithm behavior, code organization → `docs/how-it-works.md`
- New/renamed/removed files → `docs/project-structure.md`
- New/changed test files or testing approach → `docs/testing.md`

**Removed or replaced something with real user-facing functionality** (like `compare.py`) → don't just delete it silently. Add a short, honest note explaining what replaced it and why, in the most relevant doc — not a long migration guide, just enough that nobody mistakes the absence for an accident. This project has been burned by exactly this once already (see Design history above).

This project's docs have accumulated real staleness before — broken relative links, a hallucinated test count, stale script names, an undocumented functionality removal — all found only by an explicit audit after the fact, not caught as changes landed. Don't rely on a later audit; keep it in sync as you go.

## Before considering a change done

1. Run `bash tests.sh` — must pass.
2. If you touched anything in `scripts/`, make sure new logic has real unit test coverage per the conventions above (extract-then-test, not left inside an untested orchestration loop).
3. Ask explicitly: does this change alter the results JSON shape, a CLI flag, a default, model/tier definitions, or documented behavior? If yes, update the dashboard and/or the relevant doc(s) per "Keeping docs and the dashboard in sync" above — don't wait to be asked.
4. Check for now-broken relative links and stale references to renamed/removed files in anything you touched or that references what you touched.
5. Don't run `setup.sh`, `setup.bat`, or a real `run_bench.sh` invocation to "verify" — see Critical safety rules.
