[← Back to README](../README.md)

# CLI Reference

**Contents**
- [Launch modes](#launch-modes)
- [Flag details](#flag-details)
- [Examples](#examples)
- [Comparing results](#comparing-results)

```
run_bench.sh [options]  # Linux / macOS
run_bench.bat [options]   # Windows

--tests TESTS           Tests to run (default: all nine — see Flag details below)
--engine ENGINE         Inference engine to benchmark, or 'all' (default: llamacpp)
--cpu-only              Force CPU-only inference for every engine-backed test
--gpu-split-mode MODE   llama.cpp GPU mode: single, layer, or tensor (default: layer)
--llamacpp-no-repack    Disable llama.cpp weight repacking (`-nr`; default: false)
--retry-crashed-models  Ignore prior workload crash-cache skips for this run
--warmup N              Engine-backed warmups before measuring (default: 2)
--runs N                Measured runs to average (default: 3, range: 1-10)
--timeout N             Seconds per generation/chat call and engine warmup (default: 300)
--acc-timeout N         Seconds per accuracy question before giving up on it (default: 60)
--acc-token-budget N    Completion-token budget per accuracy question (default: 8192)
--maxtier TIER          Cap LLM and image models at this tier and below (default: large, no cap)
--max-prompt-tokens N   Cap the deepest prompt-processing size for llm/conv/llamabench/llamabenchconc/vllmbench (default: no cap)
--tg-tokens N [N ...]   Generation sizes for llamabench/llamabenchconc to sweep — from 128/512/1024 (default: 128 512)
--llm-models TAGS       Only test these LLM models — tags or wildcards (--models alias)
--embedding-models TAGS Only test these embedding models — tags or wildcards
--image-models SHORTS   Only test these image models — short IDs or wildcards
--list-models           List installed models, then exit
--sample N              Dev-only: run accuracy tests against an N-question subset per bank
--comfyui /path         Path to ComfyUI directory (default: ./ComfyUI)
--out filename.json     Output file (default: results/results_<hostname>_<timestamp>.json)
--force-all             Ignore slow-model cutoffs; doesn't override real failures
```

See [Flag details](#flag-details) below for what each flag actually does.

## Launch modes

Running `run_bench.sh` or `run_bench.bat` with no arguments automatically opens the graphical launcher on a usable local desktop and the terminal launcher over SSH or when no display is available. `--ui auto|gui|terminal|none` controls this explicitly; the older `--interface` spelling remains a compatibility alias. `none` requires benchmark arguments and runs noninteractively, for example `bash run_bench.sh --ui none --tests llm`. The graphical launcher is one scrollable configuration screen. Selecting a named preset applies its complete recommended configuration immediately; changing any test, model, engine, workload size, execution setting, or path changes the dropdown to Custom without disabling controls. The configuration keeps its paired sections in two columns at every supported window size; per-item Reset buttons align to the right edge while the flexible space between each label and button contracts as the window narrows. **Start Benchmark** opens a scrollable, keyboard-accessible review organized into selection, measurement settings, scope/duration, and output/environment sections before anything launches. Configuration, Run Log, and Result History repaint as soon as they are selected, including on macOS where Tk can otherwise defer drawing until another pointer or keyboard event. The GUI supervises `benchmark.py` with unbuffered child output so results and diagnostics stream into the Run Log tab while the run is active, and sends a normal interrupt when Stop is pressed so checkpointed results are retained.

The graphical launcher includes a read-only inventory and preflight section reporting the operating system, architecture, RAM, detected acceleration backend, llama.cpp tool availability, installed model counts and storage, free storage, memory-fit context, ComfyUI path, and concrete blockers. **Import Hugging Face Model** beside the installed-model list accepts an `owner/repo` identifier or full Hugging Face URL, and offers complete GGUF variants for llama.cpp or a `config.json` plus safetensors snapshot for a locally managed vLLM. Unsupported repositories leave import disabled; structurally valid imports require explicit acknowledgement that runtime compatibility remains unverified. Discovery does not install, download, start, or reconfigure anything and does not replace Setup when a required runtime or model is missing.

Engine Management exposes **Change…** beside the version of an app-managed llama.cpp runtime. It offers the ten most recent published official builds and a specific-tag field accepting `bNNNNN` or `NNNNN`; the selected upgrade or downgrade uses the same platform-specific staged update and rollback validation as the latest-version action. On macOS, the main update action migrates a detected Homebrew runtime to an app-managed official archive without removing the formula; **Change…** becomes available after that migration. Other externally managed runtimes remain inspection-only for version selection.

Supported app-managed stable-wheel vLLM environments also expose **Change…** with the ten newest non-yanked stable PyPI versions and a specific stable-version field. Exact-version installation retains staged validation and rollback. Experimental nightly environments, including DGX Spark's CUDA 13 path, omit the control because their channel cannot promise the same stable-version availability.

While a GUI-launched benchmark runs, a compact always-on-top progress window lists every selected workload in execution order and every selected model beneath its applicable workloads. Workloads and models update live as queued, running, complete, failed, interrupted, or not run. Its resource table adds a live used/total VRAM line on discrete GPUs with dedicated memory; integrated and unified-memory GPUs omit that line and use the system-RAM reading instead. The Run Log remains the detailed output and the benchmark process remains the source of truth. Normal terminal and noninteractive runs do not emit or open this progress interface.

The terminal launcher reads the selected engine and resolved ComfyUI installation, then shows its test checklist. Only installed models are shown. The test screen accepts numbers and ranges plus visible group shortcuts: `a` selects every available test, while `l`, `x`, `c`, `e`, and `i` toggle the LLM, accuracy, concurrency, embedding, and image test groups. Tests appear in the order single-shot LLM, conversation, llama-bench, vllm bench, embeddings, accuracy, concurrency, images. One menu toggle can cover more than one CLI test: **llama-bench** selects `llamabench` and `llamabenchconc` together, since they are two halves of the same native llama.cpp benchmarking. Both remain separately selectable through `--tests` on the CLI. Both frontends translate confirmed selections into the same public flags and launch the non-interactive `scripts/app/benchmark.py` CLI. Each selects one engine at a time; `--engine all` remains available through the direct CLI because inventories can differ between engines.

Passing any benchmark argument without `--ui` bypasses the launcher and forwards every argument directly to `benchmark.py`, preserving existing automation and direct CLI defaults. This includes `--help` and `--list-models`. Calling `python -m scripts.app.benchmark ...` directly is also always non-interactive.

After confirmation, the launcher saves the selected preset name plus the selected engine, tests, model IDs, `--max-prompt-tokens` cap, `--tg-tokens` selection, execution settings, and paths to the gitignored `.benchmark_frontend_state.json` in the project root. The next GUI launch restores Custom exactly, including deliberately empty model families and workload-size selections not used by the most recent run, or reapplies the remembered named preset; the terminal launcher continues to restore the compatible selections and ignores the GUI-only preset label. Delete `.benchmark_frontend_state.json` to reset the launcher to Consumer guidance and current hardware-aware defaults. A missing, malformed, or incompatible file uses current defaults, and stale entries never make an uninstalled model appear or block the interface.

For direct desktop launching, use `Run Local AI Bench.command` on macOS, `Run Local AI Bench.desktop` on Linux, or `Run Local AI Bench.bat` on Windows. The macOS launchers do not automate Terminal; after the command exits, the window follows the user's Terminal profile settings. Linux desktops may require marking the `.desktop` file trusted or allowing launching after extraction.

On Windows, double-clicking `run_bench.bat` uses a best-effort Explorer-launch check to pause after completion so the final status remains visible. Launches from an existing command prompt exit normally. The pause affects presentation only; the batch file saves and returns the benchmark's original exit code.

On first use or when no saved selection applies, the interactive launcher's state is:

| Area | Initial state |
|---|---|
| Engine | The CLI's default registered engine (currently `llamacpp`); selectable first if multiple engines exist |
| ComfyUI directory | The setup-managed `./ComfyUI` directory; there is no extra path prompt |
| Single-shot LLM and conversation | Checked when an installed catalog or custom LLM is available |
| Embeddings | Checked when an installed embedding model is available |
| Image generation | Checked when an installed catalog image checkpoint is available |
| MCQ, math, reasoning, code, and tool accuracy | Unchecked |
| Tool and chat concurrency | Unchecked |
| Installed xsmall/small/medium catalog LLMs | Checked |
| Installed large catalog LLMs | Unchecked |
| Installed custom LLMs | Unchecked; displayed by folder name and excluded from tier toggles |
| Installed embedding models | Checked; individually toggleable or grouped with `emb` |
| Installed xsmall/small/medium image models | Checked |
| Installed large-tier image models | Unchecked, including Flux.1-dev and Flux.2-dev |
| Uninstalled models | Not displayed |

The model screen uses one LLM selection for single-shot, conversation, accuracy, and concurrency tests. Number/range controls toggle individual models. `xs`, `s`, `m`, and `l` toggle installed catalog LLM and image models in that tier together: an all-selected tier becomes unselected, while a partially selected or unselected tier becomes fully selected. `custom` and `emb` independently toggle those groups. If catalog models are missing, a read-only hint reports counts by family and suggests `bash setup.sh` or `setup.bat`; it never runs setup.

After the selection summary, `Start this benchmark? [Y/n]` defaults to yes; press Enter to launch or enter `n` to cancel.

### Frontend option classification

| Classification | Options | Frontend treatment |
|---|---|---|
| Guided | `--tests`, `--engine`, `--llm-models`, `--embedding-models`, `--image-models`, `--max-prompt-tokens`, `--tg-tokens`, `--maxtier`, `--models` | Primary workload/model controls; tier and legacy model aliases are represented by the more precise installed-model checklist |
| Advanced | `--warmup`, `--runs`, `--timeout`, `--acc-timeout`, `--acc-token-budget`, `--cpu-only`, `--gpu-split-mode`, `--llamacpp-no-repack`, `--force-all`, `--retry-crashed-models`, `--offline`, `--memory-telemetry` / `--no-memory-telemetry`, `--out`, `--comfyui` | Exposed by the advanced-settings toggle because they alter execution cost, failure handling, runtime mode, privacy, or output location |
| Contextual | `--list-models` | Represented by the installed-model inventory already visible in the frontend; direct CLI retains the printable inventory command |
| Developer-only | `--sample` | Intentionally excluded because sampled accuracy results are non-comparable and intended only for development iteration |

No current public option is classified as unsafe, unsupported, or missing from the frontend inventory. An automated parser test fails if a public `benchmark.py` option is added, removed, or left unclassified.

The interactive launcher clears the terminal before its initial display, between menu screens, and before subsequent redraws while preserving the welcome banner through the first single-engine test screen and the final model choices through confirmation. It uses the native `cls` command on Windows and ANSI terminal clearing elsewhere. Launcher prompts remain untimestamped. Once execution starts, benchmark status and progress messages are prefixed with local time as `[HH:MM:SS]`. ANSI colors are limited to interactive terminals; GUI output, redirected output, and saved logs are plain UTF-8, including on Windows. Model responses, results data, answer sidecars, and generated artifacts are unchanged.

`--runs` applies only to single-shot LLM, embeddings, image generation, and llama-bench repetitions. Native llama-bench receives the configured repetition count inside its per-model streamed prefill and decode sweeps so the model remains loaded; every completed case row is checkpointed as it arrives. Conversation and each accuracy test make one measured pass, while concurrency records one measured batch per level.

## Flag details

### Interactive frontend option coverage

The launcher maintains an executable inventory of every public `benchmark.py` flag. Tests fail when a CLI option is added or removed without updating that inventory. Every option classified as `exposed` must also map to a unique concrete control identifier, including the bespoke test, engine, model-family, prompt-cap, and generation-size selectors; a declared `missing` option or an exposed option without that binding blocks release readiness. Tier selection and `--list-models` have more precise equivalents in the installed-model selection screens; `--models` is only an alias; and developer-only `--sample` and the internal fork guard are intentionally excluded.

The graphical frontend exposes warmup count, measured runs, generation and accuracy timeouts, accuracy token budget, CPU-only, multi-GPU mode, llama.cpp weight-repacking control, force-all, prior-crash retry, and offline modes, output path, and ComfyUI path behind the advanced-settings toggle. Each is validated before launch and included in the resolved plan review; the executable inventory and control-binding tests prevent a future safe public setting from disappearing silently. During a run, the progress window separates process RSS from system RAM and adds best-effort GPU utilization and NVIDIA per-process GPU memory without blocking the interface. Apple Silicon uses non-privileged AGX `ioreg` statistics, NVIDIA uses `nvidia-smi`, AMD utilization uses `rocm-smi`, and unavailable tooling is reported without affecting the benchmark.

CLI and graphical numeric constraints, choice lists, defaults, option classifications, and UI coverage policy share one typed option schema. Warmups must be zero or greater; measured runs remain 1–10; timeouts, token budgets, prompt caps, and developer sample sizes must be positive.

Reset actions exist for each individual control, each configuration section, and the entire plan. Any reset makes the configuration Custom and restores current documented defaults in memory only; saved settings are not replaced until a benchmark is reviewed and confirmed.

Alongside the use-case presets (Consumer guidance, Vendor validation, Neutral comparison, Platform optimized, Offline / private, Quick run, Full run), the dropdown offers role presets that select the tests that matter for a model's intended job: **Role: Orchestrator** (uncapped single-shot/conversation depth plus reasoning, tool, and chat concurrency), **Role: Agent / tool caller** (tool and code accuracy plus tool concurrency at a 32K cap), **Role: Coding assistant** (code and reasoning accuracy at a 32K cap), **Role: Chat assistant** (MCQ and reasoning accuracy plus chat concurrency at an 8K cap), and **Role: RAG / retrieval** (embeddings and MCQ accuracy at a 32K cap). Tests unavailable on the current machine are filtered out the same way as for any other preset.

Custom configurations can be exported as versioned portable preset JSON, imported as the current Custom configuration, or compared with the current screen. Built-in presets apply directly from the dropdown without an Apply button. Portable presets contain tests, model identifiers, prompt/generation sizes, and measurement-affecting execution settings; output and ComfyUI paths are deliberately excluded because they are machine-private. Presets never contain credentials, Hugging Face tokens, prompts, responses, or results.

No preset carries an inference engine. Presets describe *what* to run, and the engine checkboxes decide *where* it runs, so applying a built-in preset, importing a portable preset, or opening a project all leave the engine selection on screen untouched — previously any of these silently reset a multi-engine selection back to a single engine. Preset schema version 2 drops the field; version 1 files still import, with their recorded engine ignored.

The GUI can also import the `run.plan` embedded by a command-line benchmark result, or a standalone copy of that plan, and render its supported engine, tests, exact resolved models, caps, and execution settings as Custom. Machine-local output and ComfyUI paths remain unchanged. A CLI plan containing a developer-only sampled accuracy run, or referencing an engine, test, or model unavailable on the current machine, is rejected explicitly because the GUI cannot preserve that value.

The configuration screen can create and open versioned `.labproject` files for hardware comparison, model selection, acceptance validation, capacity planning, and regression workflows. Opening a project makes the displayed configuration Custom. A project binds the portable configuration to an optional local baseline result and embedded acceptance policy while retaining machine-local output and ComfyUI paths. See [Benchmark Projects](projects.md).

During a GUI-launched benchmark, the separate progress window reports the current stage and model queue in a mouse-wheel/trackpad-scrollable list while keeping estimated remaining time and the summary/resource tables fixed. The run-summary table shows finished models, usable coverage, implausible-rate retries, and invalid drops. The resource table separates benchmark-process RSS from system RAM used/total and its change since launch; GPU utilization and NVIDIA per-process GPU memory are added when supported. On unified-memory systems such as DGX Spark, system RAM and its delta are the primary memory reading because CPU and GPU allocations share the same pool and may overlap in process/GPU accounting. The Run Log remains the authoritative detailed stream, and checkpointed JSON remains the durable source after interruption.

The **Result History** tab scans local result JSON files and supports search, status and engine filters, multi-select launch into the dashboard, acceptance-policy evaluation, two-file diagnostic export, read-only recovery inspection, confirmed resume, selected context/level retry, and reviewed full-plan fork into a new result. In-place resume requires a plan made entirely of journal-owned LLM, conversation, llama-bench, or HTTP concurrency stages; selected retry applies to LLM, conversation, and HTTP concurrency cases, while native llama-bench resumes its remaining sweep. Any saved workload mix can be forked as a new run: legacy stages use the normal benchmark execution path with an internal exact-plan guard and source provenance. It does not create a second result database or sync results. See [Local Result History](result-history.md).

The Run Log provides **Pause**, **Resume**, and **Stop Benchmark** while any GUI-launched benchmark/recovery process is active, plus **Export Log** for saving the complete visible log at any time. Console timestamps use local time; under WSL the GUI reads the Windows host's current UTC offset when it launches the process instead of displaying WSL's commonly configured UTC clock. Stored result and pause metadata remain canonical UTC. A successful benchmark, recovery, retry, or fork automatically writes `log_<result suffix>.txt` beside each completed result, using the same timestamp-bearing suffix as its JSON and replacing a manual export at that canonical path. Failed and interrupted processes do not automatically replace an exported log. Pause is cooperative: the current measured operation finishes and checkpoints, then execution waits before the next safe request, conversation turn, image run, concurrency level, or native command. The loaded runtime may continue using memory while paused. Timestamped pause requests and resumes are retained in schema-4 result metadata because a long pause can change cache and thermal conditions between samples. Stop works from either running or paused state and retains the normal interruption/checkpoint guarantees.

After selecting a baseline and candidate, **Export Diagnostic** creates a reviewed `.labdiag` containing the first divergence, both environments/plans, relevant raw evidence and invalidity, source digests, and reproduction steps. The CLI equivalents are `python -m scripts.results.vendor_diagnostic_cli create BASELINE CANDIDATE OUTPUT --reviewed-metadata` and `verify OUTPUT BASELINE CANDIDATE`. See [Vendor Diagnostics](vendor-diagnostics.md).

Portable result bundles are available through the GUI's **Export Bundle** and **Import / Verify** actions or `python -m scripts.results.result_bundle_cli export RESULT BUNDLE --reviewed-metadata`, `verify BUNDLE [--source-result ORIGINAL_RESULT]`, and `import BUNDLE RESULT`. Export first previews identity metadata and writes nothing until the GUI approval or CLI acknowledgement is supplied; `--system-alias NAME` and `--hardware-alias NAME` replace exported private names. Export is deterministic, while import verifies file digests, supported embedded plan schema, locally available question-bank versions, and reproducible sample aggregates first. Supplying the private source during verification also checks the exported source-identity digest. Optional CLI `--artifact PATH` values are stored by content digest, while `--artifact-dir DIR` extracts verified artifacts under those safe digest names.

Deterministic decision reports are generated through the GUI's **Create Report** action or with `python -m scripts.results.decision_report_cli RESULT --html REPORT.html --pdf REPORT.pdf --reviewed-metadata [--policy POLICY.json]`; either CLI output may be omitted, but at least one is required. The same preview and alias flags protect private identity metadata. Reports are self-contained evidence summaries and do not calculate a hidden composite score. An optional policy adds explicit per-rule acceptance evidence. See [Decision Reports](reports.md) and [Outbound Metadata Review](outbound-review.md).

Explicit evidence thresholds can be evaluated with `python -m scripts.results.acceptance_policy_cli RESULT POLICY`. Exit code `0` means every named rule passed, `2` means evaluation completed with a rejection, and `1` means the input was invalid or unreadable. See [Acceptance Policies](acceptance-policies.md).

The GUI's **Support Bundle** action creates a separate redacted `.labsupport` archive. Before choosing the destination, a scrollable review lists every included file and field. The allowlist excludes raw results, logs, hostname, model identity, measurements, prompts, responses, tokens, credentials, and private paths; the archive contains only runtime/system compatibility facts, stage coverage, and scrubbed structured diagnostics.

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--quick` | flag | off | CLI-only smoke-test preset using the smallest xsmall LLM, the 512/2K single-shot checkpoints, one measured run, and no warmups. It is intentionally narrower than the GUI's Quick run use-case preset and overrides workload/model/depth/run-count selectors while preserving runtime, engine, and output-path options |
| `--dry-run` | flag | off | Resolves and prints every engine pass, model, workload, checkpoint sweep, run count, and an estimated duration, then exits after hardware profiling but before result creation or runtime startup. The ETA is the median of completed local results with an exact hardware, engine, workload, model, and runtime-shaping configuration match; when no exact history exists it is reported as unavailable |
| `--tests` | any of `llm conv emb img mcq math reasoning code tool`, plus `acc`, `conc_tool`, `conc_chat`, `conc`, `llamabench`, `llamabenchconc`, and `vllmbench` | all nine (`llm conv emb img mcq math reasoning code tool`) | Space-separated list; order doesn't matter. `acc` expands to every accuracy-style test (`mcq`, `math`, `reasoning`, `code`, and `tool`) and de-duplicates against any of them also listed explicitly; `conc` expands the same way to `conc_tool conc_chat`. `conc_tool` (agentic/tool-calling fan-out, 1–16-way) and `conc_chat` (many simultaneous chat users, 1–32-way) — see [Concurrency](workloads.md#concurrency) — are opt-in, not part of the default set. `llamabench` is also opt-in and runs separate native prefill and depth-aware decode sweeps, scoped like `llm`/`conv` — see [llama-bench](workloads.md#llama-bench). `llamabenchconc` is opt-in too — llama.cpp's own `llama-batched-bench` decode-throughput-vs-concurrency sweep, scoped the same way — see [llama-bench Concurrency](workloads.md#llama-bench-concurrency). `vllmbench` is the vLLM counterpart to `llamabench`, running `vllm bench latency`/`throughput`; it is skipped on a llama.cpp run and its numbers are never charted against `llamabench`'s — see [vllm bench](workloads.md#vllm-bench) |
| `--engine` | any registered engine name, or `all` | `llamacpp` | Which inference engine to benchmark against. `all` runs the full `--tests` suite once per registered engine (sorted order) and writes a separate results file for each (engine name appended to the filename). Only llama.cpp is registered today, so `all` behaves identically to the default until a second engine (e.g. MLX) is added. See [Engines](engines.md) |
| `--cpu-only` | (flag) | off | Restarts the engine with GPU devices hidden for every test that goes through it (`llm`/`conv`/`mcq`/`math`/`reasoning`/`code`/`tool`/`emb`/`conc_tool`/`conc_chat`), then restores normal GPU mode afterward — useful on GPU backends unstable under one of those workloads. `llamabench` and `llamabenchconc` also honor it (passing `-ngl 0` straight to `llama-bench`/`llama-batched-bench`) without going through that engine restart |
| `--gpu-split-mode` | `single` / `layer` / `tensor` | `layer` | Selects llama.cpp GPU execution for server-backed and native llama.cpp tests. `single` uses only the runtime's primary GPU. `layer` is the compatible multi-GPU pipeline/layer split and retains q8 KV cache plus automatic fit. `tensor` is llama.cpp's experimental tensor-parallel mode; it forces full GPU offload and f16 KV cache. The GUI offers `single` and `tensor` only when setup recorded at least two GPUs and a matching CUDA or ROCm/HIP runtime. Vulkan and unrecorded/single-GPU configurations retain `layer`; a supported topology still requires a supported model architecture |
| `--llamacpp-no-repack` | flag | off | Passes `--no-repack` (`-nr`) to `llama-server` for llama.cpp generation, conversation, accuracy, embedding, and HTTP-concurrency workloads, and to `llama-batched-bench` for llama-bench concurrency. `llama-bench` does not support this option, so the standard native llama-bench workload is unchanged. It has no effect on vLLM or ComfyUI. Disabling repacking can reduce model startup time and peak loading memory, but may reduce CPU or partially offloaded throughput. The resolved methodology records the setting for A/B comparison |
| `--offline` | (flag) | off | Allows loopback/local sockets, blocks non-loopback Python connections, and propagates offline/telemetry-disabled environment settings to managed runtimes; requires all runtimes and models to be installed already. See [Offline Mode](offline-mode.md) |
| `--memory-telemetry` / `--no-memory-telemetry` | (flag) | on | Enables or disables qualified 0.5-second schema-5 memory sampling. The sampler records idle, model-load, and measured windows plus normalized provenance; telemetry-on and telemetry-off performance is comparable when every other methodology setting matches |
| `--warmup` | integer | `2` | Discarded warmups before measurement for every engine-backed workload: once per loaded model for LLM/conversation/accuracy, per model call for embeddings, and per concurrency level. Image generation always performs one warmup at the model's first resolution and does not use this flag. The flag does not control either native llama.cpp benchmark; `llama-bench` performs its own built-in warmup for each case, while `llama-batched-bench` has no corresponding option |
| `--runs` | integer, `1`–`10` | `3` | Measured runs averaged for single-shot LLM at each context, embeddings, and images at each resolution; also isolated `llama-bench -r 1` repetitions aggregated per case. Ignored by conversation, accuracy, concurrency, and `llamabenchconc`. Warmup count is unaffected |
| `--timeout` | integer (seconds) | `300` | Per generation/chat call for single-shot, conversation, concurrency, and their engine warmups. Managed vLLM model startup uses a separate bounded window covering CPU-offload calibration. Images use twice this value (600s by default). Embedding calls retain the engine's fixed 120s timeout; accuracy questions use `--acc-timeout` |
| `--acc-timeout` | integer (seconds) | `60` | Per question for `mcq`/`math`/`reasoning`/`code`/`tool`; the partial response is scored normally, the timeout is recorded, and the bank continues — see [Accuracy](workloads.md#accuracy) |
| `--acc-token-budget` | positive integer (tokens) | `8192` | Total completion-token budget per accuracy question. The first pass receives 60%; only a literal length stop triggers a second pass with the remaining 40%. Both passes share `--acc-timeout` |
| `--maxtier` | `xsmall` / `small` / `medium` / `large` | `large` (no cap) | Cumulative — each tier includes every tier below it. Applies to `llamabench` and `llamabenchconc` the same way it applies to `llm`/`conv`. `conc_tool`/`conc_chat` ignore this — they scope to every LLM model actually downloaded locally instead, since download presence is itself a decent proxy for "this machine can try it" (see [Concurrency](workloads.md#concurrency)) |
| `--max-prompt-tokens` | positive integer (tokens) | none (no cap) | Caps the deepest prompt-processing size swept by whichever of `llm`, `conv`, `llamabench`, `llamabenchconc`, and `vllmbench` are selected in `--tests`. Drops entries above the cap from `llm`'s and `llamabench`'s depth lists, caps conversation's growth target and checkpoints, and clamps `llamabenchconc`'s fixed prompt depth. Errors out if the cap excludes every depth from `llm` or `llamabench`; conversation retains its opening `0K` checkpoint below its first nonzero checkpoint. The interactive launcher prompts for this once when any affected test is selected, as a numbered menu of the configured depths (512 up to 98304) plus a "no cap" option |
| `--tg-tokens` | space-separated subset of `128 512 1024` | `128 512` | Which generation sizes `llamabench` and `llamabenchconc` sweep at each prompt depth. Only affects those two tests. The interactive launcher shows this as a toggle checklist, only when one of those two tests is selected |
| `--llm-models` (`--models` alias) | space-separated tags and/or wildcards (e.g. `llama*`) | none (every catalog model in the selected tier) | Affects `llm`/`conv`/`mcq`/`math`/`reasoning`/`code`/`tool`/`conc_tool`/`conc_chat`/`llamabench`/`llamabenchconc` tests. Matching is case-sensitive and exact-or-wildcard (`fnmatch`-style: `*`/`?`/`[...]`), not substring. Applied after `--maxtier` (or, for concurrency, after downloaded-model scoping), narrowing catalog entries while also unioning any matching installed custom tags. `--llm-models` is canonical; `--models` remains fully backward compatible. Quote wildcards (`"llama*"`) so the shell does not expand them |
| `--embedding-models` | space-separated catalog tags and/or wildcards | none (every catalog embedding model) | Affects `emb` only. Matching is case-sensitive and exact-or-wildcard on the model's `tag` |
| `--image-models` | space-separated catalog short IDs and/or wildcards (e.g. `sd*`) | none (every image model allowed by `--maxtier`) | Affects `img` only. Matching is case-sensitive and exact-or-wildcard on the stable `short` values in `models.py`; it narrows the image list after `--maxtier` |
| `--list-models` | (flag) | off | Read-only inventory of installed catalog LLMs, embeddings, custom LLM folders, and catalog image checkpoints, then exit. It does not require or start an inference server. `--engine` selects the inventory (`all` lists each engine); image inventory always comes from Local AI Bench's managed `models/comfyui/` directory |
| `--sample` | integer `N` | none (full bank) | Dev-only. Runs `mcq`/`math`/`reasoning`/`code`/`tool` against a deterministic N-question subset of each bank, selected round-robin across categories. Every category is represented only when N is at least that bank's category count. IDs are recorded under `sample_ids`; sampled and full-bank results are not comparable — see [bank versioning](workloads.md#bank-versioning) |
| `--comfyui` | path | saved/system installation, then `./ComfyUI` | ComfyUI program directory or Windows portable root. Resolution otherwise uses `COMFYUI_DIR`, the path saved by setup, conventional user locations, and finally the managed copy. Image models remain under `models/comfyui/` regardless of this path. The launcher only passes this flag when `img` is among the selected tests — a path is rejected if it holds no ComfyUI, and a run without image tests has nothing to point it at |
| `--out` | filename | `results/results_<hostname>_<timestamp>.json` | Overrides the main JSON path entirely. Accuracy answer sidecars and generated-image folders still go under the repository's `results/` directory, named from the main output's stem — see [Project Structure](project-structure.md) |
| `--force-all` | (flag) | off | Disables the slow-TPS exits for single-shot LLM, conversation, and chat concurrency. It does not bypass timeouts, crashes, missing data, or load failures |
| `--retry-crashed-models` | (flag) | off | Ignores workload crash-cache entries for this execution so every selected model is attempted again. It does not delete the cache, suppress a new crash, or bypass missing-model and load-failure handling |

An explicitly supplied selector that resolves to no models for a selected workload is a command error and exits before hardware profiling, result-file creation, or server orchestration. This applies consistently to `--llm-models`/`--models`, `--embedding-models`, and `--image-models`. Selectors for workload families absent from `--tests` are ignored for this validation. Omitted selectors retain the defaults above; selecting a catalog model that is not downloaded still reaches the workload's existing missing-model handling.

## Examples

```bash
# Open the interactive launcher (only installed models are shown)
bash run_bench.sh

# LLM only, quick check
bash run_bench.sh --tests llm

# Skip image generation
bash run_bench.sh --tests llm conv emb

# Conversation benchmark only
bash run_bench.sh --tests conv

# Accuracy tests only — MCQ, math, reasoning, code, and tool (also: --tests acc)
bash run_bench.sh --tests mcq math reasoning code tool

# Cap at small-tier models and below — skips medium/large LLMs and
# medium/large-tier image models (SD3.5 Large, Flux.1-dev, Flux.2-dev),
# leaving SD1.5 and SDXL for the image test
bash run_bench.sh --maxtier small

# Only the Llama models, every tier — wildcard matches every Llama tag
bash run_bench.sh --tests llm --llm-models "llama*"

# One specific model plus a wildcard group
bash run_bench.sh --tests llm --llm-models qwen3.5:4b-q4_K_M "nemotron-3*"

# One embedding model only
bash run_bench.sh --tests emb --embedding-models nomic-embed-text

# SD-family image checkpoints only, still subject to --maxtier
bash run_bench.sh --tests img --image-models "sd*"

# Find the exact tag for a model you've downloaded but isn't in the catalog
bash run_bench.sh --list-models

# Run every registered engine, one pass each — writes a results file per
# engine. Currently a no-op (only llama.cpp is registered)
bash run_bench.sh --engine all --tests llm mcq

# Quick dev iteration on the accuracy tests — 10 questions per bank instead
# of the full thing; never compare this against a full-bank result
bash run_bench.sh --tests acc --sample 10

# Give slow hardware more time per run
bash run_bench.sh --timeout 600

# Give a slower model more time per accuracy question before its partial answer is scored
bash run_bench.sh --tests acc --acc-timeout 120 --acc-token-budget 12288

# Both concurrency tests — 1-16-way tool-style + 1-32-way chat-server sweeps
# on every downloaded model
bash run_bench.sh --tests conc

# Chat-server concurrency test only
bash run_bench.sh --tests conc_chat

# llama-bench prefill and depth-aware decode sweeps, opt-in — see Workloads
bash run_bench.sh --tests llamabench

# llama-batched-bench throughput-vs-concurrency sweep, opt-in — see Workloads
bash run_bench.sh --tests llamabenchconc
```

A full run takes several hours, depending on your hardware and which options you select.

## Comparing results

Copy result files from all machines to one machine, then load them into the [dashboard](dashboard.md).

---

[← Workloads](workloads.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)
