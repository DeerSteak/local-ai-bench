[← Back to README](../README.md)

# Dashboard

**Contents**
- [Loading results](#loading-results)
- [Sections](#sections)
- [Chart Style and Group By](#chart-style-and-group-by)
- [What the charts mean](#what-the-charts-mean)
- [Stats table](#stats-table)
- [Multi-file comparison](#multi-file-comparison)
- [Repeated-trial artifacts](#repeated-trial-artifacts)
- [Recommendation artifacts](#recommendation-artifacts)
- [Exporting](#exporting)
- [Development](#development)

An interactive results explorer for visualising and exporting benchmark output.

```bash
# Linux / macOS
bash launch_dashboard.sh
bash launch_dashboard.sh --port 8080   # use a different port
bash launch_dashboard.sh --result results/first.json --result results/second.json

# Windows
launch_dashboard.bat
launch_dashboard.bat --port 8080       # use a different port
launch_dashboard.bat --result results\first.json --result results\second.json
```

Desktop users can instead double-click **Launch Local AI Bench Dashboard** with the platform suffix `.command` on macOS, `.desktop` on Linux, or `.bat` on Windows. The launcher keeps the terminal open while the local dashboard server is running; closing or interrupting that server ends the dashboard session.

Requires Node.js/npm. On first run, installs npm dependencies. Every run rebuilds the app, then starts a local server on port 3000 and opens the browser automatically.

## Loading results

Drag one or more `results_*.json` files onto the drop zone in the top-right corner, click to open a file picker, or pass one or more repeatable `--result` arguments to the launcher. The benchmark GUI's **Result History** tab uses the same launcher when **Open in Dashboard** is selected. Up to six files can be loaded at once. Launcher-selected files are copied temporarily into the local dashboard build; a normal server stop removes them and the next build clears anything left by a forcibly closed terminal. The browser is never given general filesystem access. Dropping a single file when fewer than six are loaded adds it to the current set; dropping multiple at once replaces all. Sample files for testing are in `samples/`. Files must contain strict JSON; an invalid file displays an import error below the drop zone rather than failing silently.

New results record whether the run completed, remained in progress, was interrupted, or failed. Incomplete files show a warning beside their machine metadata while all valid completed measurements remain available; older files without run metadata load without a warning. Multi-file comparisons also warn when one result used llama.cpp layer splitting and another used tensor parallelism; older files are treated as the historical layer default.

A repeated-trial artifact produced by `python -m scripts.results.trial_set_cli` can be loaded by itself through the same drop zone. It switches the dashboard to the trial-set audit view; trial artifacts cannot be mixed with ordinary result files in one load.

A recommendation artifact produced by `python -m scripts.results.recommendation_cli` can likewise be loaded by itself. The static dashboard renders the authoritative Python verdict and its constraints, evidence paths, and five visually distinct candidate groups: recommended, tied, other eligible, eliminated, and unevaluated. It does not evaluate constraints or calculate a browser-side score.

## Sections

| Section | Charts |
|---|---|
| LLM | Two charts per model — Tokens/sec and TTFT — across context lengths (512 / 2K / 8K / 32K / 64K), single-shot cold-prefill test |
| LLM Conversation | Same two charts per model, but from the multi-turn conversation test, across whichever of 0 / 2K / 4K / 8K / 16K / 32K / 48K / 64K / 80K / 96K its plan reached (capped by the model's real context ceiling) |
| Sustained Load | Opt-in — one aligned timeline per system/model with throughput on the left axis and available temperature and power overlays on separate right axes, plus retention, degradation class, suspected cause, onset, and ambient summary |
| Concurrency (Tool) | Three line charts per model — Per-Request Tokens/sec, Aggregate Tokens/sec, and TTFT — at 1 / 2 / 4 / 6 / 8 / 12 / 16 simultaneous short-context requests |
| Concurrency (Chat) | The same three charts at 1 / 2 / 4 / 8 / 16 / 24 / 32 simultaneous long-context requests. See [Concurrency](workloads.md#concurrency) for how the two workloads differ |
| Accuracy | A **Test** sub-picker for MCQ / Math / Reasoning / Code / Tool Use (mirrors `ACCURACY_TESTS` in `dashboard/src/constants.ts`). Per test: one Overall accuracy-per-model chart, one Accuracy-by-Category breakdown chart per model, and — when provided by the bank — an Accuracy-by-Difficulty chart. An Accuracy Incidents chart appears for timeouts, likely loops, or exhausted token budgets. See [Accuracy](workloads.md#accuracy) |
| Embeddings | Chunks per second embedding one real document in a single call |
| Images | One grouped bar chart per resolution — all image models side by side per host |
| llama-bench | Opt-in — two line charts per model: Decode Throughput across prefilled prompt depths, with one line per tg size and system; and Prompt Processing Throughput across pp sizes, with one line per system. See [Workloads](workloads.md#llama-bench) |
| llama-bench Concurrency | Opt-in — aggregate decode throughput from `llama-batched-bench`, charted across parallel sequence counts with one chart per model and tg size. See [Workloads](workloads.md#llama-bench-concurrency) |

The **Models** filter and **Machine** labels are shared between the LLM, LLM Conversation, Sustained Load, Concurrency, Accuracy, llama-bench, and llama-bench Concurrency sections, so switching between them keeps the same models/files selected.

## Chart Style and Group By

**Chart Style** (Bar / Line) applies to the LLM, LLM Conversation, Embeddings, and Images sections — it picks the chart type. Both native llama.cpp sections are line-only because their horizontal axes are ordered numeric sweeps. **Group By** (Model / System) applies to LLM, LLM Conversation, Embeddings, Images, and llama-bench, and flips which entity becomes the chart group; llama-bench Concurrency stays grouped by model. In by-model LLM bar charts, each system remains one chart category with native group spacing, while its context bars follow the numeric checkpoint order from `CTX_ORDER` rather than lexicographic label order. Group By → System omits models for which every loaded file contains only a `no_llm_data` placeholder, while retaining a model attempted by at least one system. It also reveals a **Model Sizes** toggle (Split by tier vs. Combined) for the LLM/LLM Conversation and llama-bench sections, since a single combined chart with every model tier at once is unreadable.

Both pills are hidden on **Accuracy**, HTTP **Concurrency**, and **llama-bench Concurrency**, while only Chart Style is hidden on **llama-bench**. Accuracy charts are always bar charts grouped by model. The concurrency and native llama.cpp charts are always line charts, with concurrency level or pp depth on the horizontal axis. llama-bench's pp-size ordering is derived from each entry's own `n_prompt` or `n_depth` rather than a fixed list like `CTX_ORDER`, since the sweep is a `config.py` constant that can change across versions; pp sizes are binary-K depths and render as plain "2K"/"32K" labels rather than raw token counts.

Sustained Load is also line-only and hides Group By because each card is already one system/model soak. Baseline-percent mode is unavailable for this section: transforming timestamps and sensor values into percentages would destroy the physical alignment and units. Missing temperature or power simply omits that line; throughput and retention remain visible.

A checkpoint past a slow-model cutoff (see [Concurrency](workloads.md#concurrency) and the LLM Conversation early-exit above) renders as a "Skipped (X Too Slow)" label instead of a bar, driven entirely by each checkpoint's position in `CTX_ORDER` — this is why the dashboard already handles the cutoff firing at any depth, not just the first one, with no special-casing per depth. Adding a new checkpoint to the suite's own checkpoint list only requires adding it to `CTX_ORDER` too; the rest follows automatically.

The same position lookup drives the `crashed`, `timed_out`, and `slow_tps` cascades: the named checkpoint gets its own label and every deeper one is marked skipped, while shallower checkpoints keep the data they actually measured. A results file may name a checkpoint this dashboard build doesn't know — a depth from a newer or older schema — and an unrecognized name cascades nothing rather than being treated as shallower than every known checkpoint, which would otherwise mark every measured checkpoint as skipped. Image resolutions follow the identical rule against `RES_ORDER`.

## What the charts mean

**LLM → Tokens/sec.** Decode throughput (tokens generated per second) for the single-shot test, at each context length. Higher is better. This is generation speed *after* the prompt has already been processed — it answers "once the model starts responding, how fast does text come out?"

**LLM → TTFT.** Time to process the single-shot prompt before the first token comes back — a genuine cold prefill, since request-level cache bypass forces the stable prompt to be fully processed on every run. Lower is better. This answers "if I paste a large document and hit send, how long do I wait before anything happens?" TTFT rises sharply with context length here, since the model has to run every one of those tokens through the network with nothing cached.

**LLM Conversation → Tokens/sec.** The same decode-throughput metric, but measured mid-conversation instead of after a single cold prompt. Generally close to the single-shot number for the same model — decode speed doesn't depend much on how the context got filled.

**LLM Conversation → TTFT.** Time to process just the *next* turn in an already-long conversation, relying on the backend's KV-cache reuse (llama.cpp's slot cache) so only the new turn's tokens need to be run through the network, not the entire history again. This is **why conversation TTFT at, say, 32K is typically a small fraction of single-shot TTFT at 32K** — they're not measuring the same thing. Single-shot TTFT is "cold start with a huge prompt"; conversation TTFT is "one more message in a chat that's already this long." Both are real workloads; which one matters more depends on whether your use case looks like one-shot document Q&A or an ongoing chat/agent session.

**Accuracy → Overall.** Accuracy percentage per model on the selected test's full question bank, one bar chart with systems on the axis and one colored bar per model. Higher is better.

**Accuracy → Accuracy by Category.** The same test's questions broken down by category (e.g. arithmetic, logic, geometry — see [Accuracy](workloads.md#accuracy) for the full list per test), one chart per model. With a single file loaded, each category bar gets its own color from a fixed palette (and no legend, since there's only one system on the chart) purely to make individual bars easier to tell apart at a glance — the colors don't carry cross-chart meaning the way file/model colors do elsewhere.

All model, file, category, context, image, embedding, and fallback data colors meet the WCAG AA 4.5:1 contrast threshold against the dashboard's white chart background. Backend badge foreground colors meet the same threshold; their pale backgrounds and borders are decorative surfaces rather than foreground data marks.

**Accuracy → Accuracy Incidents.** Per model, how many questions reached `--acc-timeout`, were stopped as likely loops, or exhausted the second-pass token allowance. Lower is better. A successful final-answer nudge alone does not show this chart. The raw table adds **Nudged** and **Budget Exhausted** columns. When loaded files have different or unknown `accuracy_settings`, one warning calls out that their accuracy limits may not be comparable.

**Concurrency → Per-Request Tokens/sec.** Decode throughput for one individual request within a batch of N simultaneous requests. Higher is better. Typically drops as concurrency climbs, since requests share the same compute/memory bandwidth — this is the number that shows per-user latency degrading under load.

**Concurrency → Aggregate Tokens/sec.** Total tokens generated across every concurrent request in the batch, divided by that batch's real wall-clock duration (including each request's TTFT, not just decode time). Higher is better. This is the number that shows overall system capacity — on hardware with real batching headroom it climbs with concurrency before eventually plateauing or declining; on memory/bandwidth-constrained hardware with no spare headroom, it can decline from the very first step instead, meaning concurrency only adds contention rather than paying off.

**Concurrency → Prefill Tokens/sec.** Per-request prompt-processing throughput calculated only from an engine-reported prompt token count and server-side prompt duration. Higher is better. The chart is omitted when no attributable server timing is available; it never estimates prefill throughput from client-observed TTFT. This normally provides llama.cpp data, while concurrent vLLM requests remain absent because its shared metrics histogram cannot safely attribute prompt time to one request.

Each concurrency model group identifies its **sweet spot**: the lowest tested concurrency level that achieved the maximum aggregate throughput. The badge also reports the per-request throughput tradeoff relative to the one-request measurement when both values are available.

**Concurrency → TTFT.** Time to first token for one request in the batch, including any contention from the other simultaneous requests. Lower is better. Rises with concurrency for the same reason Per-Request Tokens/sec falls — everything in the batch is competing for the same underlying resources.

A model's sweep can stop before reaching the highest configured level — a note above its charts explains why (load failure, engine crash, or failed/timed-out batch). Chat concurrency can also stop after a measured level of 8 or higher falls below the slow-model cutoff; tool concurrency has no slow-TPS soft exit. See [Concurrency](workloads.md#concurrency).

Each loaded file's header row carries a `v<version>` badge showing the suite version that produced it, read from the results file's own top-level `version` field. Files written before that field existed simply omit the badge. The dashboard's own version — parsed from `config.py`'s `VERSION` at build time, so it is never a separately maintained copy — appears once beside the "Results Explorer" eyebrow. Comparing files whose badges disagree is supported, but a schema difference between versions is worth keeping in mind when a section renders unevenly.

The backend badge identifies the inference backend actually exposed by the selected engine build. This can differ from the machine's physical GPU family—for example, the standard Windows llama.cpp package reports Vulkan on AMD and Intel hardware, and on NVIDIA hardware without a driver new enough for any of the prebuilt CUDA builds. The raw results retain that physical classification separately as `profile.hardware_backend`.

**llama-bench → Decode Throughput.** `llama-bench`'s generation `avg_ts` after prefilling the KV cache to each configured pp depth, with one series per tg length. Higher is better. This isolates generation speed from prompt processing and shows how decode throughput changes as context grows.

**llama-bench → Prompt Processing Throughput.** `llama-bench`'s standalone prompt-processing `avg_ts` at each configured pp size. Higher is better. This measures how quickly the model ingests a prompt, independently of subsequent generation.

**llama-bench Concurrency.** `llama-batched-bench` aggregate decode throughput (`speed_tg`) as parallel sequence count rises, with one chart per tg size. Higher is better. This is a lower-level batching cross-check, not per-user throughput—the value is combined across all sequences.

**Embeddings → Chunks/sec.** Throughput embedding one real document's chunks in a single call. Higher is better.

**Images → Sec/image.** Wall-clock time to generate one image at a given resolution, per model. Lower is better.

## Stats table

Below the charts, every section also renders a sortable raw-numbers table (one row per model/context-length/category, depending on section) — click a column header to sort by it, click again to reverse direction. Useful for reading exact values or copying numbers out, where a chart is more about the overall shape.

Performance sections with sample evidence also render a collapsible **Decision-grade sample review**. It lists every available sample, whether it contributed to the aggregate, and any exclusion reason such as `implausible_server_tps`; the filter isolates valid, excluded, or legacy evidence. Historical files that contain only means and run counts are labeled **legacy aggregate-only** rather than being presented as if their raw samples were recoverable. Native llama-bench internal repetitions appear when `ts_runs` or `samples_ts` is present. When schema-4 pause evidence exists, this review opens automatically and shows each affected system's pause count and total derived paused duration; an unfinished or malformed final interval is labeled unavailable rather than treated as zero time.

Schema-5 memory results add a tightest-headroom indicator to each run card, host/process/accelerator peak and headroom columns to LLM raw tables, and a per-model peak process-RSS chart. Older files and telemetry-off schema-5 files show **Not recorded** rather than numeric zero. Process RSS and accelerator occupancy remain separate quantities; the dashboard does not merge them into a cross-platform memory score.

Schema-5 power results add measured joules, workload efficiency, and explicit power scope to raw tables; tokens-per-joule charts to LLM views; and same-scope total energy plus idle baseline to each run card. Unavailable sources show their normalized reason, older files show **Not recorded**, and mixed processor-package/accelerator/CPU-package/whole-system scopes are never plotted on one axis or summed into one run total.

## Multi-file comparison

Each file is assigned a colour (blue → orange → green → purple → red → teal). All charts use that colour to identify the host, making results from different machines directly comparable. Engine-backed chart identities render as three lines—hostname, backend, then the engine identifier and version such as `llamacpp 10362` for either an official package or a build from that official source release, `llamacpp 2026.08.11-a1b2c3d` for a non-release source build, or `vllm 0.10.2`—so results made before and after an engine update remain distinguishable without parenthetical labels; a llama.cpp run with weight repacking disabled inserts `-nr` after the engine name on workloads that consume the setting. Image charts omit runtime labels because ComfyUI runs independently. A current result whose runtime could not be identified is labeled `version unavailable`, while a historical file without the `engine_version` field is labeled `version not recorded`; the dashboard never invents a version for either case. When compared llama.cpp files disagree on weight-repacking mode, affected workload sections also show a methodology warning; Images, standard llama-bench, and vllm bench omit both `-nr` methodology cues because they do not consume the setting. Labels remain overridable per file in the header. The **Models** filter shows or hides individual models.

With two or more files loaded, **Compare As** can designate one file as the baseline. Charts then show each matching metric as a percentage of that result, with the baseline at 100%; cells absent or zero in the baseline remain absent rather than producing an invented ratio. Raw-number tables remain absolute so the underlying measurements are always available.

## Repeated-trial artifacts

The repeated-trial view shows each common metric's baseline and candidate mean, median, between-trial standard deviation, drift status, 95% change interval and method, practical threshold, pairing mode, trial counts, and verdict. Improved, regressed, unchanged, and inconclusive remain visually distinct. Missing intervals render as inconclusive rather than zero, and monotonic drift remains visible beside the affected distribution.

## Recommendation artifacts

Recommendation schema 1 is a derived artifact with `artifact_type: "recommendation"`. Load the clearly labeled synthetic [recommendation_example.json](../samples/recommendation_example.json) by itself to inspect a recommended, eliminated, and unevaluated candidate derived from [results_recommendation_synthetic.json](../samples/results_recommendation_synthetic.json). Empty outcome groups are hidden; eliminated entries show the failed hard constraint and collapsible evidence references, while unevaluated entries name missing evidence and never use failure styling. The artifact's verdict is always recommended, tied, or insufficient evidence. Interactive constraint entry and shared workspace state belong to Version 6 milestone 11 and are deliberately not implemented in the standalone dashboard.

## Exporting

Drop a logo image onto the **Logo** drop zone to embed it in the bottom-right corner of every chart. Click **Save PNG** to export all visible charts as individual files:

```
llama3.1-8b-q4_tps.png
llama3.1-8b-q4_ttft.png
llama3.1-8b-q4_conv_tps.png       # LLM Conversation section
llama3.1-8b-q4_conv_ttft.png      # LLM Conversation section
llama3.1-8b-q4_conc_tool_tps.png       # Tool concurrency, Per-Request Tokens/sec
llama3.1-8b-q4_conc_tool_aggregate.png # Tool concurrency, Aggregate Tokens/sec
llama3.1-8b-q4_conc_tool_ttft.png      # Tool concurrency, TTFT
llama3.1-8b-q4_conc_chat_tps.png       # Chat concurrency, Per-Request Tokens/sec
mcq-accuracy.png                   # Accuracy section, Overall chart
llama3.1-8b-q4_mcq-category.png    # Accuracy section, by-Category chart
mcq-incidents.png                  # Accuracy section, incident diagnostics chart
embeddings.png
1024x1024_images.png
llama3.1-8b-q4_llamabench_decode.png  # llama-bench decode section
llama3.1-8b-q4_llamabench_prefill.png # llama-bench prefill section
```

The **Chart Width** field (default 708 px) controls the capture width — increase for wider exports.

Every loaded result also renders a **Shareable Run Card** with its system, runtime, RAM, suite version, and the fastest decode/lowest-TTFT model in each represented tier. Current results use the shared 2K single-shot checkpoint; historical files without 2K use that tier's shallowest recorded canonical checkpoint and label it explicitly. **Spec Card** exports these cards as `<system>[_<suffix>]_run-card.png`, numbering repeated system names so same-host comparisons do not collide; an uploaded logo is included.

A results file is never guaranteed to have every field a newer schema might expect, since people compare files produced by different versions of this suite across different machines — `dashboard/src/utils/*.ts` leans on optional chaining (`f.data[section]?.[model]?.[ctx]`, not `f.data[section][model][ctx]`) throughout for exactly this reason. New dashboard code reading the results JSON should assume any given key might be missing on an older file.

When explicit measurement fields are present, TTFT charts and tables prefer client-observed TTFT and run tables prefer `valid_runs`; older files fall back to `ttft_mean_sec` and `n_runs`. Server-reported prompt duration remains available for auditing but is not silently substituted into the TTFT charts. Comparing legacy server-prompt conversation TTFT with explicit client-observed conversation TTFT produces a methodology warning.

## Development

```bash
cd dashboard
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

---

[← CLI Reference](cli-reference.md) · [Back to README](../README.md) · [How It Works →](how-it-works.md)
