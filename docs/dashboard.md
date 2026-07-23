[← Back to README](../README.md)

# Dashboard

**Contents**
- [Loading results](#loading-results)
- [Sections](#sections)
- [Chart Style and Group By](#chart-style-and-group-by)
- [What the charts mean](#what-the-charts-mean)
- [Stats table](#stats-table)
- [Multi-file comparison](#multi-file-comparison)
- [Exporting](#exporting)
- [Development](#development)

An interactive results explorer for visualising and exporting benchmark output.

```bash
# Linux / macOS
bash launch_dashboard.sh
bash launch_dashboard.sh --port 8080   # use a different port

# Windows
launch_dashboard.bat
launch_dashboard.bat --port 8080       # use a different port
```

Requires Node.js/npm. On first run, installs npm dependencies. Every run rebuilds the app, then starts a local server on port 3000 and opens the browser automatically.

## Loading results

Drag one or more `results_*.json` files onto the drop zone in the top-right corner, or click to open a file picker. Up to six files can be loaded at once. Dropping a single file when fewer than six are loaded adds it to the current set; dropping multiple at once replaces all. Sample files for testing are in `samples/`. Files must contain strict JSON; an invalid file now displays an import error below the drop zone rather than failing silently.

## Sections

| Section | Charts |
|---|---|
| LLM | Two charts per model — Tokens/sec and TTFT — across context lengths (512 / 2K / 8K / 32K / 64K), single-shot cold-prefill test |
| LLM Conversation | Same two charts per model, but from the multi-turn conversation test, across whichever of 0 / 2K / 4K / 8K / 16K / 32K / 48K / 64K / 80K / 96K its plan reached (xsmall/small catalog models stop at 48K) |
| Concurrency (Tool) | Three line charts per model — Per-Request Tokens/sec, Aggregate Tokens/sec, and TTFT — at 1 / 2 / 4 / 6 / 8 / 12 / 16 simultaneous short-context requests |
| Concurrency (Chat) | The same three charts at 1 / 2 / 4 / 8 / 16 / 24 / 32 simultaneous long-context requests. See [Concurrency](workloads.md#concurrency) for how the two workloads differ |
| Accuracy | A **Test** sub-picker for MCQ / Math / Reasoning / Code / Tool Use (mirrors `ACCURACY_TESTS` in `dashboard/src/constants.js`). Per test: one Overall accuracy-per-model chart, one Accuracy-by-Category breakdown chart per model, and — when provided by the bank — an Accuracy-by-Difficulty chart. A Timeouts & Likely Loops diagnostics chart appears when any incident was recorded. See [Accuracy](workloads.md#accuracy) |
| Embeddings | Chunks per second embedding one real document in a single call |
| Images | One grouped bar chart per resolution — all image models side by side per host |

The **Models** filter and **Machine** labels are shared between the LLM, LLM Conversation, Concurrency, and Accuracy sections, so switching between them keeps the same models/files selected.

## Chart Style and Group By

**Chart Style** (Bar / Line) and **Group By** (Model / System) apply to the LLM, LLM Conversation, Embeddings, and Images sections — Bar vs. Line picks the chart type, and Group By flips which axis becomes the per-chart grouping (one chart per model with systems as series, vs. one chart per system with models as series). In by-model LLM bar charts, each system remains one chart category with native group spacing, while its context bars follow the numeric checkpoint order from `CTX_ORDER` rather than lexicographic label order. Group By → System omits models for which every loaded file contains only a `no_llm_data` placeholder, while retaining a model attempted by at least one system. It also reveals a **Model Sizes** toggle (Split by tier vs. Combined) for the LLM/LLM Conversation sections, since a single combined line chart with every model tier at once is unreadable.

Both pills are hidden on the **Accuracy** and **Concurrency** sections. Accuracy charts are always bar charts grouped by model, since accuracy is a single scalar per model rather than a metric swept across context lengths or resolutions. Concurrency charts are always line charts grouped by model, with request count on the horizontal axis.

## What the charts mean

**LLM → Tokens/sec.** Decode throughput (tokens generated per second) for the single-shot test, at each context length. Higher is better. This is generation speed *after* the prompt has already been processed — it answers "once the model starts responding, how fast does text come out?"

**LLM → TTFT.** Time to process the single-shot prompt before the first token comes back — a genuine cold prefill, since every run sends fresh, never-before-seen prompt content. Lower is better. This answers "if I paste a large document and hit send, how long do I wait before anything happens?" TTFT rises sharply with context length here, since the model has to run every one of those tokens through the network with nothing cached.

**LLM Conversation → Tokens/sec.** The same decode-throughput metric, but measured mid-conversation instead of after a single cold prompt. Generally close to the single-shot number for the same model — decode speed doesn't depend much on how the context got filled.

**LLM Conversation → TTFT.** Time to process just the *next* turn in an already-long conversation, relying on the backend's KV-cache reuse (llama.cpp's slot cache) so only the new turn's tokens need to be run through the network, not the entire history again. This is **why conversation TTFT at, say, 32K is typically a small fraction of single-shot TTFT at 32K** — they're not measuring the same thing. Single-shot TTFT is "cold start with a huge prompt"; conversation TTFT is "one more message in a chat that's already this long." Both are real workloads; which one matters more depends on whether your use case looks like one-shot document Q&A or an ongoing chat/agent session.

**Accuracy → Overall.** Accuracy percentage per model on the selected test's full question bank, one bar chart with systems on the axis and one colored bar per model. Higher is better.

**Accuracy → Accuracy by Category.** The same test's questions broken down by category (e.g. arithmetic, logic, geometry — see [Accuracy](workloads.md#accuracy) for the full list per test), one chart per model. With a single file loaded, each category bar gets its own color from a fixed palette (and no legend, since there's only one system on the chart) purely to make individual bars easier to tell apart at a glance — the colors don't carry cross-chart meaning the way file/model colors do elsewhere.

**Accuracy → Timeouts & Likely Loops.** Per model, how many questions reached `--acc-timeout` (default 60s), and how many were stopped as likely generation loops (see [Accuracy → Timeouts and loop detection](workloads.md#timeouts-and-loop-detection)). These are separate diagnostics: a loop may be stopped before the timeout, and a timed-out partial response may still score correctly. Lower is better. The chart appears when either count is nonzero.

**Concurrency → Per-Request Tokens/sec.** Decode throughput for one individual request within a batch of N simultaneous requests. Higher is better. Typically drops as concurrency climbs, since requests share the same compute/memory bandwidth — this is the number that shows per-user latency degrading under load.

**Concurrency → Aggregate Tokens/sec.** Total tokens generated across every concurrent request in the batch, divided by that batch's real wall-clock duration (including each request's TTFT, not just decode time). Higher is better. This is the number that shows overall system capacity — on hardware with real batching headroom it climbs with concurrency before eventually plateauing or declining; on memory/bandwidth-constrained hardware with no spare headroom, it can decline from the very first step instead, meaning concurrency only adds contention rather than paying off.

**Concurrency → TTFT.** Time to first token for one request in the batch, including any contention from the other simultaneous requests. Lower is better. Rises with concurrency for the same reason Per-Request Tokens/sec falls — everything in the batch is competing for the same underlying resources.

A model's sweep can stop before reaching the highest configured level — a note above its charts explains why (load failure, engine crash, or failed/timed-out batch). Chat concurrency can also stop after a measured level of 8 or higher falls below the slow-model cutoff; tool concurrency has no slow-TPS soft exit. See [Concurrency](workloads.md#concurrency).

The backend badge identifies the inference backend actually exposed by the selected engine build. This can differ from the machine's physical GPU family—for example, the standard Windows llama.cpp package reports Vulkan on NVIDIA, AMD, and Intel hardware. The raw results retain that physical classification separately as `profile.hardware_backend`.

**Embeddings → Chunks/sec.** Throughput embedding one real document's chunks in a single call. Higher is better.

**Images → Sec/image.** Wall-clock time to generate one image at a given resolution, per model. Lower is better.

## Stats table

Below the charts, every section also renders a sortable raw-numbers table (one row per model/context-length/category, depending on section) — click a column header to sort by it, click again to reverse direction. Useful for reading exact values or copying numbers out, where a chart is more about the overall shape.

## Multi-file comparison

Each file is assigned a colour (blue → orange → green → purple → red → teal). All charts use that colour to identify the host, making results from different machines directly comparable. A file's `"engine"` field, when present, is folded into its default label (`hostname (llamacpp)`) so an `--engine all` pair from the same machine loads as two distinct series instead of two identically-labeled ones (currently a no-op with only one engine registered — see [Engines](engines.md#selecting-an-engine)) — still overridable per file in the header. This also keeps an older results file generated against the now-removed Ollama engine distinguishable if loaded alongside a newer llama.cpp one. The **Models** filter shows or hides individual models.

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
mcq-timeouts.png                   # Accuracy section, Timeouts & Likely Loops chart
embeddings.png
1024x1024_images.png
```

The **Chart Width** field (default 708 px) controls the capture width — increase for wider exports.

## Development

```bash
cd dashboard
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

---

[← CLI Reference](cli-reference.md) · [Back to README](../README.md) · [How It Works →](how-it-works.md)
