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
bash dashboard.sh
bash dashboard.sh --port 8080   # use a different port
bash dashboard.sh --rebuild     # force a fresh npm build

# Windows
dashboard.bat
dashboard.bat --port 8080       # use a different port
dashboard.bat --rebuild         # force a fresh npm build
```

Requires Node.js/npm. On first run, installs npm dependencies and builds the app, then starts a local server on port 3000 and opens the browser automatically.

## Loading results

Drag one or more `results_*.json` files onto the drop zone in the top-right corner, or click to open a file picker. Up to six files can be loaded at once. Dropping a single file when fewer than six are loaded adds it to the current set; dropping multiple at once replaces all. Sample files for testing are in `samples/`.

## Sections

| Section | Charts |
|---|---|
| LLM | Two charts per model — Tokens/sec and TTFT — across context lengths (2K / 8K / 32K / 64K), single-shot cold-prefill test |
| LLM Conversation | Same two charts per model, but from the multi-turn conversation test, across whichever of 0 / 2K / 4K / 8K / 16K / 32K / 64K / 96K each model's context window reached |
| Accuracy | A **Test** sub-picker for MCQ / Math / Code (mirrors `scripts/config.py`'s `ACCURACY_TESTS`). Per test: one Overall accuracy-per-model chart, one Accuracy-by-Category breakdown chart per model, and — only when at least one question actually timed out — a Timeouts & Likely Loops diagnostics chart. See [Accuracy](workloads.md#accuracy) for what these numbers measure |
| Embeddings | Chunks per second embedding one real document in a single call |
| Images | One grouped bar chart per resolution — all image models side by side per host |

The **Models** filter and **Machine** labels are shared between the LLM, LLM Conversation, and Accuracy sections, so switching between them keeps the same models/files selected.

## Chart Style and Group By

**Chart Style** (Bar / Line) and **Group By** (Model / System) apply to the LLM, LLM Conversation, Embeddings, and Images sections — Bar vs. Line picks the chart type, and Group By flips which axis becomes the per-chart grouping (one chart per model with systems as series, vs. one chart per system with models as series). Group By → System also reveals a **Model Sizes** toggle (Split by tier vs. Combined) for the LLM/LLM Conversation sections, since a single combined line chart with every model tier at once is unreadable.

Both pills are hidden on the **Accuracy** section — Accuracy charts are always bar charts grouped by model, since accuracy is a single scalar per model rather than a metric swept across context lengths or resolutions (no second axis to pivot on), so there's nothing for either control to change.

## What the charts mean

**LLM → Tokens/sec.** Decode throughput (tokens generated per second) for the single-shot test, at each context length. Higher is better. This is generation speed *after* the prompt has already been processed — it answers "once the model starts responding, how fast does text come out?"

**LLM → TTFT.** Time to process the single-shot prompt before the first token comes back — a genuine cold prefill, since every run sends fresh, never-before-seen prompt content. Lower is better. This answers "if I paste a large document and hit send, how long do I wait before anything happens?" TTFT rises sharply with context length here, since the model has to run every one of those tokens through the network with nothing cached.

**LLM Conversation → Tokens/sec.** The same decode-throughput metric, but measured mid-conversation instead of after a single cold prompt. Generally close to the single-shot number for the same model — decode speed doesn't depend much on how the context got filled.

**LLM Conversation → TTFT.** Time to process just the *next* turn in an already-long conversation, relying on the backend's KV-cache reuse (llama.cpp/Ollama's slot cache) so only the new turn's tokens need to be run through the network, not the entire history again. This is **why conversation TTFT at, say, 32K is typically a small fraction of single-shot TTFT at 32K** — they're not measuring the same thing. Single-shot TTFT is "cold start with a huge prompt"; conversation TTFT is "one more message in a chat that's already this long." Both are real workloads; which one matters more depends on whether your use case looks like one-shot document Q&A or an ongoing chat/agent session.

**Accuracy → Overall.** Accuracy percentage per model on the selected test's full question bank, one bar chart with systems on the axis and one colored bar per model. Higher is better.

**Accuracy → Accuracy by Category.** The same test's questions broken down by category (e.g. arithmetic, logic, geometry — see [Accuracy](workloads.md#accuracy) for the full list per test), one chart per model. With a single file loaded, each category bar gets its own color from a fixed palette (and no legend, since there's only one system on the chart) purely to make individual bars easier to tell apart at a glance — the colors don't carry cross-chart meaning the way file/model colors do elsewhere.

**Accuracy → Timeouts & Likely Loops.** Per model, how many questions hit `--acc-timeout` (default 60s) without answering, and how many of those were flagged as a likely generation loop (see [Accuracy → Timeouts and loop detection](workloads.md#timeouts-and-loop-detection)). Lower is better. Only rendered when at least one model/file actually had a timeout — a clean run across the board means this chart doesn't appear at all.

**Embeddings → Chunks/sec.** Throughput embedding one real document's chunks in a single call. Higher is better.

**Images → Sec/image.** Wall-clock time to generate one image at a given resolution, per model. Lower is better.

## Stats table

Below the charts, every section also renders a sortable raw-numbers table (one row per model/context-length/category, depending on section) — click a column header to sort by it, click again to reverse direction. Useful for reading exact values or copying numbers out, where a chart is more about the overall shape.

## Multi-file comparison

Each file is assigned a colour (blue → orange → green → purple → red → teal). All charts use that colour to identify the host, making results from different machines directly comparable. The **Models** filter shows or hides individual models.

## Exporting

Drop a logo image onto the **Logo** drop zone to embed it in the bottom-right corner of every chart. Click **Save PNG** to export all visible charts as individual files:

```
llama3.1-8b-q4_tps.png
llama3.1-8b-q4_ttft.png
llama3.1-8b-q4_conv_tps.png       # LLM Conversation section
llama3.1-8b-q4_conv_ttft.png      # LLM Conversation section
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
