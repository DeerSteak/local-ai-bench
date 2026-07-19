import {
  CTX_ORDER, RES_ORDER, CONCURRENCY_LEVELS, CONCURRENCY_STOP_LABELS,
  MODEL_COLORS, IMAGE_MODEL_COLORS, EMBED_MODEL_COLORS, FALLBACK_COLORS,
  FILE_COLORS, CATEGORY_COLORS, MODEL_DASH_PATTERNS,
  LLM_MODEL_LABELS, IMAGE_MODEL_LABELS, EMBED_MODEL_LABELS,
  LLM_MODEL_ORDER, IMAGE_MODEL_ORDER, EMBED_MODEL_ORDER,
  CTX_COLORS, IMAGE_BAR_COLORS, EMBED_BAR_COLORS, RES_COLORS, MODEL_SIZE_TIER,
} from "./constants";

export function parseJSON(text) {
  try { return JSON.parse(text); } catch { return null; }
}

// Turn free-typed text (or a whole joined filename stem) into something safe
// to use as a filename: whitespace and characters reserved/special on common
// filesystems — including periods, since they read as file extensions/hidden-
// file markers — collapse to a single hyphen, and any leading/trailing
// hyphens left over are trimmed.
export function sanitizeForFilename(raw) {
  return String(raw || "")
    .trim()
    .replace(/[\s<>:"/\\|?*#%&{}$!'`=+@~^.]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

export function fmt(v, unit) {
  if (v == null) return "—";
  switch (unit) {
    case "ms": {
      const ms = v * 1000;
      return ms < 10 ? `${ms.toFixed(1)}ms` : `${Math.round(ms)}ms`;
    }
    case "sec-plain":
      return `${v.toFixed(2)}s`;
    case "sec":
      if (v >= 60) return `${(v / 60).toFixed(1)}m`;
      return `${v.toFixed(2)}s`;
    case "tps":
      if (v >= 1000) return `${(v / 1000).toFixed(2)}K`;
      return v.toFixed(1);
    case "sps":
      if (v >= 10000) return `${(v / 1000).toFixed(1)}K`;
      return v.toFixed(0);
    case "pct":
      return `${v.toFixed(1)}%`;
    case "count":
      return `${Math.round(v)}`;
    default:
      return v.toFixed(2);
  }
}

// Deterministic color for an unknown model based on its name
function hashColor(key, palette) {
  const h = [...key].reduce((acc, c) => acc + c.charCodeAt(0), 0);
  return palette[h % palette.length];
}

export function getModelColor(model) {
  return MODEL_COLORS[model] || hashColor(model, FALLBACK_COLORS);
}

export function getImageModelColor(model) {
  return IMAGE_MODEL_COLORS[model] || hashColor(model, FALLBACK_COLORS);
}

export function getEmbedModelColor(model) {
  return EMBED_MODEL_COLORS[model] || hashColor(model, FALLBACK_COLORS);
}

export function modelLabel(model) {
  return LLM_MODEL_LABELS[model] || model;
}

export function imageModelLabel(model) {
  return IMAGE_MODEL_LABELS[model] || model;
}

export function embedModelLabel(model) {
  return EMBED_MODEL_LABELS[model] || model;
}

// Bucket an LLM model key into a size tier. Known models use MODEL_SIZE_TIER
// (parameter-count-based, matching models.py/README.md exactly). Unknown
// models (not in the standard roster) fall back to a param-count heuristic
// parsed from the key (e.g. "some-new-model-70b" -> 70 -> "large").
export function getModelSizeTier(model) {
  if (MODEL_SIZE_TIER[model]) return MODEL_SIZE_TIER[model];
  const match = model.match(/(\d+)b/i);
  if (!match) return "medium";
  const billions = parseInt(match[1], 10);
  if (billions <= 20) return "small";
  if (billions < 50) return "medium";
  return "large";
}

// Skip info for one (file, model) pair in a given LLM section — non-null
// only when benchmark.py intentionally excluded this model from that section
// entirely (too slow, timed out on a prior test, no single-shot data, or a
// known repeat-crasher skipped via the crash cache). Defaults to
// "llm_conversation" for existing call sites that predate the "llm" section
// also being able to produce a whole-model skip (via Shared.check_crash_cache).
export function getSkipInfo(file, model, section = "llm_conversation") {
  const d = file.data[section]?.[model];
  if (!d?.skipped) return null;
  return { reason: d.skip_reason, detail: d.skip_detail };
}

const SKIP_REASON_LABELS = {
  timed_out: "Skipped - LLM Timed Out",
  slow_tps: "Skipped - LLM Too Slow",
  no_llm_data: "Skipped - No LLM Data",
  known_crash: "Skipped - Engine Crashed",
};

// Bar-chart status label for one (file, model, context) cell: "{ctx} - Timed
// Out" for the context at which benchmark.py's run itself timed out (llm or
// llm_conversation both set a "timed_out" field), "{ctx} - Crashed" for the
// context at which the inference engine's model runner crashed (a "crashed" field, set by
// the llm and llm_conversation tests when they give up retrying after a
// repeat crash), "{ctx} - Skipped ({slowCtx} Too Slow)" for every later
// (larger) context that was never attempted because the model dropped below
// the tok/s cutoff at an earlier checkpoint (a "slow_tps" field), or a
// "Skipped - ..." label when the whole model was excluded from this section
// (a "skipped"/"skip_reason" pair, from either a slow/timed-out gate in
// benchmark.py or a known-crasher skip in the section's own crash cache). The
// slow checkpoint itself still has real data (it's the measurement that
// triggered the cutoff), so it returns null there — its actual value is shown
// rather than a status label. Returns null for cells with real data, or
// earlier contexts that simply weren't reached for unrelated reasons.
export function getBarStatusLabel(file, model, ctx, section) {
  const skip = getSkipInfo(file, model, section);
  if (skip) return SKIP_REASON_LABELS[skip.reason] || `Skipped - ${skip.detail}`;
  const sectionData = file.data[section]?.[model];
  const crashedCtx = sectionData?.crashed;
  if (crashedCtx) {
    const crashedIdx = CTX_ORDER.indexOf(crashedCtx);
    const ctxIdx = CTX_ORDER.indexOf(ctx);
    if (ctxIdx === crashedIdx) return `${ctx} - Crashed`;
    if (ctxIdx > crashedIdx) return `${ctx} - Skipped`;
  }
  const timedOutCtx = sectionData?.timed_out;
  if (timedOutCtx) {
    const timedOutIdx = CTX_ORDER.indexOf(timedOutCtx);
    const ctxIdx = CTX_ORDER.indexOf(ctx);
    if (ctxIdx === timedOutIdx) return `${ctx} - Timed Out`;
    if (ctxIdx > timedOutIdx) return `${ctx} - Skipped`;
  }
  const slowTpsCtx = sectionData?.slow_tps;
  if (slowTpsCtx) {
    const slowIdx = CTX_ORDER.indexOf(slowTpsCtx);
    const ctxIdx = CTX_ORDER.indexOf(ctx);
    if (ctxIdx > slowIdx) return `${ctx} - Skipped (${slowTpsCtx} Too Slow)`;
  }
  return null;
}

// Bar-chart status label for one (file, model, resolution) cell in the
// Images charts, mirroring getBarStatusLabel above: "{res} - Timed Out" for
// the resolution at which benchmark.py's image generation run itself timed
// out, "{res} - Skipped" for every larger resolution consequently never
// attempted. Returns null for cells with real data.
export function getImageBarStatusLabel(file, model, res) {
  const timedOutRes = file.data.images?.[model]?.timed_out;
  if (timedOutRes) {
    const timedOutIdx = RES_ORDER.indexOf(timedOutRes);
    const resIdx = RES_ORDER.indexOf(res);
    if (resIdx === timedOutIdx) return `${res} - Timed Out`;
    if (resIdx > timedOutIdx) return `${res} - Skipped`;
  }
  return null;
}

// Return all LLM model keys from the loaded files, in canonical order.
// Checks every section that runs the shared LLM roster — single-shot,
// conversation, the three accuracy tests, and concurrency — since the Models
// filter is shared UI across all of them. A model present in only one section
// (e.g. a file that only ran `--tests acc`, leaving llm/llm_conversation
// empty) should still show up rather than leaving the filter (and every
// section that depends on it) empty.
export function getAllLLMModels(files) {
  const s = new Set();
  for (const f of files) {
    for (const m of Object.keys(f.data.llm || {})) s.add(m);
    for (const m of Object.keys(f.data.llm_conversation || {})) s.add(m);
    for (const m of Object.keys(f.data.mcq || {})) s.add(m);
    for (const m of Object.keys(f.data.math || {})) s.add(m);
    for (const m of Object.keys(f.data.code || {})) s.add(m);
    for (const m of Object.keys(f.data.concurrency || {})) s.add(m);
  }
  const known   = LLM_MODEL_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !LLM_MODEL_ORDER.includes(m));
  return [...known, ...unknown];
}

// Return all image model keys from the loaded files, in canonical order
export function getAllImageModels(files) {
  const s = new Set();
  for (const f of files) for (const m of Object.keys(f.data.images || {})) s.add(m);
  const known   = IMAGE_MODEL_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !IMAGE_MODEL_ORDER.includes(m));
  return [...known, ...unknown];
}

// Return all embedding model keys from the loaded files, in canonical order
export function getAllEmbedModels(files) {
  const s = new Set();
  for (const f of files) for (const m of Object.keys(f.data.embeddings || {})) s.add(m);
  const known   = EMBED_MODEL_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !EMBED_MODEL_ORDER.includes(m));
  return [...known, ...unknown];
}

// ── Chart data builders ────────────────────────────────────────────────────────

// Per-file line configs: one line per file, color by file index. Used for all sections.
export function buildFileLineConfigs(files) {
  return files.map((f, fi) => ({
    dataKey: `f${fi}`,
    stroke: FILE_COLORS[fi % FILE_COLORS.length],
    name: f.hostname,
  }));
}

// LLM: one chart per model. X = context length, lines = files.
export function buildLLMDataForModel(files, model, metric, section = "llm") {
  const ctxSet = new Set();
  for (const f of files)
    for (const ctx of Object.keys(f.data[section]?.[model] || {})) ctxSet.add(ctx);
  const ctxLabels = CTX_ORDER.filter(c => ctxSet.has(c));
  return ctxLabels.map(ctx => {
    const row = { ctxLabel: ctx };
    files.forEach((f, fi) => {
      const s = f.data[section]?.[model]?.[ctx];
      if (s) row[`f${fi}`] = metric === "tps" ? s.tps_mean : s.ttft_mean_sec;
    });
    return row;
  });
}

// Legacy: X = context length, lines = models (+ file distinction if multi)
export function buildLLMData(files, metric, enabledModels) {
  const isSingle = files.length === 1;
  const ctxSet = new Set();
  for (const f of files)
    for (const md of Object.values(f.data.llm || {}))
      for (const ctx of Object.keys(md)) ctxSet.add(ctx);
  const ctxLabels = CTX_ORDER.filter(c => ctxSet.has(c));

  return ctxLabels.map(ctx => {
    const row = { ctxLabel: ctx };
    files.forEach((f, fi) => {
      for (const [model, md] of Object.entries(f.data.llm || {})) {
        if (!enabledModels.has(model) || !md[ctx]) continue;
        const key = isSingle ? model : `f${fi}_${model}`;
        row[key] = metric === "tps" ? md[ctx].tps_mean : md[ctx].ttft_mean_sec;
      }
    });
    return row;
  });
}

export function buildLLMLineConfigs(files, data, enabledModels) {
  const isSingle = files.length === 1;
  const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m));
  const configs = [];
  if (isSingle) {
    for (const m of allModels) {
      if (data.some(d => d[m] != null))
        configs.push({ dataKey: m, stroke: getModelColor(m), name: modelLabel(m) });
    }
  } else {
    for (let fi = 0; fi < files.length; fi++) {
      const stroke = FILE_COLORS[fi % FILE_COLORS.length];
      allModels.forEach((m, mi) => {
        const dataKey = `f${fi}_${m}`;
        if (data.some(d => d[dataKey] != null))
          configs.push({
            dataKey,
            stroke,
            strokeDasharray: MODEL_DASH_PATTERNS[mi % MODEL_DASH_PATTERNS.length],
            name: `${files[fi].hostname} — ${modelLabel(m)}`,
          });
      });
    }
  }
  return configs;
}

export function getEmbedLabel(files, model) {
  for (const f of files) {
    const d = f.data.embeddings?.[model];
    if (d?.label) return d.label;
  }
  return embedModelLabel(model);
}

// Images: one chart per model. X = resolution, lines = files.
export function buildImagesDataForModel(files, model) {
  const resSet = new Set();
  for (const f of files)
    for (const r of Object.keys(f.data.images?.[model]?.resolutions || {})) resSet.add(r);
  const resLabels = RES_ORDER.filter(r => resSet.has(r));
  return resLabels.map(res => {
    const row = { resLabel: res };
    files.forEach((f, fi) => {
      const s = f.data.images?.[model]?.resolutions?.[res];
      if (s) row[`f${fi}`] = s.sec_per_image_mean;
    });
    return row;
  });
}

export function getImageLabel(files, model) {
  for (const f of files) {
    const d = f.data.images?.[model];
    if (d?.label) return d.label;
  }
  return imageModelLabel(model);
}

// Images: one bar chart per resolution. X = model, bars = files.
export function buildImagesDataForResolution(files, resolution, enabledImageModels) {
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  return allModels
    .map(model => {
      const row = { modelLabel: getImageLabel(files, model) };
      files.forEach((f, fi) => {
        const s = f.data.images?.[model]?.resolutions?.[resolution];
        if (s) row[`f${fi}`] = s.sec_per_image_mean;
      });
      return row;
    })
    .filter(row => files.some((_, fi) => row[`f${fi}`] != null));
}

// Legacy: X = resolution, lines = image models (+ file distinction if multi)
export function buildImagesData(files, enabledImageModels) {
  const isSingle = files.length === 1;
  const resSet = new Set();
  for (const f of files)
    for (const md of Object.values(f.data.images || {}))
      for (const r of Object.keys(md.resolutions || {})) resSet.add(r);
  const resLabels = RES_ORDER.filter(r => resSet.has(r));

  return resLabels
    .map(res => {
      const row = { resLabel: res };
      files.forEach((f, fi) => {
        for (const [model, md] of Object.entries(f.data.images || {})) {
          if (!enabledImageModels.has(model) || !md.resolutions?.[res]) continue;
          const key = isSingle ? model : `f${fi}_${model}`;
          row[key] = md.resolutions[res].sec_per_image_mean;
        }
      });
      return row;
    })
    .filter(row => Object.keys(row).some(k => k !== "resLabel"));
}

export function buildImagesLineConfigs(files, data, enabledImageModels) {
  const isSingle = files.length === 1;
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  const getLabel = m => {
    for (const f of files) { const d = f.data.images?.[m]; if (d?.label) return d.label; }
    return m;
  };
  const configs = [];
  if (isSingle) {
    for (const m of allModels) {
      if (data.some(d => d[m] != null))
        configs.push({ dataKey: m, stroke: getImageModelColor(m), name: getLabel(m) });
    }
  } else {
    for (let fi = 0; fi < files.length; fi++) {
      const stroke = FILE_COLORS[fi % FILE_COLORS.length];
      allModels.forEach((m, mi) => {
        const dataKey = `f${fi}_${m}`;
        if (data.some(d => d[dataKey] != null))
          configs.push({
            dataKey,
            stroke,
            strokeDasharray: MODEL_DASH_PATTERNS[mi % MODEL_DASH_PATTERNS.length],
            name: `${files[fi].hostname} — ${getLabel(m)}`,
          });
      });
    }
  }
  return configs;
}

// ── Bar chart builders (one color per setting, X = systems) ───────────────────

// LLM bar chart: rows = files/systems, cols = context lengths
export function buildLLMBarData(files, model, metric, section = "llm") {
  return files.map(f => {
    const row = { systemLabel: f.hostname };
    const ctxData = f.data[section]?.[model] || {};
    for (const ctx of CTX_ORDER) {
      const s = ctxData[ctx];
      if (s) row[ctx] = metric === "tps" ? s.tps_mean : s.ttft_mean_sec;
      const status = getBarStatusLabel(f, model, ctx, section);
      if (status) row[`_status_${ctx}`] = status;
    }
    return row;
  });
}

export function buildLLMBarConfigs(files, model, section = "llm") {
  const ctxSet = new Set();
  for (const f of files) {
    for (const ctx of Object.keys(f.data[section]?.[model] || {})) ctxSet.add(ctx);
    const timedOutCtx = f.data[section]?.[model]?.timed_out;
    if (timedOutCtx) ctxSet.add(timedOutCtx);
    const crashedCtx = f.data[section]?.[model]?.crashed;
    if (crashedCtx) ctxSet.add(crashedCtx);
    const slowTpsCtx = f.data[section]?.[model]?.slow_tps;
    if (slowTpsCtx) ctxSet.add(slowTpsCtx);
  }
  return CTX_ORDER
    .filter(ctx => ctxSet.has(ctx))
    .map((ctx, i) => ({
      dataKey: ctx,
      name: ctx,
      fill: CTX_COLORS[ctx] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// Embeddings bar chart: rows = files/systems, cols = models
export function buildEmbedGroupedBarData(files, enabledEmbedModels) {
  const allModels = getAllEmbedModels(files).filter(m => enabledEmbedModels.has(m));
  return files
    .map(f => {
      const row = { systemLabel: f.hostname };
      for (const model of allModels) {
        const s = f.data.embeddings?.[model];
        if (s && !s.skipped) row[model] = s.chunks_per_sec_mean;
      }
      return row;
    })
    .filter(row => allModels.some(m => row[m] != null));
}

export function buildEmbedGroupedBarConfigs(files, enabledEmbedModels) {
  const allModels = getAllEmbedModels(files).filter(m => enabledEmbedModels.has(m));
  return allModels.map((m, i) => ({
    dataKey: m,
    name: getEmbedLabel(files, m),
    fill: EMBED_BAR_COLORS[m] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
  }));
}

// Images bar chart: rows = files/systems, cols = image models
export function buildImagesGroupedBarDataForResolution(files, resolution, enabledImageModels) {
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  return files
    .map(f => {
      const row = { systemLabel: f.hostname };
      for (const model of allModels) {
        const s = f.data.images?.[model]?.resolutions?.[resolution];
        if (s) row[model] = s.sec_per_image_mean;
        const status = getImageBarStatusLabel(f, model, resolution);
        if (status) row[`_status_${model}`] = status;
      }
      return row;
    })
    .filter(row => allModels.some(m => row[m] != null || row[`_status_${m}`] != null));
}

export function buildImagesGroupedBarConfigs(files, enabledImageModels) {
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  return allModels.map((m, i) => ({
    dataKey: m,
    name: getImageLabel(files, m),
    fill: IMAGE_BAR_COLORS[m] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
  }));
}

// ── "Group by System" bar chart builders (one card per system, rows = models) ──

// LLM bar chart by system: rows = models, cols = context lengths, for one file
export function buildLLMBarDataByModel(file, models, metric, section = "llm") {
  return models.map(model => {
    const row = { modelLabel: modelLabel(model) };
    const ctxData = file.data[section]?.[model] || {};
    for (const ctx of CTX_ORDER) {
      const s = ctxData[ctx];
      if (s) row[ctx] = metric === "tps" ? s.tps_mean : s.ttft_mean_sec;
      const status = getBarStatusLabel(file, model, ctx, section);
      if (status) row[`_status_${ctx}`] = status;
    }
    return row;
  });
}

export function buildLLMBarConfigsByModel(file, models, section = "llm") {
  const ctxSet = new Set();
  for (const model of models) {
    for (const ctx of Object.keys(file.data[section]?.[model] || {})) ctxSet.add(ctx);
    const timedOutCtx = file.data[section]?.[model]?.timed_out;
    if (timedOutCtx) ctxSet.add(timedOutCtx);
    const crashedCtx = file.data[section]?.[model]?.crashed;
    if (crashedCtx) ctxSet.add(crashedCtx);
    const slowTpsCtx = file.data[section]?.[model]?.slow_tps;
    if (slowTpsCtx) ctxSet.add(slowTpsCtx);
  }
  return CTX_ORDER
    .filter(ctx => ctxSet.has(ctx))
    .map((ctx, i) => ({
      dataKey: ctx,
      name: ctx,
      fill: CTX_COLORS[ctx] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// Images bar chart by system: rows = models, cols = resolutions, for one file
export function buildImagesBarDataByModel(file, models) {
  return models
    .map(model => {
      const row = { modelLabel: getImageLabel([file], model) };
      const resData = file.data.images?.[model]?.resolutions || {};
      for (const res of RES_ORDER) {
        const s = resData[res];
        if (s) row[res] = s.sec_per_image_mean;
        const status = getImageBarStatusLabel(file, model, res);
        if (status) row[`_status_${res}`] = status;
      }
      return row;
    })
    .filter(row => RES_ORDER.some(res => row[res] != null || row[`_status_${res}`] != null));
}

export function buildImagesBarConfigsByModel(file, models) {
  const resSet = new Set();
  for (const model of models) {
    for (const res of Object.keys(file.data.images?.[model]?.resolutions || {})) resSet.add(res);
    const timedOutRes = file.data.images?.[model]?.timed_out;
    if (timedOutRes) resSet.add(timedOutRes);
  }
  return RES_ORDER
    .filter(res => resSet.has(res))
    .map((res, i) => ({
      dataKey: res,
      name: res,
      fill: RES_COLORS[res] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// Embeddings bar chart by system: rows = models, single throughput value, for one file
export function buildEmbedBarDataByModel(file, models) {
  return models
    .map(model => {
      const s = file.data.embeddings?.[model];
      const row = { modelLabel: getEmbedLabel([file], model) };
      if (s && !s.skipped) row.throughput = s.chunks_per_sec_mean;
      return row;
    })
    .filter(row => row.throughput != null);
}

export function buildEmbedBarConfigsByModel(file, models) {
  const hasAny = models.some(model => file.data.embeddings?.[model] && !file.data.embeddings[model].skipped);
  return hasAny ? [{ dataKey: "throughput", name: "Chunks/sec", fill: FALLBACK_COLORS[0] }] : [];
}

// ── "Group by System" line chart builders (Y = setting, X = metric, lines = models) ──

// LLM line chart by system: rows = context lengths, one line per model, for one file
export function buildLLMLineDataByCtx(file, models, metric, section = "llm") {
  const ctxSet = new Set();
  for (const model of models)
    for (const ctx of Object.keys(file.data[section]?.[model] || {})) ctxSet.add(ctx);
  const ctxLabels = CTX_ORDER.filter(c => ctxSet.has(c));
  return ctxLabels.map(ctx => {
    const row = { ctxLabel: ctx };
    for (const model of models) {
      const s = file.data[section]?.[model]?.[ctx];
      if (s) row[model] = metric === "tps" ? s.tps_mean : s.ttft_mean_sec;
    }
    return row;
  });
}

export function buildLLMLineConfigsByCtx(models, data) {
  return models
    .filter(m => data.some(row => row[m] != null))
    .map(m => ({ dataKey: m, stroke: getModelColor(m), name: modelLabel(m) }));
}

// Images line chart by system: rows = resolutions, one line per model, for one file
export function buildImagesLineDataByRes(file, models) {
  const resSet = new Set();
  for (const model of models)
    for (const res of Object.keys(file.data.images?.[model]?.resolutions || {})) resSet.add(res);
  const resLabels = RES_ORDER.filter(r => resSet.has(r));
  return resLabels.map(res => {
    const row = { resLabel: res };
    for (const model of models) {
      const s = file.data.images?.[model]?.resolutions?.[res];
      if (s) row[model] = s.sec_per_image_mean;
    }
    return row;
  });
}

export function buildImagesLineConfigsByRes(file, models, data) {
  return models
    .filter(m => data.some(row => row[m] != null))
    .map(m => ({ dataKey: m, stroke: getImageModelColor(m), name: getImageLabel([file], m) }));
}

// ── Bar chart sorting ─────────────────────────────────────────────────────────

// Sort bar-chart rows so the fastest result is first.
// preferredKeys: ordered array of candidate sort keys; the last one present in
// the data is used (most strenuous). direction: "desc" = higher is better,
// "asc" = lower is better.
export function sortBarData(data, preferredKeys, direction) {
  let sortKey = null;
  for (let i = preferredKeys.length - 1; i >= 0; i--) {
    if (data.some(row => row[preferredKeys[i]] != null)) {
      sortKey = preferredKeys[i];
      break;
    }
  }
  if (!sortKey) return data;
  return [...data].sort((a, b) => {
    const av = a[sortKey] ?? (direction === "desc" ? -Infinity : Infinity);
    const bv = b[sortKey] ?? (direction === "desc" ? -Infinity : Infinity);
    return direction === "desc" ? bv - av : av - bv;
  });
}

// Return the key from `keys` whose maximum value across all rows is highest
// (i.e. the most strenuous setting).
export function findMostStrenuousKey(data, keys) {
  let best = null;
  let bestMax = -Infinity;
  for (const key of keys) {
    const vals = data.map(r => r[key]).filter(v => v != null);
    if (!vals.length) continue;
    const max = Math.max(...vals);
    if (max > bestMax) { bestMax = max; best = key; }
  }
  return best;
}

// ── Flat data for StatsTable ───────────────────────────────────────────────────

export function flattenLLMData(files, section = "llm") {
  return files.flatMap(f =>
    Object.entries(f.data[section] || {}).flatMap(([model, ctxData]) => {
      if (ctxData?.skipped) {
        return [{
          _fileId: f.id, model, ctx: "—", skipped: true,
          skip_reason: ctxData.skip_reason, skip_detail: ctxData.skip_detail,
        }];
      }
      return Object.entries(ctxData)
        .filter(([ctx]) => CTX_ORDER.includes(ctx))
        .map(([ctx, s]) => ({
          _fileId: f.id, model, ctx,
          tps_mean: s.tps_mean, tps_stdev: s.tps_stdev,
          ttft_mean: s.ttft_mean_sec, ttft_stdev: s.ttft_stdev_sec,
          n_runs: s.n_runs,
        }));
    })
  );
}

export function flattenEmbedData(files) {
  return files.flatMap(f =>
    Object.entries(f.data.embeddings || {}).map(([model, s]) => {
      const modelLabel = s.label || model;
      if (s.skipped) {
        return {
          _fileId: f.id, model, modelLabel, skipped: true,
          skip_reason: s.skip_reason, skip_detail: s.skip_detail,
        };
      }
      return {
        _fileId: f.id, model, modelLabel,
        cps_mean: s.chunks_per_sec_mean,
        cps_stdev: s.chunks_per_sec_stdev,
        n_chunks: s.n_chunks,
        device: s.device,
        n_runs: s.n_runs,
      };
    })
  );
}

export function flattenImageData(files) {
  return files.flatMap(f =>
    Object.entries(f.data.images || {}).flatMap(([model, md]) =>
      Object.entries(md.resolutions || {}).map(([res, s]) => ({
        _fileId: f.id, model,
        modelLabel: md.label || model,
        steps: md.steps, res,
        sec_mean: s.sec_per_image_mean,
        sec_stdev: s.sec_per_image_stdev,
        n_runs: s.n_runs,
      }))
    )
  );
}

// ── Accuracy (MCQ / Math / Code) ────────────────────────────────────────────

// Return all model keys present in a given accuracy test (mcq/math/code)
// across files, in canonical order — the same LLM roster runs every
// accuracy test, so LLM_MODEL_ORDER applies here too.
export function getAllAccuracyModels(files, testKey) {
  const s = new Set();
  for (const f of files) for (const m of Object.keys(f.data[testKey] || {})) s.add(m);
  const known   = LLM_MODEL_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !LLM_MODEL_ORDER.includes(m));
  return [...known, ...unknown];
}

// Accuracy overall-score bar chart: rows = files/systems, cols = models,
// value = accuracy_pct. A skipped model (crashed repeatedly, no score at
// all) is simply absent from that file's row rather than shown as 0%.
export function buildAccuracyGroupedBarData(files, testKey, enabledModels) {
  const allModels = getAllAccuracyModels(files, testKey).filter(m => enabledModels.has(m));
  return files
    .map(f => {
      const row = { systemLabel: f.hostname };
      for (const model of allModels) {
        const s = f.data[testKey]?.[model];
        if (s && !s.skipped && s.accuracy_pct != null) row[model] = s.accuracy_pct;
      }
      return row;
    })
    .filter(row => allModels.some(m => row[m] != null));
}

// Uses the darker CATEGORY_COLORS palette rather than getModelColor's neon
// MODEL_COLORS — this chart's bars sit side by side as flat color swatches
// (unlike the LLM line charts getModelColor is tuned for), so the pastel
// palette read as washed-out/clashing here.
export function buildAccuracyGroupedBarConfigs(files, testKey, enabledModels) {
  const allModels = getAllAccuracyModels(files, testKey).filter(m => enabledModels.has(m));
  return allModels.map((m, i) => ({
    dataKey: m,
    name: modelLabel(m),
    fill: CATEGORY_COLORS[i % CATEGORY_COLORS.length],
  }));
}

// Every category key a model's by_category breakdown has, across files, for
// one accuracy test — categories vary per test/bank version, so this is
// derived from the data rather than a fixed list, sorted alphabetically for
// a stable chart order.
function getAccuracyCategories(files, testKey, model) {
  const s = new Set();
  for (const f of files)
    for (const cat of Object.keys(f.data[testKey]?.[model]?.by_category || {})) s.add(cat);
  return [...s].sort();
}

// Accuracy per-category chart data for one model: rows = categories, bars = files.
export function buildAccuracyCategoryData(files, testKey, model) {
  const categories = getAccuracyCategories(files, testKey, model);
  return categories.map(cat => {
    const row = { categoryLabel: cat };
    files.forEach((f, fi) => {
      const c = f.data[testKey]?.[model]?.by_category?.[cat];
      if (c) row[`f${fi}`] = c.accuracy_pct;
    });
    return row;
  });
}

export function buildAccuracyCategoryConfigs(files) {
  return files.map((f, fi) => ({
    dataKey: `f${fi}`,
    name: f.hostname,
    fill: FILE_COLORS[fi % FILE_COLORS.length],
  }));
}

// Accuracy timeout/loop-detection diagnostics: one row per (file, model),
// cols = timed_out_count / likely_loop_count (0 for a model with a clean
// run, so it's still visible alongside the ones that had trouble). The whole
// chart (and its EmptyState fallback) only appears when at least one
// model/file actually had a timeout — otherwise it'd always render with
// nothing but zeroes.
export function buildAccuracyTimeoutData(files, testKey, enabledModels) {
  const isMulti = files.length > 1;
  const allModels = getAllAccuracyModels(files, testKey).filter(m => enabledModels.has(m));
  const rows = [];
  let hasIncident = false;
  for (const f of files) {
    for (const model of allModels) {
      const s = f.data[testKey]?.[model];
      const timedOut = s?.timed_out_count || 0;
      const likelyLoop = s?.likely_loop_count || 0;
      if (timedOut || likelyLoop) hasIncident = true;
      rows.push({
        rowLabel: isMulti ? `${f.hostname}\n${modelLabel(model)}` : modelLabel(model),
        timed_out_count: timedOut,
        likely_loop_count: likelyLoop,
      });
    }
  }
  return hasIncident ? rows : [];
}

// ── Concurrency ─────────────────────────────────────────────────────────────

// Return all model keys present in the concurrency section across files, in
// canonical order (same LLM roster as everything else).
export function getAllConcurrencyModels(files) {
  const s = new Set();
  for (const f of files) for (const m of Object.keys(f.data.concurrency || {})) s.add(m);
  const known   = LLM_MODEL_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !LLM_MODEL_ORDER.includes(m));
  return [...known, ...unknown];
}

// Concurrency: one chart per model. X = concurrency level, lines = files.
// metric: "tps" (per-request tokens/sec), "ttft", or "aggregate" (aggregate
// tokens/sec across the whole concurrent batch).
export function buildConcurrencyDataForModel(files, model, metric) {
  const levelSet = new Set();
  for (const f of files)
    for (const level of Object.keys(f.data.concurrency?.[model] || {}))
      if (CONCURRENCY_LEVELS.includes(level)) levelSet.add(level);
  const levels = CONCURRENCY_LEVELS.filter(l => levelSet.has(l));
  return levels.map(level => {
    const row = { levelLabel: `${level}-way` };
    files.forEach((f, fi) => {
      const s = f.data.concurrency?.[model]?.[level];
      if (!s) return;
      row[`f${fi}`] = metric === "tps" ? s.tps_mean
        : metric === "ttft" ? s.ttft_mean_sec
        : s.aggregate_tps;
    });
    return row;
  });
}

// Info about why a (file, model) concurrency sweep stopped climbing before
// CONCURRENCY_LEVELS ran out — null if it wasn't cut short (ran every level,
// or has no concurrency data at all). "slow" stops after recording the level
// that triggered it (a real measurement), the other reasons stop before ever
// recording that level's data, hence nextLevel vs lastLevel below.
export function getConcurrencyStopInfo(file, model) {
  const d = file.data.concurrency?.[model];
  const stoppedAt = d?.stopped_at;
  if (!stoppedAt) return null;
  const presentLevels = CONCURRENCY_LEVELS.filter(l => d[l] != null);
  const lastLevel = presentLevels[presentLevels.length - 1] || null;
  const lastIdx = lastLevel ? CONCURRENCY_LEVELS.indexOf(lastLevel) : -1;
  const nextLevel = stoppedAt === "slow" ? null : (CONCURRENCY_LEVELS[lastIdx + 1] || null);
  return { reason: stoppedAt, label: CONCURRENCY_STOP_LABELS[stoppedAt] || stoppedAt, lastLevel, nextLevel };
}

export function flattenConcurrencyData(files) {
  return files.flatMap(f =>
    Object.entries(f.data.concurrency || {}).flatMap(([model, d]) => {
      if (d?.skipped) {
        return [{
          _fileId: f.id, model, level: "—", skipped: true,
          skip_reason: d.skip_reason, skip_detail: d.skip_detail,
        }];
      }
      return CONCURRENCY_LEVELS.filter(l => d[l]).map(level => {
        const s = d[level];
        return {
          _fileId: f.id, model, level,
          tps_mean: s.tps_mean, tps_stdev: s.tps_stdev,
          aggregate_tps: s.aggregate_tps,
          ttft_mean: s.ttft_mean_sec, ttft_stdev: s.ttft_stdev_sec,
          total_tokens: s.total_tokens,
        };
      });
    })
  );
}

export function flattenAccuracyData(files, testKey) {
  return files.flatMap(f =>
    Object.entries(f.data[testKey] || {}).map(([model, s]) => {
      if (s.skipped) {
        return {
          _fileId: f.id, model, skipped: true,
          skip_reason: s.skip_reason, skip_detail: s.skip_detail,
        };
      }
      return {
        _fileId: f.id, model,
        correct: s.correct, total: s.total, answered: s.answered,
        accuracy_pct: s.accuracy_pct,
        timed_out_count: s.timed_out_count || 0,
        likely_loop_count: s.likely_loop_count || 0,
        crashed: s.crashed || false,
      };
    })
  );
}
