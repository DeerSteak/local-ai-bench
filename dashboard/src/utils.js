import {
  CTX_ORDER, EMBED_BATCH_KEYS, EMBED_BATCH_LABELS, RES_ORDER,
  MODEL_COLORS, IMAGE_MODEL_COLORS, FALLBACK_COLORS,
  FILE_COLORS, MODEL_DASH_PATTERNS,
  LLM_MODEL_LABELS, IMAGE_MODEL_LABELS, LLM_MODEL_ORDER, IMAGE_MODEL_ORDER,
  CTX_COLORS, BATCH_COLORS, IMAGE_BAR_COLORS, RES_COLORS, MODEL_SIZE_TIER,
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

export function modelLabel(model) {
  return LLM_MODEL_LABELS[model] || model;
}

export function imageModelLabel(model) {
  return IMAGE_MODEL_LABELS[model] || model;
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

// Return all LLM model keys from the loaded files, in canonical order.
// Checks both the single-shot and conversation LLM sections, since the
// Models filter is shared UI across both — a model present in only one of
// them should still show up and not disappear when switching sections.
export function getAllLLMModels(files) {
  const s = new Set();
  for (const f of files) {
    for (const m of Object.keys(f.data.llm || {})) s.add(m);
    for (const m of Object.keys(f.data.llm_conversation || {})) s.add(m);
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

// Embeddings: X = batch size, lines = files
export function buildEmbedData(files) {
  return EMBED_BATCH_KEYS
    .map(bk => {
      const row = { batchLabel: EMBED_BATCH_LABELS[bk] };
      files.forEach((f, fi) => {
        const stats = (f.data.embeddings || {})[bk];
        if (stats) row[`f${fi}`] = stats.sentences_per_sec_mean;
      });
      return row;
    })
    .filter(row => Object.keys(row).some(k => k !== "batchLabel"));
}

export function buildEmbedLineConfigs(files) {
  return buildFileLineConfigs(files);
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
    }
    return row;
  });
}

export function buildLLMBarConfigs(files, model, section = "llm") {
  const ctxSet = new Set();
  for (const f of files)
    for (const ctx of Object.keys(f.data[section]?.[model] || {})) ctxSet.add(ctx);
  return CTX_ORDER
    .filter(ctx => ctxSet.has(ctx))
    .map((ctx, i) => ({
      dataKey: ctx,
      name: ctx,
      fill: CTX_COLORS[ctx] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// Embeddings bar chart: rows = files/systems, cols = batch sizes
export function buildEmbedBarData(files) {
  return files.map(f => {
    const row = { systemLabel: f.hostname };
    const embedData = f.data.embeddings || {};
    for (const bk of EMBED_BATCH_KEYS) {
      const s = embedData[bk];
      if (s) row[bk] = s.sentences_per_sec_mean;
    }
    return row;
  });
}

export function buildEmbedBarConfigs(files) {
  const present = new Set();
  for (const f of files)
    for (const bk of Object.keys(f.data.embeddings || {})) present.add(bk);
  return EMBED_BATCH_KEYS
    .filter(bk => present.has(bk))
    .map((bk, i) => ({
      dataKey: bk,
      name: EMBED_BATCH_LABELS[bk],
      fill: BATCH_COLORS[bk] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
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
      }
      return row;
    })
    .filter(row => allModels.some(m => row[m] != null));
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
    }
    return row;
  });
}

export function buildLLMBarConfigsByModel(file, models, section = "llm") {
  const ctxSet = new Set();
  for (const model of models)
    for (const ctx of Object.keys(file.data[section]?.[model] || {})) ctxSet.add(ctx);
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
      }
      return row;
    })
    .filter(row => RES_ORDER.some(res => row[res] != null));
}

export function buildImagesBarConfigsByModel(file, models) {
  const resSet = new Set();
  for (const model of models)
    for (const res of Object.keys(file.data.images?.[model]?.resolutions || {})) resSet.add(res);
  return RES_ORDER
    .filter(res => resSet.has(res))
    .map((res, i) => ({
      dataKey: res,
      name: res,
      fill: RES_COLORS[res] || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// Embeddings bar chart by system: single "Embeddings" row, cols = batch sizes, for one file
export function buildEmbedBarDataByFile(file) {
  const row = { modelLabel: "Embeddings" };
  const embedData = file.data.embeddings || {};
  for (const bk of EMBED_BATCH_KEYS) {
    const s = embedData[bk];
    if (s) row[bk] = s.sentences_per_sec_mean;
  }
  return [row];
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

// Embeddings line chart by system: rows = batch sizes, single "Embeddings" line, for one file
export function buildEmbedLineDataByBatch(file) {
  const embedData = file.data.embeddings || {};
  return EMBED_BATCH_KEYS
    .filter(bk => embedData[bk])
    .map(bk => ({ batchLabel: EMBED_BATCH_LABELS[bk], value: embedData[bk].sentences_per_sec_mean }));
}

export function buildEmbedLineConfigByBatch() {
  return [{ dataKey: "value", stroke: BATCH_COLORS.batch_32, name: "Embeddings" }];
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
    Object.entries(f.data[section] || {}).flatMap(([model, ctxData]) =>
      Object.entries(ctxData)
        .filter(([ctx]) => CTX_ORDER.includes(ctx))
        .map(([ctx, s]) => ({
          _fileId: f.id, model, ctx,
          tps_mean: s.tps_mean, tps_stdev: s.tps_stdev,
          ttft_mean: s.ttft_mean_sec, ttft_stdev: s.ttft_stdev_sec,
          n_runs: s.n_runs,
        }))
    )
  );
}

export function flattenEmbedData(files) {
  return files.flatMap(f =>
    Object.entries(f.data.embeddings || {}).map(([bk, s]) => ({
      _fileId: f.id, batchKey: bk,
      batchLabel: EMBED_BATCH_LABELS[bk] || bk,
      sps_mean: s.sentences_per_sec_mean,
      sps_stdev: s.sentences_per_sec_stdev,
      peak_ram_mb: s.peak_ram_mb_mean,
      device: s.device,
      n_runs: s.n_runs,
    }))
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
