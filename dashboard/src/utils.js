import {
  CTX_ORDER, EMBED_BATCH_KEYS, EMBED_BATCH_LABELS, RES_ORDER,
  MODEL_COLORS, IMAGE_MODEL_COLORS, FALLBACK_COLORS,
  FILE_COLORS, MODEL_DASH_PATTERNS,
  LLM_MODEL_LABELS, IMAGE_MODEL_LABELS, LLM_MODEL_ORDER, IMAGE_MODEL_ORDER,
} from "./constants";

export function parseJSON(text) {
  try { return JSON.parse(text); } catch { return null; }
}

export function fmt(v, unit) {
  if (v == null) return "—";
  switch (unit) {
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

// Return all LLM model keys from the loaded files, in canonical order
export function getAllLLMModels(files) {
  const s = new Set();
  for (const f of files) for (const m of Object.keys(f.data.llm || {})) s.add(m);
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
export function buildLLMDataForModel(files, model, metric) {
  const ctxSet = new Set();
  for (const f of files)
    for (const ctx of Object.keys(f.data.llm?.[model] || {})) ctxSet.add(ctx);
  const ctxLabels = CTX_ORDER.filter(c => ctxSet.has(c));
  return ctxLabels.map(ctx => {
    const row = { ctxLabel: ctx };
    files.forEach((f, fi) => {
      const s = f.data.llm?.[model]?.[ctx];
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

// ── Flat data for StatsTable ───────────────────────────────────────────────────

export function flattenLLMData(files) {
  return files.flatMap(f =>
    Object.entries(f.data.llm || {}).flatMap(([model, ctxData]) =>
      Object.entries(ctxData).map(([ctx, s]) => ({
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
