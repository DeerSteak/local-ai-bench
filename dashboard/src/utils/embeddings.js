import { FALLBACK_COLORS, EMBED_MODEL_ORDER, EMBED_BAR_COLORS } from "../constants";
import { embedModelLabel } from "./shared";

// Return all embedding model keys from the loaded files, in canonical order
export function getAllEmbedModels(files) {
  const s = new Set();
  for (const f of files) for (const m of Object.keys(f.data.embeddings || {})) s.add(m);
  const known   = EMBED_MODEL_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !EMBED_MODEL_ORDER.includes(m));
  return [...known, ...unknown];
}

export function getEmbedLabel(files, model) {
  for (const f of files) {
    const d = f.data.embeddings?.[model];
    if (d?.label) return d.label;
  }
  return embedModelLabel(model);
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
