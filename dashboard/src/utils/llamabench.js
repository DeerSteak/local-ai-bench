import { FALLBACK_COLORS } from "../constants";
import { getModelColor, modelLabel } from "./shared";

// pp/tg sizes are a config.py constant that can change, so order is derived from
// each entry's own n_prompt/n_gen — unlike CTX_ORDER, not a hardcoded list.
export function llamaBenchCheckpointKey(entry) {
  const p = entry.n_prompt ?? 0, g = entry.n_gen ?? 0;
  if (g === 0) return `pp${p}`;
  if (p === 0) return `tg${g}`;
  return `pp${p}+tg${g}`;
}

export function llamaBenchCheckpointSortValue(entry) {
  return (entry.n_prompt ?? 0) * 1e7 + (entry.n_gen ?? 0);
}

// llama-bench pp sizes are always multiples of 1024 — they're drawn from CONTEXT_LENGTHS
// and LLMConversationBenchmark.CONV_CHECKPOINTS, both binary-K depths — so a plain "2K"/"0.5K"
// label (matching CTX_ORDER's own naming) is unambiguous without needing the tg suffix that
// llamaBenchCheckpointKey uses for the flat Raw Numbers table.
export function llamaBenchPromptLabel(nPrompt) {
  const k = (nPrompt ?? 0) / 1024;
  return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
}

// Distinct generation sizes (tg) present for `model` across files, sorted ascending — splitting
// a chart per tg value keeps pp size as its only axis/bar dimension instead of a combined
// "pp2048+tg128" label that stops fitting once there are more than a couple of pp sizes.
export function llamaBenchTgValues(files, model) {
  const set = new Set();
  for (const f of files)
    for (const entry of f.data.llamabench?.[model]?.entries || [])
      set.add(entry.n_gen ?? 0);
  return [...set].sort((a, b) => a - b);
}

export function llamaBenchTgValuesByModel(file, models) {
  const set = new Set();
  for (const model of models)
    for (const entry of file.data.llamabench?.[model]?.entries || [])
      set.add(entry.n_gen ?? 0);
  return [...set].sort((a, b) => a - b);
}

// llama-bench: one chart per (model, tg). X = system, bars = each pp size present at that tg.
export function buildLlamaBenchBarData(files, model, tg) {
  return files.map(f => {
    const row = { systemLabel: f.hostname };
    for (const entry of f.data.llamabench?.[model]?.entries || []) {
      if ((entry.n_gen ?? 0) !== tg) continue;
      row[llamaBenchPromptLabel(entry.n_prompt)] = entry.avg_ts;
    }
    return row;
  });
}

export function buildLlamaBenchBarConfigs(files, model, tg) {
  const sortValueByKey = new Map();
  for (const f of files)
    for (const entry of f.data.llamabench?.[model]?.entries || []) {
      if ((entry.n_gen ?? 0) !== tg) continue;
      sortValueByKey.set(llamaBenchPromptLabel(entry.n_prompt), entry.n_prompt ?? 0);
    }
  return [...sortValueByKey.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([key], i) => ({ dataKey: key, name: key, fill: FALLBACK_COLORS[i % FALLBACK_COLORS.length] }));
}

// llama-bench: one chart per (model, tg). X = pp size, lines = files — the same
// pp keys/ordering as the bar chart, just transposed for the Line chart style.
export function buildLlamaBenchLineData(files, model, tg) {
  const sortValueByKey = new Map();
  for (const f of files)
    for (const entry of f.data.llamabench?.[model]?.entries || []) {
      if ((entry.n_gen ?? 0) !== tg) continue;
      sortValueByKey.set(llamaBenchPromptLabel(entry.n_prompt), entry.n_prompt ?? 0);
    }
  const orderedKeys = [...sortValueByKey.entries()].sort((a, b) => a[1] - b[1]).map(([key]) => key);

  return orderedKeys.map(key => {
    const row = { promptLabel: key };
    files.forEach((f, fi) => {
      const entry = (f.data.llamabench?.[model]?.entries || [])
        .find(e => (e.n_gen ?? 0) === tg && llamaBenchPromptLabel(e.n_prompt) === key);
      if (entry) row[`f${fi}`] = entry.avg_ts;
    });
    return row;
  });
}

// llama-bench "by system" bar chart: rows = models, cols = pp sizes, for one file/tg —
// mirrors llm.js's buildLLMBarDataByModel shape.
export function buildLlamaBenchBarDataByModel(file, models, tg) {
  return models.map(model => {
    const row = { modelLabel: modelLabel(model) };
    for (const entry of file.data.llamabench?.[model]?.entries || []) {
      if ((entry.n_gen ?? 0) !== tg) continue;
      row[llamaBenchPromptLabel(entry.n_prompt)] = entry.avg_ts;
    }
    return row;
  });
}

export function buildLlamaBenchBarConfigsByModel(file, models, tg) {
  const sortValueByKey = new Map();
  for (const model of models)
    for (const entry of file.data.llamabench?.[model]?.entries || []) {
      if ((entry.n_gen ?? 0) !== tg) continue;
      sortValueByKey.set(llamaBenchPromptLabel(entry.n_prompt), entry.n_prompt ?? 0);
    }
  return [...sortValueByKey.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([key], i) => ({ dataKey: key, name: key, fill: FALLBACK_COLORS[i % FALLBACK_COLORS.length] }));
}

// llama-bench "by system" line chart: rows = pp sizes, one line per model, for one file/tg —
// mirrors llm.js's buildLLMLineDataByCtx shape.
export function buildLlamaBenchLineDataByCheckpoint(file, models, tg) {
  const sortValueByKey = new Map();
  for (const model of models)
    for (const entry of file.data.llamabench?.[model]?.entries || []) {
      if ((entry.n_gen ?? 0) !== tg) continue;
      sortValueByKey.set(llamaBenchPromptLabel(entry.n_prompt), entry.n_prompt ?? 0);
    }
  const orderedKeys = [...sortValueByKey.entries()].sort((a, b) => a[1] - b[1]).map(([key]) => key);

  return orderedKeys.map(key => {
    const row = { promptLabel: key };
    for (const model of models) {
      const entry = (file.data.llamabench?.[model]?.entries || [])
        .find(e => (e.n_gen ?? 0) === tg && llamaBenchPromptLabel(e.n_prompt) === key);
      if (entry) row[model] = entry.avg_ts;
    }
    return row;
  });
}

export function buildLlamaBenchLineConfigsByCheckpoint(models, data) {
  return models
    .filter(m => data.some(row => row[m] != null))
    .map(m => ({ dataKey: m, stroke: getModelColor(m), name: modelLabel(m) }));
}

export function flattenLlamaBenchData(files) {
  return files.flatMap(f =>
    Object.entries(f.data.llamabench || {}).flatMap(([model, modelData]) => {
      if (modelData?.error) {
        return [{ _fileId: f.id, model, ckpt: "—", skipped: true, skip_detail: modelData.error }];
      }
      return (modelData?.entries || []).map(entry => ({
        _fileId: f.id, model, ckpt: llamaBenchCheckpointKey(entry),
        avg_ts: entry.avg_ts, stddev_ts: entry.stddev_ts,
        n_gpu_layers: entry.n_gpu_layers,
      }));
    })
  );
}
