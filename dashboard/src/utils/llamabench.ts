import { FILE_COLORS, MODEL_DASH_PATTERNS } from "../constants";
import { buildFileLineConfigs, getModelColor, modelLabel, entriesOf } from "./shared";
import { memoryFields } from "./memory";
import type { JsonRecord } from "./shared";
import type { ChartRow, LineConfig, ResultsFile } from "../types";

export function llamaBenchPromptLabel(tokens: number): string {
  const k = (tokens ?? 0) / 1024;
  return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
}

export function llamaBenchPrefillEntries(modelData: JsonRecord[string]): JsonRecord[string][] {
  if (Array.isArray(modelData?.prefill_entries)) return modelData.prefill_entries;
  return (modelData?.entries || []).filter((entry: JsonRecord[string]) =>
    (entry.n_prompt ?? 0) > 0 && (entry.n_gen ?? 0) === 0);
}

export function llamaBenchDecodeEntries(modelData: JsonRecord[string]): JsonRecord[string][] {
  if (Array.isArray(modelData?.decode_entries)) return modelData.decode_entries;
  return (modelData?.entries || []).filter((entry: JsonRecord[string]) =>
    (entry.n_prompt ?? 0) === 0 && (entry.n_gen ?? 0) > 0 && (entry.n_depth ?? 0) > 0);
}

export function llamaBenchHasCombinedOnly(modelData: JsonRecord[string]): boolean {
  return llamaBenchPrefillEntries(modelData).length === 0
    && llamaBenchDecodeEntries(modelData).length === 0
    && (modelData?.entries || []).some((entry: JsonRecord[string]) =>
      (entry.n_prompt ?? 0) > 0 && (entry.n_gen ?? 0) > 0);
}

function orderedDepths(entryGroups: JsonRecord[string][][], depthKey: string): number[] {
  const depths = new Set<number>();
  for (const entries of entryGroups)
    for (const entry of entries)
      if (entry[depthKey] != null) depths.add(entry[depthKey]);
  return [...depths].sort((a, b) => a - b);
}

export function buildLlamaBenchPrefillLineData(files: ResultsFile[], model: string): ChartRow[] {
  const groups = files.map(file => llamaBenchPrefillEntries(file.data.llamabench?.[model]));
  return orderedDepths(groups, "n_prompt").map(depth => {
    const row: ChartRow = { promptLabel: llamaBenchPromptLabel(depth) };
    groups.forEach((entries, fi) => {
      const entry = entries.find(candidate => candidate.n_prompt === depth);
      if (entry?.avg_ts != null) row[`f${fi}`] = entry.avg_ts;
    });
    return row;
  });
}

export function buildLlamaBenchDecodeLineData(files: ResultsFile[], model: string): ChartRow[] {
  const groups = files.map(file => llamaBenchDecodeEntries(file.data.llamabench?.[model]));
  return orderedDepths(groups, "n_depth").map(depth => {
    const row: ChartRow = { promptLabel: llamaBenchPromptLabel(depth) };
    groups.forEach((entries, fi) => {
      for (const entry of entries) {
        if (entry.n_depth === depth && entry.n_gen != null && entry.avg_ts != null)
          row[`f${fi}_tg${entry.n_gen}`] = entry.avg_ts;
      }
    });
    return row;
  });
}

export function buildLlamaBenchDecodeLineConfigs(files: ResultsFile[], model: string, data: ChartRow[]): LineConfig[] {
  const configs: LineConfig[] = [];
  files.forEach((file, fi) => {
    const tgValues = [...new Set<number>(
      llamaBenchDecodeEntries(file.data.llamabench?.[model]).map(entry => entry.n_gen),
    )].filter(value => value != null).sort((a, b) => a - b);
    tgValues.forEach((tg, ti) => {
      const dataKey = `f${fi}_tg${tg}`;
      if (!data.some(row => row[dataKey] != null)) return;
      configs.push({
        dataKey,
        stroke: FILE_COLORS[fi % FILE_COLORS.length],
        strokeDasharray: MODEL_DASH_PATTERNS[ti % MODEL_DASH_PATTERNS.length],
        name: files.length > 1 ? `${file.hostname} — tg${tg}` : `tg${tg}`,
      });
    });
  });
  return configs;
}

export function buildLlamaBenchPrefillLineDataByModel(file: ResultsFile, models: string[]): ChartRow[] {
  const groups = models.map(model => llamaBenchPrefillEntries(file.data.llamabench?.[model]));
  return orderedDepths(groups, "n_prompt").map(depth => {
    const row: ChartRow = { promptLabel: llamaBenchPromptLabel(depth) };
    groups.forEach((entries, mi) => {
      const entry = entries.find(candidate => candidate.n_prompt === depth);
      if (entry?.avg_ts != null) row[models[mi]] = entry.avg_ts;
    });
    return row;
  });
}

export function buildLlamaBenchDecodeLineDataByModel(file: ResultsFile, models: string[]): ChartRow[] {
  const groups = models.map(model => llamaBenchDecodeEntries(file.data.llamabench?.[model]));
  return orderedDepths(groups, "n_depth").map(depth => {
    const row: ChartRow = { promptLabel: llamaBenchPromptLabel(depth) };
    groups.forEach((entries, mi) => {
      for (const entry of entries) {
        if (entry.n_depth === depth && entry.n_gen != null && entry.avg_ts != null)
          row[`${models[mi]}_tg${entry.n_gen}`] = entry.avg_ts;
      }
    });
    return row;
  });
}

export function buildLlamaBenchPrefillLineConfigsByModel(models: string[], data: ChartRow[]): LineConfig[] {
  return models
    .filter(model => data.some(row => row[model] != null))
    .map(model => ({ dataKey: model, stroke: getModelColor(model), name: modelLabel(model) }));
}

export function buildLlamaBenchDecodeLineConfigsByModel(file: ResultsFile, models: string[], data: ChartRow[]): LineConfig[] {
  const configs: LineConfig[] = [];
  for (const model of models) {
    const tgValues = [...new Set<number>(
      llamaBenchDecodeEntries(file.data.llamabench?.[model]).map(entry => entry.n_gen),
    )].filter(value => value != null).sort((a, b) => a - b);
    tgValues.forEach((tg, ti) => {
      const dataKey = `${model}_tg${tg}`;
      if (!data.some(row => row[dataKey] != null)) return;
      configs.push({
        dataKey,
        stroke: getModelColor(model),
        strokeDasharray: MODEL_DASH_PATTERNS[ti % MODEL_DASH_PATTERNS.length],
        name: `${modelLabel(model)} — tg${tg}`,
      });
    });
  }
  return configs;
}

export function buildLlamaBenchPrefillLineConfigs(files: ResultsFile[], data: ChartRow[]): LineConfig[] {
  return buildFileLineConfigs(files).filter(config => data.some(row => row[config.dataKey] != null));
}

export function flattenLlamaBenchData(files: ResultsFile[]): ChartRow[] {
  return files.flatMap(file =>
    entriesOf(file.data.llamabench).flatMap(([model, modelData]) => {
      if (modelData?.error) {
        return [{ _fileId: file.id, model, metric: "—", skipped: true, skip_detail: modelData.error }];
      }
      const prefill = llamaBenchPrefillEntries(modelData).map(entry => ({
        _fileId: file.id, model, metric: "Prefill", pp: entry.n_prompt ?? null, tg: null as number | null,
        avg_ts: entry.avg_ts, stddev_ts: entry.stddev_ts, n_gpu_layers: entry.n_gpu_layers,
        ...memoryFields(entry),
      }));
      const decode = llamaBenchDecodeEntries(modelData).map(entry => ({
        _fileId: file.id, model, metric: "Decode", pp: entry.n_depth ?? null, tg: entry.n_gen ?? null,
        avg_ts: entry.avg_ts, stddev_ts: entry.stddev_ts, n_gpu_layers: entry.n_gpu_layers,
        ...memoryFields(entry),
      }));
      if (prefill.length || decode.length) return [...prefill, ...decode];
      return (modelData?.entries || []).map((entry: JsonRecord[string]) => ({
        _fileId: file.id, model, metric: "Combined",
        pp: entry.n_prompt ?? null, tg: entry.n_gen ?? null,
        avg_ts: entry.avg_ts, stddev_ts: entry.stddev_ts, n_gpu_layers: entry.n_gpu_layers,
        ...memoryFields(entry),
      }));
    })
  );
}
