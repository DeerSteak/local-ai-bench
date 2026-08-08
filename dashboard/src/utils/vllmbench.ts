import { buildFileLineConfigs, modelLabel, entriesOf } from "./shared";
import type { JsonRecord } from "./shared";
import type { ResultsFile, ChartRow } from "../types";

// `vllm bench` results. Deliberately never share a chart with llamabench: different
// weights (AWQ/GPTQ vs GGUF) and different metric definitions — see docs/workloads.md#vllm-bench.

export function vllmBenchSizeLabel(tokens: number): string {
  const k = (tokens ?? 0) / 1024;
  return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
}

export function vllmBenchLatencyEntries(modelData: JsonRecord[string]): JsonRecord[string][] {
  return Array.isArray(modelData?.latency_entries) ? modelData.latency_entries : [];
}

export function vllmBenchThroughputEntries(modelData: JsonRecord[string]): JsonRecord[string][] {
  return Array.isArray(modelData?.throughput_entries) ? modelData.throughput_entries : [];
}

function orderedInputs(entryGroups: JsonRecord[string][][]): number[] {
  const sizes = new Set<number>();
  for (const entries of entryGroups)
    for (const entry of entries)
      if (entry?.input_len != null) sizes.add(entry.input_len);
  return [...sizes].sort((a, b) => a - b);
}

function buildLineData(
  files: ResultsFile[], model: string,
  pick: (modelData: JsonRecord[string]) => JsonRecord[string][], metric: string,
): ChartRow[] {
  const groups = files.map(file => pick(file.data.vllmbench?.[model]));
  return orderedInputs(groups).map(size => {
    const row: ChartRow = { promptLabel: vllmBenchSizeLabel(size) };
    groups.forEach((entries, fi) => {
      for (const entry of entries) {
        if (entry.input_len === size && entry[metric] != null)
          row[`f${fi}_out${entry.output_len}`] = entry[metric];
      }
    });
    return row;
  });
}

export function buildVllmBenchLatencyData(files: ResultsFile[], model: string): ChartRow[] {
  return buildLineData(files, model, vllmBenchLatencyEntries, "avg_latency_sec");
}

export function buildVllmBenchThroughputData(files: ResultsFile[], model: string): ChartRow[] {
  return buildLineData(files, model, vllmBenchThroughputEntries, "output_tps");
}

// One series per (file, output length), since each output length is its own curve.
function buildLineConfigs(
  files: ResultsFile[], model: string, data: ChartRow[],
  pick: (modelData: JsonRecord[string]) => JsonRecord[string][],
) {
  const configs: { dataKey: string, name: string, stroke?: string }[] = [];
  const fileConfigs = buildFileLineConfigs(files);
  files.forEach((file, fi) => {
    const outputs = new Set<number>(
      pick(file.data.vllmbench?.[model]).map(entry => entry.output_len).filter(v => v != null),
    );
    [...outputs].sort((a, b) => a - b).forEach(output => {
      const dataKey = `f${fi}_out${output}`;
      if (!data.some(row => row[dataKey] != null)) return;
      configs.push({
        dataKey,
        name: files.length > 1
          ? `${fileConfigs[fi]?.name ?? file.hostname} · out${output}`
          : `out${output}`,
        stroke: fileConfigs[fi]?.stroke,
      });
    });
  });
  return configs;
}

export function buildVllmBenchLatencyConfigs(files: ResultsFile[], model: string, data: ChartRow[]) {
  return buildLineConfigs(files, model, data, vllmBenchLatencyEntries);
}

export function buildVllmBenchThroughputConfigs(files: ResultsFile[], model: string, data: ChartRow[]) {
  return buildLineConfigs(files, model, data, vllmBenchThroughputEntries);
}

export function getVllmBenchModels(files: ResultsFile[]): string[] {
  const models = new Set<string>();
  for (const file of files)
    for (const model of Object.keys(file.data.vllmbench || {})) models.add(model);
  return [...models];
}

export function flattenVllmBenchData(files: ResultsFile[]) {
  return files.flatMap(file =>
    entriesOf(file.data.vllmbench).flatMap(([model, modelData]) => {
      const byShape = new Map<string, JsonRecord[string]>();
      for (const entry of vllmBenchLatencyEntries(modelData)) {
        byShape.set(`${entry.input_len}/${entry.output_len}`, {
          _fileId: file.id, model, modelLabel: modelLabel(model),
          input_len: entry.input_len, output_len: entry.output_len,
          avg_latency_sec: entry.avg_latency_sec, latency_output_tps: entry.output_tps,
        });
      }
      for (const entry of vllmBenchThroughputEntries(modelData)) {
        const key = `${entry.input_len}/${entry.output_len}`;
        const row = byShape.get(key) || {
          _fileId: file.id, model, modelLabel: modelLabel(model),
          input_len: entry.input_len, output_len: entry.output_len,
        };
        row.requests_per_sec = entry.requests_per_sec;
        row.throughput_output_tps = entry.output_tps;
        byShape.set(key, row);
      }
      return [...byShape.values()];
    })
  );
}
