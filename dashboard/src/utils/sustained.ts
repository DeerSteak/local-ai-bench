import { entriesOf } from "./shared";
import type { ChartRow, ResultsFile } from "../types";
import type { JsonRecord } from "./shared";

export const SUSTAINED_TEMPERATURE_KEYS = ["cpu_package_c", "gpu_die_c", "gpu_hotspot_c"] as const;

export function sustainedSeries(modelData: JsonRecord[string]): JsonRecord[string][] {
  return Array.isArray(modelData?.series) ? modelData.series : [];
}

export function buildSustainedTimeline(modelData: JsonRecord[string]): ChartRow[] {
  return sustainedSeries(modelData).map(window => ({
    elapsed_min: typeof window.timestamp_sec === "number" ? window.timestamp_sec / 60 : null,
    tokens_per_sec: window.tokens_per_sec,
    power_watts: window.power_watts,
    cpu_package_c: window.cpu_package_c,
    gpu_die_c: window.gpu_die_c,
    gpu_hotspot_c: window.gpu_hotspot_c,
  }));
}

export function preferredTemperatureKey(data: ChartRow[]): typeof SUSTAINED_TEMPERATURE_KEYS[number] | null {
  return SUSTAINED_TEMPERATURE_KEYS.find(key => data.some(row => typeof row[key] === "number")) ?? null;
}

export function flattenSustainedData(files: ResultsFile[]): ChartRow[] {
  return files.flatMap(file => entriesOf(file.data.sustained).map(([model, result]) => ({
    _fileId: file.id,
    model,
    initial_tokens_per_sec: result?.analysis?.initial_tokens_per_sec,
    steady_state_tokens_per_sec: result?.analysis?.steady_state_tokens_per_sec,
    retention_pct: typeof result?.analysis?.retention_ratio === "number"
      ? result.analysis.retention_ratio * 100 : null,
    throttle_onset_sec: result?.analysis?.throttle_onset_sec,
    performance: result?.analysis?.performance,
    cause: result?.analysis?.cause,
    actual_duration_sec: result?.actual_duration_sec,
    request_count: result?.request_count,
    valid_request_count: result?.valid_request_count,
    ambient_temp_c: result?.ambient_temp_c,
    pause_invalidated: result?.pause_invalidated,
    skipped: result?.skipped,
    skip_detail: result?.skipped,
    error: result?.unexpected_error ?? result?.error ?? result?.crashed,
  })));
}
