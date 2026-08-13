import type { ChartRow, ResultsFile } from "../types";
import { CTX_ORDER } from "../constants";
import { entriesOf } from "./shared";
import type { JsonRecord } from "./shared";

export const memoryChannelPeak = (sample: JsonRecord[string], channel: string): number | null => {
  const value = sample?.memory?.summary?.[channel]?.peak_gb;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

export const memoryHeadroom = (sample: JsonRecord[string]): number | null => {
  const value = sample?.memory?.headroom?.absolute_gb;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

export const memoryHeadroomState = (sample: JsonRecord[string]): string => {
  const state = sample?.memory?.headroom?.state;
  return typeof state === "string" ? state : "not_recorded";
};

export function buildProcessMemoryDataForModel(
  files: ResultsFile[], model: string, section = "llm",
): ChartRow[] {
  const contexts = new Set<string>();
  for (const file of files) {
    for (const ctx of Object.keys(file.data[section]?.[model] || {})) contexts.add(ctx);
  }
  return CTX_ORDER.filter(ctx => contexts.has(ctx)).map(ctx => {
    const row: ChartRow = { ctxLabel: ctx };
    files.forEach((file, index) => {
      const value = memoryChannelPeak(
        file.data[section]?.[model]?.[ctx], "process_rss_gb",
      );
      if (value != null) row[`f${index}`] = value;
    });
    return row;
  });
}

export function getMemoryRecordingState(file: ResultsFile, section = "llm"): string {
  for (const [, model] of entriesOf(file.data[section])) {
    for (const ctx of CTX_ORDER) {
      if (model?.[ctx]?.memory) return "recorded";
    }
  }
  return "not_recorded";
}

export function runHeadroomSummary(file: ResultsFile): {
  state: string, absoluteGb: number | null, casePath: string | null,
} {
  const tightest = file.data?.run?.memory_summary?.tightest_headroom;
  const absolute = tightest?.absolute_gb;
  return {
    state: typeof tightest?.state === "string" ? tightest.state : "not_recorded",
    absoluteGb: typeof absolute === "number" && Number.isFinite(absolute) ? absolute : null,
    casePath: typeof tightest?.case_path === "string" ? tightest.case_path : null,
  };
}
