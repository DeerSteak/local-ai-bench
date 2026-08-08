// llama-batched-bench: aggregate decode throughput (speed_tg) vs. parallel-sequence count (pl).
// Levels come from each entry's own pl, not a constants list — fit_npl drops levels per-model.
import { entriesOf } from "./shared";
import type { JsonRecord } from "./shared";
import type { ResultsFile, ChartRow } from "../types";

// Distinct generation sizes (tg) present for `model` across files, sorted ascending —
// one chart per tg keeps concurrency level as the chart's only axis dimension.
export function llamaBenchConcTgValues(files: ResultsFile[], model: string): number[] {
  const set = new Set<number>();
  for (const f of files)
    for (const entry of f.data.llamabenchconc?.[model]?.entries || [])
      set.add(entry.tg ?? 0);
  return [...set].sort((a, b) => a - b);
}

export function llamaBenchConcLevels(files: ResultsFile[], model: string, tg: number): number[] {
  const set = new Set<number>();
  for (const f of files)
    for (const entry of f.data.llamabenchconc?.[model]?.entries || [])
      if ((entry.tg ?? 0) === tg && entry.pl != null) set.add(entry.pl);
  return [...set].sort((a, b) => a - b);
}

// One chart per (model, tg). X = concurrency level, lines = files — same
// "N-way" level labelling the conc_tool/conc_chat charts use.
export function buildLlamaBenchConcLineData(files: ResultsFile[], model: string, tg: number): ChartRow[] {
  return llamaBenchConcLevels(files, model, tg).map(level => {
    const row: ChartRow = { levelLabel: `${level}-way` };
    files.forEach((f, fi) => {
      const entry = (f.data.llamabenchconc?.[model]?.entries || [])
        .find((e: JsonRecord[string]) => (e.tg ?? 0) === tg && e.pl === level);
      if (entry?.speed_tg != null) row[`f${fi}`] = entry.speed_tg;
    });
    return row;
  });
}

// Prompt depth actually swept for a model in one file — fit_npl clamps it on
// short-context models, so it's worth showing rather than assuming the config value.
export function llamaBenchConcPromptDepth(file: ResultsFile, model: string): number | null {
  return file.data.llamabenchconc?.[model]?.pp ?? null;
}

export function flattenLlamaBenchConcData(files: ResultsFile[]) {
  return files.flatMap(f =>
    entriesOf(f.data.llamabenchconc).flatMap(([model, modelData]) => {
      if (modelData?.error) {
        return [{ _fileId: f.id, model, level: "—", skipped: true, skip_detail: modelData.error }];
      }
      return (modelData?.entries || []).map((entry: JsonRecord[string]) => ({
        _fileId: f.id, model,
        level: entry.pl ?? null,
        pp: entry.pp ?? modelData?.pp ?? null,
        tg: entry.tg ?? null,
        speed_tg: entry.speed_tg,
        speed_pp: entry.speed_pp,
        speed: entry.speed,
      }));
    })
  );
}

// `level` is numeric, so it must sort numerically; a skipped row's "—" is pinned to
// +Infinity for comparator consistency — same rule as concurrency.js's own sort value.
export function llamaBenchConcSortValue(row: ChartRow, key: string) {
  if (key !== "level") return row[key] ?? "";
  const n = Number(row.level);
  return Number.isNaN(n) ? Infinity : n;
}
