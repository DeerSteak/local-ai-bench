import { CONCURRENCY_LEVELS, CONCURRENCY_STOP_LABELS, LLM_DISPLAY_ORDER } from "../constants";
import { entriesOf, lookup } from "./shared";
import type { JsonRecord } from "./shared";
import type { ResultsFile, ChartRow } from "../types";

const ttftMean = (sample: JsonRecord[string]) => sample?.client_ttft_mean_sec ?? sample?.ttft_mean_sec;

// `section` is "concurrency_tool" or "concurrency_chat" — the two
// concurrency tests share this shape but have different level ladders (see
// CONCURRENCY_LEVELS in constants.js) and live under separate results JSON
// keys.

// Return all model keys present in a concurrency section across files, in
// canonical order (same LLM roster as everything else).
export function getAllConcurrencyModels(files: ResultsFile[], section: string): string[] {
  const s = new Set<string>();
  for (const f of files) for (const m of Object.keys(f.data[section] || {})) s.add(m);
  const known   = LLM_DISPLAY_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !LLM_DISPLAY_ORDER.includes(m));
  return [...known, ...unknown];
}

// Concurrency: one chart per model. X = concurrency level, lines = files.
// metric: "tps" (per-request tokens/sec), "ttft", or "aggregate" (aggregate
// tokens/sec across the whole concurrent batch).
export function buildConcurrencyDataForModel(files: ResultsFile[], section: string, model: string, metric: string): ChartRow[] {
  const allLevels: string[] = CONCURRENCY_LEVELS[section as keyof typeof CONCURRENCY_LEVELS];
  const levelSet = new Set<string>();
  for (const f of files)
    for (const level of Object.keys(f.data[section]?.[model] || {}))
      if (allLevels.includes(level)) levelSet.add(level);
  const levels = allLevels.filter(l => levelSet.has(l));
  return levels.map(level => {
    const row: ChartRow = { levelLabel: `${level}-way` };
    files.forEach((f, fi) => {
      const s = f.data[section]?.[model]?.[level];
      if (!s) return;
      row[`f${fi}`] = metric === "tps" ? s.tps_mean
        : metric === "ttft" ? ttftMean(s)
        : s.aggregate_tps;
    });
    return row;
  });
}

// Info about why a (file, model) concurrency sweep stopped climbing before
// its level ladder ran out — null if it wasn't cut short (ran every level,
// or has no concurrency data at all). "slow" stops after recording the level
// that triggered it (a real measurement), the other reasons stop before ever
// recording that level's data, hence nextLevel vs lastLevel below.
export function getConcurrencyStopInfo(file: ResultsFile, section: string, model: string) {
  const allLevels: string[] = CONCURRENCY_LEVELS[section as keyof typeof CONCURRENCY_LEVELS];
  const d = file.data[section]?.[model];
  const stoppedAt = d?.stopped_at;
  if (!stoppedAt) return null;
  const presentLevels = allLevels.filter(l => d[l] != null);
  const lastLevel = presentLevels[presentLevels.length - 1] || null;
  const lastIdx = lastLevel ? allLevels.indexOf(lastLevel) : -1;
  const nextLevel = stoppedAt === "slow" ? null : (allLevels[lastIdx + 1] || null);
  return { reason: stoppedAt, label: lookup(CONCURRENCY_STOP_LABELS, stoppedAt) || stoppedAt, lastLevel, nextLevel };
}

export function getConcurrencySweetSpot(file: ResultsFile, section: string, model: string) {
  const levels: string[] = CONCURRENCY_LEVELS[section as keyof typeof CONCURRENCY_LEVELS];
  const data = file.data[section]?.[model];
  const candidates = levels
    .map(level => ({ level, aggregateTps: data?.[level]?.aggregate_tps }))
    .filter((entry): entry is { level: string, aggregateTps: number } =>
      typeof entry.aggregateTps === "number" && Number.isFinite(entry.aggregateTps));
  if (!candidates.length) return null;
  const best = candidates.reduce((winner, entry) =>
    entry.aggregateTps > winner.aggregateTps ? entry : winner);
  const baseTps = Number(data?.[levels[0]]?.tps_mean);
  const bestTps = Number(data?.[best.level]?.tps_mean);
  const sacrificePct = Number.isFinite(baseTps) && baseTps > 0 && Number.isFinite(bestTps)
    ? Math.max(0, (1 - bestTps / baseTps) * 100)
    : null;
  return { ...best, sacrificePct };
}

export function flattenConcurrencyData(files: ResultsFile[], section: string) {
  const allLevels: string[] = CONCURRENCY_LEVELS[section as keyof typeof CONCURRENCY_LEVELS];
  return files.flatMap(f =>
    entriesOf(f.data[section]).flatMap(([model, d]): ChartRow[] => {
      if (d?.skipped) {
        return [{
          _fileId: f.id, model, level: "—", skipped: true,
          skip_reason: d.skip_reason, skip_detail: d.skip_detail,
        }];
      }
      return allLevels.filter(l => d[l]).map(level => {
        const s = d[level];
        return {
          _fileId: f.id, model, level,
          tps_mean: s.tps_mean, tps_stdev: s.tps_stdev,
          aggregate_tps: s.aggregate_tps,
          ttft_mean: ttftMean(s),
          ttft_stdev: s.client_ttft_stdev_sec ?? s.ttft_stdev_sec,
          total_tokens: s.total_tokens,
        };
      });
    })
  );
}

// level is a numeric-string sweep value ("1".."32"), so it must sort
// numerically rather than lexicographically ("12" before "2"). Every other
// column sorts by its raw value, same as the other stats tables. A skipped
// row's level is "—" (non-numeric) — Number(NaN) compares false in both
// directions, which breaks comparator consistency, so it's pinned to
// +Infinity: last ascending and first descending.
export function concurrencySortValue(row: ChartRow, key: string) {
  if (key !== "level") return row[key] ?? "";
  const n = Number(row.level);
  return Number.isNaN(n) ? Infinity : n;
}
