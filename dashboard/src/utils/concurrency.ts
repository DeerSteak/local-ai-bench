import { CONCURRENCY_LEVELS, CONCURRENCY_STOP_LABELS, LLM_DISPLAY_ORDER } from "../constants";
import { entriesOf } from "./shared";

const ttftMean = sample => sample?.client_ttft_mean_sec ?? sample?.ttft_mean_sec;

// `section` is "concurrency_tool" or "concurrency_chat" — the two
// concurrency tests share this shape but have different level ladders (see
// CONCURRENCY_LEVELS in constants.js) and live under separate results JSON
// keys.

// Return all model keys present in a concurrency section across files, in
// canonical order (same LLM roster as everything else).
export function getAllConcurrencyModels(files, section) {
  const s = new Set<string>();
  for (const f of files) for (const m of Object.keys(f.data[section] || {})) s.add(m);
  const known   = LLM_DISPLAY_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !LLM_DISPLAY_ORDER.includes(m));
  return [...known, ...unknown];
}

// Concurrency: one chart per model. X = concurrency level, lines = files.
// metric: "tps" (per-request tokens/sec), "ttft", or "aggregate" (aggregate
// tokens/sec across the whole concurrent batch).
export function buildConcurrencyDataForModel(files, section, model, metric) {
  const allLevels = CONCURRENCY_LEVELS[section];
  const levelSet = new Set<string>();
  for (const f of files)
    for (const level of Object.keys(f.data[section]?.[model] || {}))
      if (allLevels.includes(level)) levelSet.add(level);
  const levels = allLevels.filter(l => levelSet.has(l));
  return levels.map(level => {
    const row = { levelLabel: `${level}-way` };
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
export function getConcurrencyStopInfo(file, section, model) {
  const allLevels = CONCURRENCY_LEVELS[section];
  const d = file.data[section]?.[model];
  const stoppedAt = d?.stopped_at;
  if (!stoppedAt) return null;
  const presentLevels = allLevels.filter(l => d[l] != null);
  const lastLevel = presentLevels[presentLevels.length - 1] || null;
  const lastIdx = lastLevel ? allLevels.indexOf(lastLevel) : -1;
  const nextLevel = stoppedAt === "slow" ? null : (allLevels[lastIdx + 1] || null);
  return { reason: stoppedAt, label: CONCURRENCY_STOP_LABELS[stoppedAt] || stoppedAt, lastLevel, nextLevel };
}

export function flattenConcurrencyData(files, section) {
  const allLevels = CONCURRENCY_LEVELS[section];
  return files.flatMap(f =>
    entriesOf(f.data[section]).flatMap(([model, d]) => {
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
export function concurrencySortValue(row, key) {
  if (key !== "level") return row[key] ?? "";
  const n = Number(row.level);
  return Number.isNaN(n) ? Infinity : n;
}
