import { LLM_DISPLAY_ORDER, CATEGORY_COLORS, FILE_COLORS } from "../constants";
import { modelLabel, entriesOf } from "./shared";
import { memoryFields } from "./memory";
import type { BarConfig, ChartRow, ResultsFile } from "../types";

// Return all model keys present in a given accuracy test (mcq/math/code)
// across files, in canonical order — the same LLM roster runs every
// accuracy test, so the current-plus-legacy display order applies here too.
export function getAllAccuracyModels(files: ResultsFile[], testKey: string): string[] {
  const s = new Set<string>();
  for (const f of files) for (const m of Object.keys(f.data[testKey] || {})) s.add(m);
  const known   = LLM_DISPLAY_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !LLM_DISPLAY_ORDER.includes(m));
  return [...known, ...unknown];
}

// Accuracy overall-score bar chart: rows = files/systems, cols = models,
// value = accuracy_pct. A skipped model (crashed repeatedly, no score at
// all) is simply absent from that file's row rather than shown as 0%.
export function buildAccuracyGroupedBarData(files: ResultsFile[], testKey: string, enabledModels: Set<string>): ChartRow[] {
  const allModels = getAllAccuracyModels(files, testKey).filter(m => enabledModels.has(m));
  return files
    .map(f => {
      const row: ChartRow = { systemLabel: f.hostname };
      for (const model of allModels) {
        const s = f.data[testKey]?.[model];
        if (s && !s.skipped && s.accuracy_pct != null) row[model] = s.accuracy_pct;
      }
      return row;
    })
    .filter(row => allModels.some(m => row[m] != null));
}

// Uses the darker CATEGORY_COLORS palette rather than getModelColor's neon
// MODEL_COLORS — this chart's bars sit side by side as flat color swatches
// (unlike the LLM line charts getModelColor is tuned for), so the pastel
// palette read as washed-out/clashing here.
export function buildAccuracyGroupedBarConfigs(files: ResultsFile[], testKey: string, enabledModels: Set<string>): BarConfig[] {
  const allModels = getAllAccuracyModels(files, testKey).filter(m => enabledModels.has(m));
  return allModels.map((m, i) => ({
    dataKey: m,
    name: modelLabel(m),
    fill: CATEGORY_COLORS[i % CATEGORY_COLORS.length],
  }));
}

// Every category key a model's by_category breakdown has, across files, for
// one accuracy test — categories vary per test/bank version, so this is
// derived from the data rather than a fixed list, sorted alphabetically for
// a stable chart order.
function getAccuracyCategories(files: ResultsFile[], testKey: string, model: string): string[] {
  const s = new Set<string>();
  for (const f of files)
    for (const cat of Object.keys(f.data[testKey]?.[model]?.by_category || {})) s.add(cat);
  return [...s].sort();
}

// Accuracy per-category chart data for one model: rows = categories, bars = files.
export function buildAccuracyCategoryData(files: ResultsFile[], testKey: string, model: string): ChartRow[] {
  const categories = getAccuracyCategories(files, testKey, model);
  return categories.map(cat => {
    const row: ChartRow = { categoryLabel: cat };
    files.forEach((f, fi) => {
      const c = f.data[testKey]?.[model]?.by_category?.[cat];
      if (c) row[`f${fi}`] = c.accuracy_pct;
    });
    return row;
  });
}

export function buildAccuracyCategoryConfigs(files: ResultsFile[]): BarConfig[] {
  return files.map((f, fi) => ({
    dataKey: `f${fi}`,
    name: f.hostname ?? "Unknown",
    fill: FILE_COLORS[fi % FILE_COLORS.length],
  }));
}

const DIFFICULTY_ORDER = ["easy", "medium", "hard", "very_hard"];

export function buildAccuracyDifficultyData(files: ResultsFile[], testKey: string, model: string): ChartRow[] {
  const found = new Set<string>();
  for (const f of files)
    for (const difficulty of Object.keys(f.data[testKey]?.[model]?.by_difficulty || {}))
      found.add(difficulty);
  const difficulties = [
    ...DIFFICULTY_ORDER.filter(difficulty => found.has(difficulty)),
    ...[...found].filter(difficulty => !DIFFICULTY_ORDER.includes(difficulty)).sort(),
  ];
  return difficulties.map(difficulty => {
    const row: ChartRow = {
      difficultyLabel: difficulty.replaceAll("_", " ").replace(/^./, c => c.toUpperCase()),
    };
    files.forEach((f, fi) => {
      const score = f.data[testKey]?.[model]?.by_difficulty?.[difficulty];
      if (score) row[`f${fi}`] = score.accuracy_pct;
    });
    return row;
  });
}

// Accuracy cutoff diagnostics: one row per (file, model),
// cols = timeout / loop / exhausted budget (0 for a model with a clean
// run, so it's still visible alongside the ones that had trouble). The whole
// chart (and its EmptyState fallback) only appears when at least one
// model/file had either incident — otherwise it'd always render with
// nothing but zeroes.
export function buildAccuracyTimeoutData(files: ResultsFile[], testKey: string, enabledModels: Set<string>): ChartRow[] {
  const isMulti = files.length > 1;
  const allModels = getAllAccuracyModels(files, testKey).filter(m => enabledModels.has(m));
  const rows: ChartRow[] = [];
  let hasIncident = false;
  for (const f of files) {
    for (const model of allModels) {
      const s = f.data[testKey]?.[model];
      const timedOut = s?.timed_out_count || 0;
      const likelyLoop = s?.likely_loop_count || 0;
      const budgetExceeded = s?.budget_exceeded_count || 0;
      if (timedOut || likelyLoop || budgetExceeded) hasIncident = true;
      rows.push({
        rowLabel: isMulti ? `${f.hostname}\n${modelLabel(model)}` : modelLabel(model),
        timed_out_count: timedOut,
        likely_loop_count: likelyLoop,
        budget_exceeded_count: budgetExceeded,
      });
    }
  }
  return hasIncident ? rows : [];
}

export function flattenAccuracyData(files: ResultsFile[], testKey: string): ChartRow[] {
  return files.flatMap(f =>
    entriesOf(f.data[testKey]).map(([model, s]) => {
      if (s.skipped) {
        return {
          _fileId: f.id, model, skipped: true,
          skip_reason: s.skip_reason, skip_detail: s.skip_detail,
        };
      }
      return {
        _fileId: f.id, model,
        correct: s.correct, total: s.total, answered: s.answered,
        accuracy_pct: s.accuracy_pct,
        timed_out_count: s.timed_out_count || 0,
        likely_loop_count: s.likely_loop_count || 0,
        budget_nudged_count: s.budget_nudged_count || 0,
        budget_exceeded_count: s.budget_exceeded_count || 0,
        crashed: s.crashed || false,
        ...memoryFields(s),
      };
    })
  );
}

export function getAccuracySettingsWarning(files: ResultsFile[]): string {
  if (!files.length) return "";
  const settings = files.map(file => file.data?.accuracy_settings);
  const valid = settings.every(s =>
    s != null
    && Number.isFinite(s.timeout_seconds)
    && Number.isInteger(s.token_budget)
    && Number.isFinite(s.first_pass_fraction)
  );
  if (!valid) {
    return "Accuracy settings are unknown for one or more loaded files; accuracy comparisons may use different limits.";
  }
  const first = settings[0];
  const matches = settings.every(s =>
    s.timeout_seconds === first.timeout_seconds
    && s.token_budget === first.token_budget
    && s.first_pass_fraction === first.first_pass_fraction
  );
  return matches ? "" : "Loaded files use different accuracy timeout or token-budget settings.";
}
