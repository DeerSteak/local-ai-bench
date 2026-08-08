import {
  CTX_ORDER, FALLBACK_COLORS, FILE_COLORS, MODEL_DASH_PATTERNS, LLM_DISPLAY_ORDER,
  CTX_COLORS, ACCURACY_TESTS,
} from "../constants";
import { getModelColor, modelLabel, getSkipInfo, entriesOf, valuesOf, lookup } from "./shared";
import type { JsonRecord } from "./shared";
import type { ResultsFile, ChartRow } from "../types";

const SKIP_REASON_LABELS: Record<string, string> = {
  timed_out: "Skipped - LLM Timed Out",
  slow_tps: "Skipped - LLM Too Slow",
  no_llm_data: "Skipped - No LLM Data",
  known_crash: "Skipped - Engine Crashed",
  tool_calls_unsupported: "Skipped - No Tool Parser",
};

export const llmTTFTMean = (sample: JsonRecord[string]) => sample?.client_ttft_mean_sec ?? sample?.ttft_mean_sec;
// Prompt-processing throughput. Absent on results from before it was recorded, and
// on any run whose engine reported no prompt duration — see docs/engines.md#prefill-timing.
export const llmPrefillTPS = (sample: JsonRecord[string]) => sample?.prefill_tps_mean;

export function llmMetricValue(sample: JsonRecord[string], metric: string) {
  if (metric === "tps") return sample?.tps_mean;
  if (metric === "prefill") return llmPrefillTPS(sample);
  return llmTTFTMean(sample);
}
export const llmValidRuns = (sample: JsonRecord[string]) => sample?.valid_runs ?? sample?.n_runs;

// Bar-chart status label for one (file, model, context) cell: "{ctx} - Timed
// Out" for the context at which benchmark.py's run itself timed out (llm or
// llm_conversation both set a "timed_out" field), "{ctx} - Crashed" for the
// context at which the inference engine's model runner crashed (a "crashed" field, set by
// the llm and llm_conversation tests when they give up retrying after a
// repeat crash), "{ctx} - Skipped ({slowCtx} Too Slow)" for every later
// (larger) context that was never attempted because the model dropped below
// the tok/s cutoff at an earlier checkpoint (a "slow_tps" field), or a
// "Skipped - ..." label when the whole model was excluded from this section
// (a "skipped"/"skip_reason" pair, from either a slow/timed-out gate in
// benchmark.py or a known-crasher skip in the section's own crash cache). The
// slow checkpoint itself still has real data (it's the measurement that
// triggered the cutoff), so it returns null there — its actual value is shown
// rather than a status label. Returns null for cells with real data, or
// earlier contexts that simply weren't reached for unrelated reasons.
export function getBarStatusLabel(file: ResultsFile, model: string, ctx: string, section: string): string | null {
  const skip = getSkipInfo(file, model, section);
  if (skip) return SKIP_REASON_LABELS[skip.reason] || `Skipped - ${skip.detail}`;
  const sectionData = file.data[section]?.[model];
  const crashedCtx = sectionData?.crashed;
  if (crashedCtx) {
    const crashedIdx = CTX_ORDER.indexOf(crashedCtx);
    const ctxIdx = CTX_ORDER.indexOf(ctx);
    if (ctxIdx === crashedIdx) return `${ctx} - Crashed`;
    if (ctxIdx > crashedIdx) return `${ctx} - Skipped`;
  }
  const timedOutCtx = sectionData?.timed_out;
  if (timedOutCtx) {
    const timedOutIdx = CTX_ORDER.indexOf(timedOutCtx);
    const ctxIdx = CTX_ORDER.indexOf(ctx);
    if (ctxIdx === timedOutIdx) return `${ctx} - Timed Out`;
    if (ctxIdx > timedOutIdx) return `${ctx} - Skipped`;
  }
  const slowTpsCtx = sectionData?.slow_tps;
  if (slowTpsCtx) {
    const slowIdx = CTX_ORDER.indexOf(slowTpsCtx);
    const ctxIdx = CTX_ORDER.indexOf(ctx);
    if (ctxIdx > slowIdx) return `${ctx} - Skipped (${slowTpsCtx} Too Slow)`;
  }
  return null;
}

// Return all LLM model keys from the loaded files, in canonical order.
// Checks every section that runs the shared LLM roster — single-shot,
// conversation, all accuracy tests, and both concurrency tests — since
// the Models filter is shared UI across all of them. A model present in only
// one section (e.g. a file that only ran `--tests acc`, leaving
// llm/llm_conversation empty) should still show up rather than leaving the
// filter (and every section that depends on it) empty.
export function getAllLLMModels(files: ResultsFile[]): string[] {
  const s = new Set<string>();
  for (const f of files) {
    for (const m of Object.keys(f.data.llm || {})) s.add(m);
    for (const m of Object.keys(f.data.llm_conversation || {})) s.add(m);
    for (const test of ACCURACY_TESTS)
      for (const m of Object.keys(f.data[test] || {})) s.add(m);
    for (const m of Object.keys(f.data.concurrency_tool || {})) s.add(m);
    for (const m of Object.keys(f.data.concurrency_chat || {})) s.add(m);
    for (const m of Object.keys(f.data.llamabench || {})) s.add(m);
    for (const m of Object.keys(f.data.llamabenchconc || {})) s.add(m);
  }
  const known   = LLM_DISPLAY_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !LLM_DISPLAY_ORDER.includes(m));
  return [...known, ...unknown];
}

export function getLLMModelsWithSectionResults(files: ResultsFile[], section: string): string[] {
  return getAllLLMModels(files).filter(model => files.some(file => {
    const result = file.data[section]?.[model];
    if (!result) return false;
    if (result.skipped) return result.skip_reason !== "no_llm_data";
    return CTX_ORDER.some(ctx => result[ctx] != null)
      || result.crashed != null
      || result.timed_out != null
      || result.slow_tps != null;
  }));
}

// LLM: one chart per model. X = context length, lines = files.
export function buildLLMDataForModel(files: ResultsFile[], model: string, metric: string, section = "llm"): ChartRow[] {
  const ctxSet = new Set<string>();
  for (const f of files)
    for (const ctx of Object.keys(f.data[section]?.[model] || {})) ctxSet.add(ctx);
  const ctxLabels = CTX_ORDER.filter(c => ctxSet.has(c));
  return ctxLabels.map(ctx => {
    const row: ChartRow = { ctxLabel: ctx };
    files.forEach((f, fi) => {
      const s = f.data[section]?.[model]?.[ctx];
      if (s) row[`f${fi}`] = llmMetricValue(s, metric);
    });
    return row;
  });
}

// Legacy: X = context length, lines = models (+ file distinction if multi)
export function buildLLMData(files: ResultsFile[], metric: string, enabledModels: Set<string>): ChartRow[] {
  const isSingle = files.length === 1;
  const ctxSet = new Set<string>();
  for (const f of files)
    for (const md of valuesOf(f.data.llm))
      for (const ctx of Object.keys(md)) ctxSet.add(ctx);
  const ctxLabels = CTX_ORDER.filter(c => ctxSet.has(c));

  return ctxLabels.map(ctx => {
    const row: ChartRow = { ctxLabel: ctx };
    files.forEach((f, fi) => {
      for (const [model, md] of entriesOf(f.data.llm)) {
        if (!enabledModels.has(model) || !md[ctx]) continue;
        const key = isSingle ? model : `f${fi}_${model}`;
        row[key] = llmMetricValue(md[ctx], metric);
      }
    });
    return row;
  });
}

export function buildLLMLineConfigs(files: ResultsFile[], data: ChartRow[], enabledModels: Set<string>) {
  const isSingle = files.length === 1;
  const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m));
  const configs = [];
  if (isSingle) {
    for (const m of allModels) {
      if (data.some(d => d[m] != null))
        configs.push({ dataKey: m, stroke: getModelColor(m), name: modelLabel(m) });
    }
  } else {
    for (let fi = 0; fi < files.length; fi++) {
      const stroke = FILE_COLORS[fi % FILE_COLORS.length];
      allModels.forEach((m, mi) => {
        const dataKey = `f${fi}_${m}`;
        if (data.some(d => d[dataKey] != null))
          configs.push({
            dataKey,
            stroke,
            strokeDasharray: MODEL_DASH_PATTERNS[mi % MODEL_DASH_PATTERNS.length],
            name: `${files[fi].hostname} — ${modelLabel(m)}`,
          });
      });
    }
  }
  return configs;
}

// LLM bar chart: rows = files/systems, cols = context lengths
export function buildLLMBarData(files: ResultsFile[], model: string, metric: string, section = "llm"): ChartRow[] {
  return files.map(f => {
    const row: ChartRow = { systemLabel: f.hostname };
    const ctxData = f.data[section]?.[model] || {};
    for (const ctx of CTX_ORDER) {
      const s = ctxData[ctx];
      if (s) row[ctx] = llmMetricValue(s, metric);
      const status = getBarStatusLabel(f, model, ctx, section);
      if (status) row[`_status_${ctx}`] = status;
    }
    return row;
  });
}

export function buildLLMBarConfigs(files: ResultsFile[], model: string, section = "llm") {
  const ctxSet = new Set<string>();
  for (const f of files) {
    for (const ctx of Object.keys(f.data[section]?.[model] || {})) ctxSet.add(ctx);
    const timedOutCtx = f.data[section]?.[model]?.timed_out;
    if (timedOutCtx) ctxSet.add(timedOutCtx);
    const crashedCtx = f.data[section]?.[model]?.crashed;
    if (crashedCtx) ctxSet.add(crashedCtx);
    const slowTpsCtx = f.data[section]?.[model]?.slow_tps;
    if (slowTpsCtx) ctxSet.add(slowTpsCtx);
  }
  return CTX_ORDER
    .filter(ctx => ctxSet.has(ctx))
    .map((ctx, i) => ({
      dataKey: ctx,
      name: ctx,
      fill: lookup(CTX_COLORS, ctx) || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// LLM bar chart by system: rows = models, cols = context lengths, for one file
export function buildLLMBarDataByModel(file: ResultsFile, models: string[], metric: string, section = "llm"): ChartRow[] {
  return models.map(model => {
    const row: ChartRow = { modelLabel: modelLabel(model) };
    const ctxData = file.data[section]?.[model] || {};
    for (const ctx of CTX_ORDER) {
      const s = ctxData[ctx];
      if (s) row[ctx] = llmMetricValue(s, metric);
      const status = getBarStatusLabel(file, model, ctx, section);
      if (status) row[`_status_${ctx}`] = status;
    }
    return row;
  });
}

export function buildLLMBarConfigsByModel(file: ResultsFile, models: string[], section = "llm") {
  const ctxSet = new Set<string>();
  for (const model of models) {
    for (const ctx of Object.keys(file.data[section]?.[model] || {})) ctxSet.add(ctx);
    const timedOutCtx = file.data[section]?.[model]?.timed_out;
    if (timedOutCtx) ctxSet.add(timedOutCtx);
    const crashedCtx = file.data[section]?.[model]?.crashed;
    if (crashedCtx) ctxSet.add(crashedCtx);
    const slowTpsCtx = file.data[section]?.[model]?.slow_tps;
    if (slowTpsCtx) ctxSet.add(slowTpsCtx);
  }
  return CTX_ORDER
    .filter(ctx => ctxSet.has(ctx))
    .map((ctx, i) => ({
      dataKey: ctx,
      name: ctx,
      fill: lookup(CTX_COLORS, ctx) || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// LLM line chart by system: rows = context lengths, one line per model, for one file
export function buildLLMLineDataByCtx(file: ResultsFile, models: string[], metric: string, section = "llm"): ChartRow[] {
  const ctxSet = new Set<string>();
  for (const model of models)
    for (const ctx of Object.keys(file.data[section]?.[model] || {})) ctxSet.add(ctx);
  const ctxLabels = CTX_ORDER.filter(c => ctxSet.has(c));
  return ctxLabels.map(ctx => {
    const row: ChartRow = { ctxLabel: ctx };
    for (const model of models) {
      const s = file.data[section]?.[model]?.[ctx];
      if (s) row[model] = llmMetricValue(s, metric);
    }
    return row;
  });
}

export function buildLLMLineConfigsByCtx(models: string[], data: ChartRow[]) {
  return models
    .filter(m => data.some(row => row[m] != null))
    .map(m => ({ dataKey: m, stroke: getModelColor(m), name: modelLabel(m) }));
}

export function flattenLLMData(files: ResultsFile[], section = "llm") {
  return files.flatMap(f =>
    entriesOf(f.data[section]).flatMap(([model, ctxData]): ChartRow[] => {
      if (ctxData?.skipped) {
        return [{
          _fileId: f.id, model, ctx: "—", skipped: true,
          skip_reason: ctxData.skip_reason, skip_detail: ctxData.skip_detail,
        }];
      }
      return entriesOf(ctxData)
        .filter(([ctx]) => CTX_ORDER.includes(ctx))
        .map(([ctx, s]) => ({
          _fileId: f.id, model, ctx,
          tps_mean: s.tps_mean, tps_stdev: s.tps_stdev,
          ttft_mean: llmTTFTMean(s),
          ttft_stdev: s.client_ttft_stdev_sec ?? s.ttft_stdev_sec,
          prefill_tps: llmPrefillTPS(s), prefill_tps_stdev: s.prefill_tps_stdev,
          n_runs: llmValidRuns(s),
        }));
    })
  );
}
