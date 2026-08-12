import { FILE_COLORS, MODEL_COLORS, IMAGE_MODEL_COLORS, EMBED_MODEL_COLORS, FALLBACK_COLORS,
  LLM_MODEL_LABELS, IMAGE_MODEL_LABELS, EMBED_MODEL_LABELS, MODEL_SIZE_TIER } from "../constants";
import type { ChartRow, LineConfig, ResultsFile } from "../types";

// The one sanctioned `any` in the dashboard — see AGENTS.md's TypeScript section.
// Reference `JsonRecord`/`JsonRecord[string]` instead of writing `any` directly.
export type JsonRecord = Record<string, any>;

// Object.entries on an `any`-typed value can infer T as `unknown` rather than
// a usable type (a TS overload-resolution quirk) — this pins the value type.
export function entriesOf(obj: JsonRecord | null | undefined): [string, JsonRecord[string]][] {
  return Object.entries(obj || {});
}

export function valuesOf(obj: JsonRecord | null | undefined): JsonRecord[string][] {
  return Object.values(obj || {});
}

export function parseJSON(text: string): unknown {
  try { return JSON.parse(text); } catch { return null; }
}

export function parseResultsJSON(text: string): { data: JsonRecord | null, error: string | null } {
  try {
    const data = JSON.parse(text);
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      return { data: null, error: "Expected a results JSON object." };
    }
    return { data, error: null };
  } catch {
    return {
      data: null,
      error: "Invalid JSON. Non-finite values such as Infinity are not supported.",
    };
  }
}

export function getRunReliabilityWarning(data: JsonRecord | null | undefined): string {
  const run = data?.run;
  if (run == null) return "";
  if (typeof run !== "object" || Array.isArray(run)) return "Run metadata is malformed.";
  if (run.status === "complete") return "";
  const labels: Record<string, string> = {
    running: "This result was saved while the benchmark was still running.",
    partial: "This benchmark ended with partial results.",
    interrupted: "This benchmark was interrupted; completed measurements are still shown.",
    failed: "This benchmark failed before every selected stage completed.",
  };
  return labels[run.status] || "This result has an unknown completion state.";
}

export function getLlamaBenchMethodologyWarning(files: ResultsFile[]): string {
  const relevant = files.filter(file => Object.keys(file.data?.llamabench || {}).length > 0);
  if (relevant.length < 2) return "";
  const modes = new Set(relevant.map(file =>
    file.data?.run?.llamabench_repetition_mode || "legacy_internal_repetitions"));
  return modes.size > 1
    ? "Loaded llama-bench files use different repetition methodologies."
    : "";
}

export function getConversationTTFTMethodologyWarning(files: ResultsFile[]): string {
  const modes = new Set<string>();
  for (const file of files) {
    const samples = valuesOf(file.data?.llm_conversation)
      .flatMap(model => valuesOf(model))
      .filter(sample => sample && typeof sample === "object");
    if (samples.some(sample => sample.client_ttft_mean_sec != null)) modes.add("client");
    else if (samples.some(sample => sample.ttft_mean_sec != null)) modes.add("legacy_server");
  }
  return modes.size > 1
    ? "Loaded conversation files use different TTFT methodologies (client-observed versus legacy server prompt time)."
    : "";
}

export function getGpuSplitMethodologyWarning(files: ResultsFile[]): string {
  if (files.length < 2) return "";
  const modes = new Set(files.map(file =>
    file.data?.run?.effective_config?.gpu_split_mode || "layer"));
  return modes.size > 1
    ? "Loaded files use different multi-GPU modes (layer versus tensor parallelism)."
    : "";
}

export function getNoRepackMethodologyWarning(files: ResultsFile[], section?: string): string {
  if (section && ["images", "llamabench", "vllmbench"].includes(section)) return "";
  const relevant = files.filter(file => file.engine === "llamacpp" || file.engine == null);
  if (relevant.length < 2) return "";
  const modes = new Set(relevant.map(file =>
    file.data?.run?.effective_config?.llamacpp_no_repack === true));
  return modes.size > 1
    ? "Loaded llama.cpp files use different weight-repacking modes."
    : "";
}

// Cross-engine comparison compares different weight files, not just different
// runtimes: llama.cpp measures Q4_K_M GGUFs, vLLM measures 4-bit AWQ/GPTQ/W4A16
// safetensors of the same base model. Matching bit width is as close as they get.
export function getCrossEngineWeightsWarning(files: ResultsFile[]): string {
  const engines = new Set(files.map(file => file.engine).filter(Boolean));
  return engines.size > 1
    ? "Loaded files span multiple engines. They do not measure the same weights: "
      + "llama.cpp runs Q4_K_M GGUFs and vLLM runs 4-bit AWQ/GPTQ safetensors of the "
      + "same base model, so differences reflect the quantization as well as the runtime."
    : "";
}

// Turn free-typed text (or a whole joined filename stem) into something safe
// to use as a filename: whitespace and characters reserved/special on common
// filesystems — including periods, since they read as file extensions/hidden-
// file markers — collapse to a single hyphen, and any leading/trailing
// hyphens left over are trimmed.
export function sanitizeForFilename(raw: string | null | undefined): string {
  return String(raw || "")
    .trim()
    .replace(/[\s<>:"/\\|?*#%&{}$!'`=+@~^.]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

export function backendLabel(backend: string | null | undefined): string {
  const labels: Record<string, string> = {
    cpu: "CPU", cuda: "CUDA", directml: "DirectML", hip: "HIP", metal: "Metal",
    mlx: "MLX", mps: "MPS", opencl: "OpenCL", rocm: "ROCm", sycl: "SYCL", vulkan: "Vulkan",
  };
  const key = String(backend || "").toLowerCase();
  return labels[key] || String(backend || "").replace(/(^|[-_])([a-z])/g,
    (_match, separator: string, letter: string) => `${separator}${letter.toUpperCase()}`);
}

export function engineLabel(engine: string | null | undefined): string {
  const labels: Record<string, string> = { llamacpp: "llama.cpp", vllm: "vLLM" };
  const key = String(engine || "").toLowerCase();
  return labels[key] || String(engine || "");
}

export function measuredCategoryAxisWidth(
  rows: ChartRow[], key: string, measure: (text: string) => number, tickSpace = 11,
): number {
  const lines = rows.flatMap(row => String(row[key] ?? "").split("\n"));
  return Math.ceil(Math.max(0, ...lines.map(measure))) + tickSpace;
}

// Runtime versions are always material comparison context.
export function applyEngineLabels<T extends ResultsFile>(files: T[], section?: string): T[] {
  const multiEngine = new Set(files.map(f => f.engine).filter(Boolean)).size > 1;
  return files.map(f => {
    const noRepack = f.engine === "llamacpp"
      && !["llamabench", "vllmbench"].includes(section || "")
      && f.data?.run?.effective_config?.llamacpp_no_repack === true;
    const displayEngine = engineLabel(f.engine);
    const engine = noRepack ? `${displayEngine} -nr` : displayEngine;
    const identity = (runtime: string) => {
  const runtimeLabel = [backendLabel(f.backend), runtime].filter(Boolean).join(" / ");
      return [f.hostname, runtimeLabel].filter(Boolean).join("\n");
    };
    if (f.engineVersion) {
      const runtime = [engine, f.engineVersion].filter(Boolean).join(" ");
      return { ...f, hostname: identity(runtime) };
    }
    if (f.engine && f.engineVersionRecorded === false) {
      return { ...f, hostname: identity(`${engine} version not recorded`) };
    }
    if (f.engine && f.engineVersionRecorded === true) {
      return { ...f, hostname: identity(`${engine} version unavailable`) };
    }
    return multiEngine && f.engine ? { ...f, hostname: identity(engine || f.engine) } : f;
  });
}

export function filesForSection<T extends ResultsFile>(files: T[], section: string): T[] {
  return section === "images" ? files : applyEngineLabels(files, section);
}

export function fmt(v: number | null | undefined, unit: string): string {
  if (v == null) return "—";
  switch (unit) {
    case "ms": {
      const ms = v * 1000;
      return ms < 10 ? `${ms.toFixed(1)}ms` : `${Math.round(ms)}ms`;
    }
    case "sec-plain":
      return `${v.toFixed(2)}s`;
    case "sec":
      if (v >= 60) return `${(v / 60).toFixed(1)}m`;
      return `${v.toFixed(2)}s`;
    case "tps":
      if (v >= 1000) return `${(v / 1000).toFixed(2)}K`;
      return v.toFixed(1);
    case "sps":
      if (v >= 10000) return `${(v / 1000).toFixed(1)}K`;
      return v.toFixed(0);
    case "pct":
      return `${v.toFixed(1)}%`;
    case "count":
      return `${Math.round(v)}`;
    default:
      return v.toFixed(2);
  }
}

// Deterministic color for an unknown model based on its name
function hashColor(key: string, palette: readonly string[]): string {
  const h = [...key].reduce((acc, c) => acc + c.charCodeAt(0), 0);
  return palette[h % palette.length];
}

// A type-guard for `.filter(Boolean)` after a `.map()` that returns `T | null`
// — `Boolean` itself narrows nothing, so TS still sees `(T | null)[]` after it.
export function isNotNull<T>(x: T | null | undefined): x is T {
  return x != null;
}

// *_COLORS/*_LABELS/*_TIER constants have fixed literal keys; callers here
// look up an arbitrary model string that may not be one of them.
export function lookup<T>(dict: Record<string, T>, key: string): T | undefined {
  return (dict as Record<string, T>)[key];
}

export function getModelColor(model: string): string {
  return lookup(MODEL_COLORS, model) || hashColor(model, FALLBACK_COLORS);
}

export function getImageModelColor(model: string): string {
  return lookup(IMAGE_MODEL_COLORS, model) || hashColor(model, FALLBACK_COLORS);
}

export function getEmbedModelColor(model: string): string {
  return lookup(EMBED_MODEL_COLORS, model) || hashColor(model, FALLBACK_COLORS);
}

export function modelLabel(model: string): string {
  return lookup(LLM_MODEL_LABELS, model) || model;
}

export function imageModelLabel(model: string): string {
  return lookup(IMAGE_MODEL_LABELS, model) || model;
}

export function embedModelLabel(model: string): string {
  return lookup(EMBED_MODEL_LABELS, model) || model;
}

// Bucket an LLM model key into a size tier. Known models use MODEL_SIZE_TIER
// (parameter-count-based, matching models.py/README.md exactly). Unknown
// models (not in the standard roster) fall back to a param-count heuristic
// parsed from the key (e.g. "some-new-model-70b" -> 70 -> "large").
export function getModelSizeTier(model: string): string {
  const tier = lookup(MODEL_SIZE_TIER, model);
  if (tier) return tier;
  const match = model.match(/(\d+)b/i);
  if (!match) return "medium";
  const billions = parseInt(match[1], 10);
  if (billions <= 20) return "small";
  if (billions < 50) return "medium";
  return "large";
}

// Skip info for one (file, model) pair in a given LLM section — non-null
// only when benchmark.py intentionally excluded this model from that section
// entirely (too slow, timed out on a prior test, no single-shot data, or a
// known repeat-crasher skipped via the crash cache). Defaults to
// "llm_conversation" for existing call sites that predate the "llm" section
// also being able to produce a whole-model skip (via Shared.check_crash_cache).
export function getSkipInfo(file: ResultsFile, model: string, section = "llm_conversation")
  : { reason: string, detail: string } | null {
  const d = file.data[section]?.[model];
  if (!d?.skipped) return null;
  return { reason: d.skip_reason, detail: d.skip_detail };
}

// Per-file line configs: one line per file, color by file index. Used for all sections.
export function buildFileLineConfigs(files: ResultsFile[]): LineConfig[] {
  return files.map((f, fi) => ({
    dataKey: `f${fi}`,
    stroke: FILE_COLORS[fi % FILE_COLORS.length],
    name: f.hostname ?? "Unknown",
  }));
}

export function prepareOrderedBarGroupData(data: ChartRow[], barConfigs: { dataKey: string }[]): ChartRow[] {
  return data.map(row => ({
    ...row,
    _groupMax: Math.max(0, ...barConfigs.map(config => row[config.dataKey] ?? 0)),
  }));
}

// Sort bar-chart rows so the fastest result is first.
// preferredKeys: ordered array of candidate sort keys; the last one present in
// the data is used (most strenuous). direction: "desc" = higher is better,
// "asc" = lower is better.
export function sortBarData(data: ChartRow[], preferredKeys: string[], direction: "desc" | "asc"): ChartRow[] {
  let sortKey: string | null = null;
  for (let i = preferredKeys.length - 1; i >= 0; i--) {
    if (data.some(row => row[preferredKeys[i]] != null)) {
      sortKey = preferredKeys[i];
      break;
    }
  }
  if (!sortKey) return data;
  const key = sortKey;
  return [...data].sort((a, b) => {
    const av = a[key] ?? (direction === "desc" ? -Infinity : Infinity);
    const bv = b[key] ?? (direction === "desc" ? -Infinity : Infinity);
    return direction === "desc" ? bv - av : av - bv;
  });
}

// TTFT switches units by scale (values are wildly different at 2K vs 96K context)
// — all-sub-second data reads better in ms, minutes-long prefills read better as
// plain seconds with no decimals.
export function deriveTtftUnit(values: number[]): { ttftUnit: string, ttftYLabel: string } {
  const ttftUnit = values.some(v => v >= 60) ? "sec-plain"
    : values.length && values.every(v => v < 1) ? "ms"
    : "sec";
  return { ttftUnit, ttftYLabel: ttftUnit === "ms" ? "TTFT (ms)" : "TTFT (sec)" };
}

// A bar-chart series counts as present if it has either a real value or a
// skip-status placeholder (`_status_<key>`) to render instead.
export function hasValueOrStatus(rows: ChartRow[], key: string): boolean {
  return rows.some(r => r[key] != null || r[`_status_${key}`] != null);
}

// Return the key from `keys` whose maximum value across all rows is highest
// (i.e. the most strenuous setting).
export function findMostStrenuousKey(data: ChartRow[], keys: string[]): string | null {
  let best: string | null = null;
  let bestMax = -Infinity;
  for (const key of keys) {
    const vals = data.map(r => r[key]).filter(v => v != null);
    if (!vals.length) continue;
    const max = Math.max(...vals);
    if (max > bestMax) { bestMax = max; best = key; }
  }
  return best;
}

// Shared comparator behind every StatsTable variant's column sort — `valueFn`
// lets concurrency's numeric level column override the default row[key] lookup.
export function sortRows<T extends ChartRow>(
  rows: T[], sortConfig: { key: string, dir: 1 | -1 },
  valueFn: (row: T, key: string) => JsonRecord[string] = (row, key) => row[key] ?? "",
): T[] {
  return [...rows].sort((a, b) => {
    const av = valueFn(a, sortConfig.key);
    const bv = valueFn(b, sortConfig.key);
    return (av < bv ? -1 : av > bv ? 1 : 0) * sortConfig.dir;
  });
}
