import { FILE_COLORS, MODEL_COLORS, IMAGE_MODEL_COLORS, EMBED_MODEL_COLORS, FALLBACK_COLORS,
  LLM_MODEL_LABELS, IMAGE_MODEL_LABELS, EMBED_MODEL_LABELS, MODEL_SIZE_TIER } from "../constants";

export function parseJSON(text) {
  try { return JSON.parse(text); } catch { return null; }
}

export function parseResultsJSON(text) {
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

export function getRunReliabilityWarning(data) {
  const run = data?.run;
  if (run == null) return "";
  if (!run || typeof run !== "object" || Array.isArray(run)) return "Run metadata is malformed.";
  if (run.status === "complete") return "";
  const labels = {
    running: "This result was saved while the benchmark was still running.",
    partial: "This benchmark ended with partial results.",
    interrupted: "This benchmark was interrupted; completed measurements are still shown.",
    failed: "This benchmark failed before every selected stage completed.",
  };
  return labels[run.status] || "This result has an unknown completion state.";
}

export function getLlamaBenchMethodologyWarning(files) {
  const relevant = files.filter(file => Object.keys(file.data?.llamabench || {}).length > 0);
  if (relevant.length < 2) return "";
  const modes = new Set(relevant.map(file =>
    file.data?.run?.llamabench_repetition_mode || "legacy_internal_repetitions"));
  return modes.size > 1
    ? "Loaded llama-bench files use different repetition methodologies."
    : "";
}

export function getConversationTTFTMethodologyWarning(files) {
  const modes = new Set();
  for (const file of files) {
    const samples = Object.values(file.data?.llm_conversation || {})
      .flatMap(model => Object.values(model || {}))
      .filter(sample => sample && typeof sample === "object");
    if (samples.some(sample => sample.client_ttft_mean_sec != null)) modes.add("client");
    else if (samples.some(sample => sample.ttft_mean_sec != null)) modes.add("legacy_server");
  }
  return modes.size > 1
    ? "Loaded conversation files use different TTFT methodologies (client-observed versus legacy server prompt time)."
    : "";
}

// Turn free-typed text (or a whole joined filename stem) into something safe
// to use as a filename: whitespace and characters reserved/special on common
// filesystems — including periods, since they read as file extensions/hidden-
// file markers — collapse to a single hyphen, and any leading/trailing
// hyphens left over are trimmed.
export function sanitizeForFilename(raw) {
  return String(raw || "")
    .trim()
    .replace(/[\s<>:"/\\|?*#%&{}$!'`=+@~^.]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

// Fold each file's engine into its hostname label, but only when needed to
// disambiguate — e.g. two --engine runs off the same host (identical
// profile.hostname) loaded side by side. With a single engine among the
// loaded files, appending "(llamacpp)" to every label is just noise.
export function applyEngineLabels(files) {
  const multiEngine = new Set(files.map(f => f.engine).filter(Boolean)).size > 1;
  if (!multiEngine) return files;
  return files.map(f => f.engine ? { ...f, hostname: `${f.hostname} (${f.engine})` } : f);
}

export function fmt(v, unit) {
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
function hashColor(key, palette) {
  const h = [...key].reduce((acc, c) => acc + c.charCodeAt(0), 0);
  return palette[h % palette.length];
}

export function getModelColor(model) {
  return MODEL_COLORS[model] || hashColor(model, FALLBACK_COLORS);
}

export function getImageModelColor(model) {
  return IMAGE_MODEL_COLORS[model] || hashColor(model, FALLBACK_COLORS);
}

export function getEmbedModelColor(model) {
  return EMBED_MODEL_COLORS[model] || hashColor(model, FALLBACK_COLORS);
}

export function modelLabel(model) {
  return LLM_MODEL_LABELS[model] || model;
}

export function imageModelLabel(model) {
  return IMAGE_MODEL_LABELS[model] || model;
}

export function embedModelLabel(model) {
  return EMBED_MODEL_LABELS[model] || model;
}

// Bucket an LLM model key into a size tier. Known models use MODEL_SIZE_TIER
// (parameter-count-based, matching models.py/README.md exactly). Unknown
// models (not in the standard roster) fall back to a param-count heuristic
// parsed from the key (e.g. "some-new-model-70b" -> 70 -> "large").
export function getModelSizeTier(model) {
  if (MODEL_SIZE_TIER[model]) return MODEL_SIZE_TIER[model];
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
export function getSkipInfo(file, model, section = "llm_conversation") {
  const d = file.data[section]?.[model];
  if (!d?.skipped) return null;
  return { reason: d.skip_reason, detail: d.skip_detail };
}

// Per-file line configs: one line per file, color by file index. Used for all sections.
export function buildFileLineConfigs(files) {
  return files.map((f, fi) => ({
    dataKey: `f${fi}`,
    stroke: FILE_COLORS[fi % FILE_COLORS.length],
    name: f.hostname,
  }));
}

export function prepareOrderedBarGroupData(data, barConfigs) {
  return data.map(row => ({
    ...row,
    _groupMax: Math.max(0, ...barConfigs.map(config => row[config.dataKey] ?? 0)),
  }));
}

// Sort bar-chart rows so the fastest result is first.
// preferredKeys: ordered array of candidate sort keys; the last one present in
// the data is used (most strenuous). direction: "desc" = higher is better,
// "asc" = lower is better.
export function sortBarData(data, preferredKeys, direction) {
  let sortKey = null;
  for (let i = preferredKeys.length - 1; i >= 0; i--) {
    if (data.some(row => row[preferredKeys[i]] != null)) {
      sortKey = preferredKeys[i];
      break;
    }
  }
  if (!sortKey) return data;
  return [...data].sort((a, b) => {
    const av = a[sortKey] ?? (direction === "desc" ? -Infinity : Infinity);
    const bv = b[sortKey] ?? (direction === "desc" ? -Infinity : Infinity);
    return direction === "desc" ? bv - av : av - bv;
  });
}

// TTFT switches units by scale (values are wildly different at 2K vs 96K context)
// — all-sub-second data reads better in ms, minutes-long prefills read better as
// plain seconds with no decimals.
export function deriveTtftUnit(values) {
  const ttftUnit = values.some(v => v >= 60) ? "sec-plain"
    : values.length && values.every(v => v < 1) ? "ms"
    : "sec";
  return { ttftUnit, ttftYLabel: ttftUnit === "ms" ? "TTFT (ms)" : "TTFT (sec)" };
}

// A bar-chart series counts as present if it has either a real value or a
// skip-status placeholder (`_status_<key>`) to render instead.
export function hasValueOrStatus(rows, key) {
  return rows.some(r => r[key] != null || r[`_status_${key}`] != null);
}

// Return the key from `keys` whose maximum value across all rows is highest
// (i.e. the most strenuous setting).
export function findMostStrenuousKey(data, keys) {
  let best = null;
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
export function sortRows(rows, sortConfig, valueFn = (row, key) => row[key] ?? "") {
  return [...rows].sort((a, b) => {
    const av = valueFn(a, sortConfig.key);
    const bv = valueFn(b, sortConfig.key);
    return (av < bv ? -1 : av > bv ? 1 : 0) * sortConfig.dir;
  });
}
