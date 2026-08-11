import type { JsonRecord } from "./shared";
import type { ResultsFile } from "../types";

// Every numeric source field rendered by a chart must be registered and covered by the ratchet test.
export const DELTA_FIELD_NAMES = [
  "tps_mean", "prefill_tps_mean", "client_ttft_mean_sec", "ttft_mean_sec",
  "aggregate_tps", "accuracy_pct", "chunks_per_sec_mean", "sec_per_image_mean",
  "avg_ts", "speed_tg", "avg_latency_sec", "output_tps",
] as const;
const DELTA_FIELDS = new Set<string>(DELTA_FIELD_NAMES);
const IDENTITY_FIELDS = ["n_prompt", "n_depth", "n_gen", "pp", "tg", "pl", "input_len", "output_len"];

function arrayPeer(candidate: JsonRecord, baseline: JsonRecord[]): JsonRecord | undefined {
  if (!candidate || typeof candidate !== "object") return undefined;
  const keys = IDENTITY_FIELDS.filter(key => candidate[key] != null);
  if (!keys.length) return undefined;
  return baseline.find(peer => keys.every(key => peer?.[key] === candidate[key]));
}

function relativeNode(candidate: JsonRecord[string], baseline: JsonRecord[string], key = ""): JsonRecord[string] {
  if (DELTA_FIELDS.has(key)) {
    if (typeof candidate !== "number" || typeof baseline !== "number" || !Number.isFinite(candidate)
        || !Number.isFinite(baseline) || baseline === 0) return undefined;
    return candidate / baseline * 100;
  }
  if (Array.isArray(candidate)) {
    const peers = Array.isArray(baseline) ? baseline : [];
    return candidate.map((value, index) =>
      relativeNode(value, arrayPeer(value, peers) ?? peers[index], key));
  }
  if (!candidate || typeof candidate !== "object") return candidate;
  const result: JsonRecord = {};
  for (const [childKey, value] of Object.entries(candidate)) {
    const transformed = relativeNode(value, baseline?.[childKey], childKey);
    if (transformed !== undefined) result[childKey] = transformed;
  }
  return result;
}

export function applyBaselineDeltas<T extends ResultsFile>(files: T[], baselineId: string | null): T[] {
  if (!baselineId) return files;
  const baseline = files.find(file => String(file.id) === baselineId);
  if (!baseline) return files;
  return files.map(file => ({
    ...file,
    hostname: file === baseline ? `${file.hostname ?? "Baseline"} (baseline)` : file.hostname,
    data: relativeNode(file.data, baseline.data),
  }));
}
