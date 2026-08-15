import type { JsonRecord } from "./shared";

export interface TrialDisplayRow {
  key: string;
  baselineMean: number | null;
  baselineMedian: number | null;
  baselineStdev: number | null;
  baselineDrift: string;
  candidateMean: number | null;
  candidateMedian: number | null;
  candidateStdev: number | null;
  candidateDrift: string;
  interval: [number, number] | null;
  intervalMethod: string;
  threshold: number | null;
  verdict: string;
}

function record(value: unknown): JsonRecord | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function isTrialSetArtifact(value: unknown): value is JsonRecord {
  const data = record(value);
  return data?.schema_version === 1
    && data?.compatible === true
    && Array.isArray(data?.rows)
    && typeof data?.comparison_mode === "string";
}

export function trialArtifactLoadMode(values: unknown[]): "none" | "single" | "mixed" {
  const count = values.filter(isTrialSetArtifact).length;
  if (count === 0) return "none";
  return count === 1 && values.length === 1 ? "single" : "mixed";
}

export function buildTrialDisplayRows(data: JsonRecord): TrialDisplayRow[] {
  if (!isTrialSetArtifact(data)) return [];
  return data.rows.flatMap((value: unknown) => {
    const row = record(value);
    const baseline = record(row?.baseline);
    const candidate = record(row?.candidate);
    if (!row || !baseline || !candidate || typeof row.key !== "string") return [];
    const intervalValues = Array.isArray(row.change_interval_pct)
      ? row.change_interval_pct.map(finite)
      : [];
    const interval: [number, number] | null = intervalValues.length === 2
      && intervalValues[0] != null && intervalValues[1] != null
      ? [intervalValues[0], intervalValues[1]]
      : null;
    return [{
      key: row.key,
      baselineMean: finite(baseline.mean),
      baselineMedian: finite(baseline.median),
      baselineStdev: finite(baseline.stdev),
      baselineDrift: typeof baseline.drift === "string" ? baseline.drift : "unknown",
      candidateMean: finite(candidate.mean),
      candidateMedian: finite(candidate.median),
      candidateStdev: finite(candidate.stdev),
      candidateDrift: typeof candidate.drift === "string" ? candidate.drift : "unknown",
      interval,
      intervalMethod: typeof row.interval_method === "string" ? row.interval_method : "unavailable",
      threshold: finite(row.practical_threshold_pct),
      verdict: typeof row.verdict === "string" ? row.verdict : "inconclusive",
    }];
  });
}
