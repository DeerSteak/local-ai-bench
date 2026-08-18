import type { JsonRecord } from "./shared";

export type RecommendationGroup = "recommended" | "tied" | "eliminated" | "unevaluated";

export interface RecommendationDisplayItem {
  group: RecommendationGroup;
  candidate: string;
  detail: string;
  evidencePath: string | null;
}

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  efficiency: "Efficiency",
  memory: "Peak memory",
  memory_headroom: "Memory headroom",
  throughput: "Throughput",
  ttft: "TTFT",
};

const UNIT_LABELS: Record<string, string> = {
  GB: "GB",
  images_per_second: "images/s",
  percent: "%",
  seconds: "sec",
  tokens_per_joule: "tokens/J",
  tokens_per_second: "tokens/s",
};

function formattedValue(value: number, unit: string): string {
  const label = UNIT_LABELS[unit] || unit;
  return unit === "percent" ? `${value}%` : `${value} ${label}`.trim();
}

function record(value: unknown): JsonRecord | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

export function isRecommendationArtifact(value: unknown): value is JsonRecord {
  const data = record(value);
  return data?.artifact_type === "recommendation"
    && data?.schema_version === 1
    && ["recommended", "tied", "insufficient_evidence"].includes(String(data?.verdict))
    && Array.isArray(data?.recommended)
    && Array.isArray(data?.tied)
    && Array.isArray(data?.eliminated)
    && Array.isArray(data?.unevaluated);
}

export function recommendationArtifactLoadMode(values: unknown[]): "none" | "single" | "mixed" {
  const count = values.filter(isRecommendationArtifact).length;
  if (count === 0) return "none";
  return count === 1 && values.length === 1 ? "single" : "mixed";
}

function evidenceDetail(item: JsonRecord, objective: string): { detail: string, path: string | null } {
  const evidence = record(record(item.evidence)?.[objective]);
  const value = evidence?.value;
  const unit = typeof evidence?.unit === "string" ? evidence.unit : "";
  return {
    detail: typeof value === "number"
      ? `${METRIC_LABELS[objective] || objective}: ${formattedValue(value, unit)}`
      : "Evidence recorded",
    path: typeof evidence?.evidence_path === "string" ? [evidence.evidence_path, ...(
      Array.isArray(evidence.raw_evidence_paths) ? evidence.raw_evidence_paths.map(String) : []
    )].join(" · ") : null,
  };
}

export function buildRecommendationDisplayItems(data: JsonRecord): RecommendationDisplayItem[] {
  if (!isRecommendationArtifact(data)) return [];
  const constraints = record(data.constraints);
  const objective = typeof constraints?.primary_objective === "string"
    ? constraints.primary_objective
    : "objective";
  const items: RecommendationDisplayItem[] = [];
  for (const group of ["recommended", "tied"] as const) {
    for (const value of data[group]) {
      const item = record(value);
      if (!item || typeof item.candidate !== "string") continue;
      const evidence = evidenceDetail(item, objective);
      items.push({ group, candidate: item.candidate, detail: evidence.detail, evidencePath: evidence.path });
    }
  }
  for (const value of data.eliminated) {
    const item = record(value);
    if (!item || typeof item.candidate !== "string") continue;
    const reasons = Array.isArray(item.reasons) ? item.reasons.map(record).filter(Boolean) : [];
    const detail = reasons.map(reason => {
      const metric = String(reason?.constraint);
      const operator = reason?.operator === "minimum" ? "below minimum" : "above maximum";
      const measurement = record(reason?.measurement);
      const unit = typeof measurement?.unit === "string" ? measurement.unit : "";
      const value = typeof measurement?.value === "number"
        ? formattedValue(measurement.value, unit)
        : "measurement unavailable";
      const threshold = typeof reason?.threshold === "number"
        ? formattedValue(reason.threshold, unit)
        : String(reason?.threshold);
      return `${METRIC_LABELS[metric] || metric}: ${value} (${operator} ${threshold})`;
    }).join("; ") || "Constraint not met";
    const measurement = record(reasons[0]?.measurement);
    items.push({
      group: "eliminated", candidate: item.candidate, detail,
      evidencePath: typeof measurement?.evidence_path === "string" ? [measurement.evidence_path, ...(
        Array.isArray(measurement.raw_evidence_paths) ? measurement.raw_evidence_paths.map(String) : []
      )].join(" · ") : null,
    });
  }
  for (const value of data.unevaluated) {
    const item = record(value);
    if (!item || typeof item.candidate !== "string") continue;
    const missing = Array.isArray(item.missing_evidence)
      ? item.missing_evidence.map(value => METRIC_LABELS[String(value)] || String(value).replaceAll("_", " ")).join(", ")
      : "missing evidence";
    const resolution = record(item.resolution);
    items.push({
      group: "unevaluated", candidate: item.candidate, detail: `Needs: ${missing}`,
      evidencePath: typeof resolution?.evidence_path === "string" ? resolution.evidence_path : null,
    });
  }
  return items;
}
