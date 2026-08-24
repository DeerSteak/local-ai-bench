import type { JsonRecord } from "./shared";

export interface VariantDisplayRow {
  variant: string;
  reference: boolean;
  qualityVerdict: string;
  qualityRanked: boolean;
  quality: string;
  throughput: string;
  memory: string;
  energy: string;
}

function record(value: unknown): JsonRecord | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

export function isVariantComparisonArtifact(value: unknown): value is JsonRecord {
  const data = record(value);
  return data?.artifact_type === "variant_comparison" && data?.schema_version === 1
    && typeof data?.base_model === "string" && typeof data?.reference_variant === "string"
    && Array.isArray(data?.variants);
}

export function variantArtifactLoadMode(values: unknown[]): "none" | "single" | "mixed" {
  const count = values.filter(isVariantComparisonArtifact).length;
  if (count === 0) return "none";
  return count === 1 && values.length === 1 ? "single" : "mixed";
}

function deltaLabel(value: unknown, suffix: string): string {
  const metric = record(value);
  if (typeof metric?.value !== "number" || typeof metric?.delta !== "number") return "Not recorded";
  const sign = metric.delta > 0 ? "+" : "";
  return `${metric.value} (${sign}${metric.delta}${suffix})`;
}

export function buildVariantDisplayRows(value: JsonRecord): VariantDisplayRow[] {
  if (!isVariantComparisonArtifact(value)) return [];
  return (value.variants as unknown[]).flatMap(item => {
    const row = record(item);
    if (!row || typeof row.variant !== "string") return [];
    return [{
      variant: row.variant,
      reference: row.reference === true,
      qualityVerdict: typeof row.quality_verdict === "string" ? row.quality_verdict : "inconclusive",
      qualityRanked: row.quality_ranked === true,
      quality: deltaLabel(row.quality, " pp"),
      throughput: deltaLabel(row.throughput, "%"),
      memory: deltaLabel(row.memory, "%"),
      energy: deltaLabel(row.energy, "%"),
    }];
  });
}
