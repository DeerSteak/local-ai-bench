import { CTX_ORDER } from "../constants";
import type { ChartRow, ResultsFile } from "../types";
import type { JsonRecord } from "./shared";

const finiteNumber = (value: JsonRecord[string]): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

export const powerEnergy = (sample: JsonRecord[string]): number | null =>
  finiteNumber(sample?.power?.energy_joules);

export const powerEfficiency = (sample: JsonRecord[string]): number | null =>
  finiteNumber(sample?.power?.efficiency?.per_joule);

export const powerScope = (sample: JsonRecord[string]): string | null =>
  typeof sample?.power?.scope === "string" ? sample.power.scope : null;

export const powerFields = (sample: JsonRecord[string]): ChartRow => sample?.power ? ({
  energy_joules: powerEnergy(sample),
  efficiency_per_joule: powerEfficiency(sample),
  efficiency_unit: typeof sample.power.efficiency?.unit === "string"
    ? sample.power.efficiency.unit : null,
  power_scope: powerScope(sample),
  power_status: typeof sample.power.status === "string" ? sample.power.status : "unavailable",
  power_reason: typeof sample.power.reason === "string" ? sample.power.reason : null,
}) : ({});

export function buildPowerEfficiencyDataForModel(
  files: ResultsFile[], model: string, section = "llm",
): ChartRow[] {
  const scopes = new Set<string>();
  for (const file of files) {
    for (const ctx of CTX_ORDER) {
      const scope = powerScope(file.data[section]?.[model]?.[ctx]);
      if (scope) scopes.add(scope);
    }
  }
  if (scopes.size > 1) return [];
  return CTX_ORDER.map(ctx => {
    const row: ChartRow = { ctxLabel: ctx };
    files.forEach((file, index) => {
      const value = powerEfficiency(file.data[section]?.[model]?.[ctx]);
      if (value != null) row[`f${index}`] = value;
    });
    return row;
  }).filter(row => files.some((_, index) => row[`f${index}`] != null));
}

export function hasMixedPowerScopes(
  files: ResultsFile[], models: string[], section = "llm",
): boolean {
  const scopes = new Set<string>();
  for (const file of files) {
    for (const model of models) {
      for (const ctx of CTX_ORDER) {
        const scope = powerScope(file.data[section]?.[model]?.[ctx]);
        if (scope) scopes.add(scope);
      }
    }
  }
  return scopes.size > 1;
}

export function runPowerSummary(file: ResultsFile): {
  status: string, energyJoules: number | null, idleWatts: number | null,
  scope: string | null, reason: string | null,
} {
  const summary = file.data?.run?.power_summary;
  return {
    status: typeof summary?.status === "string" ? summary.status : "not_recorded",
    energyJoules: finiteNumber(summary?.energy_joules),
    idleWatts: finiteNumber(summary?.idle_baseline_watts),
    scope: typeof summary?.scope === "string" ? summary.scope : null,
    reason: typeof summary?.reason === "string" ? summary.reason : null,
  };
}

export const powerScopeLabel = (scope: string | null): string => ({
  processor_package: "Processor package",
  accelerator: "Accelerator",
  cpu_package: "CPU package",
  whole_system: "Whole system",
  mixed: "Mixed scopes",
}[scope || ""] || "Scope not recorded");
