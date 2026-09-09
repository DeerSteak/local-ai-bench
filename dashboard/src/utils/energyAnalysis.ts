import { FILE_COLORS } from "../constants";
import { entriesOf, modelLabel, imageModelLabel, embedModelLabel } from "./shared";
import type { JsonRecord } from "./shared";
import { powerScopeLabel } from "./power";
import { llamaBenchPrefillEntries, llamaBenchDecodeEntries, llamaBenchPromptLabel } from "./llamabench";
import type { ChartRow, LineConfig, ResultsFile } from "../types";

export const ENERGY_SECTIONS = ["llamabench", "llamabenchconc", "images", "embeddings"];
const UNITS: Record<string, string> = {
  tokens_per_joule: "Tokens / Joule", images_per_joule: "Images / Joule",
  embeddings_per_joule: "Embeddings / Joule",
};
const SCOPES = new Set(["processor_package", "accelerator", "cpu_package", "whole_system"]);
const positive = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value > 0;

interface EnergyCase {
  sample: JsonRecord[string]; phase: string; label: string; order: number;
}
export interface EnergyChartGroup {
  id: string; model: string; description: string; unit: string;
  data: ChartRow[]; configs: LineConfig[];
}

function energyCases(section: string, data: JsonRecord[string]): EnergyCase[] {
  if (section === "llamabench") {
    return [
      ...llamaBenchPrefillEntries(data).filter(sample => sample && typeof sample === "object").map(sample => ({
        sample, phase: `Prefill · ${sample.completed_reps ?? "unknown"} repetitions`,
        label: llamaBenchPromptLabel(sample.n_prompt), order: sample.n_prompt,
      })),
      ...llamaBenchDecodeEntries(data).filter(sample => sample && typeof sample === "object").map(sample => ({
        sample, phase: `Decode · ${sample.n_gen} generated tokens · ${sample.completed_reps ?? "unknown"} repetitions`,
        label: llamaBenchPromptLabel(sample.n_depth), order: sample.n_depth,
      })),
    ];
  }
  if (section === "llamabenchconc") {
    return (Array.isArray(data?.entries) ? data.entries : []).filter((sample: JsonRecord[string]) => sample && typeof sample === "object").map((sample: JsonRecord[string]) => ({
      sample, phase: `Concurrency · pp ${sample.pp ?? data.pp ?? "unknown"} · tg ${sample.tg ?? "unknown"}`,
      label: `${sample.pl}-way`, order: sample.pl,
    }));
  }
  const resolutions = Object.keys(data?.resolutions || {}).sort().join(", ");
  return [{ sample: data, phase: section === "images"
    ? `All measured resolutions: ${resolutions || "not recorded"} · ${data?.steps ?? "unknown"} steps`
    : "Measured embedding workload", label: "Measured workload", order: 0 }];
}

export function buildEnergyAnalysis(
  files: ResultsFile[], section: string, enabled: Set<string>, bySystem = false,
): { groups: EnergyChartGroup[], notices: string[] } {
  const groups = new Map<string, EnergyChartGroup>();
  const notices = new Set<string>();
  const seenCases = new Set<string>();
  if (!ENERGY_SECTIONS.includes(section)) return { groups: [], notices: [] };
  const expectedUnit = section === "images" ? "images_per_joule"
    : section === "embeddings" ? "embeddings_per_joule" : "tokens_per_joule";
  files.forEach((file, fi) => {
    for (const [model, data] of entriesOf(file.data[section])) {
      if (!enabled.has(model) || !data) continue;
      const label = section === "images" ? imageModelLabel(model)
        : section === "embeddings" ? embedModelLabel(model) : modelLabel(model);
      const identity = `${file.hostname || "Unknown system"} · ${label}`;
      const cases = energyCases(section, data);
      if (!cases.length) notices.add(`${identity}: energy not recorded for these cases.`);
      for (const entry of cases) {
        if (typeof entry.order !== "number" || !Number.isFinite(entry.order) || entry.order < 0) {
          notices.add(`${identity}: energy case dimensions are not recorded.`);
          continue;
        }
        const power = entry.sample?.power;
        if (!SCOPES.has(power?.scope)) {
          const reason = typeof power?.reason === "string" ? power.reason
            : !power ? "not recorded; requires power telemetry during the run"
              : "power scope unavailable";
          notices.add(`${identity}: ${reason}.`);
          continue;
        }
        const names: string[] = (Array.isArray(power.windows) ? power.windows : [])
          .map((window: JsonRecord[string]) => window?.name).filter((name: unknown): name is string => typeof name === "string");
        const basis = names.some(name => name.startsWith("measured:") && (name.includes("includes-load") || name.startsWith("measured:native-sweep")))
          ? "Full case, including model load"
          : names.some(name => name.startsWith("measured:")) ? "Measured work, excluding model load" : "Measurement window not recorded";
        const id = JSON.stringify([bySystem ? fi : null, model, entry.phase, power.scope, basis]);
        let group = groups.get(id);
        if (!group) {
          group = { id, model: bySystem ? identity : label,
            description: `${entry.phase} · ${powerScopeLabel(power.scope)} · ${basis}`,
            unit: UNITS[expectedUnit], data: [], configs: [] };
          groups.set(id, group);
        }
        let row = group.data.find(row => row.caseLabel === entry.label);
        if (!row) {
          row = { caseLabel: entry.label, order: entry.order };
          group.data.push(row);
        }
        const key = `f${fi}`;
        const caseId = JSON.stringify([id, fi, entry.label]);
        if (seenCases.has(caseId)) {
          notices.add(`${identity}: duplicate energy case ${entry.label}; ambiguous case omitted.`);
          row[`${key}_energy`] = null;
          row[key] = null;
          continue;
        }
        seenCases.add(caseId);
        if (power.status !== "recorded" || !positive(power.energy_joules)) {
          notices.add(`${identity}: ${typeof power.reason === "string" ? power.reason : "valid measured energy unavailable"}.`);
          row[key] = null;
          row[`${key}_energy`] = null;
          continue;
        }
        row[`${key}_energy`] = power.energy_joules;
        const efficiency = power.efficiency;
        if (efficiency?.unit === expectedUnit && positive(efficiency.per_joule) && positive(efficiency.work_count)) {
          row[key] = efficiency.per_joule;
        } else {
          notices.add(`${identity}: valid ${UNITS[expectedUnit]} not recorded.`);
        }
        if (!group.configs.some(config => config.dataKey === key)) group.configs.push({
          dataKey: key, name: file.hostname || "Unknown system", stroke: FILE_COLORS[fi % FILE_COLORS.length],
        });
      }
    }
  });
  return { groups: [...groups.values()].map(group => ({
    ...group, data: group.data.sort((a, b) => a.order - b.order),
  })).filter(group => energyChartSeries(group, true).length > 0), notices: [...notices] };
}

export function energyChartSeries(group: EnergyChartGroup, energy = false): LineConfig[] {
  return group.configs.map(config => ({
    ...config, dataKey: energy ? `${config.dataKey}_energy` : config.dataKey,
  })).filter(config => group.data.some(row => positive(row[config.dataKey])));
}
