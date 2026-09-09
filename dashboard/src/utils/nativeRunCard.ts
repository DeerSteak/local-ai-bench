import { SIZE_TIER_ORDER } from "../constants";
import type { ResultsFile } from "../types";
import { entriesOf, getModelSizeTier, modelLabel } from "./shared";
import type { JsonRecord } from "./shared";
import { runHeadroomSummary } from "./memory";
import { runPowerSummary } from "./power";

interface NativeCase {
  model: string; sample: JsonRecord; kind: string; dimensions: number[]; checkpoint: string;
}
export interface NativeLeader {
  tier: string; kind: string; checkpoint: string; model: string; value: number;
}
const positive = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v > 0;

function nativeCases(file: ResultsFile, section: string): NativeCase[] {
  const cases: NativeCase[] = [];
  for (const [model, data] of entriesOf(file.data[section])) {
    if (!data || data.error || data.skipped) continue;
    const arrays = section === "llamabenchconc" ? [data.entries]
      : Array.isArray(data.prefill_entries) || Array.isArray(data.decode_entries)
        ? [data.prefill_entries, data.decode_entries] : [data.entries];
    for (const array of arrays) {
      if (!Array.isArray(array)) continue;
      for (const sample of array) {
        if (!sample || typeof sample !== "object" || sample.error || sample.skipped) continue;
        if (section === "llamabenchconc") {
          const pp = sample.pp ?? data.pp, tg = sample.tg, pl = sample.pl;
          if (![pp, tg, pl].every(positive)) continue;
          cases.push({ model, sample, kind: "Aggregate decode", dimensions: [-pl, pp, tg],
            checkpoint: `${pl}-way · pp${pp} · tg${tg}` });
        } else {
          const pp = sample.n_prompt ?? 0, tg = sample.n_gen ?? 0, depth = sample.n_depth ?? 0;
          const reps = sample.completed_reps;
          const suffix = positive(reps) ? ` · ${reps} repetitions` : " · repetitions not recorded";
          if (positive(pp) && tg === 0) cases.push({ model, sample, kind: "Prefill",
            dimensions: [pp], checkpoint: `pp${pp}${suffix}` });
          else if (pp === 0 && positive(tg) && positive(depth)) cases.push({ model, sample, kind: "Decode",
            dimensions: [depth, tg], checkpoint: `depth${depth} · tg${tg}${suffix}` });
          else if (positive(pp) && positive(tg)) cases.push({ model, sample, kind: "Combined",
            dimensions: [pp, tg], checkpoint: `pp${pp} · tg${tg}${suffix}` });
        }
      }
    }
  }
  const counts = new Map<string, number>();
  const key = (item: NativeCase) => JSON.stringify([item.model, item.kind, item.checkpoint]);
  for (const item of cases) counts.set(key(item), (counts.get(key(item)) ?? 0) + 1);
  return cases.filter(item => counts.get(key(item)) === 1);
}

export function nativeRunCardSummary(file: ResultsFile, section: string): {
  leaders: NativeLeader[]; headroom: ReturnType<typeof runHeadroomSummary>; power: ReturnType<typeof runPowerSummary>;
} {
  const cases = nativeCases(file, section);
  const groups = new Map<string, { item: NativeCase; tier: string; values: Map<string, number> }>();
  for (const item of cases) {
    const tier = getModelSizeTier(item.model);
    const value = section === "llamabenchconc" ? item.sample.speed_tg : item.sample.avg_ts;
    if (!positive(value)) continue;
    const key = JSON.stringify([tier, item.kind, item.checkpoint]);
    const group = groups.get(key) ?? { item, tier, values: new Map<string, number>() };
    group.values.set(item.model, group.values.has(item.model) ? NaN : value);
    groups.set(key, group);
  }
  const leaders: NativeLeader[] = [];
  for (const tier of [...SIZE_TIER_ORDER, "unknown"]) {
    for (const kind of ["Prefill", "Decode", "Combined", "Aggregate decode"]) {
      const candidates = [...groups.values()].filter(group => group.tier === tier && group.item.kind === kind)
        .map(group => ({ ...group, valid: [...group.values].filter(([, value]) => positive(value)) }))
        .filter(group => group.valid.length)
        .sort((a, b) => b.valid.length - a.valid.length
          || a.item.dimensions.reduce((delta, value, index) => delta || value - b.item.dimensions[index], 0)
          || a.item.checkpoint.localeCompare(b.item.checkpoint));
      const chosen = candidates[0];
      if (!chosen) continue;
      const [model, value] = chosen.valid.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0];
      leaders.push({ tier, kind, checkpoint: chosen.item.checkpoint, model: modelLabel(model), value });
    }
  }
  const headroom: ReturnType<typeof runHeadroomSummary> = { state: "not_recorded", absoluteGb: null, casePath: null };
  for (const item of cases) {
    const memory = item.sample.memory?.headroom;
    if (typeof memory?.absolute_gb === "number" && Number.isFinite(memory.absolute_gb)
        && (headroom.absoluteGb == null || memory.absolute_gb < headroom.absoluteGb)) {
      headroom.absoluteGb = memory.absolute_gb;
      headroom.state = typeof memory.state === "string" ? memory.state : "not_recorded";
      headroom.casePath = `${item.model} · ${item.checkpoint}`;
    }
  }
  const measured = cases.map(item => item.sample.power).filter(power => power?.status === "recorded"
    && positive(power.energy_joules) && ["accelerator", "processor_package", "cpu_package", "whole_system"].includes(power.scope));
  const scopes = new Set(measured.map(power => power.scope));
  const power: ReturnType<typeof runPowerSummary> = { status: "not_recorded", energyJoules: null,
    idleWatts: null, scope: null, reason: "Not recorded for this tab" };
  if (scopes.size > 1) Object.assign(power, { status: "unavailable", scope: "mixed", reason: "Mixed scopes; no total" });
  else if (measured.length) Object.assign(power, { status: "recorded", scope: measured[0].scope,
    energyJoules: measured.reduce((sum, value) => sum + value.energy_joules, 0),
    reason: measured.length < cases.length ? "Recorded cases only; some energy unavailable" : null });
  return { leaders, headroom, power };
}
