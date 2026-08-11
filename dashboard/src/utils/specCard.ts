import { CTX_ORDER, MODEL_SIZE_TIER, SIZE_TIER_ORDER, SPEC_CARD_PREFERRED_CTX } from "../constants";
import type { DisplayFile } from "../types";
import { entriesOf, lookup, modelLabel, sanitizeForFilename } from "./shared";
import type { JsonRecord } from "./shared";
import { llmTTFTMean } from "./llm";

interface Winner { model: string; value: number }

export interface TierSummary {
  tier: string;
  checkpoint: string;
  fastest: Winner;
  lowestTtft: Winner;
}

export function buildRunCardFilename(names: string[], index: number, suffix: string): string {
  const name = sanitizeForFilename(names[index] || `run-${index + 1}`);
  const occurrence = names.slice(0, index + 1)
    .filter(candidate => sanitizeForFilename(candidate) === name).length;
  const parts = [name, occurrence > 1 ? String(occurrence) : "", sanitizeForFilename(suffix), "run-card"];
  return `${parts.filter(Boolean).join("_")}.png`;
}

export function runCardGpuLabels(file: DisplayFile): string[] {
  const profile = file.data.profile;
  const recorded = profile?.gpu;
  if (typeof recorded === "string" && recorded.trim()) return [recorded.trim()];
  if (Array.isArray(recorded)) {
    const labels = recorded.filter((gpu): gpu is string => typeof gpu === "string" && Boolean(gpu.trim()))
      .map(gpu => gpu.trim());
    if (labels.length) return [...new Set(labels)];
  }
  const recordedHostname = typeof profile?.hostname === "string" ? profile.hostname : file.hostname;
  return [...new Set(String(recordedHostname || "").split("\n").slice(1).map(line => line.trim()).filter(Boolean))];
}

function winner(entries: Winner[], direction: "max" | "min"): Winner | null {
  if (!entries.length) return null;
  return entries.reduce((best, entry) =>
    direction === "max" ? (entry.value > best.value ? entry : best)
      : (entry.value < best.value ? entry : best));
}

export function buildSpecCardSummary(file: DisplayFile): TierSummary[] {
  const tiers = new Map<string, { model: string, contexts: JsonRecord }[]>();
  for (const [model, contexts] of entriesOf(file.data.llm)) {
    const tier = lookup(MODEL_SIZE_TIER, model);
    if (!tier) continue;
    const values = tiers.get(tier) ?? [];
    values.push({ model, contexts });
    tiers.set(tier, values);
  }
  return SIZE_TIER_ORDER.flatMap(tier => {
    const models = tiers.get(tier) ?? [];
    const checkpoints = [SPEC_CARD_PREFERRED_CTX,
      ...CTX_ORDER.filter(context => context !== SPEC_CARD_PREFERRED_CTX)];
    for (const checkpoint of checkpoints) {
      const tps: Winner[] = [];
      const ttft: Winner[] = [];
      for (const entry of models) {
        const sample = entry.contexts[checkpoint];
        if (!sample || typeof sample !== "object") continue;
        if (typeof sample.tps_mean === "number" && Number.isFinite(sample.tps_mean))
          tps.push({ model: modelLabel(entry.model), value: sample.tps_mean });
        const value = llmTTFTMean(sample);
        if (typeof value === "number" && Number.isFinite(value))
          ttft.push({ model: modelLabel(entry.model), value });
      }
      const fastest = winner(tps, "max");
      const lowestTtft = winner(ttft, "min");
      if (fastest && lowestTtft) return [{ tier, checkpoint, fastest, lowestTtft }];
    }
    return [];
  });
}
