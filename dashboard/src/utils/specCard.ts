import { CTX_ORDER, MODEL_SIZE_TIER, SIZE_TIER_ORDER } from "../constants";
import type { DisplayFile } from "../types";
import { entriesOf, lookup, modelLabel } from "./shared";
import type { JsonRecord } from "./shared";
import { llmTTFTMean } from "./llm";

interface Winner { model: string; value: number }

export interface TierSummary {
  tier: string;
  checkpoint: string;
  fastest: Winner;
  lowestTtft: Winner;
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
    const checkpoints = ["2K", ...CTX_ORDER.filter(context => context !== "2K")];
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
