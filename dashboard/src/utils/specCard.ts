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

export function runCardHostname(file: DisplayFile): string {
  const hostname = String(file.hostname || "").split("\n")[0];
  if (!file.os.toLowerCase().startsWith("windows")) return hostname;
  return hostname
    .replace(/\s*\/\s*\d+(?:\.\d+)?\s*GB RAM.*$/i, "")
    .replace(/^Intel(?:\(R\))?\s+/i, "")
    .replace(/Core(?:\(TM\))?\s+/i, "Core ")
    .replace(/^AMD\s+/i, "")
    .replace(/\s+\d+-core\s+Processor$/i, "")
    .replace(/\s+Plus$/i, "")
    .trim();
}

function compactGpuLabel(label: string): string {
  return label
    .replace(/^(?:NVIDIA|AMD|Intel(?:\(R\))?)\s+/i, "")
    .replace(/\s*\/\s*\d+(?:\.\d+)?\s*GB VRAM.*$/i, "")
    .trim();
}

function countedGpuLabels(labels: string[]): string[] {
  const counts = new Map<string, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  return [...counts].map(([label, count]) => count > 1 ? `${count}x ${label}` : label);
}

function llamaBenchGpuLabels(file: DisplayFile): string[] {
  for (const [, modelData] of entriesOf(file.data.llamabench)) {
    for (const entryKey of ["prefill_entries", "decode_entries"]) {
      const samples = modelData[entryKey];
      if (!Array.isArray(samples)) continue;
      for (const sample of samples) {
        if (!sample || typeof sample !== "object") continue;
        const gpuInfo = (sample as JsonRecord).gpu_info;
        if (typeof gpuInfo === "string" && gpuInfo.trim()) {
          return gpuInfo.split(/\s*,\s*/).map(compactGpuLabel).filter(Boolean);
        }
      }
    }
  }
  return [];
}

export function runCardGpuLabels(file: DisplayFile): string[] {
  const profile = file.data.profile;
  const recorded = profile?.gpu;
  if (typeof recorded === "string" && recorded.trim()) return [compactGpuLabel(recorded)];
  if (Array.isArray(recorded)) {
    const labels = recorded.filter((gpu): gpu is string => typeof gpu === "string" && Boolean(gpu.trim()))
      .map(compactGpuLabel);
    if (labels.length) return countedGpuLabels(labels);
  }
  const llamaBenchLabels = llamaBenchGpuLabels(file);
  if (llamaBenchLabels.length) return countedGpuLabels(llamaBenchLabels);
  const recordedHostname = typeof profile?.hostname === "string" ? profile.hostname : file.hostname;
  return countedGpuLabels(String(recordedHostname || "").split("\n").slice(1)
    .map(compactGpuLabel).filter(Boolean));
}

export function dashboardHostname(file: DisplayFile): string {
  if (!file.os.toLowerCase().startsWith("windows")) return String(file.hostname || "");
  const recordedHostname = typeof file.data.profile?.hostname === "string"
    ? file.data.profile.hostname : String(file.hostname || "");
  const capacityLabel = (kind: "RAM" | "VRAM") => {
    const match = recordedHostname.match(new RegExp(`\\b(\\d+(?:\\.\\d+)?)\\s*GB ${kind}\\b`, "i"));
    return match ? `${Math.round(Number(match[1]))} GB ${kind}` : "";
  };
  const ram = capacityLabel("RAM");
  const vram = capacityLabel("VRAM");
  const cpuLabel = [runCardHostname(file), ram].filter(Boolean).join(" / ");
  const gpuLabel = [runCardGpuLabels(file).join(", "), vram].filter(Boolean).join(" / ");
  return [cpuLabel, gpuLabel].filter(Boolean).join("\n");
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
