import { describe, expect, it } from "vitest";

import { buildRunCardFilename, buildSpecCardSummary } from "./specCard";

function file(llm: object) {
  return {
    name: "result.json", backend: "cuda", os: "Linux", wsl: false, ram_gb: 32,
    version: "5.0", timestamp: null, reliabilityWarning: "", data: { llm },
  };
}

describe("buildSpecCardSummary", () => {
  it("selects independent speed and latency winners within each tier at 2K", () => {
    const summary = buildSpecCardSummary(file({
      "gemma3-1b": { "2K": { tps_mean: 80, client_ttft_mean_sec: 0.4 } },
      "granite4.1-3b-q4": { "2K": { tps_mean: 70, client_ttft_mean_sec: 0.2 } },
      "granite4.1-8b-q4": { "2K": { tps_mean: 40, client_ttft_mean_sec: 0.5 } },
    }));
    expect(summary[0]).toMatchObject({
      tier: "xsmall", checkpoint: "2K", fastest: { value: 80 }, lowestTtft: { value: 0.2 },
    });
    expect(summary[1]).toMatchObject({ tier: "small", fastest: { value: 40 } });
  });

  it("falls back to the shallowest canonical checkpoint and ignores unusable models", () => {
    const summary = buildSpecCardSummary(file({
      "gemma3-1b": { "8K": { tps_mean: 80, client_ttft_mean_sec: 0.4 } },
      unknown: { "2K": { tps_mean: 90, client_ttft_mean_sec: 0.1 } },
      "granite4.1-3b-q4": { "2K": { tps_mean: Number.NaN, client_ttft_mean_sec: 0.2 } },
    }));
    expect(summary).toEqual([{
      tier: "xsmall", checkpoint: "8K",
      fastest: { model: "Gemma 3 1B", value: 80 },
      lowestTtft: { model: "Gemma 3 1B", value: 0.4 },
    }]);
  });
});

describe("buildRunCardFilename", () => {
  it("uses the export suffix and disambiguates repeated hostnames", () => {
    const names = ["My Host", "My Host"];
    expect(buildRunCardFilename(names, 0, "before upgrade")).toBe(
      "My-Host_before-upgrade_run-card.png",
    );
    expect(buildRunCardFilename(names, 1, "after upgrade")).toBe(
      "My-Host_2_after-upgrade_run-card.png",
    );
  });
});
