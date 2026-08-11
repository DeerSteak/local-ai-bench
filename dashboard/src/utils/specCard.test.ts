import { describe, expect, it } from "vitest";

import type { DisplayFile } from "../types";
import {
  buildRunCardFilename, buildSpecCardSummary, dashboardHostname, runCardGpuLabels, runCardHostname,
} from "./specCard";

function file(llm: object): DisplayFile {
  return {
    name: "result.json", hostname: "Host", backend: "cuda", os: "Linux", wsl: false, ram_gb: 32,
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

describe("runCardGpuLabels", () => {
  it("prefers explicit multi-GPU profile metadata", () => {
    const result = file({});
    result.data.profile = { gpu: ["NVIDIA RTX 5090", "RTX 5090", "AMD Radeon RX 9070 XT"] };
    expect(runCardGpuLabels(result)).toEqual(["2x RTX 5090", "Radeon RX 9070 XT"]);
  });

  it("falls back to the hardware lines embedded in legacy hostnames", () => {
    const result = file({});
    result.hostname = "AMD Ryzen / 64 GB RAM\nNVIDIA RTX 5080 / 32 GB VRAM";
    result.data.profile = { hostname: result.hostname };
    expect(runCardGpuLabels(result)).toEqual(["RTX 5080"]);
  });

  it("uses legacy llama-bench device metadata before aggregated hostname VRAM", () => {
    const result = file({});
    result.hostname = "Core Ultra / 64 GB RAM\nNVIDIA GeForce RTX 5060 Ti / 31.8 GB VRAM";
    result.data.profile = { hostname: result.hostname };
    result.data.llamabench = {
      model: { prefill_entries: [{
        gpu_info: "NVIDIA GeForce RTX 5060 Ti, NVIDIA GeForce RTX 5060 Ti",
      }] },
    };
    expect(runCardGpuLabels(result)).toEqual(["2x GeForce RTX 5060 Ti"]);
  });
});

describe("runCardHostname", () => {
  it.each([
    ["Intel(R) Core(TM) Ultra 7 270K Plus / 64 GB RAM", "Core Ultra 7 270K"],
    ["AMD Ryzen 7 9850X3D 8-Core Processor / 64 GB RAM", "Ryzen 7 9850X3D"],
  ])("compacts Windows processor labels", (hostname, expected) => {
    const result = file({});
    result.os = "Windows 11";
    result.hostname = hostname;
    expect(runCardHostname(result)).toBe(expected);
  });

  it("preserves non-Windows hostnames", () => {
    const result = file({});
    result.hostname = "AMD Ryzen workstation / 64 GB RAM";
    expect(runCardHostname(result)).toBe(result.hostname);
  });
});

describe("dashboardHostname", () => {
  it("compacts both lines of a Windows hardware label", () => {
    const result = file({});
    result.os = "Windows 11";
    result.hostname = "Intel(R) Core(TM) Ultra 7 270K Plus / 64 GB RAM\nNVIDIA GeForce RTX 5060 Ti / 31.8 GB VRAM";
    result.data.profile = { hostname: result.hostname };
    result.data.llamabench = { model: { prefill_entries: [{
      gpu_info: "NVIDIA GeForce RTX 5060 Ti, NVIDIA GeForce RTX 5060 Ti",
    }] } };
    expect(dashboardHostname(result)).toBe(
      "Core Ultra 7 270K\n2x GeForce RTX 5060 Ti\n64 GB RAM / 32 GB VRAM",
    );
  });

  it("preserves the complete non-Windows hostname", () => {
    const result = file({});
    result.hostname = "Linux host\nNVIDIA GPU";
    expect(dashboardHostname(result)).toBe(result.hostname);
  });
});
