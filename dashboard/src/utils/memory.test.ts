import { describe, expect, it } from "vitest";

import {
  buildProcessMemoryDataForModel, getMemoryRecordingState, memoryChannelPeak,
  memoryFields, memoryHeadroom, memoryHeadroomState, runHeadroomSummary,
} from "./memory";

const memory = {
  summary: {
    host_ram_used_gb: { peak_gb: 20 },
    process_rss_gb: { peak_gb: 4 },
    accelerator_memory_used_gb: { peak_gb: 8 },
  },
  headroom: { absolute_gb: 14, fraction: 0.5, state: "comfortable" },
};

describe("memory telemetry", () => {
  it("reads normalized channel peaks and headroom", () => {
    const sample = { memory };
    expect(memoryChannelPeak(sample, "host_ram_used_gb")).toBe(20);
    expect(memoryHeadroom(sample)).toBe(14);
    expect(memoryHeadroomState(sample)).toBe("comfortable");
    expect(memoryFields(sample)).toEqual({
      host_ram_peak_gb: 20,
      process_rss_peak_gb: 4,
      accelerator_memory_peak_gb: 8,
      headroom_gb: 14,
      headroom_state: "comfortable",
    });
    expect(memoryFields({})).toEqual({});
  });

  it("treats missing, malformed, and non-finite values as not recorded", () => {
    expect(memoryChannelPeak({}, "process_rss_gb")).toBeNull();
    expect(memoryChannelPeak({ memory: { summary: {
      process_rss_gb: { peak_gb: Number.NaN },
    } } }, "process_rss_gb")).toBeNull();
    expect(memoryHeadroom({ memory: { headroom: { absolute_gb: "4" } } })).toBeNull();
    expect(memoryHeadroomState({})).toBe("not_recorded");
  });

  it("builds process RSS chart data without zero bars for old files", () => {
    const files = [
      { data: { llm: { model: { "2K": { memory }, "8K": {} } } } },
      { data: { llm: { model: { "2K": { tps_mean: 10 } } } } },
    ];
    expect(buildProcessMemoryDataForModel(files, "model")).toEqual([
      { ctxLabel: "2K", f0: 4 }, { ctxLabel: "8K" },
    ]);
    expect(getMemoryRecordingState(files[0])).toBe("recorded");
    expect(getMemoryRecordingState(files[1])).toBe("not_recorded");
  });

  it("summarizes run headroom and names missing telemetry explicitly", () => {
    expect(runHeadroomSummary({ data: { run: { memory_summary: {
      tightest_headroom: { state: "tight", absolute_gb: 2, case_path: "llm/model/8K" },
    } } } })).toEqual({ state: "tight", absoluteGb: 2, casePath: "llm/model/8K" });
    expect(runHeadroomSummary({ data: {} })).toEqual({
      state: "not_recorded", absoluteGb: null, casePath: null,
    });
  });
});
