import { describe, it, expect } from "vitest";
import {
  llamaBenchCheckpointKey, llamaBenchCheckpointSortValue, llamaBenchPromptLabel,
  llamaBenchTgValues, llamaBenchTgValuesByModel,
  buildLlamaBenchBarData, buildLlamaBenchBarConfigs, buildLlamaBenchLineData,
  buildLlamaBenchBarDataByModel, buildLlamaBenchBarConfigsByModel,
  buildLlamaBenchLineDataByCheckpoint, buildLlamaBenchLineConfigsByCheckpoint,
  flattenLlamaBenchData,
} from "./llamabench";

describe("llamaBenchCheckpointKey", () => {
  it("labels a pp+tg combo, the shape every real sweep actually produces", () => {
    expect(llamaBenchCheckpointKey({ n_prompt: 2048, n_gen: 128 })).toBe("pp2048+tg128");
  });
  it("labels a pure prompt-processing entry (n_gen 0)", () => {
    expect(llamaBenchCheckpointKey({ n_prompt: 512, n_gen: 0 })).toBe("pp512");
  });
  it("labels a pure generation entry (n_prompt 0)", () => {
    expect(llamaBenchCheckpointKey({ n_prompt: 0, n_gen: 128 })).toBe("tg128");
  });
});

describe("llamaBenchCheckpointSortValue", () => {
  it("orders primarily by prompt size, then generation size", () => {
    const a = llamaBenchCheckpointSortValue({ n_prompt: 2048, n_gen: 512 });
    const b = llamaBenchCheckpointSortValue({ n_prompt: 8192, n_gen: 128 });
    expect(a).toBeLessThan(b);
  });
});

describe("llamaBenchPromptLabel", () => {
  it("formats a binary-K prompt size as \"<K>K\"", () => {
    expect(llamaBenchPromptLabel(2048)).toBe("2K");
    expect(llamaBenchPromptLabel(98304)).toBe("96K");
  });
  it("keeps a fractional K for sizes under 1024", () => {
    expect(llamaBenchPromptLabel(512)).toBe("0.5K");
  });
});

describe("llamaBenchTgValues", () => {
  it("returns distinct generation sizes across files, sorted ascending", () => {
    const files = [
      { data: { llamabench: { m: { entries: [{ n_prompt: 2048, n_gen: 512 }, { n_prompt: 2048, n_gen: 128 }] } } } },
      { data: { llamabench: { m: { entries: [{ n_prompt: 8192, n_gen: 128 }] } } } },
    ];
    expect(llamaBenchTgValues(files, "m")).toEqual([128, 512]);
  });
});

describe("llamaBenchTgValuesByModel", () => {
  it("returns distinct generation sizes across every model in the group, sorted ascending", () => {
    const file = { data: { llamabench: {
      m1: { entries: [{ n_prompt: 2048, n_gen: 512 }] },
      m2: { entries: [{ n_prompt: 2048, n_gen: 128 }] },
    } } };
    expect(llamaBenchTgValuesByModel(file, ["m1", "m2"])).toEqual([128, 512]);
  });
});

describe("buildLlamaBenchBarData", () => {
  it("keys each row by hostname and each pp size's avg_ts, filtered to the given tg", () => {
    const files = [{
      hostname: "TestHost",
      data: { llamabench: { m: { entries: [
        { n_prompt: 2048, n_gen: 128, avg_ts: 100.5 },
        { n_prompt: 8192, n_gen: 128, avg_ts: 50.25 },
        { n_prompt: 2048, n_gen: 512, avg_ts: 90 },
      ] } } },
    }];
    const rows = buildLlamaBenchBarData(files, "m", 128);
    expect(rows).toEqual([{ systemLabel: "TestHost", "2K": 100.5, "8K": 50.25 }]);
  });
  it("produces an empty row (just the system label) for a model with an error instead of entries", () => {
    const files = [{ hostname: "TestHost", data: { llamabench: { m: { error: "timed out after 1800s" } } } }];
    expect(buildLlamaBenchBarData(files, "m", 128)).toEqual([{ systemLabel: "TestHost" }]);
  });
});

describe("buildLlamaBenchBarConfigs", () => {
  it("orders pp sizes numerically, not insertion order, filtered to the given tg", () => {
    const files = [{ data: { llamabench: { m: { entries: [
      { n_prompt: 8192, n_gen: 128 },
      { n_prompt: 2048, n_gen: 512 },
      { n_prompt: 2048, n_gen: 128 },
    ] } } } }];
    const configs = buildLlamaBenchBarConfigs(files, "m", 128);
    expect(configs.map(c => c.dataKey)).toEqual(["2K", "8K"]);
  });
  it("aggregates pp sizes across files, so one file stopping early still gets the other's columns", () => {
    const files = [
      { data: { llamabench: { m: { entries: [{ n_prompt: 2048, n_gen: 128 }] } } } },
      { data: { llamabench: { m: { entries: [{ n_prompt: 2048, n_gen: 128 }, { n_prompt: 8192, n_gen: 128 }] } } } },
    ];
    const configs = buildLlamaBenchBarConfigs(files, "m", 128);
    expect(configs.map(c => c.dataKey)).toEqual(["2K", "8K"]);
  });
});

describe("buildLlamaBenchLineData", () => {
  it("transposes the bar shape: one row per pp size, one column per file, filtered to the given tg", () => {
    const files = [
      { data: { llamabench: { m: { entries: [
        { n_prompt: 2048, n_gen: 128, avg_ts: 100 },
        { n_prompt: 8192, n_gen: 128, avg_ts: 50 },
        { n_prompt: 2048, n_gen: 512, avg_ts: 80 },
      ] } } } },
      { data: { llamabench: { m: { entries: [
        { n_prompt: 2048, n_gen: 128, avg_ts: 120 },
      ] } } } },
    ];
    const rows = buildLlamaBenchLineData(files, "m", 128);
    expect(rows).toEqual([
      { promptLabel: "2K", f0: 100, f1: 120 },
      { promptLabel: "8K", f0: 50 },
    ]);
  });
  it("returns no rows for a model with no entries anywhere", () => {
    const files = [{ data: { llamabench: { m: { error: "timed out" } } } }];
    expect(buildLlamaBenchLineData(files, "m", 128)).toEqual([]);
  });
});

describe("buildLlamaBenchBarDataByModel", () => {
  it("keys each row by model label and each pp size's avg_ts, for one file/tg", () => {
    const file = { data: { llamabench: {
      m1: { entries: [{ n_prompt: 2048, n_gen: 128, avg_ts: 100 }] },
      m2: { entries: [{ n_prompt: 2048, n_gen: 128, avg_ts: 80 }, { n_prompt: 8192, n_gen: 128, avg_ts: 40 }] },
    } } };
    const rows = buildLlamaBenchBarDataByModel(file, ["m1", "m2"], 128);
    expect(rows).toEqual([
      { modelLabel: "m1", "2K": 100 },
      { modelLabel: "m2", "2K": 80, "8K": 40 },
    ]);
  });
});

describe("buildLlamaBenchBarConfigsByModel", () => {
  it("orders pp sizes numerically across every model in the group, filtered to the given tg", () => {
    const file = { data: { llamabench: {
      m1: { entries: [{ n_prompt: 8192, n_gen: 128 }] },
      m2: { entries: [{ n_prompt: 2048, n_gen: 128 }] },
    } } };
    const configs = buildLlamaBenchBarConfigsByModel(file, ["m1", "m2"], 128);
    expect(configs.map(c => c.dataKey)).toEqual(["2K", "8K"]);
  });
});

describe("buildLlamaBenchLineDataByCheckpoint", () => {
  it("transposes the by-model bar shape: one row per pp size, one column per model, filtered to the given tg", () => {
    const file = { data: { llamabench: {
      m1: { entries: [{ n_prompt: 2048, n_gen: 128, avg_ts: 100 }, { n_prompt: 8192, n_gen: 128, avg_ts: 50 }] },
      m2: { entries: [{ n_prompt: 2048, n_gen: 128, avg_ts: 120 }] },
    } } };
    const rows = buildLlamaBenchLineDataByCheckpoint(file, ["m1", "m2"], 128);
    expect(rows).toEqual([
      { promptLabel: "2K", m1: 100, m2: 120 },
      { promptLabel: "8K", m1: 50 },
    ]);
  });
});

describe("buildLlamaBenchLineConfigsByCheckpoint", () => {
  it("omits a model with no data in the given rows", () => {
    const data = [{ promptLabel: "2K", m1: 100 }];
    const configs = buildLlamaBenchLineConfigsByCheckpoint(["m1", "m2"], data);
    expect(configs.map(c => c.dataKey)).toEqual(["m1"]);
  });
});

describe("flattenLlamaBenchData", () => {
  it("produces one row per checkpoint entry", () => {
    const files = [{
      id: "f1",
      data: { llamabench: { m: { entries: [
        { n_prompt: 2048, n_gen: 128, avg_ts: 100, stddev_ts: 2, n_gpu_layers: 999 },
      ] } } },
    }];
    expect(flattenLlamaBenchData(files)).toEqual([
      { _fileId: "f1", model: "m", ckpt: "pp2048+tg128", avg_ts: 100, stddev_ts: 2, n_gpu_layers: 999 },
    ]);
  });
  it("produces a single skipped row for a model that errored instead of one row per checkpoint", () => {
    const files = [{ id: "f1", data: { llamabench: { m: { error: "llama-bench not found" } } } }];
    expect(flattenLlamaBenchData(files)).toEqual([
      { _fileId: "f1", model: "m", ckpt: "—", skipped: true, skip_detail: "llama-bench not found" },
    ]);
  });
});
