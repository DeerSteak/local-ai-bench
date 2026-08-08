import { describe, it, expect } from "vitest";
import {
  llamaBenchPromptLabel,
  llamaBenchPrefillEntries,
  llamaBenchDecodeEntries,
  llamaBenchHasCombinedOnly,
  buildLlamaBenchPrefillLineData,
  buildLlamaBenchDecodeLineData,
  buildLlamaBenchDecodeLineConfigs,
  buildLlamaBenchPrefillLineDataByModel,
  buildLlamaBenchDecodeLineDataByModel,
  buildLlamaBenchPrefillLineConfigsByModel,
  buildLlamaBenchDecodeLineConfigsByModel,
  buildLlamaBenchPrefillLineConfigs,
  flattenLlamaBenchData,
} from "./llamabench";

const prefill = (pp: number, speed: number) => ({
  n_prompt: pp, n_gen: 0, n_depth: 0, avg_ts: speed, stddev_ts: 2, n_gpu_layers: 999,
});
const decode = (pp: number, tg: number, speed: number) => ({
  n_prompt: 0, n_gen: tg, n_depth: pp, avg_ts: speed, stddev_ts: 1, n_gpu_layers: 999,
});

const fileA = {
  id: "a", hostname: "alpha",
  data: { llamabench: {
    m1: {
      prefill_entries: [prefill(512, 1000), prefill(2048, 1800)],
      decode_entries: [
        decode(512, 128, 80), decode(512, 512, 75),
        decode(2048, 128, 70), decode(2048, 512, 64),
      ],
    },
    m2: {
      prefill_entries: [prefill(512, 700)],
      decode_entries: [decode(512, 128, 45)],
    },
  } },
};

const fileB = {
  id: "b", hostname: "beta",
  data: { llamabench: {
    m1: {
      prefill_entries: [prefill(512, 1200), prefill(8192, 2200)],
      decode_entries: [decode(512, 128, 95), decode(8192, 128, 60)],
    },
    failed: { error: "idle timeout" },
  } },
};

describe("llamaBenchPromptLabel", () => {
  it("formats binary-K prompt depths", () => {
    expect(llamaBenchPromptLabel(512)).toBe("0.5K");
    expect(llamaBenchPromptLabel(2048)).toBe("2K");
    expect(llamaBenchPromptLabel(98304)).toBe("96K");
  });
});

describe("schema compatibility", () => {
  it("reads the explicit prefill/decode arrays", () => {
    expect(llamaBenchPrefillEntries(fileA.data.llamabench.m1)).toHaveLength(2);
    expect(llamaBenchDecodeEntries(fileA.data.llamabench.m1)).toHaveLength(4);
  });

  it("recovers separate metrics from legacy entries when their fields are unambiguous", () => {
    const modelData = { entries: [prefill(512, 100), decode(512, 128, 50)] };
    expect(llamaBenchPrefillEntries(modelData)).toEqual([prefill(512, 100)]);
    expect(llamaBenchDecodeEntries(modelData)).toEqual([decode(512, 128, 50)]);
  });

  it("identifies combined-only matrix results without pretending they contain separate metrics", () => {
    const modelData = { entries: [{ n_prompt: 512, n_gen: 128, n_depth: 0, avg_ts: 90 }] };
    expect(llamaBenchHasCombinedOnly(modelData)).toBe(true);
    expect(llamaBenchPrefillEntries(modelData)).toEqual([]);
    expect(llamaBenchDecodeEntries(modelData)).toEqual([]);
  });
});

describe("by-model line data", () => {
  it("builds one prefill row per pp size and one series per system", () => {
    expect(buildLlamaBenchPrefillLineData([fileA, fileB], "m1")).toEqual([
      { promptLabel: "0.5K", f0: 1000, f1: 1200 },
      { promptLabel: "2K", f0: 1800 },
      { promptLabel: "8K", f1: 2200 },
    ]);
  });

  it("builds decode rows by pp depth with a series per system and tg size", () => {
    expect(buildLlamaBenchDecodeLineData([fileA, fileB], "m1")).toEqual([
      { promptLabel: "0.5K", f0_tg128: 80, f0_tg512: 75, f1_tg128: 95 },
      { promptLabel: "2K", f0_tg128: 70, f0_tg512: 64 },
      { promptLabel: "8K", f1_tg128: 60 },
    ]);
  });

  it("uses file color for system identity and dash patterns for tg identity", () => {
    const data = buildLlamaBenchDecodeLineData([fileA, fileB], "m1");
    const configs = buildLlamaBenchDecodeLineConfigs([fileA, fileB], "m1", data);
    expect(configs.map(config => config.name)).toEqual([
      "alpha — tg128", "alpha — tg512", "beta — tg128",
    ]);
    expect(configs[0].stroke).toBe(configs[1].stroke);
    expect(configs[0].strokeDasharray).not.toBe(configs[1].strokeDasharray);
    expect(configs[0].stroke).not.toBe(configs[2].stroke);
  });

  it("filters prefill file configs that have no values", () => {
    const data = buildLlamaBenchPrefillLineData([fileA, fileB], "m2");
    expect(buildLlamaBenchPrefillLineConfigs([fileA, fileB], data).map(config => config.name))
      .toEqual(["alpha"]);
  });
});

describe("by-system line data", () => {
  it("builds prefill and decode series across models", () => {
    expect(buildLlamaBenchPrefillLineDataByModel(fileA, ["m1", "m2"])).toEqual([
      { promptLabel: "0.5K", m1: 1000, m2: 700 },
      { promptLabel: "2K", m1: 1800 },
    ]);
    expect(buildLlamaBenchDecodeLineDataByModel(fileA, ["m1", "m2"])[0]).toEqual({
      promptLabel: "0.5K", m1_tg128: 80, m1_tg512: 75, m2_tg128: 45,
    });
  });

  it("builds only configs backed by data", () => {
    const prefillData = buildLlamaBenchPrefillLineDataByModel(fileA, ["m1", "missing"]);
    expect(buildLlamaBenchPrefillLineConfigsByModel(["m1", "missing"], prefillData)
      .map(config => config.dataKey)).toEqual(["m1"]);

    const decodeData = buildLlamaBenchDecodeLineDataByModel(fileA, ["m1", "missing"]);
    expect(buildLlamaBenchDecodeLineConfigsByModel(fileA, ["m1", "missing"], decodeData)
      .map(config => config.dataKey)).toEqual(["m1_tg128", "m1_tg512"]);
  });
});

describe("flattenLlamaBenchData", () => {
  it("emits explicit prefill and decode rows", () => {
    const rows = flattenLlamaBenchData([fileA]).filter(row => row.model === "m1");
    expect(rows).toHaveLength(6);
    expect(rows[0]).toEqual({
      _fileId: "a", model: "m1", metric: "Prefill", pp: 512, tg: null,
      avg_ts: 1000, stddev_ts: 2, n_gpu_layers: 999,
    });
    expect(rows[2]).toEqual({
      _fileId: "a", model: "m1", metric: "Decode", pp: 512, tg: 128,
      avg_ts: 80, stddev_ts: 1, n_gpu_layers: 999,
    });
  });

  it("preserves error and combined-only legacy rows", () => {
    const legacy = {
      id: "legacy", data: { llamabench: {
        old: { entries: [{ n_prompt: 512, n_gen: 128, avg_ts: 90 }] },
        failed: { error: "boom" },
      } },
    };
    expect(flattenLlamaBenchData([legacy])).toEqual([
      {
        _fileId: "legacy", model: "old", metric: "Combined", pp: 512, tg: 128,
        avg_ts: 90, stddev_ts: undefined, n_gpu_layers: undefined,
      },
      {
        _fileId: "legacy", model: "failed", metric: "—", skipped: true,
        skip_detail: "boom",
      },
    ]);
  });
});
