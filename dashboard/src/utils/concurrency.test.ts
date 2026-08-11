import { describe, it, expect } from "vitest";
import {
  getAllConcurrencyModels, buildConcurrencyDataForModel,
  getConcurrencyStopInfo, getConcurrencySweetSpot, flattenConcurrencyData, concurrencySortValue,
} from "./concurrency";
import { applyBaselineDeltas } from "./baseline";

describe("getAllConcurrencyModels", () => {
  it("returns known models in canonical order, unknowns appended after", () => {
    const files = [{ data: { concurrency_chat: { "phi4-mini": {}, "llama3.2-3b-q4": {}, "brand-new-model": {} } } }];
    expect(getAllConcurrencyModels(files, "concurrency_chat")).toEqual(["llama3.2-3b-q4", "phi4-mini", "brand-new-model"]);
  });
  it("returns an empty array when no file has concurrency data", () => {
    expect(getAllConcurrencyModels([{ data: { llm: { m: {} } } }], "concurrency_chat")).toEqual([]);
  });
  it("reads from the matching section only", () => {
    const files = [{ data: { concurrency_tool: { "phi4-mini": {} } } }];
    expect(getAllConcurrencyModels(files, "concurrency_chat")).toEqual([]);
    expect(getAllConcurrencyModels(files, "concurrency_tool")).toEqual(["phi4-mini"]);
  });
});

describe("getConcurrencySweetSpot", () => {
  it("selects peak aggregate throughput and reports per-request sacrifice from one-way", () => {
    const file = { data: { concurrency_tool: { m: {
      "1": { aggregate_tps: 20, tps_mean: 24 },
      "2": { aggregate_tps: 35, tps_mean: 18 },
      "4": { aggregate_tps: 32, tps_mean: 11 },
    } } } };
    expect(getConcurrencySweetSpot(file, "concurrency_tool", "m")).toEqual({
      level: "2", aggregateTps: 35, sacrificePct: 25,
    });
  });

  it("prefers the lower concurrency level on a throughput tie", () => {
    const file = { data: { concurrency_chat: { m: {
      "1": { aggregate_tps: 10, tps_mean: 10 },
      "2": { aggregate_tps: 20, tps_mean: 8 },
      "4": { aggregate_tps: 20, tps_mean: 7 },
    } } } };
    expect(getConcurrencySweetSpot(file, "concurrency_chat", "m")?.level).toBe("2");
  });

  it("returns null without finite aggregate measurements", () => {
    expect(getConcurrencySweetSpot({ data: {} }, "concurrency_chat", "m")).toBeNull();
    const file = { data: { concurrency_chat: { m: { "1": { aggregate_tps: null } } } } };
    expect(getConcurrencySweetSpot(file, "concurrency_chat", "m")).toBeNull();
  });

  it("keeps the absolute sweet spot when chart data is transformed to deltas", () => {
    const files = [
      { id: "base", data: { concurrency_tool: { m: {
        "1": { aggregate_tps: 20, tps_mean: 10 },
        "2": { aggregate_tps: 40, tps_mean: 8 },
        "4": { aggregate_tps: 20, tps_mean: 6 },
      } } } },
      { id: "next", data: { concurrency_tool: { m: {
        "1": { aggregate_tps: 20, tps_mean: 12 },
        "2": { aggregate_tps: 35, tps_mean: 9 },
        "4": { aggregate_tps: 30, tps_mean: 7 },
      } } } },
    ];
    const chartFiles = applyBaselineDeltas(files, "base");
    expect(getConcurrencySweetSpot(files[1], "concurrency_tool", "m")).toEqual({
      level: "2", aggregateTps: 35, sacrificePct: 25,
    });
    expect(getConcurrencySweetSpot(chartFiles[1], "concurrency_tool", "m")?.level).toBe("4");
  });
});

describe("buildConcurrencyDataForModel", () => {
  const files = [{
    hostname: "TestHost",
    data: {
      concurrency_chat: {
        m: {
          "1": { tps_mean: 28.3, prefill_tps_mean: 412.6, ttft_mean_sec: 31.35, aggregate_tps: 7.79, client_ttft_mean_sec: undefined as number | undefined },
          "2": { tps_mean: 11.4, ttft_mean_sec: 36.29, aggregate_tps: 7.53 },
          "4": { tps_mean: 3.9, ttft_mean_sec: 45.29, aggregate_tps: 6.13 },
          stopped_at: "failed",
        },
      },
    },
  }];
  it("builds one row per recorded level, in level-ladder order, labeled as N-way", () => {
    const rows = buildConcurrencyDataForModel(files, "concurrency_chat", "m", "tps");
    expect(rows.map(r => r.levelLabel)).toEqual(["1-way", "2-way", "4-way"]);
    expect(rows[0].f0).toBe(28.3);
  });
  it("picks ttft_mean_sec for the ttft metric and aggregate_tps for the aggregate metric", () => {
    expect(buildConcurrencyDataForModel(files, "concurrency_chat", "m", "ttft")[0].f0).toBe(31.35);
    expect(buildConcurrencyDataForModel(files, "concurrency_chat", "m", "aggregate")[0].f0).toBe(7.79);
  });
  it("uses only genuine server-reported prefill throughput and preserves missing values", () => {
    const rows = buildConcurrencyDataForModel(files, "concurrency_chat", "m", "prefill");
    expect(rows[0].f0).toBe(412.6);
    expect(rows[1].f0).toBeUndefined();
  });
  it("prefers explicit client TTFT over the legacy field", () => {
    const explicit = structuredClone(files);
    explicit[0].data.concurrency_chat.m["1"].client_ttft_mean_sec = 12.5;
    expect(buildConcurrencyDataForModel(explicit, "concurrency_chat", "m", "ttft")[0].f0).toBe(12.5);
  });
  it("excludes non-level keys like stopped_at from the level rows", () => {
    const rows = buildConcurrencyDataForModel(files, "concurrency_chat", "m", "tps");
    expect(rows).toHaveLength(3);
  });
  it("returns an empty array for a model with no concurrency data", () => {
    expect(buildConcurrencyDataForModel(files, "concurrency_chat", "missing-model", "tps")).toEqual([]);
  });
});

describe("getConcurrencyStopInfo", () => {
  it("returns null when the sweep wasn't cut short", () => {
    const file = { data: { concurrency_chat: { m: { "1": {}, "2": {} } } } };
    expect(getConcurrencyStopInfo(file, "concurrency_chat", "m")).toBeNull();
  });
  it("returns null for a model with no concurrency data at all", () => {
    expect(getConcurrencyStopInfo({ data: {} }, "concurrency_chat", "m")).toBeNull();
  });
  it("points to the next level for a load/crash/failure stop, since that level's data was never recorded", () => {
    const file = { data: { concurrency_chat: { m: { "1": {}, "2": {}, "4": {}, stopped_at: "failed" } } } };
    const info = getConcurrencyStopInfo(file, "concurrency_chat", "m");
    expect(info).toEqual({ reason: "failed", label: expect.any(String), lastLevel: "4", nextLevel: "8" });
  });
  it("has no next level for a slow stop, since the triggering level's real data was already recorded", () => {
    const file = { data: { concurrency_chat: { m: { "1": {}, "2": {}, "4": {}, "8": {}, stopped_at: "slow" } } } };
    const info = getConcurrencyStopInfo(file, "concurrency_chat", "m");
    expect(info?.lastLevel).toBe("8");
    expect(info?.nextLevel).toBeNull();
  });
  it("has no next level when the failure happened at the very first level (no data recorded yet)", () => {
    const file = { data: { concurrency_chat: { m: { stopped_at: "load_failed" } } } };
    const info = getConcurrencyStopInfo(file, "concurrency_chat", "m");
    expect(info?.lastLevel).toBeNull();
    expect(info?.nextLevel).toBe("1");
  });
  it("renders a human-readable label when every completed measurement was invalid", () => {
    const file = { data: { concurrency_chat: { m: { "1": {}, stopped_at: "invalid" } } } };
    const info = getConcurrencyStopInfo(file, "concurrency_chat", "m");
    expect(info?.label).toContain("no valid measurements");
  });
});

describe("flattenConcurrencyData", () => {
  it("produces a single skipped row for a whole-model skip, not one row per level", () => {
    const files = [{ id: "f1", data: { concurrency_chat: { m: { skipped: true, skip_reason: "known_crash", skip_detail: "x" } } } }];
    expect(flattenConcurrencyData(files, "concurrency_chat")).toEqual([
      { _fileId: "f1", model: "m", level: "—", skipped: true, skip_reason: "known_crash", skip_detail: "x" },
    ]);
  });
  it("produces one row per real level, excluding non-level keys like stopped_at", () => {
    const files = [{
      id: "f1",
      data: {
        concurrency_chat: {
          m: {
            "1": { tps_mean: 28.3, tps_stdev: 0, prefill_tps_mean: 412.6, prefill_tps_stdev: 8.2, aggregate_tps: 7.79, ttft_mean_sec: 31.35, ttft_stdev_sec: 0, total_tokens: 337 },
            stopped_at: "failed",
          },
        },
      },
    }];
    const rows = flattenConcurrencyData(files, "concurrency_chat");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toEqual({
      _fileId: "f1", model: "m", level: "1",
      tps_mean: 28.3, tps_stdev: 0, aggregate_tps: 7.79,
      prefill_tps_mean: 412.6, prefill_tps_stdev: 8.2,
      ttft_mean: 31.35, ttft_stdev: 0, total_tokens: 337,
    });
  });
});

describe("concurrencySortValue", () => {
  it("coerces level to a number so sweep order beats lexicographic order", () => {
    const levels = ["1", "2", "4", "6", "8", "12", "16"].map(level => ({ level }));
    const sorted = [...levels].sort(
      (a, b) => concurrencySortValue(a, "level") - concurrencySortValue(b, "level"),
    );
    expect(sorted.map(r => r.level)).toEqual(["1", "2", "4", "6", "8", "12", "16"]);
  });
  it("passes other columns through unchanged", () => {
    expect(concurrencySortValue({ tps_mean: 12.5 }, "tps_mean")).toBe(12.5);
    expect(concurrencySortValue({ model: "m" }, "model")).toBe("m");
  });
  it("falls back to empty string for a missing non-level field", () => {
    expect(concurrencySortValue({}, "tps_mean")).toBe("");
  });
  it("pins a skipped row's non-numeric level to +Infinity instead of NaN", () => {
    // Number("—") is NaN, and NaN compares false in both directions, which
    // breaks comparator consistency (a<b and a>b both false, yet a !== b).
    expect(concurrencySortValue({ level: "—" }, "level")).toBe(Infinity);
    const rows = [{ level: "16" }, { level: "—" }, { level: "1" }, { level: "4" }];
    const ascending = [...rows].sort(
      (a, b) => concurrencySortValue(a, "level") - concurrencySortValue(b, "level"),
    );
    expect(ascending.map(r => r.level)).toEqual(["1", "4", "16", "—"]);
    const descending = [...rows].sort(
      (a, b) => concurrencySortValue(b, "level") - concurrencySortValue(a, "level"),
    );
    expect(descending.map(r => r.level)).toEqual(["—", "16", "4", "1"]);
  });
});
