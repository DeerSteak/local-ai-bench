import { describe, it, expect } from "vitest";
import {
  getAllConcurrencyModels, buildConcurrencyDataForModel,
  getConcurrencyStopInfo, flattenConcurrencyData, concurrencySortValue,
} from "./concurrency";

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

describe("buildConcurrencyDataForModel", () => {
  const files = [{
    hostname: "TestHost",
    data: {
      concurrency_chat: {
        m: {
          "1": { tps_mean: 28.3, ttft_mean_sec: 31.35, aggregate_tps: 7.79 },
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
    expect(info.lastLevel).toBe("8");
    expect(info.nextLevel).toBeNull();
  });
  it("has no next level when the failure happened at the very first level (no data recorded yet)", () => {
    const file = { data: { concurrency_chat: { m: { stopped_at: "load_failed" } } } };
    const info = getConcurrencyStopInfo(file, "concurrency_chat", "m");
    expect(info.lastLevel).toBeNull();
    expect(info.nextLevel).toBe("1");
  });
  it("renders a human-readable label when every completed measurement was invalid", () => {
    const file = { data: { concurrency_chat: { m: { "1": {}, stopped_at: "invalid" } } } };
    const info = getConcurrencyStopInfo(file, "concurrency_chat", "m");
    expect(info.label).toContain("no valid measurements");
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
            "1": { tps_mean: 28.3, tps_stdev: 0, aggregate_tps: 7.79, ttft_mean_sec: 31.35, ttft_stdev_sec: 0, total_tokens: 337 },
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
