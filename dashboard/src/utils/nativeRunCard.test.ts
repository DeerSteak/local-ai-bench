import { describe, it, expect } from "vitest";
import { nativeRunCardSummary } from "./nativeRunCard";
const a = "gemma3-1b", b = "granite4.1-3b-q4";
const pp = (n_prompt: number, avg_ts: number) => ({ n_prompt, n_gen: 0, avg_ts, completed_reps: 3 });
const power = (scope = "accelerator") => ({ status: "recorded", energy_joules: 10, scope });

describe("nativeRunCardSummary", () => {
  it("uses native data only and independently selects prefill and decode leaders at matching cases", () => {
    const data = { llm: { [a]: { "2K": { tps_mean: 9999 } } }, llamabench: {
      [a]: { prefill_entries: [pp(8192, 100)], decode_entries: [{ n_depth: 8192, n_gen: 512, avg_ts: 20 }] },
      [b]: { prefill_entries: [pp(8192, 200), pp(512, 999)], decode_entries: [{ n_depth: 8192, n_gen: 512, avg_ts: 10 }] },
    } };
    const result = nativeRunCardSummary({ data }, "llamabench");
    expect(result.leaders.map(x => [x.kind, x.value])).toEqual([["Prefill", 200], ["Decode", 20]]);
    expect(result.leaders[0].checkpoint).toContain("pp8192");
  });
  it("compares aggregate concurrency at identical prompt, generation, and parallel dimensions", () => {
    const entry = (pl: number, speed_tg: number, pp = 8192) => ({ pl, pp, tg: 512, speed_tg });
    const result = nativeRunCardSummary({ data: { llamabenchconc: {
      [a]: { entries: [entry(1, 30), entry(4, 100), entry(8, 500)] },
      [b]: { entries: [entry(1, 40), entry(4, 120), entry(8, 999, 512)] },
    } } }, "llamabenchconc");
    expect(result.leaders).toHaveLength(1);
    expect(result.leaders[0]).toMatchObject({ kind: "Aggregate decode", checkpoint: "4-way · pp8192 · tg512", value: 120 });
  });
  it("derives telemetry from this tab and never adds mixed power scopes", () => {
    const data = { run: { power_summary: { energy_joules: 9999 } }, llamabench: {
      [a]: { prefill_entries: [{ ...pp(8192, 10), power: power(), memory: { headroom: { absolute_gb: 2, state: "low" } } },
        { ...pp(16384, 9), power: power() }] },
    } };
    const result = nativeRunCardSummary({ data }, "llamabench");
    expect(result.power.energyJoules).toBe(20);
    expect(result.headroom.absoluteGb).toBe(2);
    data.llamabench[a].prefill_entries[1].power = power("processor_package");
    expect(nativeRunCardSummary({ data }, "llamabench").power.energyJoules).toBeNull();
  });
  it("supports legacy combined data without calling it prefill or decode", () => {
    const result = nativeRunCardSummary({ data: { llamabench: { [a]: { entries: [{ ...pp(512, 20), n_gen: 128 }] } } } }, "llamabench");
    expect(result.leaders[0].kind).toBe("Combined");
  });
  it("omits duplicates, invalid values, skipped models and malformed data", () => {
    const sample = { ...pp(8192, 10), power: power() };
    const result = nativeRunCardSummary({ data: { llamabench: {
      [a]: { prefill_entries: [sample, sample, null, pp(512, Infinity)] },
      [b]: { error: "failed", prefill_entries: [sample] }, bad: null, missing: { entries: "bad" },
    } } }, "llamabench");
    expect(result.leaders).toEqual([]);
    expect(result.power.energyJoules).toBeNull();
    expect(nativeRunCardSummary({ data: {} }, "llamabenchconc").headroom.absoluteGb).toBeNull();
  });
  it("labels partial energy totals and retains negative memory headroom", () => {
    const result = nativeRunCardSummary({ data: { llamabench: { [a]: { prefill_entries: [
      { ...pp(512, 10), power: power(), memory: { headroom: { absolute_gb: -1, state: "exceeded" } } }, pp(8192, 5),
    ] } } } }, "llamabench");
    expect(result.power.reason).toContain("some energy unavailable");
    expect(result.headroom.absoluteGb).toBe(-1);
  });
});
