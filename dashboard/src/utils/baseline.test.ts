import { describe, expect, it } from "vitest";

import { applyBaselineDeltas, DELTA_FIELD_NAMES } from "./baseline";

describe("applyBaselineDeltas", () => {
  it("keeps the chart metric registry explicit", () => {
    expect(DELTA_FIELD_NAMES).toEqual([
      "tps_mean", "prefill_tps_mean", "client_ttft_mean_sec", "ttft_mean_sec",
      "aggregate_tps", "accuracy_pct", "chunks_per_sec_mean", "sec_per_image_mean",
      "avg_ts", "speed_tg", "avg_latency_sec", "output_tps",
    ]);
  });
  const files = [
    { id: "a", hostname: "Old", data: { llm: { m: { "2K": { tps_mean: 40, n_runs: 3 } } } } },
    { id: "b", hostname: "New", data: { llm: { m: { "2K": { tps_mean: 50, n_runs: 3 } } } } },
  ];

  it("turns matching chart metrics into percentage changes and keeps metadata absolute", () => {
    const result = applyBaselineDeltas(files, "a");
    expect(result[0].data.llm.m["2K"].tps_mean).toBe(0);
    expect(result[1].data.llm.m["2K"].tps_mean).toBe(25);
    expect(result[1].data.llm.m["2K"].n_runs).toBe(3);
    expect(result[0].hostname).toBe("Old (baseline)");
  });

  it("matches native-benchmark array entries by their shape instead of array position", () => {
    const baseline = { id: "a", data: { llamabench: { m: { entries: [
      { n_prompt: 512, n_gen: 0, avg_ts: 100 }, { n_prompt: 2048, n_gen: 0, avg_ts: 80 },
    ] } } } };
    const candidate = { id: "b", data: { llamabench: { m: { entries: [
      { n_prompt: 2048, n_gen: 0, avg_ts: 100 },
    ] } } } };
    const result = applyBaselineDeltas([baseline, candidate], "a");
    expect(result[1].data.llamabench.m.entries[0].avg_ts).toBe(25);
  });

  it("drops a metric when the baseline is missing or zero and leaves files unchanged without a baseline", () => {
    const baseline = { id: "a", data: { llm: { m: { "2K": { tps_mean: 0 } } } } };
    const candidate = { id: "b", data: { llm: { m: { "2K": { tps_mean: 10 } } } } };
    expect(applyBaselineDeltas([baseline, candidate], "a")[1].data.llm.m["2K"].tps_mean).toBeUndefined();
    expect(applyBaselineDeltas(files, null)).toBe(files);
  });

  it("preserves nulls and primitives in unrelated evidence arrays", () => {
    const baseline = { id: "a", data: { evidence: [null, 3, "legacy"] } };
    const candidate = { id: "b", data: { evidence: [null, 4, "current"] } };
    expect(applyBaselineDeltas([baseline, candidate], "a")[1].data.evidence).toEqual([
      null, 4, "current",
    ]);
  });
});
