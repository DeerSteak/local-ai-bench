import { describe, expect, it } from "vitest";
import { buildEnergyAnalysis, energyChartSeries } from "./energyAnalysis";

const power = (scope = "accelerator", unit = "tokens_per_joule") => ({
  status: "recorded", scope, energy_joules: 100,
  efficiency: { unit, per_joule: 2, work_count: 200 },
  windows: [{ name: "measured:native-sweep-includes-load" }],
});
const sample = (depth = 8192) => ({ n_prompt: depth, completed_reps: 3, power: power() });
const file = (hostname = "GPU", scope = "accelerator") => ({ hostname, data: { llamabench: {
  m: { prefill_entries: [{ ...sample(), power: power(scope) }], decode_entries: [] },
} } });
const enabled = new Set(["m"]);

describe("workload energy analysis", () => {
  it("exposes recorded tokens per joule and energy with scope and load inclusion", () => {
    const { groups, notices } = buildEnergyAnalysis([file()], "llamabench", enabled);
    expect(notices).toEqual([]);
    expect(groups).toHaveLength(1);
    expect(groups[0].unit).toBe("Tokens / Joule");
    expect(groups[0].description).toContain("Accelerator · Full case, including model load");
    expect(groups[0].data).toEqual([{ caseLabel: "8K", order: 8192, f0: 2, f0_energy: 100 }]);
    expect(energyChartSeries(groups[0], true)[0].dataKey).toBe("f0_energy");
  });

  it("compares same-scope systems but splits incompatible scopes and System grouping", () => {
    const same = [file("A"), file("B")];
    expect(buildEnergyAnalysis(same, "llamabench", enabled).groups).toHaveLength(1);
    expect(buildEnergyAnalysis(same, "llamabench", enabled, true).groups).toHaveLength(2);
    const mixed = buildEnergyAnalysis([file("GPU"), file("Mac", "processor_package")], "llamabench", enabled);
    expect(mixed.groups).toHaveLength(2);
    expect(mixed.groups.map(group => group.configs.length)).toEqual([1, 1]);
  });

  it("keeps prefill, decode, generation sizes, repetitions, and measurement windows distinct", () => {
    const f = file();
    const other = { hostname: "B", data: { llamabench: { m: {
      prefill_entries: [{ ...sample(), completed_reps: 1 }],
      decode_entries: [256, 512].map(n_gen => ({ n_prompt: 0, n_depth: 8192, n_gen, power: power() })),
    } } } };
    expect(buildEnergyAnalysis([f, other], "llamabench", enabled).groups).toHaveLength(4);
    const windows = file("C");
    windows.data.llamabench.m.prefill_entries[0].power.windows = [{ name: "measured:request" }];
    expect(buildEnergyAnalysis([f, windows], "llamabench", enabled).groups).toHaveLength(2);
  });

  it("retains unavailable gaps and reports why instead of plotting zero", () => {
    const f = file();
    const missing = { ...sample(16384), power: { ...power(), status: "unavailable", reason: "insufficient samples" } };
    f.data.llamabench.m.prefill_entries = [sample(32768), missing, sample()];
    const { groups, notices } = buildEnergyAnalysis([f], "llamabench", enabled);
    expect(groups[0].data.map(row => row.caseLabel)).toEqual(["8K", "16K", "32K"]);
    expect(groups[0].data[1].f0).toBeNull();
    expect(groups[0].data[1].f0_energy).toBeNull();
    expect(notices.join()).toContain("insufficient samples");
  });

  it.each([0, -1, NaN, Infinity])("rejects invalid energy %s", energy => {
    const f = file();
    f.data.llamabench.m.prefill_entries[0].power.energy_joules = energy;
    const result = buildEnergyAnalysis([f], "llamabench", enabled);
    expect(result.groups).toEqual([]);
    expect(result.notices).toHaveLength(1);
  });

  it("does not label the wrong efficiency unit as tokens per joule", () => {
    const f = file();
    f.data.llamabench.m.prefill_entries[0].power.efficiency.unit = "images_per_joule";
    const result = buildEnergyAnalysis([f], "llamabench", enabled);
    expect(energyChartSeries(result.groups[0])).toEqual([]);
    expect(energyChartSeries(result.groups[0], true)).toHaveLength(1);
    expect(result.notices.join()).toContain("Tokens / Joule not recorded");
  });

  it("ignores filtered models and handles legacy/missing/null power", () => {
    expect(buildEnergyAnalysis([file()], "llamabench", new Set()).groups).toEqual([]);
    expect(buildEnergyAnalysis([{ data: {} }], "llamabench", enabled).groups).toEqual([]);
    const old = { data: { embeddings: { m: {}, empty: null } } };
    expect(buildEnergyAnalysis([old], "embeddings", enabled).notices.join()).toContain("not recorded");
    expect(buildEnergyAnalysis([file()], "llm", enabled)).toEqual({ groups: [], notices: [] });
  });

  it("keeps concurrency prompt/generation settings distinct and sorts levels numerically", () => {
    const f = { data: { llamabenchconc: { m: { entries: [16, 2, 1].map(pl => ({
      pp: 8192, tg: 512, pl, power: power(),
    })) } } } };
    const result = buildEnergyAnalysis([f], "llamabenchconc", enabled);
    expect(result.groups[0].data.map(row => row.caseLabel)).toEqual(["1-way", "2-way", "16-way"]);
    expect(result.groups[0].description).toContain("pp 8192 · tg 512");
  });

  it("labels image energy as an aggregate and separates different workload resolutions", () => {
    const a = { data: { images: { m: { steps: 20, resolutions: { "512x512": {} }, power: power("accelerator", "images_per_joule") } } } };
    const b = { data: { images: { m: { ...a.data.images.m, resolutions: { "768x768": {} } } } } };
    const result = buildEnergyAnalysis([a, b], "images", enabled);
    expect(result.groups).toHaveLength(2);
    expect(result.groups[0].unit).toBe("Images / Joule");
    expect(result.groups[0].description).toContain("All measured resolutions: 512x512 · 20 steps");
  });

  it("exposes embeddings per joule", () => {
    const f = { data: { embeddings: { m: { power: power("accelerator", "embeddings_per_joule") } } } };
    expect(buildEnergyAnalysis([f], "embeddings", enabled).groups[0].unit).toBe("Embeddings / Joule");
  });

  it("omits ambiguous duplicate cases rather than replacing their measurements", () => {
    const f = file();
    f.data.llamabench.m.prefill_entries = [sample(), sample(), sample()];
    const result = buildEnergyAnalysis([f], "llamabench", enabled);
    expect(result.groups).toEqual([]);
    expect(result.notices.join()).toContain("duplicate energy case");
  });
});

it("recognizes legacy native-sweep energy as including hidden subprocess model load", () => {
  const f = file();
  f.data.llamabench.m.prefill_entries.push({ ...sample(16384), power: {
    ...power(), windows: [{ name: "measured:native-sweep" }],
  } });
  const result = buildEnergyAnalysis([f], "llamabench", enabled);
  expect(result.groups).toHaveLength(1);
  expect(result.groups[0].data).toHaveLength(2);
  expect(result.groups[0].description).toContain("including model load");
});

it("handles malformed entries and rejects missing case dimensions", () => {
  const f = { data: { llamabenchconc: { m: { entries: [null, "bad", { power: power() }] } } } };
  const result = buildEnergyAnalysis([f], "llamabenchconc", enabled);
  expect(result.groups).toEqual([]);
  expect(result.notices.join()).toContain("dimensions are not recorded");
});

it.each([true, false])("rejects duplicates even when one copy is unavailable (first: %s)", unavailableFirst => {
  const f = file();
  const unavailable = { ...sample(), power: { ...power(), status: "unavailable" } };
  f.data.llamabench.m.prefill_entries = unavailableFirst ? [unavailable, sample()] : [sample(), unavailable];
  const result = buildEnergyAnalysis([f], "llamabench", enabled);
  expect(result.groups).toEqual([]);
  expect(result.notices.join()).toContain("duplicate energy case");
});
