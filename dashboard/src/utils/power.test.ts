import { describe, expect, it } from "vitest";

import {
  buildPowerEnergyCostDataForModel, hasMixedPowerScopes, powerFields, powerScopeLabel,
  runPowerSummary, powerEnergyCost,
} from "./power";

const power = {
  status: "recorded", source: "powermetrics", scope: "processor_package",
  energy_joules: 20, efficiency: { unit: "tokens_per_joule", per_joule: 5 },
};

describe("power telemetry", () => {
  it("extracts only finite optional values", () => {
    expect(powerFields({ power })).toEqual({
      energy_joules: 20, efficiency_per_joule: 5,
      efficiency_unit: "tokens_per_joule", power_scope: "processor_package",
      power_status: "recorded", power_reason: null,
    });
    expect(powerFields({ power: { energy_joules: "20", efficiency: { per_joule: Infinity } } }))
      .toMatchObject({ energy_joules: null, efficiency_per_joule: null });
    expect(powerFields({})).toEqual({});
  });

  it("builds a context series and refuses mixed scopes on one axis", () => {
    const files = [
      { id: "a", data: { llm: { model: { "2K": { power } } } } },
      { id: "b", data: { llm: { model: { "2K": { power: {
        ...power, energy_joules: 30, efficiency: { unit: "tokens_per_joule", per_joule: 7 },
      } } } } } },
    ];
    expect(buildPowerEnergyCostDataForModel(files, "model")).toEqual([
      { ctxLabel: "2K", f0: 200, f1: 1000 / 7 },
    ]);
    files[1].data.llm.model["2K"].power.scope = "accelerator";
    expect(buildPowerEnergyCostDataForModel(files, "model")).toEqual([]);
    expect(hasMixedPowerScopes(files, ["model"])).toBe(true);
  });

  it("renders run totals, unavailable reasons, and scope labels", () => {
    expect(runPowerSummary({ data: { run: { power_summary: {
      status: "recorded", energy_joules: 42, idle_baseline_watts: 3.5,
      scope: "processor_package", reason: null,
    } } } })).toEqual({
      status: "recorded", energyJoules: 42, idleWatts: 3.5,
      scope: "processor_package", reason: null,
    });
    expect(runPowerSummary({ data: {} })).toMatchObject({
      status: "not_recorded", energyJoules: null,
    });
    expect(powerScopeLabel("accelerator")).toBe("Accelerator");
    expect(powerScopeLabel(null)).toBe("Scope not recorded");
  });
});

it.each([
  ["tokens_per_joule", 5, 200],
  ["images_per_joule", 0.005, 200],
  ["embeddings_per_joule", 20, 50],
] as const)("converts %s to the chart's energy cost", (unit, per_joule, expected) => {
  const sample = { power: { ...power, efficiency: { unit, per_joule } } };
  expect(powerEnergyCost(sample, unit)).toBe(expected);
  expect(sample.power.efficiency.per_joule).toBe(per_joule);
  expect(powerFields(sample).efficiency_per_joule).toBe(per_joule);
});

it.each([0, -1, NaN, Infinity, Number.MIN_VALUE, null, "5"])(
  "omits non-invertible efficiency %s", per_joule => {
    expect(powerEnergyCost({ power: { ...power, efficiency: { unit: "tokens_per_joule", per_joule } } }, "tokens_per_joule")).toBeNull();
  },
);

it("does not convert unavailable, missing, or mismatched-unit measurements", () => {
  expect(powerEnergyCost({}, "tokens_per_joule")).toBeNull();
  expect(powerEnergyCost({ power: { ...power, status: "unavailable" } }, "tokens_per_joule")).toBeNull();
  expect(powerEnergyCost({ power }, "images_per_joule")).toBeNull();
});

it.each(["llm", "llm_cached", "llm_conversation"])("uses the same energy cost in both grouping modes for %s", async section => {
  const { buildLLMBarDataByModel, buildLLMLineDataByCtx } = await import("./llm");
  const file = { data: { [section]: { model: { "2K": { power } } } } };
  expect(buildPowerEnergyCostDataForModel([file], "model", section)[0].f0).toBe(200);
  expect(buildLLMBarDataByModel(file, ["model"], "efficiency", section)[0]["2K"]).toBe(200);
  expect(buildLLMLineDataByCtx(file, ["model"], "efficiency", section)[0].model).toBe(200);
});
