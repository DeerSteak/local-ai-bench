import { describe, expect, it } from "vitest";

import {
  buildPowerEfficiencyDataForModel, hasMixedPowerScopes, powerFields, powerScopeLabel,
  runPowerSummary,
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
        ...power, energy_joules: 30, efficiency: { per_joule: 7 },
      } } } } } },
    ];
    expect(buildPowerEfficiencyDataForModel(files, "model")).toEqual([
      { ctxLabel: "2K", f0: 5, f1: 7 },
    ]);
    files[1].data.llm.model["2K"].power.scope = "accelerator";
    expect(buildPowerEfficiencyDataForModel(files, "model")).toEqual([]);
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
