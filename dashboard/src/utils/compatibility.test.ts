import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { buildLLMDataForModel, getAllLLMModels } from "./llm";
import {
  buildLlamaBenchDecodeLineData,
  buildLlamaBenchPrefillLineData,
} from "./llamabench";
import { getRunReliabilityWarning, parseResultsJSON } from "./shared";
import { getMemoryRecordingState, runHeadroomSummary } from "./memory";
import { buildPowerEfficiencyDataForModel, runPowerSummary } from "./power";


function loadGolden(name: string) {
  const url = new URL(`../../../tests/fixtures/${name}`, import.meta.url);
  const parsed = parseResultsJSON(readFileSync(url, "utf8"));
  expect(parsed.error).toBeNull();
  if (!parsed.data) throw new Error(`${name}: no data despite no error`);
  return parsed.data;
}


describe("4.1 golden result compatibility", () => {
  it("renders a schema 1 aggregate-only result with missing newer sections", () => {
    const data = loadGolden("results_v4_1_schema1_legacy.json");
    const files = [{ id: "schema1", hostname: "Golden", data }];

    expect(data.run.schema_version).toBe(1);
    expect(buildLLMDataForModel(files, "golden", "tps"))
      .toEqual([{ ctxLabel: "2K", f0: 50 }]);
    expect(getMemoryRecordingState(files[0])).toBe("not_recorded");
  });

  it("renders schema 5 memory while preserving the benchmark measurement", () => {
    const data = loadGolden("results_v6_schema5_memory.json");
    const files = [{ id: "schema5", hostname: "Golden", data }];

    expect(data.run.schema_version).toBe(5);
    expect(buildLLMDataForModel(files, "golden", "tps"))
      .toEqual([{ ctxLabel: "2K", f0: 50 }]);
    expect(getMemoryRecordingState(files[0])).toBe("recorded");
    expect(runHeadroomSummary(files[0])).toEqual({
      state: "comfortable", absoluteGb: 12, casePath: "llm/golden/2K",
    });
    expect(buildPowerEfficiencyDataForModel(files, "golden"))
      .toEqual([{ ctxLabel: "2K", f0: 10 }]);
    expect(runPowerSummary(files[0])).toEqual({
      status: "recorded", energyJoules: 12, idleWatts: 4,
      scope: "accelerator", reason: null,
    });
  });

  it("renders complete LLM, conversation, and llama-bench measurements", () => {
    const data = loadGolden("results_v4_1_complete.json");
    const files = [{ id: "golden", hostname: "Golden", data }];

    expect(getRunReliabilityWarning(data)).toBe("");
    expect(getAllLLMModels(files)).toEqual(["golden"]);
    expect(buildLLMDataForModel(files, "golden", "tps")).toEqual([
      { ctxLabel: "2K", f0: 50 },
    ]);
    expect(buildLLMDataForModel(files, "golden", "ttft", "llm_conversation"))
      .toEqual([{ ctxLabel: "0K", f0: 0.1 }]);
    expect(buildLlamaBenchPrefillLineData(files, "golden"))
      .toEqual([{ promptLabel: "2K", f0: 1000 }]);
    expect(buildLlamaBenchDecodeLineData(files, "golden"))
      .toEqual([{ promptLabel: "2K", f0_tg128: 50 }]);
  });

  it("warns about interruption while retaining chartable partial data", () => {
    const data = loadGolden("results_v4_1_interrupted.json");
    const files = [{ id: "interrupted", hostname: "Golden", data }];

    expect(getRunReliabilityWarning(data))
      .toBe("This benchmark was interrupted; completed measurements are still shown.");
    expect(buildLLMDataForModel(files, "golden", "tps"))
      .toEqual([{ ctxLabel: "0.5K", f0: 50 }]);
  });

  it("renders schema 3 plan-backed results without changing chart behavior", () => {
    const data = loadGolden("results_v4_1_schema3_plan.json");
    const files = [{ id: "schema3", hostname: "Golden", data }];

    expect(data.run.plan.schema_version).toBe(1);
    expect(data.run.plan_id).toHaveLength(64);
    expect(getRunReliabilityWarning(data)).toBe("");
    expect(buildLLMDataForModel(files, "golden", "tps"))
      .toEqual([{ ctxLabel: "2K", f0: 50 }]);
  });

  it("renders schema 4 pause-backed results without changing chart behavior", () => {
    const data = loadGolden("results_v4_1_schema4_pause.json");
    const files = [{ id: "schema4", hostname: "Golden", data }];

    expect(data.run.schema_version).toBe(4);
    expect(data.run.pause.control_transitions).toHaveLength(2);
    expect(getRunReliabilityWarning(data)).toBe("");
    expect(buildLLMDataForModel(files, "golden", "tps"))
      .toEqual([{ ctxLabel: "2K", f0: 50 }]);
  });
});
