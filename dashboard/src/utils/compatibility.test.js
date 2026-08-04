import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { buildLLMDataForModel, getAllLLMModels } from "./llm";
import {
  buildLlamaBenchDecodeLineData,
  buildLlamaBenchPrefillLineData,
} from "./llamabench";
import { getRunReliabilityWarning, parseResultsJSON } from "./shared";


function loadGolden(name) {
  const url = new URL(`../../../tests/fixtures/${name}`, import.meta.url);
  const parsed = parseResultsJSON(readFileSync(url, "utf8"));
  expect(parsed.error).toBeNull();
  return parsed.data;
}


describe("4.1 golden result compatibility", () => {
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
});
