import { describe, it, expect } from "vitest";
import {
  getBarStatusLabel, getAllLLMModels, getLLMModelsWithSectionResults,
  buildLLMBarData, buildLLMBarConfigs, flattenLLMData, llmTTFTMean, llmValidRuns,
} from "./llm";

describe("explicit measurement fields", () => {
  it("prefers explicit client TTFT and valid-run counts with legacy fallback", () => {
    expect(llmTTFTMean({ client_ttft_mean_sec: 0.4, ttft_mean_sec: 9 })).toBe(0.4);
    expect(llmTTFTMean({ ttft_mean_sec: 0.8 })).toBe(0.8);
    expect(llmValidRuns({ valid_runs: 2, n_runs: 3 })).toBe(2);
    expect(llmValidRuns({ n_runs: 3 })).toBe(3);
  });
});

describe("getBarStatusLabel", () => {
  it("returns a whole-model skip label when the model was excluded from this section entirely", () => {
    const file = { data: { llm_conversation: { m: { skipped: true, skip_reason: "timed_out" } } } };
    expect(getBarStatusLabel(file, "m", "2K", "llm_conversation")).toBe("Skipped - LLM Timed Out");
  });
  it("falls back to a detail-based label for an unrecognized skip_reason", () => {
    const file = {
      data: { llm_conversation: { m: { skipped: true, skip_reason: "weird", skip_detail: "custom reason" } } },
    };
    expect(getBarStatusLabel(file, "m", "2K", "llm_conversation")).toBe("Skipped - custom reason");
  });
  it("labels exactly the crashed checkpoint as Crashed, deeper ones as Skipped, earlier ones unaffected", () => {
    const file = { data: { llm: { m: { crashed: "8K" } } } };
    expect(getBarStatusLabel(file, "m", "8K", "llm")).toBe("8K - Crashed");
    expect(getBarStatusLabel(file, "m", "16K", "llm")).toBe("16K - Skipped");
    expect(getBarStatusLabel(file, "m", "2K", "llm")).toBeNull();
  });
  it("labels exactly the timed-out checkpoint as Timed Out, deeper ones as Skipped", () => {
    const file = { data: { llm: { m: { timed_out: "32K" } } } };
    expect(getBarStatusLabel(file, "m", "32K", "llm")).toBe("32K - Timed Out");
    expect(getBarStatusLabel(file, "m", "64K", "llm")).toBe("64K - Skipped");
    expect(getBarStatusLabel(file, "m", "8K", "llm")).toBeNull();
  });
  it("shows the slow checkpoint's own real data (null status), labels only deeper checkpoints as skipped", () => {
    const file = { data: { llm_conversation: { m: { slow_tps: "32K" } } } };
    expect(getBarStatusLabel(file, "m", "32K", "llm_conversation")).toBeNull();
    expect(getBarStatusLabel(file, "m", "64K", "llm_conversation")).toBe("64K - Skipped (32K Too Slow)");
    expect(getBarStatusLabel(file, "m", "16K", "llm_conversation")).toBeNull();
  });
  it("returns null when nothing unusual applies to this cell", () => {
    const file = { data: { llm: { m: { "2K": { tps_mean: 40 } } } } };
    expect(getBarStatusLabel(file, "m", "2K", "llm")).toBeNull();
  });
});

describe("getAllLLMModels", () => {
  it("returns known models in canonical order, with unknown models appended after", () => {
    const files = [{ data: { llm: { "mistral-7b-q4": {}, "llama3.2-3b-q4": {}, "brand-new-model": {} } } }];
    expect(getAllLLMModels(files)).toEqual(["llama3.2-3b-q4", "mistral-7b-q4", "brand-new-model"]);
  });
  it("includes a model present only in llm_conversation, not llm, since the Models filter is shared", () => {
    const files = [{ data: { llm: {}, llm_conversation: { "phi4-mini": {} } } }];
    expect(getAllLLMModels(files)).toContain("phi4-mini");
  });
  it("deduplicates a model present in multiple files/sections", () => {
    const files = [
      { data: { llm: { "phi4-mini": {} } } },
      { data: { llm_conversation: { "phi4-mini": {} } } },
    ];
    expect(getAllLLMModels(files).filter(m => m === "phi4-mini")).toHaveLength(1);
  });
  it("includes a model present only in an accuracy test (e.g. an --tests acc-only run, leaving llm/llm_conversation empty)", () => {
    const files = [{ data: { llm: {}, llm_conversation: {}, mcq: { "phi4-mini": {} } } }];
    expect(getAllLLMModels(files)).toContain("phi4-mini");
  });
  it("includes a model present only in tool accuracy results", () => {
    const files = [{ data: { tool: { "phi4-mini": {} } } }];
    expect(getAllLLMModels(files)).toEqual(["phi4-mini"]);
  });
  it("includes a model present only in reasoning accuracy results", () => {
    const files = [{ data: { reasoning: { "phi4-mini": {} } } }];
    expect(getAllLLMModels(files)).toEqual(["phi4-mini"]);
  });
  it("includes a model present only in concurrency_tool, leaving llm/llm_conversation empty", () => {
    const files = [{ data: { llm: {}, llm_conversation: {}, concurrency_tool: { "phi4-mini": {} } } }];
    expect(getAllLLMModels(files)).toContain("phi4-mini");
  });
  it("includes a model present only in concurrency_chat, leaving llm/llm_conversation empty", () => {
    const files = [{ data: { llm: {}, llm_conversation: {}, concurrency_chat: { "phi4-mini": {} } } }];
    expect(getAllLLMModels(files)).toContain("phi4-mini");
  });
  it("includes a model present only in llamabench, leaving llm/llm_conversation empty", () => {
    const files = [{ data: { llm: {}, llm_conversation: {}, llamabench: { "phi4-mini": {} } } }];
    expect(getAllLLMModels(files)).toContain("phi4-mini");
  });
});

describe("getLLMModelsWithSectionResults", () => {
  it("keeps a model attempted by one system and excludes models with only no-data placeholders", () => {
    const files = [
      { data: { llm: {
        "qwen3.6-27b-q4": { "0.5K": { tps_mean: 0 }, slow_tps: "0.5K" },
      }, llm_conversation: {
        "qwen3.6-27b-q4": { skipped: true, skip_reason: "slow_tps" },
        "nemotron3-nano-30b-a3b": { skipped: true, skip_reason: "no_llm_data" },
      } } },
      { data: { llm: {}, llm_conversation: {
        "qwen3.6-27b-q4": { skipped: true, skip_reason: "no_llm_data" },
        "nemotron3-nano-30b-a3b": { skipped: true, skip_reason: "no_llm_data" },
      } } },
    ];

    expect(getLLMModelsWithSectionResults(files, "llm")).toEqual(["qwen3.6-27b-q4"]);
    expect(getLLMModelsWithSectionResults(files, "llm_conversation")).toEqual(["qwen3.6-27b-q4"]);
  });

  it("retains meaningful whole-model skip outcomes", () => {
    const files = [{ data: { llm: {
      "phi4-mini": { skipped: true, skip_reason: "known_crash" },
    } } }];

    expect(getLLMModelsWithSectionResults(files, "llm")).toEqual(["phi4-mini"]);
  });
});

describe("buildLLMBarData", () => {
  it("shows real values at every recorded checkpoint, including the slow one itself, with no status overlay there", () => {
    const files = [{
      hostname: "TestHost",
      data: { llm_conversation: { m: { "0K": { tps_mean: 50 }, "2K": { tps_mean: 4 }, slow_tps: "2K" } } },
    }];
    const rows = buildLLMBarData(files, "m", "tps", "llm_conversation");
    expect(rows).toHaveLength(1);
    expect(rows[0].systemLabel).toBe("TestHost");
    expect(rows[0]["0K"]).toBe(50);
    expect(rows[0]["2K"]).toBe(4);
    expect(rows[0]["_status_2K"]).toBeUndefined();
    expect(rows[0]["_status_4K"]).toBe("4K - Skipped (2K Too Slow)");
  });
});

describe("buildLLMBarConfigs", () => {
  it("aggregates columns across files, so a file that stopped early still gets columns another file reached", () => {
    const files = [
      { data: { llm_conversation: { m: { "0K": {}, "2K": {}, slow_tps: "2K" } } } },
      { data: { llm_conversation: { m: { "0K": {}, "2K": {}, "4K": {}, "8K": {} } } } },
    ];
    const configs = buildLLMBarConfigs(files, "m", "llm_conversation");
    expect(configs.map(c => c.dataKey)).toEqual(["0K", "2K", "4K", "8K"]);
  });
  it("orders columns by CTX_ORDER, not by insertion order in the source data", () => {
    const files = [{ data: { llm: { m: { "8K": {}, "2K": {} } } } }];
    const configs = buildLLMBarConfigs(files, "m", "llm");
    expect(configs.map(c => c.dataKey)).toEqual(["2K", "8K"]);
  });
});

describe("flattenLLMData", () => {
  it("produces a single skipped row for a whole-model skip, not one row per checkpoint", () => {
    const files = [{ id: "f1", data: { llm: { m: { skipped: true, skip_reason: "timed_out", skip_detail: "x" } } } }];
    expect(flattenLLMData(files)).toEqual([
      { _fileId: "f1", model: "m", ctx: "—", skipped: true, skip_reason: "timed_out", skip_detail: "x" },
    ]);
  });
  it("produces one row per real checkpoint, excluding non-checkpoint keys like timed_out/crashed/slow_tps", () => {
    const files = [{
      id: "f1",
      data: {
        llm: {
          m: {
            "2K": { tps_mean: 10, tps_stdev: 1, ttft_mean_sec: 0.5, ttft_stdev_sec: 0.1, n_runs: 3 },
            timed_out: "8K",
          },
        },
      },
    }];
    const rows = flattenLLMData(files);
    expect(rows).toHaveLength(1);
    expect(rows[0].ctx).toBe("2K");
    expect(rows[0].tps_mean).toBe(10);
  });
});
