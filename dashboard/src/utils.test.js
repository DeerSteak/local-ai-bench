import { describe, it, expect } from "vitest";
import {
  parseJSON, parseResultsJSON, sanitizeForFilename, fmt,
  getModelColor, modelLabel, imageModelLabel, embedModelLabel,
  getModelSizeTier,
  getSkipInfo, getBarStatusLabel, getImageBarStatusLabel,
  getAllLLMModels, getLLMModelsWithSectionResults,
  buildImagesDataForModel, buildImagesBarDataByModel,
  getEmbedLabel,
  buildLLMBarData, buildLLMBarConfigs, prepareOrderedBarGroupData,
  sortBarData, findMostStrenuousKey,
  flattenLLMData,
  getAllAccuracyModels,
  buildAccuracyGroupedBarData, buildAccuracyGroupedBarConfigs,
  buildAccuracyCategoryData, buildAccuracyCategoryConfigs,
  buildAccuracyDifficultyData,
  buildAccuracyTimeoutData,
  flattenAccuracyData,
  getAllConcurrencyModels, buildConcurrencyDataForModel,
  getConcurrencyStopInfo, flattenConcurrencyData, concurrencySortValue,
} from "./utils";

describe("parseJSON", () => {
  it("parses valid JSON", () => {
    expect(parseJSON('{"a":1}')).toEqual({ a: 1 });
  });
  it("returns null for invalid JSON instead of throwing", () => {
    expect(parseJSON("not json")).toBeNull();
  });
});

describe("parseResultsJSON", () => {
  it("returns a results object with no error", () => {
    expect(parseResultsJSON('{"profile":{"hostname":"host"}}')).toEqual({
      data: { profile: { hostname: "host" } }, error: null,
    });
  });
  it("rejects non-object JSON roots", () => {
    expect(parseResultsJSON("[]")).toEqual({
      data: null, error: "Expected a results JSON object.",
    });
  });
  it("explains invalid JSON including Python's non-standard Infinity token", () => {
    const expected = {
      data: null,
      error: "Invalid JSON. Non-finite values such as Infinity are not supported.",
    };
    expect(parseResultsJSON("not json")).toEqual(expected);
    expect(parseResultsJSON('{"given":Infinity}')).toEqual(expected);
  });
});

describe("sanitizeForFilename", () => {
  it("collapses whitespace and special characters to a single hyphen", () => {
    expect(sanitizeForFilename("My Model: v1.0")).toBe("My-Model-v1-0");
  });
  it("trims leading/trailing hyphens left over after sanitizing, and outer whitespace", () => {
    expect(sanitizeForFilename("  .hidden-file.  ")).toBe("hidden-file");
  });
  it("handles null/undefined/empty input gracefully", () => {
    expect(sanitizeForFilename(null)).toBe("");
    expect(sanitizeForFilename(undefined)).toBe("");
    expect(sanitizeForFilename("")).toBe("");
  });
  it("collapses a run of consecutive special characters into one hyphen, not several", () => {
    expect(sanitizeForFilename("a///b")).toBe("a-b");
  });
  it("preserves hyphens already present in the input", () => {
    expect(sanitizeForFilename("already-hyphenated")).toBe("already-hyphenated");
  });
});

describe("fmt", () => {
  it("returns an em dash for null/undefined regardless of unit", () => {
    expect(fmt(null, "sec")).toBe("—");
    expect(fmt(undefined, "tps")).toBe("—");
  });
  it("formats ms with one decimal under 10ms, rounded to an integer above", () => {
    expect(fmt(0.005, "ms")).toBe("5.0ms");
    expect(fmt(0.123, "ms")).toBe("123ms");
  });
  it("formats sec-plain with two decimals always, never converting to minutes", () => {
    expect(fmt(1.5, "sec-plain")).toBe("1.50s");
    expect(fmt(90, "sec-plain")).toBe("90.00s");
  });
  it("formats sec as minutes once at or above 60s", () => {
    expect(fmt(90, "sec")).toBe("1.5m");
    expect(fmt(59.99, "sec")).toBe("59.99s");
  });
  it("formats tps in K notation at or above 1000", () => {
    expect(fmt(1500, "tps")).toBe("1.50K");
    expect(fmt(50, "tps")).toBe("50.0");
  });
  it("formats sps in K notation at or above 10000", () => {
    expect(fmt(15000, "sps")).toBe("15.0K");
    expect(fmt(500, "sps")).toBe("500");
  });
  it("falls back to two-decimal formatting for an unrecognized unit", () => {
    expect(fmt(3.14159, "unknown")).toBe("3.14");
  });
  it("formats pct with one decimal and a percent sign", () => {
    expect(fmt(87.34, "pct")).toBe("87.3%");
    expect(fmt(0, "pct")).toBe("0.0%");
  });
  it("formats count as a rounded integer with no decoration", () => {
    expect(fmt(2, "count")).toBe("2");
    expect(fmt(2.9, "count")).toBe("3");
  });
});

describe("getModelSizeTier", () => {
  it("uses the known MODEL_SIZE_TIER map for known models", () => {
    expect(getModelSizeTier("llama3.2-3b-q4")).toBe("xsmall");
    expect(getModelSizeTier("nemotron3-super-120b")).toBe("large");
  });
  it("falls back to a param-count heuristic parsed from an unknown model's key", () => {
    expect(getModelSizeTier("some-new-model-15b")).toBe("small");
    expect(getModelSizeTier("some-new-model-30b")).toBe("medium");
    expect(getModelSizeTier("some-new-model-70b")).toBe("large");
  });
  it("defaults to medium when no param count can be parsed at all", () => {
    expect(getModelSizeTier("mystery-model")).toBe("medium");
  });
  it("boundaries: <=20b small, <50b medium, >=50b large", () => {
    expect(getModelSizeTier("x-20b")).toBe("small");
    expect(getModelSizeTier("x-21b")).toBe("medium");
    expect(getModelSizeTier("x-49b")).toBe("medium");
    expect(getModelSizeTier("x-50b")).toBe("large");
  });
});

describe("color/label lookups", () => {
  it("returns the known color/label for a known model", () => {
    expect(getModelColor("llama3.2-3b-q4")).toBe("#0550ae");
    expect(modelLabel("llama3.2-3b-q4")).toBe("Llama 3.2 3B Q4_K_M");
  });
  it("falls back to a deterministic hash color for an unknown model", () => {
    expect(getModelColor("totally-unknown-model")).toBe(getModelColor("totally-unknown-model"));
  });
  it("falls back to the raw key as the label for unknown models", () => {
    expect(modelLabel("totally-unknown-model")).toBe("totally-unknown-model");
    expect(imageModelLabel("unknown-image-model")).toBe("unknown-image-model");
    expect(embedModelLabel("unknown-embed-model")).toBe("unknown-embed-model");
  });
});

describe("getSkipInfo", () => {
  const file = {
    data: {
      llm_conversation: {
        "slow-model": { skipped: true, skip_reason: "slow_tps", skip_detail: "too slow" },
        "fine-model": { "0K": { tps_mean: 50 } },
      },
    },
  };
  it("returns the skip reason/detail for a skipped model", () => {
    expect(getSkipInfo(file, "slow-model")).toEqual({ reason: "slow_tps", detail: "too slow" });
  });
  it("returns null for a model with real data", () => {
    expect(getSkipInfo(file, "fine-model")).toBeNull();
  });
  it("returns null for a model with no data at all", () => {
    expect(getSkipInfo(file, "missing-model")).toBeNull();
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

describe("getImageBarStatusLabel", () => {
  it("labels the timed-out resolution and every larger one as skipped", () => {
    const file = { data: { images: { m: { timed_out: "1024x1024" } } } };
    expect(getImageBarStatusLabel(file, "m", "1024x1024")).toBe("1024x1024 - Timed Out");
    expect(getImageBarStatusLabel(file, "m", "1536x1536")).toBe("1536x1536 - Skipped");
  });
  it("returns null when nothing is unusual", () => {
    const file = { data: { images: { m: {} } } };
    expect(getImageBarStatusLabel(file, "m", "1024x1024")).toBeNull();
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

describe("SD 1.5 image resolutions", () => {
  const file = {
    data: { images: { sd15: { label: "Stable Diffusion 1.5", resolutions: {
      "512x512": { sec_per_image_mean: 1.25 },
      "768x768": { sec_per_image_mean: 2.5 },
    } } } },
  };

  it("builds by-model line data at both native resolutions", () => {
    expect(buildImagesDataForModel([file], "sd15")).toEqual([
      { resLabel: "512x512", f0: 1.25 },
      { resLabel: "768x768", f0: 2.5 },
    ]);
  });

  it("builds by-system bar data from an SD 1.5-only result", () => {
    expect(buildImagesBarDataByModel(file, ["sd15"])).toEqual([{
      modelLabel: "Stable Diffusion 1.5",
      "512x512": 1.25,
      "768x768": 2.5,
    }]);
  });
});

describe("getEmbedLabel", () => {
  it("uses the file-provided label when present, since results files carry their own labels", () => {
    const files = [{ data: { embeddings: { m: { label: "Custom Label" } } } }];
    expect(getEmbedLabel(files, "m")).toBe("Custom Label");
  });
  it("falls back to the static label map when no loaded file provides one", () => {
    const files = [{ data: { embeddings: { "nomic-embed-text": {} } } }];
    expect(getEmbedLabel(files, "nomic-embed-text")).toBe("Nomic Embed Text");
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

describe("prepareOrderedBarGroupData", () => {
  it.each(["llm", "llm_conversation"])(
    "keeps each system as one native group while deriving its scale for %s",
    (section) => {
      const files = [{
        hostname: "Host",
        data: { [section]: { m: {
          "32K": { tps_mean: 32 }, "8K": { tps_mean: 8 },
          "2K": { tps_mean: 2 }, "16K": { tps_mean: 16 },
        } } },
      }];
      const configs = buildLLMBarConfigs(files, "m", section);
      const rows = buildLLMBarData(files, "m", "tps", section);
      const prepared = prepareOrderedBarGroupData(rows, configs);

      expect(configs.map(config => config.dataKey)).toEqual(["2K", "8K", "16K", "32K"]);
      expect(prepared).toEqual([{
        systemLabel: "Host", "2K": 2, "8K": 8, "16K": 16, "32K": 32, _groupMax: 32,
      }]);
      expect(rows[0]).not.toHaveProperty("_groupMax");
    },
  );

  it("ignores missing values while preserving status metadata", () => {
    const rows = [{ systemLabel: "System", "2K": null, "8K": 9, _status_2K: "Skipped" }];
    const configs = [
      { dataKey: "2K", name: "2K", fill: "red" },
      { dataKey: "8K", name: "8K", fill: "blue" },
    ];

    const prepared = prepareOrderedBarGroupData(rows, configs);

    expect(prepared[0]).toMatchObject({ systemLabel: "System", _status_2K: "Skipped", _groupMax: 9 });
  });
});

describe("sortBarData", () => {
  const data = [
    { systemLabel: "A", "2K": 10 },
    { systemLabel: "B", "2K": 30 },
    { systemLabel: "C", "2K": 20 },
  ];
  it("sorts descending (higher is better) using the deepest preferred key present in the data", () => {
    expect(sortBarData(data, ["2K", "8K"], "desc").map(r => r.systemLabel)).toEqual(["B", "C", "A"]);
  });
  it("sorts ascending (lower is better)", () => {
    expect(sortBarData(data, ["2K"], "asc").map(r => r.systemLabel)).toEqual(["A", "C", "B"]);
  });
  it("returns the data unchanged (same reference) if none of the preferred keys are present", () => {
    expect(sortBarData(data, ["64K"], "desc")).toBe(data);
  });
  it("picks the deepest key that has data even when only some rows have it, and sorts missing values last", () => {
    const mixed = [{ "2K": 5 }, { "8K": 10 }];
    const sorted = sortBarData(mixed, ["2K", "8K"], "desc");
    expect(sorted[0]["8K"]).toBe(10);
  });
});

describe("findMostStrenuousKey", () => {
  it("returns the key whose max value across all rows is highest", () => {
    expect(findMostStrenuousKey([{ a: 5, b: 100 }, { a: 8, b: 50 }], ["a", "b"])).toBe("b");
  });
  it("returns null when no row has any of the candidate keys", () => {
    expect(findMostStrenuousKey([{ c: 1 }], ["a", "b"])).toBeNull();
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

describe("getAllAccuracyModels", () => {
  it("returns known models in canonical order for the given test only, unknowns appended after", () => {
    const files = [{ data: {
      mcq: { "phi4-mini": {}, "llama3.2-3b-q4": {}, "brand-new-model": {} },
      math: { "phi4-mini": {} },
    } }];
    expect(getAllAccuracyModels(files, "mcq")).toEqual(["llama3.2-3b-q4", "phi4-mini", "brand-new-model"]);
  });
  it("doesn't pull in a model that only ran a different accuracy test", () => {
    const files = [{ data: { mcq: { "phi4-mini": {} }, math: { "qwen3.5-4b": {} } } }];
    expect(getAllAccuracyModels(files, "mcq")).toEqual(["phi4-mini"]);
  });
});

describe("buildAccuracyGroupedBarData / buildAccuracyGroupedBarConfigs", () => {
  const enabledModels = new Set(["phi4-mini", "qwen3.5-4b"]);
  it("builds one row per file with one column per enabled model's accuracy_pct", () => {
    const files = [{
      hostname: "TestHost",
      data: { mcq: { "phi4-mini": { accuracy_pct: 87.3 }, "qwen3.5-4b": { accuracy_pct: 89.3 } } },
    }];
    const rows = buildAccuracyGroupedBarData(files, "mcq", enabledModels);
    expect(rows).toEqual([{ systemLabel: "TestHost", "phi4-mini": 87.3, "qwen3.5-4b": 89.3 }]);
  });
  it("omits a skipped model's column entirely rather than showing it as 0%", () => {
    const files = [{
      hostname: "TestHost",
      data: { mcq: { "phi4-mini": { skipped: true, skip_reason: "known_crash" } } },
    }];
    expect(buildAccuracyGroupedBarData(files, "mcq", new Set(["phi4-mini"]))).toEqual([]);
  });
  it("respects the enabled-models filter", () => {
    const configs = buildAccuracyGroupedBarConfigs(
      [{ data: { mcq: { "phi4-mini": {}, "qwen3.5-4b": {} } } }],
      "mcq", new Set(["phi4-mini"]),
    );
    expect(configs.map(c => c.dataKey)).toEqual(["phi4-mini"]);
  });
});

describe("buildAccuracyCategoryData / buildAccuracyCategoryConfigs", () => {
  it("builds one row per category with one column per file", () => {
    const files = [
      { hostname: "HostA", data: { mcq: { m: { by_category: { logic: { accuracy_pct: 60 }, science: { accuracy_pct: 90 } } } } } },
      { hostname: "HostB", data: { mcq: { m: { by_category: { logic: { accuracy_pct: 80 } } } } } },
    ];
    const rows = buildAccuracyCategoryData(files, "mcq", "m");
    expect(rows).toEqual([
      { categoryLabel: "logic", f0: 60, f1: 80 },
      { categoryLabel: "science", f0: 90 },
    ]);
    expect(buildAccuracyCategoryConfigs(files).map(c => c.name)).toEqual(["HostA", "HostB"]);
  });
  it("returns an empty array for a model with no category breakdown at all", () => {
    const files = [{ data: { mcq: { m: {} } } }];
    expect(buildAccuracyCategoryData(files, "mcq", "m")).toEqual([]);
  });
});

describe("buildAccuracyDifficultyData", () => {
  it("uses semantic difficulty order, keeps missing file values absent, and labels underscores", () => {
    const files = [
      { hostname: "HostA", data: { reasoning: { m: { by_difficulty: {
        very_hard: { accuracy_pct: 25 }, easy: { accuracy_pct: 100 },
      } } } } },
      { hostname: "HostB", data: { reasoning: { m: { by_difficulty: {
        hard: { accuracy_pct: 50 }, novel_level: { accuracy_pct: 75 },
      } } } } },
    ];
    expect(buildAccuracyDifficultyData(files, "reasoning", "m")).toEqual([
      { difficultyLabel: "Easy", f0: 100 },
      { difficultyLabel: "Hard", f1: 50 },
      { difficultyLabel: "Very hard", f0: 25 },
      { difficultyLabel: "Novel level", f1: 75 },
    ]);
  });

  it("returns an empty array for older accuracy results without a difficulty breakdown", () => {
    expect(buildAccuracyDifficultyData(
      [{ data: { reasoning: { m: { by_category: {} } } } }], "reasoning", "m",
    )).toEqual([]);
  });
});

describe("buildAccuracyTimeoutData", () => {
  it("includes every enabled model once at least one had a timed-out question, zero-filling the rest", () => {
    const files = [{
      hostname: "TestHost",
      data: {
        mcq: {
          "phi4-mini": { timed_out_count: 2, likely_loop_count: 2 },
          "mistral-7b-q4": { timed_out_count: 0, likely_loop_count: 0 },
        },
      },
    }];
    const rows = buildAccuracyTimeoutData(files, "mcq", new Set(["phi4-mini", "mistral-7b-q4"]));
    expect(rows).toEqual([
      { rowLabel: "Phi 4 Mini", timed_out_count: 2, likely_loop_count: 2 },
      { rowLabel: "Mistral 7B v0.3 Q4_K_M", timed_out_count: 0, likely_loop_count: 0 },
    ]);
  });
  it("prefixes the row label with hostname across multiple files, to tell them apart", () => {
    const files = [
      { hostname: "HostA", data: { mcq: { m: { timed_out_count: 1, likely_loop_count: 0 } } } },
      { hostname: "HostB", data: { mcq: { m: { timed_out_count: 3, likely_loop_count: 1 } } } },
    ];
    const rows = buildAccuracyTimeoutData(files, "mcq", new Set(["m"]));
    expect(rows.map(r => r.rowLabel)).toEqual(["HostA\nm", "HostB\nm"]);
  });
  it("returns an empty array when nothing timed out anywhere, so the chart cleanly disappears", () => {
    const files = [{ hostname: "TestHost", data: { mcq: { m: { timed_out_count: 0 } } } }];
    expect(buildAccuracyTimeoutData(files, "mcq", new Set(["m"]))).toEqual([]);
  });
});

describe("flattenAccuracyData", () => {
  it("produces a skipped row for a whole-model skip", () => {
    const files = [{ id: "f1", data: { mcq: { m: { skipped: true, skip_reason: "known_crash", skip_detail: "x" } } } }];
    expect(flattenAccuracyData(files, "mcq")).toEqual([
      { _fileId: "f1", model: "m", skipped: true, skip_reason: "known_crash", skip_detail: "x" },
    ]);
  });
  it("defaults timed_out_count/likely_loop_count to 0 and crashed to false when absent", () => {
    const files = [{ id: "f1", data: { mcq: { m: { correct: 10, total: 20, answered: 20, accuracy_pct: 50 } } } }];
    const rows = flattenAccuracyData(files, "mcq");
    expect(rows[0]).toEqual({
      _fileId: "f1", model: "m", correct: 10, total: 20, answered: 20, accuracy_pct: 50,
      timed_out_count: 0, likely_loop_count: 0, crashed: false,
    });
  });
  it("passes through timed_out_count/likely_loop_count/crashed when present", () => {
    const files = [{ id: "f1", data: { mcq: { m: {
      correct: 5, total: 10, answered: 8, accuracy_pct: 50,
      timed_out_count: 2, likely_loop_count: 1, crashed: true,
    } } } }];
    const rows = flattenAccuracyData(files, "mcq");
    expect(rows[0].timed_out_count).toBe(2);
    expect(rows[0].likely_loop_count).toBe(1);
    expect(rows[0].crashed).toBe(true);
  });
});

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
