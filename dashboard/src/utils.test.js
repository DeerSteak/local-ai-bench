import { describe, it, expect } from "vitest";
import {
  parseJSON, sanitizeForFilename, fmt,
  getModelColor, modelLabel, imageModelLabel, embedModelLabel,
  getModelSizeTier,
  getSkipInfo, getBarStatusLabel, getImageBarStatusLabel,
  getAllLLMModels,
  getEmbedLabel,
  buildLLMBarData, buildLLMBarConfigs,
  sortBarData, findMostStrenuousKey,
  flattenLLMData,
} from "./utils";

describe("parseJSON", () => {
  it("parses valid JSON", () => {
    expect(parseJSON('{"a":1}')).toEqual({ a: 1 });
  });
  it("returns null for invalid JSON instead of throwing", () => {
    expect(parseJSON("not json")).toBeNull();
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
});

describe("getModelSizeTier", () => {
  it("uses the known MODEL_SIZE_TIER map for known models", () => {
    expect(getModelSizeTier("llama3.2-3b-q4")).toBe("xsmall");
    expect(getModelSizeTier("gpt-oss-120b")).toBe("large");
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
    expect(getModelColor("llama3.2-3b-q4")).toBe("#536dfe");
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
    const files = [{ data: { llm: { "gpt-oss-20b": {}, "llama3.2-3b-q4": {}, "brand-new-model": {} } } }];
    expect(getAllLLMModels(files)).toEqual(["llama3.2-3b-q4", "gpt-oss-20b", "brand-new-model"]);
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
