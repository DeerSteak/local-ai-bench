import { describe, it, expect } from "vitest";
import {
  parseJSON, parseResultsJSON, getRunReliabilityWarning, getLlamaBenchMethodologyWarning,
  getConversationTTFTMethodologyWarning, getGpuSplitMethodologyWarning,
  getNoRepackMethodologyWarning,
  sanitizeForFilename, applyEngineLabels, filesForSection, fmt, getCrossEngineWeightsWarning,
  getModelColor, modelLabel, imageModelLabel, embedModelLabel,
  getModelSizeTier, getSkipInfo, prepareOrderedBarGroupData,
  sortBarData, sortRows, deriveTtftUnit, hasValueOrStatus, findMostStrenuousKey,
  entriesOf, valuesOf, isNotNull,
} from "./shared";
import { buildLLMBarData, buildLLMBarConfigs } from "./llm";
import type { ResultsFile, ChartRow } from "../types";

describe("entriesOf", () => {
  it("returns Object.entries for a populated object", () => {
    expect(entriesOf({ a: 1, b: 2 })).toEqual([["a", 1], ["b", 2]]);
  });
  it("returns an empty array for null or undefined", () => {
    expect(entriesOf(null)).toEqual([]);
    expect(entriesOf(undefined)).toEqual([]);
  });
});

describe("isNotNull", () => {
  it("keeps non-null, non-undefined values and drops the rest", () => {
    expect([1, null, 2, undefined, 0, ""].filter(isNotNull)).toEqual([1, 2, 0, ""]);
  });
});

describe("valuesOf", () => {
  it("returns Object.values for a populated object", () => {
    expect(valuesOf({ a: 1, b: 2 })).toEqual([1, 2]);
  });
  it("returns an empty array for null or undefined", () => {
    expect(valuesOf(null)).toEqual([]);
    expect(valuesOf(undefined)).toEqual([]);
  });
});

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
    const expected: { data: null, error: string } = {
      data: null,
      error: "Invalid JSON. Non-finite values such as Infinity are not supported.",
    };
    expect(parseResultsJSON("not json")).toEqual(expected);
    expect(parseResultsJSON('{"given":Infinity}')).toEqual(expected);
  });
});

describe("getRunReliabilityWarning", () => {
  it("keeps legacy and complete results quiet", () => {
    expect(getRunReliabilityWarning({})).toBe("");
    expect(getRunReliabilityWarning({ run: { status: "complete" } })).toBe("");
  });

  it("warns without hiding partial measurements", () => {
    expect(getRunReliabilityWarning({ run: { status: "running" } })).toContain("still running");
    expect(getRunReliabilityWarning({ run: { status: "partial" } })).toContain("partial");
    expect(getRunReliabilityWarning({ run: { status: "interrupted" } })).toContain("interrupted");
    expect(getRunReliabilityWarning({ run: { status: "failed" } })).toContain("failed");
  });

  it("warns for malformed run metadata", () => {
    expect(getRunReliabilityWarning({ run: [] })).toContain("malformed");
    expect(getRunReliabilityWarning({ run: 0 })).toContain("malformed");
  });
});

describe("getLlamaBenchMethodologyWarning", () => {
  it("warns when legacy and isolated-repetition files are compared", () => {
    const files = [
      { data: { llamabench: { m: {} } } },
      { data: { llamabench: { m: {} }, run: { llamabench_repetition_mode: "separate_process_r1" } } },
    ];
    expect(getLlamaBenchMethodologyWarning(files)).toContain("different repetition");
  });

  it("does not warn for one file or matching modes", () => {
    const legacy = { data: { llamabench: { m: {} } } };
    expect(getLlamaBenchMethodologyWarning([legacy])).toBe("");
    expect(getLlamaBenchMethodologyWarning([legacy, legacy])).toBe("");
    const current = { data: { llamabench: { m: {} }, run: {
      llamabench_repetition_mode: "streamed_internal_repetitions",
    } } };
    expect(getLlamaBenchMethodologyWarning([current, current])).toBe("");
  });

  it("warns when per-case and streamed internal-repetition v4.1 files are compared", () => {
    const perCase = { data: { llamabench: { m: {} }, run: {
      llamabench_repetition_mode: "separate_process_r1",
    } } };
    const streamed = { data: { llamabench: { m: {} }, run: {
      llamabench_repetition_mode: "streamed_internal_repetitions",
    } } };
    expect(getLlamaBenchMethodologyWarning([perCase, streamed])).toContain("different repetition");
  });
});

describe("getConversationTTFTMethodologyWarning", () => {
  const legacy = { data: { llm_conversation: { m: { "2K": { ttft_mean_sec: 0.2 } } } } };
  const current = { data: { llm_conversation: { m: {
    "2K": { ttft_mean_sec: 0.4, client_ttft_mean_sec: 0.4 },
  } } } };

  it("warns when client-observed and legacy server-prompt TTFT files are compared", () => {
    expect(getConversationTTFTMethodologyWarning([legacy, current])).toContain("different TTFT");
  });

  it("does not warn for one file, matching modes, or files without conversation measurements", () => {
    expect(getConversationTTFTMethodologyWarning([legacy])).toBe("");
    expect(getConversationTTFTMethodologyWarning([legacy, legacy])).toBe("");
    expect(getConversationTTFTMethodologyWarning([current, current])).toBe("");
    expect(getConversationTTFTMethodologyWarning([{ data: {} }, current])).toBe("");
  });
});

describe("getGpuSplitMethodologyWarning", () => {
  const legacy = { data: { run: { effective_config: {} } } };
  const tensor = { data: { run: { effective_config: { gpu_split_mode: "tensor" } } } };

  it("warns when tensor and legacy layer results are compared", () => {
    expect(getGpuSplitMethodologyWarning([legacy, tensor])).toContain("different multi-GPU");
  });

  it("stays quiet for one file or matching modes", () => {
    expect(getGpuSplitMethodologyWarning([tensor])).toBe("");
    expect(getGpuSplitMethodologyWarning([tensor, tensor])).toBe("");
    expect(getGpuSplitMethodologyWarning([legacy, legacy])).toBe("");
  });
});

describe("getNoRepackMethodologyWarning", () => {
  const enabled = { engine: "llamacpp", data: { run: { effective_config: {} } } };
  const disabled = { engine: "llamacpp", data: { run: {
    effective_config: { llamacpp_no_repack: true },
  } } };

  it("warns when llama.cpp repack modes differ", () => {
    expect(getNoRepackMethodologyWarning([enabled, disabled])).toContain("weight-repacking");
  });

  it("treats a missing legacy setting as repacking enabled", () => {
    const legacy = { data: { run: { effective_config: {} } } };
    expect(getNoRepackMethodologyWarning([legacy, disabled])).toContain("weight-repacking");
  });

  it("ignores vLLM settings and matching llama.cpp modes", () => {
    const vllm = { engine: "vllm", data: { run: {
      effective_config: { llamacpp_no_repack: true },
    } } };
    expect(getNoRepackMethodologyWarning([enabled, enabled, vllm])).toBe("");
  });

  it("stays quiet for workloads that do not consume the setting", () => {
    expect(getNoRepackMethodologyWarning([enabled, disabled], "images")).toBe("");
    expect(getNoRepackMethodologyWarning([enabled, disabled], "llamabench")).toBe("");
    expect(getNoRepackMethodologyWarning([enabled, disabled], "vllmbench")).toBe("");
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

describe("applyEngineLabels", () => {
  it("leaves hostnames untouched when only one engine is present", () => {
    const files = [
      { id: 1, hostname: "host-a", engine: "llamacpp", data: {} },
      { id: 2, hostname: "host-b", engine: "llamacpp", data: {} },
    ];
    expect(applyEngineLabels(files)).toEqual(files);
  });

  it("always labels chart systems with a recorded engine version", () => {
    const files: ResultsFile[] = [
      { id: 1, hostname: "host-a", backend: "cuda", engine: "llamacpp", engineVersion: "7000", data: {} },
      { id: 2, hostname: "host-b", backend: "metal", engine: "llamacpp", engineVersion: "7001", data: {} },
    ];
    expect(applyEngineLabels(files).map(file => file.hostname)).toEqual([
      "host-a\ncuda\nllamacpp 7000", "host-b\nmetal\nllamacpp 7001",
    ]);
  });

  it("marks llama.cpp no-repack runs in the engine label", () => {
    const files: ResultsFile[] = [{
      id: 1, hostname: "host-a", engine: "llamacpp", engineVersion: "7000",
      data: { run: { effective_config: { llamacpp_no_repack: true } } },
    }];
    expect(applyEngineLabels(files)[0].hostname).toBe("host-a\nllamacpp -nr 7000");
  });

  it("omits no-repack from labels for workloads that do not consume it", () => {
    const files: ResultsFile[] = [{
      id: 1, hostname: "host-a", engine: "llamacpp", engineVersion: "7000",
      data: { run: { effective_config: { llamacpp_no_repack: true } } },
    }];
    expect(filesForSection(files, "llamabench")[0].hostname).toBe("host-a\nllamacpp 7000");
    expect(filesForSection(files, "vllmbench")[0].hostname).toBe("host-a\nllamacpp 7000");
  });

  it("labels a version even when an older result omitted its engine name", () => {
    const files: ResultsFile[] = [
      { id: 1, hostname: "host-a", engineVersion: "0.10.2", data: {} },
    ];
    expect(applyEngineLabels(files)[0].hostname).toBe("host-a\n0.10.2");
  });
  it("distinguishes historical results that predate engine version recording", () => {
    const files: ResultsFile[] = [{
      id: 1, hostname: "host-a", engine: "llamacpp", engineVersionRecorded: false, data: {},
    }];
    expect(applyEngineLabels(files)[0].hostname)
      .toBe("host-a\nllamacpp version not recorded");
  });
  it("labels a current result whose runtime version could not be discovered", () => {
    const files: ResultsFile[] = [{
      id: 1, hostname: "host-a", engine: "vllm", engineVersionRecorded: true, data: {},
    }];
    expect(applyEngineLabels(files)[0].hostname)
      .toBe("host-a\nvllm version unavailable");
  });
  it("leaves hostnames untouched when no file has an engine field", () => {
    const files: ResultsFile[] = [{ id: 1, hostname: "host-a", engine: null, data: {} }];
    expect(applyEngineLabels(files)).toEqual(files);
  });
  it("appends the engine when two distinct engines are loaded together", () => {
    const files = [
      { id: 1, hostname: "host-a", engine: "llamacpp", data: {} },
      { id: 2, hostname: "host-a", engine: "mlx", data: {} },
    ];
    expect(applyEngineLabels(files)).toEqual([
      { id: 1, hostname: "host-a\nllamacpp", engine: "llamacpp", data: {} },
      { id: 2, hostname: "host-a\nmlx", engine: "mlx", data: {} },
    ]);
  });
  it("skips labeling a file with no engine even when others disagree", () => {
    const files = [
      { id: 1, hostname: "host-a", engine: "llamacpp", data: {} },
      { id: 2, hostname: "host-b", engine: "mlx", data: {} },
      { id: 3, hostname: "host-c", engine: null, data: {} },
    ];
    expect(applyEngineLabels(files)).toEqual([
      { id: 1, hostname: "host-a\nllamacpp", engine: "llamacpp", data: {} },
      { id: 2, hostname: "host-b\nmlx", engine: "mlx", data: {} },
      { id: 3, hostname: "host-c", engine: null, data: {} },
    ]);
  });
});

describe("filesForSection", () => {
  const files: ResultsFile[] = [{
    hostname: "host-a", engine: "llamacpp", engineVersion: "7000", data: {},
  }];

  it("includes engine versions on engine-backed charts", () => {
    expect(filesForSection(files, "llm")[0].hostname).toBe("host-a\nllamacpp 7000");
  });

  it("does not associate LLM runtime versions with ComfyUI image charts", () => {
    expect(filesForSection(files, "images")).toEqual(files);
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
    expect(modelLabel("llama3.2-3b-q4")).toBe("Llama 3.2 3B 4-Bit Quantization");
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
    const rows: ChartRow[] = [{ systemLabel: "System", "2K": null, "8K": 9, _status_2K: "Skipped" }];
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

describe("sortRows", () => {
  const rows = [{ model: "b", n: 2 }, { model: "a", n: 30 }, { model: "c", n: 1 }];
  it("sorts ascending by the string default value lookup", () => {
    expect(sortRows(rows, { key: "model", dir: 1 }).map(r => r.model)).toEqual(["a", "b", "c"]);
  });
  it("sorts descending", () => {
    expect(sortRows(rows, { key: "model", dir: -1 }).map(r => r.model)).toEqual(["c", "b", "a"]);
  });
  it("does not mutate the input array", () => {
    const copy = [...rows];
    sortRows(rows, { key: "model", dir: -1 });
    expect(rows).toEqual(copy);
  });
  it("supports a custom valueFn, e.g. concurrency's numeric level override", () => {
    const withLevel = [{ level: "2" }, { level: "10" }];
    expect(sortRows(withLevel, { key: "level", dir: 1 }, (r) => Number(r.level)).map(r => r.level))
      .toEqual(["2", "10"]);
  });
});

describe("deriveTtftUnit", () => {
  it("uses ms when every value is sub-second", () => {
    expect(deriveTtftUnit([0.1, 0.5])).toEqual({ ttftUnit: "ms", ttftYLabel: "TTFT (ms)" });
  });
  it("uses sec-plain once any value reaches a full minute", () => {
    expect(deriveTtftUnit([5, 65])).toEqual({ ttftUnit: "sec-plain", ttftYLabel: "TTFT (sec)" });
  });
  it("falls back to sec for values in between", () => {
    expect(deriveTtftUnit([2, 5])).toEqual({ ttftUnit: "sec", ttftYLabel: "TTFT (sec)" });
  });
  it("defaults to sec on an empty array rather than dividing by nothing", () => {
    expect(deriveTtftUnit([])).toEqual({ ttftUnit: "sec", ttftYLabel: "TTFT (sec)" });
  });
});

describe("hasValueOrStatus", () => {
  it("is true when a row has a real value for the key", () => {
    expect(hasValueOrStatus([{ "2K": 10 }], "2K")).toBe(true);
  });
  it("is true when a row has only a status placeholder for the key", () => {
    expect(hasValueOrStatus([{ _status_2K: "skipped" }], "2K")).toBe(true);
  });
  it("is false when no row has either", () => {
    expect(hasValueOrStatus([{ "8K": 10 }], "2K")).toBe(false);
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

describe("getCrossEngineWeightsWarning", () => {
  const file = (engine: string): ResultsFile => ({ id: engine, engine, data: {} });

  it("warns when files span more than one engine", () => {
    const warning = getCrossEngineWeightsWarning([file("llamacpp"), file("vllm")]);
    expect(warning).toContain("do not measure the same weights");
    expect(warning).toContain("Q4_K_M");
    expect(warning).toContain("AWQ");
  });

  it("stays silent for a single engine", () => {
    expect(getCrossEngineWeightsWarning([file("vllm"), file("vllm")])).toBe("");
    expect(getCrossEngineWeightsWarning([file("llamacpp")])).toBe("");
  });

  it("stays silent when no file records an engine", () => {
    expect(getCrossEngineWeightsWarning([{ id: "a", data: {} }, { id: "b", data: {} }])).toBe("");
  });

  it("ignores files with no engine while still comparing the rest", () => {
    expect(getCrossEngineWeightsWarning([file("vllm"), { id: "old", data: {} }])).toBe("");
  });
});
