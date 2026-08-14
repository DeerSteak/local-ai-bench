import { describe, it, expect } from "vitest";
import {
  getAllAccuracyModels,
  buildAccuracyGroupedBarData, buildAccuracyGroupedBarConfigs,
  buildAccuracyCategoryData, buildAccuracyCategoryConfigs,
  buildAccuracyDifficultyData, buildAccuracyTimeoutData,
  flattenAccuracyData, getAccuracySettingsWarning, getTemplateWarning,
  modelHasTemplateWarning,
} from "./accuracy";

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
  it("marks a model whose preflight recorded a chat-template warning", () => {
    const files = [{ data: {
      mcq: { "phi4-mini": {} },
      preflight: { models: { "phi4-mini": { checks: [{
        name: "chat_template", severity: "warning", detail: "No embedded template.",
      }] } } },
    } }];
    const configs = buildAccuracyGroupedBarConfigs(files, "mcq", new Set(["phi4-mini"]));
    expect(configs[0].name).toContain("⚠ template");
    expect(getTemplateWarning(files[0], "phi4-mini")).toBe("No embedded template.");
    expect(modelHasTemplateWarning(files, "phi4-mini")).toBe(true);
    expect(modelHasTemplateWarning([{ data: {} }], "phi4-mini")).toBe(false);
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
      { rowLabel: "Phi 4 Mini", timed_out_count: 2, likely_loop_count: 2, budget_exceeded_count: 0 },
      { rowLabel: "Mistral 7B v0.3 4-Bit Quantization", timed_out_count: 0, likely_loop_count: 0, budget_exceeded_count: 0 },
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
  it("shows budget exhaustion but not a successful nudge by itself", () => {
    const nudged = [{ hostname: "Host", data: { mcq: { m: { budget_nudged_count: 2 } } } }];
    expect(buildAccuracyTimeoutData(nudged, "mcq", new Set(["m"]))).toEqual([]);
    const exhausted = [{
      hostname: "Host",
      data: { mcq: { m: { budget_nudged_count: 2, budget_exceeded_count: 1 } } },
    }];
    expect(buildAccuracyTimeoutData(exhausted, "mcq", new Set(["m"]))[0]).toEqual({
      rowLabel: "m",
      timed_out_count: 0,
      likely_loop_count: 0,
      budget_exceeded_count: 1,
    });
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
      timed_out_count: 0, likely_loop_count: 0,
      budget_nudged_count: 0, budget_exceeded_count: 0, crashed: false,
      preflight_warning: "",
    });
  });
  it("attaches a template warning to the matching file and model only", () => {
    const files = [{ id: "f1", data: {
      mcq: { m: { accuracy_pct: 50 } },
      preflight: { models: { m: { checks: [{
        name: "chat_template", severity: "warning", detail: "Fallback formatting.",
      }] } } },
    } }];
    expect(flattenAccuracyData(files, "mcq")[0].preflight_warning).toBe("Fallback formatting.");
  });
  it("passes through timed_out_count/likely_loop_count/crashed when present", () => {
    const files = [{ id: "f1", data: { mcq: { m: {
      correct: 5, total: 10, answered: 8, accuracy_pct: 50,
      timed_out_count: 2, likely_loop_count: 1,
      budget_nudged_count: 3, budget_exceeded_count: 1, crashed: true,
    } } } }];
    const rows = flattenAccuracyData(files, "mcq");
    expect(rows[0].timed_out_count).toBe(2);
    expect(rows[0].likely_loop_count).toBe(1);
    expect(rows[0].budget_nudged_count).toBe(3);
    expect(rows[0].budget_exceeded_count).toBe(1);
    expect(rows[0].crashed).toBe(true);
  });
});

describe("getAccuracySettingsWarning", () => {
  const settings = { timeout_seconds: 60, token_budget: 8192, first_pass_fraction: 0.6 };
  it("returns no warning when settings match", () => {
    const files = [{ data: { accuracy_settings: settings } }, { data: { accuracy_settings: { ...settings } } }];
    expect(getAccuracySettingsWarning(files)).toBe("");
  });
  it("warns when settings differ or are missing from an old result", () => {
    const mismatch = [
      { data: { accuracy_settings: settings } },
      { data: { accuracy_settings: { ...settings, token_budget: 4096 } } },
    ];
    expect(getAccuracySettingsWarning(mismatch)).toMatch(/different/i);
    expect(getAccuracySettingsWarning([{ data: {} }])).toMatch(/unknown/i);
  });
});
