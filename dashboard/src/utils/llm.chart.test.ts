import { describe, it, expect } from "vitest";

import {
  buildLLMBarConfigsByModel,
  buildLLMBarDataByModel,
  buildLLMCacheComparisonConfigs,
  buildLLMCacheComparisonData,
  buildLLMData,
  buildLLMLineConfigs,
  buildLLMLineConfigsByCtx,
  buildLLMLineDataByCtx,
  getBarStatusLabel,
} from "./llm";
import { CTX_COLORS } from "../constants";

const sample = (tps_mean: number) => ({ tps_mean });

const file = (hostname: string, llm: object | undefined, section = "llm") =>
  ({ id: hostname, hostname, data: llm === undefined ? {} : { [section]: llm } });

const dash = (config: object): string | undefined =>
  (config as { strokeDasharray?: string }).strokeDasharray;

const ALL = new Set(["gemma3-1b", "gemma3-27b-q4"]);

describe("cached versus uncached comparison", () => {
  it("pairs prefill and generation values by model and context", () => {
    const files = [{ id: "a", hostname: "alpha", data: {
      llm: { m: { "2K": { tps_mean: 40, prefill_tps_mean: 900 } } },
      llm_cached: { m: { "2K": { tps_mean: 55, prefill_tps_mean: 12000 } } },
    } }];
    expect(buildLLMCacheComparisonData(files, "m", "prefill")).toEqual([
      { ctxLabel: "2K", f0_uncached: 900, f0_cached: 12000 },
    ]);
    expect(buildLLMCacheComparisonData(files, "m", "tps")).toEqual([
      { ctxLabel: "2K", f0_uncached: 40, f0_cached: 55 },
    ]);
  });

  it("uses cache-state colors for one file and file colors plus line styles for several", () => {
    expect(buildLLMCacheComparisonConfigs([file("alpha", {})]).map(config => config.name))
      .toEqual(["Uncached", "Cached"]);
    const configs = buildLLMCacheComparisonConfigs([file("alpha", {}), file("beta", {})]);
    expect(configs.map(config => config.name)).toEqual([
      "alpha — Uncached", "alpha — Cached", "beta — Uncached", "beta — Cached",
    ]);
    expect(dash(configs[0])).toBe("8 5");
    expect(dash(configs[1])).toBeUndefined();
  });
});


// An unrecognized anchor cascades nothing — see docs/dashboard.md.
describe("getBarStatusLabel with an unrecognized cascade anchor", () => {
  it("does not mark measured checkpoints skipped when 'crashed' is unrecognized", () => {
    const f = file("a", { m: { crashed: "256K", "2K": sample(40) } });
    expect(getBarStatusLabel(f, "m", "2K", "llm")).toBeNull();
  });

  it("does not mark measured checkpoints skipped when 'timed_out' is unrecognized", () => {
    const f = file("a", { m: { timed_out: "256K", "2K": sample(40) } });
    expect(getBarStatusLabel(f, "m", "2K", "llm")).toBeNull();
  });

  it("does not mark measured checkpoints skipped when 'slow_tps' is unrecognized", () => {
    const f = file("a", { m: { slow_tps: "256K", "2K": sample(40) } });
    expect(getBarStatusLabel(f, "m", "2K", "llm")).toBeNull();
  });
});


describe("buildLLMData (legacy combined chart)", () => {
  it("keys series by bare model for a single file, ordered by CTX_ORDER", () => {
    const files = [file("alpha", { "gemma3-1b": { "8K": sample(30), "2K": sample(40) } })];
    expect(buildLLMData(files, "tps", ALL)).toEqual([
      { ctxLabel: "2K", "gemma3-1b": 40 },
      { ctxLabel: "8K", "gemma3-1b": 30 },
    ]);
  });

  it("namespaces series by file index when several files are loaded", () => {
    const files = [
      file("alpha", { "gemma3-1b": { "2K": sample(40) } }),
      file("beta", { "gemma3-1b": { "2K": sample(35) } }),
    ];
    expect(buildLLMData(files, "tps", ALL)).toEqual([
      { ctxLabel: "2K", "f0_gemma3-1b": 40, "f1_gemma3-1b": 35 },
    ]);
  });

  it("honours the Models filter", () => {
    const files = [file("alpha", {
      "gemma3-1b": { "2K": sample(40) }, "gemma3-27b-q4": { "2K": sample(12) },
    })];
    expect(buildLLMData(files, "tps", new Set(["gemma3-1b"])))
      .toEqual([{ ctxLabel: "2K", "gemma3-1b": 40 }]);
  });

  it("ignores non-checkpoint bookkeeping keys such as timed_out", () => {
    const files = [file("alpha", { "gemma3-1b": { "2K": sample(40), timed_out: "8K" } })];
    expect(buildLLMData(files, "tps", ALL).map(row => row.ctxLabel)).toEqual(["2K"]);
  });

  it("reads the metric the caller asked for", () => {
    const files = [file("alpha", {
      "gemma3-1b": { "2K": { tps_mean: 40, client_ttft_mean_sec: 0.5, prefill_tps_mean: 900 } },
    })];
    expect(buildLLMData(files, "ttft", ALL)[0]["gemma3-1b"]).toBe(0.5);
    expect(buildLLMData(files, "prefill", ALL)[0]["gemma3-1b"]).toBe(900);
  });

  it("tolerates a file with no llm section at all", () => {
    expect(buildLLMData([file("alpha", undefined)], "tps", ALL)).toEqual([]);
  });
});


describe("buildLLMLineConfigs", () => {
  const single = [file("alpha", { "gemma3-1b": { "2K": sample(40) } })];

  it("colors by model with no dash pattern for a single file", () => {
    const configs = buildLLMLineConfigs(single, buildLLMData(single, "tps", ALL), ALL);
    expect(configs).toHaveLength(1);
    expect(configs[0].dataKey).toBe("gemma3-1b");
    expect(configs[0].name).toBe("Gemma 3 1B — Q4_K_M");
    expect(dash(configs[0])).toBeUndefined();
  });

  it("distinguishes files by color and models by dash pattern when comparing systems", () => {
    const files = [
      file("alpha", { "gemma3-1b": { "2K": sample(40) }, "gemma3-27b-q4": { "2K": sample(12) } }),
      file("beta", { "gemma3-1b": { "2K": sample(35) }, "gemma3-27b-q4": { "2K": sample(10) } }),
    ];
    const configs = buildLLMLineConfigs(files, buildLLMData(files, "tps", ALL), ALL);
    expect(configs.map(c => c.dataKey)).toEqual([
      "f0_gemma3-1b", "f0_gemma3-27b-q4", "f1_gemma3-1b", "f1_gemma3-27b-q4",
    ]);
    expect(configs[0].stroke).toBe(configs[1].stroke);
    expect(configs[0].stroke).not.toBe(configs[2].stroke);
    expect(dash(configs[0])).not.toBe(dash(configs[1]));
    expect(configs[0].name).toBe("alpha — Gemma 3 1B — Q4_K_M");
  });

  it("emits no config for an enabled model with nothing plotted", () => {
    const configs = buildLLMLineConfigs(single, buildLLMData(single, "tps", ALL), ALL);
    expect(configs.map(c => c.dataKey)).not.toContain("gemma3-27b-q4");
  });
});


describe("buildLLMBarDataByModel", () => {
  it("emits one row per model with a column per checkpoint, ordered by CTX_ORDER", () => {
    const single = file("alpha", { "gemma3-1b": { "8K": sample(30), "2K": sample(40) } });
    expect(buildLLMBarDataByModel(single, ["gemma3-1b"], "tps")).toEqual([
      { modelLabel: "Gemma 3 1B — Q4_K_M", "2K": 40, "8K": 30 },
    ]);
  });

  it("keeps the slow checkpoint's own measurement and labels only deeper ones", () => {
    const single = file("alpha", {
      "gemma3-1b": { "2K": sample(40), "8K": sample(9), slow_tps: "8K" },
    });
    const [row] = buildLLMBarDataByModel(single, ["gemma3-1b"], "tps");
    expect(row["8K"]).toBe(9);
    expect(row["_status_8K"]).toBeUndefined();
    expect(row["_status_32K"]).toBe("32K - Skipped (8K Too Slow)");
  });

  it("annotates the crashed checkpoint and the deeper ones it prevented", () => {
    const single = file("alpha", { "gemma3-1b": { "2K": sample(40), crashed: "8K" } });
    const [row] = buildLLMBarDataByModel(single, ["gemma3-1b"], "tps");
    expect(row["_status_2K"]).toBeUndefined();
    expect(row["_status_8K"]).toBe("8K - Crashed");
    expect(row["_status_32K"]).toBe("32K - Skipped");
  });

  it("carries a whole-model skip label for a model excluded from the section", () => {
    const single = file("alpha", {
      "gemma3-1b": { skipped: true, skip_reason: "slow_tps" },
    });
    const [row] = buildLLMBarDataByModel(single, ["gemma3-1b"], "tps");
    expect(row["_status_2K"]).toBe("Skipped - LLM Too Slow");
  });

  it("reads a non-default section such as llm_conversation", () => {
    const single = file("alpha", { "gemma3-1b": { "32K": sample(22) } }, "llm_conversation");
    expect(buildLLMBarDataByModel(single, ["gemma3-1b"], "tps", "llm_conversation"))
      .toEqual([{ modelLabel: "Gemma 3 1B — Q4_K_M", "32K": 22 }]);
  });

  it("emits a labelled row even for a model the file has no data for", () => {
    const single = file("alpha", undefined);
    expect(buildLLMBarDataByModel(single, ["gemma3-1b"], "tps"))
      .toEqual([{ modelLabel: "Gemma 3 1B — Q4_K_M" }]);
  });
});


describe("buildLLMBarConfigsByModel", () => {
  it("emits a column per recorded checkpoint, in CTX_ORDER with registered colors", () => {
    const single = file("alpha", { "gemma3-1b": { "8K": sample(30), "2K": sample(40) } });
    expect(buildLLMBarConfigsByModel(single, ["gemma3-1b"])).toEqual([
      { dataKey: "2K", name: "2K", fill: CTX_COLORS["2K"] },
      { dataKey: "8K", name: "8K", fill: CTX_COLORS["8K"] },
    ]);
  });

  // Without a column, the corresponding status label has nowhere to render.
  it.each([
    ["timed_out", "timed_out"],
    ["crashed", "crashed"],
    ["slow_tps", "slow_tps"],
  ])("includes the %s checkpoint as a column even with no measurement there", (_name, key) => {
    const single = file("alpha", { "gemma3-1b": { "2K": sample(40), [key]: "8K" } });
    expect(buildLLMBarConfigsByModel(single, ["gemma3-1b"]).map(c => c.dataKey))
      .toEqual(["2K", "8K"]);
  });

  it("unions checkpoints across the requested models", () => {
    const single = file("alpha", {
      "gemma3-1b": { "2K": sample(40) },
      "gemma3-27b-q4": { "32K": sample(8) },
    });
    expect(buildLLMBarConfigsByModel(single, ["gemma3-1b", "gemma3-27b-q4"]).map(c => c.dataKey))
      .toEqual(["2K", "32K"]);
  });

  it("emits no columns when the file never ran the section", () => {
    expect(buildLLMBarConfigsByModel(file("alpha", undefined), ["gemma3-1b"])).toEqual([]);
  });

  it("declares every checkpoint column its data builder populates", () => {
    const single = file("alpha", {
      "gemma3-1b": { "2K": sample(40), "8K": sample(9), slow_tps: "8K" },
    });
    const [row] = buildLLMBarDataByModel(single, ["gemma3-1b"], "tps");
    const declared = buildLLMBarConfigsByModel(single, ["gemma3-1b"]).map(c => c.dataKey);
    for (const key of Object.keys(row).filter(k => k !== "modelLabel" && !k.startsWith("_status_"))) {
      expect(declared).toContain(key);
    }
  });
});


describe("buildLLMLineDataByCtx / buildLLMLineConfigsByCtx", () => {
  const single = file("alpha", {
    "gemma3-1b": { "8K": sample(30), "2K": sample(40) },
    "gemma3-27b-q4": { "2K": sample(12) },
  });

  it("orders rows by CTX_ORDER with one key per model", () => {
    expect(buildLLMLineDataByCtx(single, ["gemma3-1b", "gemma3-27b-q4"], "tps")).toEqual([
      { ctxLabel: "2K", "gemma3-1b": 40, "gemma3-27b-q4": 12 },
      { ctxLabel: "8K", "gemma3-1b": 30 },
    ]);
  });

  it("names each line with its display label", () => {
    const data = buildLLMLineDataByCtx(single, ["gemma3-1b", "gemma3-27b-q4"], "tps");
    expect(buildLLMLineConfigsByCtx(["gemma3-1b", "gemma3-27b-q4"], data).map(c => c.name))
      .toEqual(["Gemma 3 1B — Q4_K_M", "Gemma 3 27B 4-Bit Quantization"]);
  });

  it("omits a requested model with no points to plot", () => {
    const data = buildLLMLineDataByCtx(single, ["gemma3-1b", "llama3.3-70b-q4"], "tps");
    expect(buildLLMLineConfigsByCtx(["gemma3-1b", "llama3.3-70b-q4"], data).map(c => c.dataKey))
      .toEqual(["gemma3-1b"]);
  });

  it("ignores non-checkpoint bookkeeping keys when deriving the axis", () => {
    const withStatus = file("alpha", { "gemma3-1b": { "2K": sample(40), crashed: "8K" } });
    expect(buildLLMLineDataByCtx(withStatus, ["gemma3-1b"], "tps").map(r => r.ctxLabel))
      .toEqual(["2K"]);
  });

  it("returns no rows when the file never ran the section", () => {
    expect(buildLLMLineDataByCtx(file("alpha", undefined), ["gemma3-1b"], "tps")).toEqual([]);
  });
});
