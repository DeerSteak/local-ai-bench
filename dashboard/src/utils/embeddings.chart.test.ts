import { describe, it, expect } from "vitest";

import {
  buildEmbedBarConfigsByModel,
  buildEmbedBarDataByModel,
  buildEmbedGroupedBarConfigs,
  buildEmbedGroupedBarData,
  getAllEmbedModels,
} from "./embeddings";
import { EMBED_BAR_COLORS, EMBED_MODEL_ORDER } from "../constants";

// A results file is never guaranteed to carry every field a newer schema expects, so each
// builder is exercised against sparse and unknown-key inputs as well as complete ones.
const file = (hostname: string, embeddings: object | undefined) =>
  ({ id: hostname, hostname, data: embeddings === undefined ? {} : { embeddings } });

const throughput = (chunks_per_sec_mean: number) => ({ chunks_per_sec_mean });

const ALL = new Set(EMBED_MODEL_ORDER);


describe("getAllEmbedModels", () => {
  it("returns known models in EMBED_MODEL_ORDER regardless of their order in the file", () => {
    const files = [file("a", { "mxbai-embed-large": {}, "nomic-embed-text": {} })];
    expect(getAllEmbedModels(files)).toEqual(["nomic-embed-text", "mxbai-embed-large"]);
  });

  it("appends unknown models after the canonical roster instead of dropping them", () => {
    const files = [file("a", { "custom-embed": {}, "nomic-embed-text": {} })];
    expect(getAllEmbedModels(files)).toEqual(["nomic-embed-text", "custom-embed"]);
  });

  it("unions models across files and deduplicates a model present in several", () => {
    const files = [
      file("a", { "nomic-embed-text": {} }),
      file("b", { "nomic-embed-text": {}, "mxbai-embed-large": {} }),
    ];
    expect(getAllEmbedModels(files)).toEqual(["nomic-embed-text", "mxbai-embed-large"]);
  });

  it("returns nothing for a file that never ran the embeddings section", () => {
    expect(getAllEmbedModels([file("a", undefined)])).toEqual([]);
  });
});


describe("buildEmbedGroupedBarData", () => {
  it("emits one row per system, keyed by model, for the models the filter enables", () => {
    const files = [
      file("alpha", { "nomic-embed-text": throughput(12), "mxbai-embed-large": throughput(7) }),
      file("beta", { "nomic-embed-text": throughput(9) }),
    ];
    expect(buildEmbedGroupedBarData(files, ALL)).toEqual([
      { systemLabel: "alpha", "nomic-embed-text": 12, "mxbai-embed-large": 7 },
      { systemLabel: "beta", "nomic-embed-text": 9 },
    ]);
  });

  it("omits a model the Models filter has disabled", () => {
    const files = [file("alpha", {
      "nomic-embed-text": throughput(12), "mxbai-embed-large": throughput(7),
    })];
    expect(buildEmbedGroupedBarData(files, new Set(["nomic-embed-text"])))
      .toEqual([{ systemLabel: "alpha", "nomic-embed-text": 12 }]);
  });

  it("leaves a skipped model's bar absent rather than plotting it as zero throughput", () => {
    const files = [file("alpha", {
      "nomic-embed-text": { skipped: true, skip_reason: "known_crash", chunks_per_sec_mean: 12 },
      "mxbai-embed-large": throughput(7),
    })];
    const [row] = buildEmbedGroupedBarData(files, ALL);
    expect(row["nomic-embed-text"]).toBeUndefined();
    expect(row["mxbai-embed-large"]).toBe(7);
  });

  it("drops a system with no enabled-model data so it does not render an empty group", () => {
    const files = [
      file("alpha", { "nomic-embed-text": throughput(12) }),
      file("beta", { "nomic-embed-text": { skipped: true } }),
      file("gamma", undefined),
    ];
    expect(buildEmbedGroupedBarData(files, ALL).map(row => row.systemLabel)).toEqual(["alpha"]);
  });

  it("preserves a genuine zero measurement, which is distinct from an absent one", () => {
    const files = [file("alpha", { "nomic-embed-text": throughput(0) })];
    expect(buildEmbedGroupedBarData(files, ALL)).toEqual([
      { systemLabel: "alpha", "nomic-embed-text": 0 },
    ]);
  });
});


describe("buildEmbedGroupedBarConfigs", () => {
  it("assigns each model its registered bar color and display label, in canonical order", () => {
    const files = [file("a", { "mxbai-embed-large": throughput(7), "nomic-embed-text": throughput(12) })];
    expect(buildEmbedGroupedBarConfigs(files, ALL)).toEqual([
      { dataKey: "nomic-embed-text", name: "Nomic Embed Text", fill: EMBED_BAR_COLORS["nomic-embed-text"] },
      { dataKey: "mxbai-embed-large", name: "MixedBread Embed Large", fill: EMBED_BAR_COLORS["mxbai-embed-large"] },
    ]);
  });

  it("gives an unknown model a fallback color rather than no color at all", () => {
    const files = [file("a", { "custom-embed": throughput(3) })];
    const [config] = buildEmbedGroupedBarConfigs(files, new Set(["custom-embed"]));
    expect(config.dataKey).toBe("custom-embed");
    expect(config.fill).toMatch(/^#[0-9a-f]{6}$/i);
  });

  it("prefers a label the results file carries over the static label map", () => {
    const files = [file("a", { "nomic-embed-text": { ...throughput(12), label: "Nomic v2" } })];
    expect(buildEmbedGroupedBarConfigs(files, ALL)[0].name).toBe("Nomic v2");
  });

  it("produces configs whose dataKeys match the keys the data builder emits", () => {
    const files = [file("alpha", { "nomic-embed-text": throughput(12) })];
    const [row] = buildEmbedGroupedBarData(files, ALL);
    for (const config of buildEmbedGroupedBarConfigs(files, ALL)) {
      expect(Object.keys(row)).toContain(config.dataKey);
    }
  });
});


describe("buildEmbedBarDataByModel", () => {
  it("emits one row per model with its throughput, labelled for display", () => {
    const single = file("alpha", {
      "nomic-embed-text": throughput(12), "mxbai-embed-large": throughput(7),
    });
    expect(buildEmbedBarDataByModel(single, ["nomic-embed-text", "mxbai-embed-large"])).toEqual([
      { modelLabel: "Nomic Embed Text", throughput: 12 },
      { modelLabel: "MixedBread Embed Large", throughput: 7 },
    ]);
  });

  it("drops a skipped model rather than charting it as zero", () => {
    const single = file("alpha", {
      "nomic-embed-text": { skipped: true, chunks_per_sec_mean: 12 },
      "mxbai-embed-large": throughput(7),
    });
    expect(buildEmbedBarDataByModel(single, ["nomic-embed-text", "mxbai-embed-large"]))
      .toEqual([{ modelLabel: "MixedBread Embed Large", throughput: 7 }]);
  });

  it("returns nothing when the file never ran embeddings", () => {
    expect(buildEmbedBarDataByModel(file("alpha", undefined), ["nomic-embed-text"])).toEqual([]);
  });
});


describe("buildEmbedBarConfigsByModel", () => {
  it("declares the single throughput series when at least one model has data", () => {
    const single = file("alpha", { "nomic-embed-text": throughput(12) });
    const configs = buildEmbedBarConfigsByModel(single, ["nomic-embed-text"]);
    expect(configs).toHaveLength(1);
    expect(configs[0].dataKey).toBe("throughput");
  });

  it("declares no series when every requested model was skipped, so no empty axis renders", () => {
    const single = file("alpha", { "nomic-embed-text": { skipped: true } });
    expect(buildEmbedBarConfigsByModel(single, ["nomic-embed-text"])).toEqual([]);
  });

  it("declares no series when the file never ran embeddings", () => {
    expect(buildEmbedBarConfigsByModel(file("alpha", undefined), ["nomic-embed-text"])).toEqual([]);
  });

  it("stays consistent with its data builder: series exist exactly when rows do", () => {
    for (const embeddings of [
      { "nomic-embed-text": throughput(12) },
      { "nomic-embed-text": { skipped: true } },
      undefined,
    ]) {
      const single = file("alpha", embeddings);
      const rows = buildEmbedBarDataByModel(single, ["nomic-embed-text"]);
      const configs = buildEmbedBarConfigsByModel(single, ["nomic-embed-text"]);
      expect(configs.length > 0).toBe(rows.length > 0);
    }
  });
});
