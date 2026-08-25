import { describe, it, expect } from "vitest";

import {
  buildImagesBarConfigsByModel,
  buildImagesBarDataByModel,
  buildImagesData,
  buildImagesDataForModel,
  buildImagesDataForResolution,
  buildImagesGroupedBarConfigs,
  buildImagesGroupedBarDataForResolution,
  buildImagesLineConfigs,
  buildImagesLineConfigsByRes,
  buildImagesLineDataByRes,
  getAllImageModels,
  getImageBarStatusLabel,
} from "./images";
import { IMAGE_BAR_COLORS, IMAGE_MODEL_ORDER, RES_COLORS, RES_ORDER } from "../constants";

const res = (sec_per_image_mean: number) => ({ sec_per_image_mean });

const file = (hostname: string, images: object | undefined) =>
  ({ id: hostname, hostname, data: images === undefined ? {} : { images } });

const dash = (config: object): string | undefined =>
  (config as { strokeDasharray?: string }).strokeDasharray;

const ALL = new Set(IMAGE_MODEL_ORDER);


// benchmark.py records the resolution at which image generation timed out; every larger
// resolution is consequently never attempted. Mirrors llm.ts's getBarStatusLabel contract.
describe("getImageBarStatusLabel", () => {
  const timedOut = file("a", { sd15: { timed_out: "768x768" } });

  it("labels exactly the timed-out resolution as Timed Out", () => {
    expect(getImageBarStatusLabel(timedOut, "sd15", "768x768")).toBe("768x768 - Timed Out");
  });

  it("labels every larger resolution as Skipped, since none was attempted", () => {
    expect(getImageBarStatusLabel(timedOut, "sd15", "1024x1024")).toBe("1024x1024 - Skipped");
    expect(getImageBarStatusLabel(timedOut, "sd15", "1536x1536")).toBe("1536x1536 - Skipped");
  });

  it("leaves smaller resolutions unlabelled, because they completed before the timeout", () => {
    expect(getImageBarStatusLabel(timedOut, "sd15", "512x512")).toBeNull();
  });

  it("returns null for a model that never timed out", () => {
    const clean = file("a", { sd15: { resolutions: { "512x512": res(3) } } });
    expect(getImageBarStatusLabel(clean, "sd15", "512x512")).toBeNull();
  });

  it("returns null for a file with no images section at all", () => {
    expect(getImageBarStatusLabel(file("a", undefined), "sd15", "512x512")).toBeNull();
  });

  // An unrecognized resolution cascades nothing — see docs/dashboard.md.
  it("does not mislabel known resolutions when the timed-out resolution is unrecognized", () => {
    const unknown = file("a", { sd15: { timed_out: "2048x2048" } });
    expect(getImageBarStatusLabel(unknown, "sd15", "512x512")).toBeNull();
    expect(getImageBarStatusLabel(unknown, "sd15", "1536x1536")).toBeNull();
  });
});


describe("getAllImageModels", () => {
  it("returns known models in IMAGE_MODEL_ORDER regardless of file key order", () => {
    const files = [file("a", { "flux-dev": {}, sd15: {}, sdxl: {} })];
    expect(getAllImageModels(files)).toEqual(["sd15", "sdxl", "flux-dev"]);
  });

  it("appends unknown models after the canonical roster rather than dropping them", () => {
    const files = [file("a", { "my-model": {}, sd15: {} })];
    expect(getAllImageModels(files)).toEqual(["sd15", "my-model"]);
  });

  it("keeps legacy SD3.5 results in stable order after current models", () => {
    const files = [file("a", { "sd35-large": {}, "z-image-turbo": {}, sdxl: {} })];
    expect(getAllImageModels(files)).toEqual(["sdxl", "z-image-turbo", "sd35-large"]);
  });

  it("unions across files without duplicating a shared model", () => {
    const files = [file("a", { sd15: {} }), file("b", { sd15: {}, sdxl: {} })];
    expect(getAllImageModels(files)).toEqual(["sd15", "sdxl"]);
  });

  it("returns nothing for a file that never ran images", () => {
    expect(getAllImageModels([file("a", undefined)])).toEqual([]);
  });
});


describe("buildImagesDataForModel", () => {
  it("orders rows by RES_ORDER and keys each system by file index", () => {
    const files = [
      file("alpha", { sd15: { resolutions: { "1024x1024": res(9), "512x512": res(3) } } }),
      file("beta", { sd15: { resolutions: { "512x512": res(4) } } }),
    ];
    expect(buildImagesDataForModel(files, "sd15")).toEqual([
      { resLabel: "512x512", f0: 3, f1: 4 },
      { resLabel: "1024x1024", f0: 9 },
    ]);
  });

  it("emits no rows for a model absent from every file", () => {
    const files = [file("alpha", { sd15: { resolutions: { "512x512": res(3) } } })];
    expect(buildImagesDataForModel(files, "flux-dev")).toEqual([]);
  });

  it("tolerates a file with no images section alongside one that has data", () => {
    const files = [
      file("alpha", undefined),
      file("beta", { sd15: { resolutions: { "512x512": res(4) } } }),
    ];
    expect(buildImagesDataForModel(files, "sd15")).toEqual([{ resLabel: "512x512", f1: 4 }]);
  });
});


describe("buildImagesDataForResolution", () => {
  it("emits one row per model at the requested resolution, in canonical model order", () => {
    const files = [file("alpha", {
      sdxl: { resolutions: { "512x512": res(6) } },
      sd15: { resolutions: { "512x512": res(3) } },
    })];
    expect(buildImagesDataForResolution(files, "512x512", ALL)).toEqual([
      { modelLabel: "Stable Diffusion 1.5", f0: 3 },
      { modelLabel: "SDXL", f0: 6 },
    ]);
  });

  it("drops a model with no data at that resolution instead of rendering an empty bar", () => {
    const files = [file("alpha", {
      sd15: { resolutions: { "512x512": res(3) } },
      sdxl: { resolutions: { "1024x1024": res(6) } },
    })];
    expect(buildImagesDataForResolution(files, "512x512", ALL))
      .toEqual([{ modelLabel: "Stable Diffusion 1.5", f0: 3 }]);
  });

  it("honours the Models filter", () => {
    const files = [file("alpha", {
      sd15: { resolutions: { "512x512": res(3) } },
      sdxl: { resolutions: { "512x512": res(6) } },
    })];
    expect(buildImagesDataForResolution(files, "512x512", new Set(["sdxl"])))
      .toEqual([{ modelLabel: "SDXL", f0: 6 }]);
  });
});


describe("buildImagesData (legacy combined chart)", () => {
  it("keys series by bare model when a single file is loaded", () => {
    const files = [file("alpha", { sd15: { resolutions: { "512x512": res(3) } } })];
    expect(buildImagesData(files, ALL)).toEqual([{ resLabel: "512x512", sd15: 3 }]);
  });

  it("namespaces series by file index when several files are loaded", () => {
    const files = [
      file("alpha", { sd15: { resolutions: { "512x512": res(3) } } }),
      file("beta", { sd15: { resolutions: { "512x512": res(4) } } }),
    ];
    expect(buildImagesData(files, ALL)).toEqual([
      { resLabel: "512x512", f0_sd15: 3, f1_sd15: 4 },
    ]);
  });

  it("orders rows by RES_ORDER and omits resolutions nothing recorded", () => {
    const files = [file("alpha", {
      sd15: { resolutions: { "1536x1536": res(20), "512x512": res(3) } },
    })];
    expect(buildImagesData(files, ALL).map(row => row.resLabel))
      .toEqual(["512x512", "1536x1536"]);
  });

  it("drops a resolution row left empty by the Models filter", () => {
    const files = [file("alpha", {
      sd15: { resolutions: { "512x512": res(3) } },
      sdxl: { resolutions: { "1024x1024": res(6) } },
    })];
    expect(buildImagesData(files, new Set(["sd15"])))
      .toEqual([{ resLabel: "512x512", sd15: 3 }]);
  });
});


describe("buildImagesLineConfigs", () => {
  const single = [file("alpha", { sd15: { resolutions: { "512x512": res(3) } } })];

  it("colors by model and omits a dash pattern for a single file", () => {
    const data = buildImagesData(single, ALL);
    const [config] = buildImagesLineConfigs(single, data, ALL);
    expect(config.dataKey).toBe("sd15");
    expect(dash(config)).toBeUndefined();
  });

  it("names a series with its display label, not the raw model key", () => {
    const data = buildImagesData(single, ALL);
    expect(buildImagesLineConfigs(single, data, ALL)[0].name).toBe("Stable Diffusion 1.5");
  });

  it("prefers a label carried by the results file", () => {
    const labelled = [file("alpha", {
      sd15: { label: "SD 1.5 (custom)", resolutions: { "512x512": res(3) } },
    })];
    const data = buildImagesData(labelled, ALL);
    expect(buildImagesLineConfigs(labelled, data, ALL)[0].name).toBe("SD 1.5 (custom)");
  });

  it("distinguishes multiple files by color and multiple models by dash pattern", () => {
    const files = [
      file("alpha", { sd15: { resolutions: { "512x512": res(3) } }, sdxl: { resolutions: { "512x512": res(6) } } }),
      file("beta", { sd15: { resolutions: { "512x512": res(4) } }, sdxl: { resolutions: { "512x512": res(7) } } }),
    ];
    const configs = buildImagesLineConfigs(files, buildImagesData(files, ALL), ALL);
    expect(configs.map(c => c.dataKey))
      .toEqual(["f0_sd15", "f0_sdxl", "f1_sd15", "f1_sdxl"]);
    expect(configs[0].stroke).toBe(configs[1].stroke);
    expect(configs[0].stroke).not.toBe(configs[2].stroke);
    expect(dash(configs[0])).not.toBe(dash(configs[1]));
    expect(configs[0].name).toContain("alpha");
  });

  it("emits no config for a model with no plotted points", () => {
    const data = buildImagesData(single, ALL);
    expect(buildImagesLineConfigs(single, data, ALL).map(c => c.dataKey)).not.toContain("sdxl");
  });
});


describe("buildImagesGroupedBarDataForResolution", () => {
  it("emits one row per system keyed by model", () => {
    const files = [
      file("alpha", { sd15: { resolutions: { "512x512": res(3) } } }),
      file("beta", { sd15: { resolutions: { "512x512": res(4) } } }),
    ];
    expect(buildImagesGroupedBarDataForResolution(files, "512x512", ALL)).toEqual([
      { systemLabel: "alpha", sd15: 3 },
      { systemLabel: "beta", sd15: 4 },
    ]);
  });

  it("carries a status label for a timed-out cell so the gap is explained, not blank", () => {
    const files = [file("alpha", { sd15: { timed_out: "768x768", resolutions: { "512x512": res(3) } } })];
    const [row] = buildImagesGroupedBarDataForResolution(files, "768x768", ALL);
    expect(row._status_sd15).toBe("768x768 - Timed Out");
    expect(row.sd15).toBeUndefined();
  });

  it("keeps a system whose only outcome at this resolution is a status label", () => {
    const files = [file("alpha", { sd15: { timed_out: "512x512" } })];
    expect(buildImagesGroupedBarDataForResolution(files, "512x512", ALL))
      .toHaveLength(1);
  });

  it("drops a system with neither data nor a status at this resolution", () => {
    const files = [
      file("alpha", { sd15: { resolutions: { "512x512": res(3) } } }),
      file("beta", { sd15: { resolutions: { "1024x1024": res(9) } } }),
    ];
    expect(buildImagesGroupedBarDataForResolution(files, "512x512", ALL).map(r => r.systemLabel))
      .toEqual(["alpha"]);
  });
});


describe("buildImagesGroupedBarConfigs", () => {
  it("assigns registered colors and display labels in canonical model order", () => {
    const files = [file("a", {
      sdxl: { resolutions: { "512x512": res(6) } },
      sd15: { resolutions: { "512x512": res(3) } },
    })];
    expect(buildImagesGroupedBarConfigs(files, ALL)).toEqual([
      { dataKey: "sd15", name: "Stable Diffusion 1.5", fill: IMAGE_BAR_COLORS.sd15 },
      { dataKey: "sdxl", name: "SDXL", fill: IMAGE_BAR_COLORS.sdxl },
    ]);
  });

  it("gives an unknown model a fallback color", () => {
    const files = [file("a", { "my-model": { resolutions: { "512x512": res(3) } } })];
    expect(buildImagesGroupedBarConfigs(files, new Set(["my-model"]))[0].fill)
      .toMatch(/^#[0-9a-f]{6}$/i);
  });
});


describe("buildImagesBarDataByModel", () => {
  it("emits one row per model with a column per resolution, ordered by RES_ORDER", () => {
    const single = file("alpha", {
      sd15: { resolutions: { "1024x1024": res(9), "512x512": res(3) } },
    });
    expect(buildImagesBarDataByModel(single, ["sd15"])).toEqual([
      { modelLabel: "Stable Diffusion 1.5", "512x512": 3, "1024x1024": 9 },
    ]);
  });

  it("annotates the timed-out resolution and the larger ones it prevented", () => {
    const single = file("alpha", {
      sd15: { timed_out: "768x768", resolutions: { "512x512": res(3) } },
    });
    const [row] = buildImagesBarDataByModel(single, ["sd15"]);
    expect(row["512x512"]).toBe(3);
    expect(row._status_512x512).toBeUndefined();
    expect(row["_status_768x768"]).toBe("768x768 - Timed Out");
    expect(row["_status_1024x1024"]).toBe("1024x1024 - Skipped");
  });

  it("drops a model with neither data nor status rather than rendering an empty row", () => {
    const single = file("alpha", { sd15: { resolutions: { "512x512": res(3) } } });
    expect(buildImagesBarDataByModel(single, ["sd15", "flux-dev"]).map(r => r.modelLabel))
      .toEqual(["Stable Diffusion 1.5"]);
  });
});


describe("buildImagesBarConfigsByModel", () => {
  it("emits a column per recorded resolution, ordered by RES_ORDER with registered colors", () => {
    const single = file("alpha", {
      sd15: { resolutions: { "1024x1024": res(9), "512x512": res(3) } },
    });
    expect(buildImagesBarConfigsByModel(single, ["sd15"])).toEqual([
      { dataKey: "512x512", name: "512x512", fill: RES_COLORS["512x512"] },
      { dataKey: "1024x1024", name: "1024x1024", fill: RES_COLORS["1024x1024"] },
    ]);
  });

  // Without a column the timed-out status label has nowhere to render.
  it("includes the timed-out resolution as a column even though it has no measurement", () => {
    const single = file("alpha", {
      sd15: { timed_out: "768x768", resolutions: { "512x512": res(3) } },
    });
    expect(buildImagesBarConfigsByModel(single, ["sd15"]).map(c => c.dataKey))
      .toEqual(["512x512", "768x768"]);
  });

  it("unions resolutions across the requested models", () => {
    const single = file("alpha", {
      sd15: { resolutions: { "512x512": res(3) } },
      sdxl: { resolutions: { "1024x1024": res(9) } },
    });
    expect(buildImagesBarConfigsByModel(single, ["sd15", "sdxl"]).map(c => c.dataKey))
      .toEqual(["512x512", "1024x1024"]);
  });

  it("emits no columns when the file never ran images", () => {
    expect(buildImagesBarConfigsByModel(file("alpha", undefined), ["sd15"])).toEqual([]);
  });

  it("declares every column its data builder actually populates", () => {
    const single = file("alpha", {
      sd15: { timed_out: "768x768", resolutions: { "512x512": res(3) } },
    });
    const [row] = buildImagesBarDataByModel(single, ["sd15"]);
    const declared = buildImagesBarConfigsByModel(single, ["sd15"]).map(c => c.dataKey);
    for (const key of RES_ORDER.filter(r => row[r] != null)) {
      expect(declared).toContain(key);
    }
  });
});


describe("buildImagesLineDataByRes / buildImagesLineConfigsByRes", () => {
  const single = file("alpha", {
    sd15: { resolutions: { "1024x1024": res(9), "512x512": res(3) } },
    sdxl: { resolutions: { "512x512": res(6) } },
  });

  it("orders rows by RES_ORDER with one key per model", () => {
    expect(buildImagesLineDataByRes(single, ["sd15", "sdxl"])).toEqual([
      { resLabel: "512x512", sd15: 3, sdxl: 6 },
      { resLabel: "1024x1024", sd15: 9 },
    ]);
  });

  it("names each line with its display label", () => {
    const data = buildImagesLineDataByRes(single, ["sd15", "sdxl"]);
    expect(buildImagesLineConfigsByRes(single, ["sd15", "sdxl"], data).map(c => c.name))
      .toEqual(["Stable Diffusion 1.5", "SDXL"]);
  });

  it("omits a requested model that has no points to plot", () => {
    const data = buildImagesLineDataByRes(single, ["sd15", "flux-dev"]);
    expect(buildImagesLineConfigsByRes(single, ["sd15", "flux-dev"], data).map(c => c.dataKey))
      .toEqual(["sd15"]);
  });

  it("returns no rows when the file never ran images", () => {
    expect(buildImagesLineDataByRes(file("alpha", undefined), ["sd15"])).toEqual([]);
  });
});
