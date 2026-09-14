import { describe, it, expect } from "vitest";
import { getImageBarStatusLabel, buildImagesDataForModel, buildImagesBarDataByModel, buildImagesGroupedBarConfigs, getImageResolutions } from "./images";

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

describe("resolution-specific image series", () => {
  const files = [{ hostname: "alpha", data: { images: {
    sd15: { resolutions: { "512x512": { sec_per_image_mean: 1 }, "768x768": { sec_per_image_mean: 2 } } },
    sdxl: { resolutions: { "1024x1024": { sec_per_image_mean: 3 }, "1536x1536": { sec_per_image_mean: 4 } } },
    "z-image-turbo": { resolutions: { "1024x1024": { sec_per_image_mean: 5 }, "1536x1536": { sec_per_image_mean: 6 } } },
  } } }];
  const enabled = new Set(["sd15", "sdxl", "z-image-turbo"]);

  it.each([
    ["512x512", ["sd15"]], ["768x768", ["sd15"]],
    ["1024x1024", ["sdxl", "z-image-turbo"]], ["1536x1536", ["sdxl", "z-image-turbo"]],
  ])("includes only recorded models at %s in both combined and per-system charts", (resolution, models) => {
    expect(buildImagesGroupedBarConfigs(files, resolution as string, enabled).map(c => c.dataKey)).toEqual(models);
    expect(getImageResolutions(files, [...enabled])).toContain(resolution);
  });

  it("unions actual series across systems and preserves older non-default measurements", () => {
    const older = { data: { images: { sdxl: { resolutions: { "512x512": { sec_per_image_mean: 8 } } } } } };
    expect(buildImagesGroupedBarConfigs([...files, older], "512x512", enabled).map(c => c.dataKey)).toEqual(["sd15", "sdxl"]);
    expect(buildImagesGroupedBarConfigs([older], "1536x1536", enabled)).toEqual([]);
  });

  it("respects filters and tolerates missing, null, and unknown models", () => {
    const sparse = [{ data: {} }, { data: { images: { sd15: null, custom: {
      resolutions: { "768x768": { sec_per_image_mean: 0 } },
    } } } }];
    expect(buildImagesGroupedBarConfigs(sparse, "768x768", new Set(["sd15", "custom"])).map(c => c.dataKey)).toEqual(["custom"]);
    expect(buildImagesGroupedBarConfigs(files, "512x512", new Set(["sdxl"]))).toEqual([]);
    expect(getImageResolutions(files, ["sdxl"])).toEqual(["1024x1024", "1536x1536"]);
  });

  it("retains timeout-only series and inferred skips only within the workload range", () => {
    const timed = [{ data: { images: {
      sd15: { timed_out: "512x512" }, sdxl: { timed_out: "1024x1024" },
    } } }];
    expect(buildImagesGroupedBarConfigs(timed, "768x768", enabled).map(c => c.dataKey)).toEqual(["sd15"]);
    expect(buildImagesGroupedBarConfigs(timed, "1536x1536", enabled).map(c => c.dataKey)).toEqual(["sdxl"]);
    expect(getImageResolutions(timed, ["sd15"])).toEqual(["512x512", "768x768"]);
    expect(getImageBarStatusLabel(timed[0], "sdxl", "1536x1536")).toBe("1536x1536 - Skipped");
  });

  it("does not overwrite a recorded measurement with a stale timeout", () => {
    const stale = { data: { images: { sd15: { timed_out: "512x512", resolutions: {
      "768x768": { sec_per_image_mean: 2 },
    } } } } };
    expect(getImageBarStatusLabel(stale, "sd15", "768x768")).toBeNull();
  });
});


it("keeps unknown model colors stable across system rosters and filters", () => {
  const a = { data: { images: { custom: { resolutions: { "512x512": { sec_per_image_mean: 2 } } } } } };
  const b = { data: { images: { sd15: { resolutions: { "512x512": { sec_per_image_mean: 1 } } }, ...a.data.images } } };
  const enabled = new Set(["sd15", "custom"]);
  const color = buildImagesGroupedBarConfigs([a], "512x512", enabled)[0].fill;
  expect(buildImagesGroupedBarConfigs([b], "512x512", enabled).find(c => c.dataKey === "custom")?.fill).toBe(color);
  expect(buildImagesGroupedBarConfigs([b], "512x512", new Set(["custom"]))[0].fill).toBe(color);
});
