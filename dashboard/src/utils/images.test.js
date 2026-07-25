import { describe, it, expect } from "vitest";
import { getImageBarStatusLabel, buildImagesDataForModel, buildImagesBarDataByModel } from "./images";

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
