import { describe, expect, it } from "vitest";
import { buildVariantDisplayRows, isVariantComparisonArtifact, variantArtifactLoadMode } from "./variants";

const artifact = {
  artifact_type: "variant_comparison", schema_version: 1, base_model: "demo",
  reference_variant: "Q4_K_M", variants: [
    {
      variant: "Q4_K_M", reference: true, quality_verdict: "reference", quality_ranked: false,
      quality: { value: 70, delta: 0 }, throughput: { value: 20, delta: 0 },
      memory: { value: 4, delta: 0 }, energy: { value: 10, delta: 0 },
    },
    {
      variant: "Q8_0", reference: false, quality_verdict: "unchanged", quality_ranked: false,
      quality: { value: 72, delta: 2 }, throughput: { value: 15, delta: -25 },
      memory: { value: 8, delta: 100 }, energy: { value: null, delta: null },
    },
  ],
};

describe("variant comparison artifact", () => {
  it("recognizes a standalone artifact and formats reference-relative metrics", () => {
    expect(isVariantComparisonArtifact(artifact)).toBe(true);
    expect(buildVariantDisplayRows(artifact)).toEqual([
      {
        variant: "Q4_K_M", reference: true, qualityVerdict: "reference", qualityRanked: false,
        quality: "70 (0 pp)", throughput: "20 (0%)", memory: "4 (0%)", energy: "10 (0%)",
      },
      {
        variant: "Q8_0", reference: false, qualityVerdict: "unchanged", qualityRanked: false,
        quality: "72 (+2 pp)", throughput: "15 (-25%)", memory: "8 (+100%)",
        energy: "Not recorded",
      },
    ]);
  });

  it("requires the artifact to load alone and rejects malformed rows", () => {
    expect(variantArtifactLoadMode([artifact])).toBe("single");
    expect(variantArtifactLoadMode([artifact, { profile: {} }])).toBe("mixed");
    expect(variantArtifactLoadMode([{ profile: {} }])).toBe("none");
    expect(buildVariantDisplayRows({ ...artifact, variants: [null] })).toEqual([]);
  });
});
