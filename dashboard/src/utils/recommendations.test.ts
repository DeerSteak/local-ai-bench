import { describe, expect, it } from "vitest";
import { buildRecommendationDisplayItems, isRecommendationArtifact, recommendationArtifactLoadMode } from "./recommendations";
import recommendationExample from "../../../samples/recommendation_example.json";

const artifact = {
  artifact_type: "recommendation", schema_version: 1, verdict: "recommended",
  constraints: { primary_objective: "throughput" }, source_sha256: ["abc"],
  recommended: [{ candidate: "fast", evidence: { throughput: {
    value: 80, unit: "tokens_per_second", evidence_path: "llm/fast/8K/tps_mean",
  } } }],
  tied: [],
  eliminated: [{ candidate: "small", reasons: [{
    constraint: "accuracy", operator: "minimum", threshold: 80,
    measurement: { evidence_path: "code/small/accuracy_pct" },
  }] }],
  unevaluated: [{ candidate: "unknown", missing_evidence: ["memory"], resolution: {} }],
};

describe("recommendation artifact view", () => {
  it("recognizes the versioned artifact and preserves four result groups", () => {
    expect(isRecommendationArtifact(artifact)).toBe(true);
    expect(buildRecommendationDisplayItems(artifact)).toEqual([
      { group: "recommended", candidate: "fast", detail: "80 tokens_per_second", evidencePath: "llm/fast/8K/tps_mean" },
      { group: "eliminated", candidate: "small", detail: "accuracy minimum 80", evidencePath: "code/small/accuracy_pct" },
      { group: "unevaluated", candidate: "unknown", detail: "Needs: memory", evidencePath: null },
    ]);
  });

  it("rejects malformed artifacts and malformed candidate rows", () => {
    expect(isRecommendationArtifact({ ...artifact, verdict: "best" })).toBe(false);
    expect(buildRecommendationDisplayItems({ ...artifact, recommended: [null] })).toEqual(
      buildRecommendationDisplayItems({ ...artifact, recommended: [] }),
    );
  });

  it("requires a recommendation artifact to be loaded by itself", () => {
    expect(recommendationArtifactLoadMode([artifact])).toBe("single");
    expect(recommendationArtifactLoadMode([artifact, { profile: {} }])).toBe("mixed");
    expect(recommendationArtifactLoadMode([{ profile: {} }])).toBe("none");
  });

  it("renders the shared Python-generated conformance artifact", () => {
    expect(isRecommendationArtifact(recommendationExample)).toBe(true);
    expect(buildRecommendationDisplayItems(recommendationExample)[0]).toMatchObject({
      group: "recommended", candidate: "qwen3.5-4b-q4",
      evidencePath: "llm/qwen3.5-4b-q4/8K/tps_mean",
    });
  });
});
