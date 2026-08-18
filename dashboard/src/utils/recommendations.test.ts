import { describe, expect, it } from "vitest";
import { buildRecommendationDisplayItems, formatRecommendationConstraint, isRecommendationArtifact, recommendationArtifactLoadMode } from "./recommendations";
import recommendationExample from "../../../samples/recommendation_example.json";

const artifact = {
  artifact_type: "recommendation", schema_version: 1, verdict: "recommended",
  constraints: { primary_objective: "throughput" }, source_sha256: ["abc"],
  candidates: {
    recommended: [{ candidate: "fast", evidence: { throughput: {
      value: 80, unit: "tokens_per_second", evidence_path: "llm/fast/8K/tps_mean",
      raw_evidence_paths: ["llm/fast/8K/valid_samples"],
    } } }],
    tied: [],
    other_eligible: [{ candidate: "steady", evidence: { throughput: {
      value: 70, unit: "tokens_per_second", evidence_path: "llm/steady/8K/tps_mean",
      raw_evidence_paths: ["llm/steady/8K/valid_samples"],
    } } }],
    eliminated: [{ candidate: "small", reasons: [{
      constraint: "accuracy", operator: "minimum", threshold: 80,
      measurement: { value: 70, unit: "percent", evidence_path: "code/small/accuracy_pct" },
    }] }],
    unevaluated: [{ candidate: "unknown", missing_evidence: ["memory"], resolution: {} }],
  },
};

describe("recommendation artifact view", () => {
  it("recognizes the versioned artifact and preserves four result groups", () => {
    expect(isRecommendationArtifact(artifact)).toBe(true);
    expect(buildRecommendationDisplayItems(artifact)).toEqual([
      { group: "recommended", candidate: "fast", detail: "Throughput: 80 tokens/s", evidencePath: "llm/fast/8K/tps_mean · llm/fast/8K/valid_samples" },
      { group: "other_eligible", candidate: "steady", detail: "Throughput: 70 tokens/s", evidencePath: "llm/steady/8K/tps_mean · llm/steady/8K/valid_samples" },
      { group: "eliminated", candidate: "small", detail: "Accuracy: 70% (below minimum 80%)", evidencePath: "code/small/accuracy_pct" },
      { group: "unevaluated", candidate: "unknown", detail: "Needs: Peak memory", evidencePath: null },
    ]);
  });

  it("rejects malformed artifacts and malformed candidate rows", () => {
    expect(isRecommendationArtifact({ ...artifact, verdict: "best" })).toBe(false);
    expect(buildRecommendationDisplayItems({
      ...artifact, candidates: { ...artifact.candidates, recommended: [null] },
    })).toEqual(
      buildRecommendationDisplayItems({
        ...artifact, candidates: { ...artifact.candidates, recommended: [] },
      }),
    );
  });

  it("requires a recommendation artifact to be loaded by itself", () => {
    expect(recommendationArtifactLoadMode([artifact])).toBe("single");
    expect(recommendationArtifactLoadMode([artifact, { profile: {} }])).toBe("mixed");
    expect(recommendationArtifactLoadMode([{ profile: {} }])).toBe("none");
  });

  it("renders the shared recommendation conformance artifact", () => {
    expect(isRecommendationArtifact(recommendationExample)).toBe(true);
    expect(buildRecommendationDisplayItems(recommendationExample)[0]).toMatchObject({
      group: "recommended", candidate: "qwen3.5-4b-q4",
      evidencePath: "llm/qwen3.5-4b-q4/8K/tps_mean · llm/qwen3.5-4b-q4/8K/valid_samples",
    });
    expect(buildRecommendationDisplayItems(recommendationExample).map(item => item.group)).toEqual([
      "recommended", "eliminated", "unevaluated",
    ]);
  });

  it("formats throughput constraints in workload-specific units", () => {
    expect(formatRecommendationConstraint("minimum_throughput", 0.4, "images")).toBe("0.4 images/s");
    expect(formatRecommendationConstraint("minimum_throughput", 20, "llm")).toBe("20 tokens/s");
  });
});
