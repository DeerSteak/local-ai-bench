import { describe, expect, it } from "vitest";
import { buildTrialDisplayRows, isTrialSetArtifact, trialArtifactLoadMode } from "./trials";

const artifact = {
  schema_version: 1,
  compatible: true,
  comparison_mode: "paired",
  rows: [{
    key: "llm/model/2K/tps_mean",
    baseline: { mean: 50, median: 50.1, stdev: 0.5, drift: "none" },
    candidate: { mean: 55, median: 55.1, stdev: 0.6, drift: "none" },
    change_interval_pct: [9.2, 10.8],
    interval_method: "student_t_95",
    practical_threshold_pct: 3,
    verdict: "improved",
  }],
};

describe("trial-set artifact view", () => {
  it("recognizes and normalizes a valid artifact", () => {
    expect(isTrialSetArtifact(artifact)).toBe(true);
    expect(buildTrialDisplayRows(artifact)[0]).toMatchObject({
      baselineMean: 50,
      candidateMean: 55,
      interval: [9.2, 10.8],
      verdict: "improved",
    });
  });

  it("keeps missing intervals unavailable instead of zero", () => {
    const incomplete = { ...artifact, rows: [{ ...artifact.rows[0], change_interval_pct: null }] };
    expect(buildTrialDisplayRows(incomplete)[0].interval).toBeNull();
  });

  it("rejects ordinary results and malformed rows", () => {
    expect(isTrialSetArtifact({ profile: {} })).toBe(false);
    expect(buildTrialDisplayRows({ ...artifact, rows: [null] })).toEqual([]);
  });

  it("requires a trial artifact to be loaded by itself", () => {
    expect(trialArtifactLoadMode([artifact])).toBe("single");
    expect(trialArtifactLoadMode([artifact, { profile: {} }])).toBe("mixed");
    expect(trialArtifactLoadMode([{ profile: {} }])).toBe("none");
  });
});
