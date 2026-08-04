import { describe, expect, it } from "vitest";
import {
  buildPauseSummaries, buildValidityRows, formatPausedDuration, validitySummary,
} from "./validity";

const file = {
  id: "f1", hostname: "System A", data: { llm: { model: { "2K": {
    completed_runs: 3,
    valid_samples: [
      { tokens_per_sec: 50, client_ttft_sec: 0.2, client_wall_sec: 2.2, generated_tokens: 100 },
      { tokens_per_sec: 48, client_ttft_sec: 0.3, client_wall_sec: 2.3, generated_tokens: 96 },
    ],
    invalid_runs: [{ run: 3, errors: ["implausible_server_tps"] }],
  } } } },
};

describe("validity inspection", () => {
  it("exposes valid samples and excluded runs without treating invalid as zero", () => {
    const rows = buildValidityRows([file], "llm");
    expect(rows).toHaveLength(3);
    expect(rows[0]).toMatchObject({ model: "model", caseLabel: "2K", status: "valid" });
    expect(rows[0].summary).toContain("50 tok/s");
    expect(rows[2]).toMatchObject({ sample: 3, status: "invalid" });
    expect(rows[2].errors).toEqual(["implausible_server_tps"]);
    expect(validitySummary(rows)).toEqual({ total: 3, valid: 2, invalid: 1, legacy: 0 });
  });

  it("labels historical aggregate-only cases instead of inventing samples", () => {
    const legacy = { ...file, data: { llm: { model: { "2K": { n_runs: 2 } } } } };
    expect(buildValidityRows([legacy], "llm")).toEqual([expect.objectContaining({
      status: "legacy", summary: "2 completed; raw samples unavailable",
    })]);
  });

  it("shows native internal repetitions and ignores unsupported sections", () => {
    const native = { ...file, data: { llamabench: { model: { prefill_entries: [
      { n_prompt: 512, ts_runs: [100, 101] },
    ] } } } };
    expect(buildValidityRows([native], "llamabench").map(row => row.summary))
      .toEqual(["100 tok/s", "101 tok/s"]);
    expect(buildValidityRows([native], "accuracy")).toEqual([]);
    expect(buildValidityRows(null, "llm")).toEqual([]);
  });

  it("summarizes pause count and total duration from run transitions", () => {
    const paused = { ...file, data: { ...file.data, run: {
      pause: { control_transitions: [
        { state: "running", at: "2026-08-04T10:00:00Z" },
        { state: "paused", at: "2026-08-04T10:05:00Z" },
        { state: "running", at: "2026-08-04T10:35:30Z" },
        { state: "paused", at: "2026-08-04T11:00:00Z" },
        { state: "running", at: "2026-08-04T12:30:00Z" },
      ] },
    } } };
    expect(buildPauseSummaries([paused])).toEqual([{
      fileId: "f1", system: "System A", count: 2,
      totalPausedSeconds: 7230, incomplete: false,
    }]);
    expect(formatPausedDuration(7230)).toBe("2h 0m");
  });

  it("uses run completion for a final pause and flags an unknown open duration", () => {
    const transitions = [{ state: "paused", at: "2026-08-04T10:00:00Z" }];
    const finished = { ...file, data: { run: {
      finished_at: "2026-08-04T10:10:00Z", pause: { control_transitions: transitions },
    } } };
    expect(buildPauseSummaries([finished])[0]).toMatchObject({
      count: 1, totalPausedSeconds: 600, incomplete: false,
    });
    const open = { ...file, data: { run: { pause: { control_transitions: transitions } } } };
    expect(buildPauseSummaries([open])[0]).toMatchObject({ incomplete: true });
    expect(buildPauseSummaries([{ ...file, data: { run: { pause: {
      control_transitions: [{ state: "paused", at: "invalid" }],
    } } } }])).toEqual([expect.objectContaining({
      count: 1, totalPausedSeconds: 0, incomplete: true,
    })]);
    expect(buildPauseSummaries(null)).toEqual([]);
  });
});
