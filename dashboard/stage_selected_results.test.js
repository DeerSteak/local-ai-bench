import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { stageSelectedResults } from "./stage_selected_results.mjs";

describe("stageSelectedResults", () => {
  it("copies only selected JSON files and writes browser-safe URLs", () => {
    const root = mkdtempSync(join(tmpdir(), "lab-dashboard-"));
    const dist = join(root, "dist");
    mkdirSync(dist);
    const first = join(root, "first result.json");
    const second = join(root, "second.json");
    writeFileSync(first, '{"version":"4.1"}');
    writeFileSync(second, '{"version":"4.0"}');

    expect(stageSelectedResults(dist, [first, second])).toEqual([
      { name: "first result.json", url: "/__selected_results__/0.json" },
      { name: "second.json", url: "/__selected_results__/1.json" },
    ]);
    expect(readFileSync(join(dist, "__selected_results__", "0.json"), "utf8"))
      .toBe('{"version":"4.1"}');
  });

  it("removes a previous selection when launched without files", () => {
    const root = mkdtempSync(join(tmpdir(), "lab-dashboard-"));
    const dist = join(root, "dist");
    mkdirSync(dist);
    const result = join(root, "result.json");
    writeFileSync(result, "{}");
    stageSelectedResults(dist, [result]);

    expect(stageSelectedResults(dist, [])).toEqual([]);
    expect(() => readFileSync(join(dist, "__selected_results__.json"))).toThrow();
  });

  it("rejects missing and excessive selections", () => {
    const root = mkdtempSync(join(tmpdir(), "lab-dashboard-"));
    const dist = join(root, "dist");
    mkdirSync(dist);
    expect(() => stageSelectedResults(dist, [join(root, "missing.json")]))
      .toThrow("Result file not found");
    expect(() => stageSelectedResults(dist, Array(7).fill(join(root, "missing.json"))))
      .toThrow("no more than 6");
  });
});
