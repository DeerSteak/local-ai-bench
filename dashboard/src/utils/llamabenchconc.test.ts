import { describe, it, expect } from "vitest";
import {
  llamaBenchConcTgValues, llamaBenchConcLevels, buildLlamaBenchConcLineData,
  llamaBenchConcPromptDepth, flattenLlamaBenchConcData, llamaBenchConcSortValue,
} from "./llamabenchconc";

const entry = (pl: number, tg: number, speed_tg: number | null) => ({ pp: 8192, tg, pl, speed_tg, speed_pp: 400, speed: 500 });

const fileA = {
  id: "a", hostname: "alpha",
  data: {
    llamabenchconc: {
      "m1": {
        pp: 8192, ctx_size: 139264,
        entries: [entry(1, 128, 40), entry(4, 128, 120), entry(1, 512, 38), entry(4, 512, 110)],
      },
    },
  },
};

const fileB = {
  id: "b", hostname: "beta",
  data: {
    llamabenchconc: {
      "m1": { pp: 8192, entries: [entry(1, 128, 60), entry(2, 128, 100), entry(4, 128, 150)] },
      "m2": { error: "no output for 1800s (idle timeout)" },
    },
  },
};

const emptyFile = { id: "e", hostname: "empty", data: {} };

describe("llamaBenchConcTgValues", () => {
  it("returns distinct tg values ascending", () => {
    expect(llamaBenchConcTgValues([fileA, fileB], "m1")).toEqual([128, 512]);
  });

  it("returns nothing for a model with no data or a missing section", () => {
    expect(llamaBenchConcTgValues([fileA], "nope")).toEqual([]);
    expect(llamaBenchConcTgValues([emptyFile], "m1")).toEqual([]);
  });

  it("treats a missing tg field as 0 rather than dropping the row", () => {
    const f = { id: "x", hostname: "x", data: { llamabenchconc: { m1: { entries: [{ pl: 1, speed_tg: 5 }] } } } };
    expect(llamaBenchConcTgValues([f], "m1")).toEqual([0]);
  });
});

describe("llamaBenchConcLevels", () => {
  it("unions levels across files, numerically sorted", () => {
    expect(llamaBenchConcLevels([fileA, fileB], "m1", 128)).toEqual([1, 2, 4]);
  });

  it("is scoped per tg value", () => {
    expect(llamaBenchConcLevels([fileA, fileB], "m1", 512)).toEqual([1, 4]);
  });

  it("skips entries with no pl", () => {
    const f = { id: "x", hostname: "x", data: { llamabenchconc: { m1: { entries: [{ tg: 128, speed_tg: 5 }] } } } };
    expect(llamaBenchConcLevels([f], "m1", 128)).toEqual([]);
  });

  it("returns nothing for an error-only model", () => {
    expect(llamaBenchConcLevels([fileB], "m2", 128)).toEqual([]);
  });
});

describe("buildLlamaBenchConcLineData", () => {
  it("builds one row per level with a series per file", () => {
    expect(buildLlamaBenchConcLineData([fileA, fileB], "m1", 128)).toEqual([
      { levelLabel: "1-way", f0: 40, f1: 60 },
      { levelLabel: "2-way", f1: 100 },
      { levelLabel: "4-way", f0: 120, f1: 150 },
    ]);
  });

  it("omits a file's series entirely when speed_tg is null", () => {
    const f = { id: "x", hostname: "x", data: { llamabenchconc: { m1: { entries: [{ tg: 128, pl: 1, speed_tg: null as number | null }] } } } };
    expect(buildLlamaBenchConcLineData([f], "m1", 128)).toEqual([{ levelLabel: "1-way" }]);
  });

  it("returns an empty series list for an unknown model", () => {
    expect(buildLlamaBenchConcLineData([fileA, fileB], "ghost", 128)).toEqual([]);
  });
});

describe("llamaBenchConcPromptDepth", () => {
  it("returns the effective prompt depth recorded for the model", () => {
    expect(llamaBenchConcPromptDepth(fileA, "m1")).toBe(8192);
  });

  it("returns null when absent or the section is missing", () => {
    expect(llamaBenchConcPromptDepth(fileB, "m2")).toBeNull();
    expect(llamaBenchConcPromptDepth(emptyFile, "m1")).toBeNull();
  });
});

describe("flattenLlamaBenchConcData", () => {
  it("flattens entries with per-row fields", () => {
    const rows = flattenLlamaBenchConcData([fileA]);
    expect(rows).toHaveLength(4);
    expect(rows[0]).toEqual({
      _fileId: "a", model: "m1", level: 1, pp: 8192, tg: 128,
      speed_tg: 40, speed_pp: 400, speed: 500,
    });
  });

  it("emits a single skipped row for an error model", () => {
    const rows = flattenLlamaBenchConcData([fileB]).filter(r => r.model === "m2");
    expect(rows).toEqual([{
      _fileId: "b", model: "m2", level: "—", skipped: true,
      skip_detail: "no output for 1800s (idle timeout)",
    }]);
  });

  it("falls back to the model-level pp when a row omits it", () => {
    const f = { id: "x", hostname: "x", data: { llamabenchconc: { m1: { pp: 4096, entries: [{ pl: 1, tg: 128, speed_tg: 9 }] } } } };
    expect(flattenLlamaBenchConcData([f])[0].pp).toBe(4096);
  });

  it("returns nothing for files without the section", () => {
    expect(flattenLlamaBenchConcData([emptyFile])).toEqual([]);
  });
});

describe("llamaBenchConcSortValue", () => {
  it("sorts levels numerically, not lexicographically", () => {
    expect(llamaBenchConcSortValue({ level: 16 }, "level")).toBe(16);
    expect(llamaBenchConcSortValue({ level: 2 }, "level")).toBe(2);
  });

  it("pins a non-numeric level to Infinity", () => {
    expect(llamaBenchConcSortValue({ level: "—" }, "level")).toBe(Infinity);
  });

  it("passes other keys through, with a stable fallback for missing values", () => {
    expect(llamaBenchConcSortValue({ speed_tg: 12 }, "speed_tg")).toBe(12);
    expect(llamaBenchConcSortValue({}, "speed_tg")).toBe("");
  });
});
