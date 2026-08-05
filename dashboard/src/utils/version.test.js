import { describe, it, expect } from "vitest";
import { parseSuiteVersion } from "./version";

describe("parseSuiteVersion", () => {
  it("reads VERSION from a config.py-shaped source", () => {
    expect(parseSuiteVersion('VERSION        = "4.1"\n')).toBe("4.1");
  });

  it("accepts single quotes and no padding", () => {
    expect(parseSuiteVersion("VERSION='4.1'")).toBe("4.1");
  });

  it("finds VERSION among surrounding assignments", () => {
    const source = [
      "N_RUNS = 3",
      "# VERSION bumps alongside README's title",
      'VERSION        = "4.12"',
      'LLAMACPP_KV_CACHE_TYPE = "q8_0"',
    ].join("\n");
    expect(parseSuiteVersion(source)).toBe("4.12");
  });

  it("does not match a differently-named key that ends in VERSION", () => {
    expect(parseSuiteVersion('CATALOG_VERSION = "1"')).toBe(null);
    expect(parseSuiteVersion('FRONTEND_STATE_VERSION = "2"')).toBe(null);
  });

  it("does not match an indented or commented assignment", () => {
    expect(parseSuiteVersion('    VERSION = "4.1"')).toBe(null);
    expect(parseSuiteVersion('# VERSION = "4.1"')).toBe(null);
  });

  it("returns null when VERSION is absent or the source is empty/missing", () => {
    expect(parseSuiteVersion("N_RUNS = 3")).toBe(null);
    expect(parseSuiteVersion("")).toBe(null);
    expect(parseSuiteVersion(null)).toBe(null);
    expect(parseSuiteVersion(undefined)).toBe(null);
  });
});
