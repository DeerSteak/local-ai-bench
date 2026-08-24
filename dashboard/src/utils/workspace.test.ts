import { describe, expect, it } from "vitest";

import type { DisplayFile } from "../types";
import { buildWorkspaceSelection } from "./workspace";

const digest = (character: string) => character.repeat(64);
const file = (id: string, sha256: string): DisplayFile => ({
  id, name: `${id}.json`, sourceSha256: sha256, hostname: id,
  backend: "cpu", os: "", wsl: false, ram_gb: null, version: null,
  timestamp: null, reliabilityWarning: "", data: {},
});
const view = {
  section: "llm", accuracy_test: "mcq", enabled_models: ["model"],
  enabled_image_models: [], enabled_embedding_models: [], hostname_overrides: {},
};

describe("workspace selection", () => {
  it("binds the baseline and view to exact source digests", () => {
    const selection = buildWorkspaceSelection(
      [file("first", digest("a")), file("second", digest("b"))], "second", view,
    );
    expect(selection.results).toEqual([
      { name: "first.json", sha256: digest("a") },
      { name: "second.json", sha256: digest("b") },
    ]);
    expect(selection.baseline_sha256).toBe(digest("b"));
    expect(selection.view).toEqual(view);
  });

  it("rejects empty, duplicate, malformed, and foreign selections", () => {
    expect(() => buildWorkspaceSelection([], null, view)).toThrow("at least one");
    expect(() => buildWorkspaceSelection([
      file("first", digest("a")), file("second", digest("a")),
    ], null, view)).toThrow("distinct valid");
    expect(() => buildWorkspaceSelection([file("first", "bad")], null, view))
      .toThrow("distinct valid");
    expect(() => buildWorkspaceSelection([file("first", digest("a"))], "missing", view))
      .toThrow("not selected");
  });
});
