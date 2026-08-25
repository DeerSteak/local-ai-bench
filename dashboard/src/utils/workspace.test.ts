import { describe, expect, it } from "vitest";

import type { DisplayFile } from "../types";
import {
  buildWorkspaceSelection, requestWorkspaceEvaluation, requestWorkspaceExport,
  isWorkspaceSelection,
} from "./workspace";

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

  it("recognizes only versioned workspace-selection artifacts", () => {
    const selection = buildWorkspaceSelection([file("first", digest("a"))], null, view);
    expect(isWorkspaceSelection(selection)).toBe(true);
    expect(isWorkspaceSelection({ ...selection, schema_version: 2 })).toBe(false);
    expect(isWorkspaceSelection({ ...selection, results: [{ name: "x", sha256: "bad" }] }))
      .toBe(false);
  });

  it("sends the exact selection and original sources to the bounded service", async () => {
    const selected = file("first", digest("a"));
    selected.sourceText = '{"value":1}';
    const selection = buildWorkspaceSelection([selected], null, view);
    const requests: { input: string, init?: RequestInit }[] = [];
    const exported = new Blob(["report"]);
    const result = await requestWorkspaceExport(selection, [selected], "html", async (input, init) => {
      requests.push({ input, init });
      return input.includes("config")
        ? { ok: true, json: async () => ({ token: "token" }), blob: async () => new Blob() }
        : { ok: true, json: async () => ({}), blob: async () => exported };
    });
    expect(result).toEqual({ blob: exported, filename: "decision.html" });
    expect(requests[1].init?.headers).toEqual({
      "Authorization": "Bearer token", "Content-Type": "application/json",
    });
    expect(JSON.parse(String(requests[1].init?.body))).toEqual({
      format: "html", selection, results: [{ name: "first.json", text: '{"value":1}' }],
    });
  });

  it("rejects export when original source text is unavailable", async () => {
    const selected = file("first", digest("a"));
    const selection = buildWorkspaceSelection([selected], null, view);
    await expect(requestWorkspaceExport(selection, [selected], "bundle"))
      .rejects.toThrow("Original result text");
  });

  it("returns authoritative workspace evaluation without browser-side policy logic", async () => {
    const selected = file("first", digest("a"));
    selected.sourceText = "{}";
    const selection = buildWorkspaceSelection([selected], null, view);
    const evaluation = { acceptance: { decision: "accepted" }, recommendation: null };
    const result = await requestWorkspaceEvaluation(selection, [selected], async input => (
      input.includes("config")
        ? { ok: true, json: async () => ({ token: "token" }), blob: async () => new Blob() }
        : { ok: true, json: async () => evaluation, blob: async () => new Blob() }
    ));
    expect(result).toEqual(evaluation);
  });
});
