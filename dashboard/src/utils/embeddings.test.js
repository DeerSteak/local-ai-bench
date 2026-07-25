import { describe, it, expect } from "vitest";
import { getEmbedLabel } from "./embeddings";

describe("getEmbedLabel", () => {
  it("uses the file-provided label when present, since results files carry their own labels", () => {
    const files = [{ data: { embeddings: { m: { label: "Custom Label" } } } }];
    expect(getEmbedLabel(files, "m")).toBe("Custom Label");
  });
  it("falls back to the static label map when no loaded file provides one", () => {
    const files = [{ data: { embeddings: { "nomic-embed-text": {} } } }];
    expect(getEmbedLabel(files, "nomic-embed-text")).toBe("Nomic Embed Text");
  });
});
