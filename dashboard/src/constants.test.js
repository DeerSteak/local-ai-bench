import { describe, it, expect } from "vitest";
import {
  LLM_MODEL_ORDER, LLM_MODEL_LABELS, MODEL_COLORS, MODEL_SIZE_TIER,
  IMAGE_MODEL_ORDER, IMAGE_MODEL_LABELS, IMAGE_MODEL_COLORS,
  EMBED_MODEL_ORDER, EMBED_MODEL_LABELS, EMBED_MODEL_COLORS,
  SIZE_TIER_ORDER,
} from "./constants";

// These catch the most common maintenance mistake in this file: adding a
// model to one registry (e.g. LLM_MODEL_ORDER, to make it show up at all)
// without also adding it to the others (label, color, size tier) that other
// code assumes are present for every ordered model.
describe("model registry consistency", () => {
  it("every LLM model in LLM_MODEL_ORDER has a label, a color, and a valid size tier", () => {
    for (const model of LLM_MODEL_ORDER) {
      expect(LLM_MODEL_LABELS[model], `${model} missing a label`).toBeDefined();
      expect(MODEL_COLORS[model], `${model} missing a color`).toBeDefined();
      expect(MODEL_SIZE_TIER[model], `${model} missing a size tier`).toBeDefined();
      expect(SIZE_TIER_ORDER, `${model}'s tier "${MODEL_SIZE_TIER[model]}" is not a recognized tier`)
        .toContain(MODEL_SIZE_TIER[model]);
    }
  });

  it("every image model in IMAGE_MODEL_ORDER has a label and a color", () => {
    for (const model of IMAGE_MODEL_ORDER) {
      expect(IMAGE_MODEL_LABELS[model], `${model} missing a label`).toBeDefined();
      expect(IMAGE_MODEL_COLORS[model], `${model} missing a color`).toBeDefined();
    }
  });

  it("every embedding model in EMBED_MODEL_ORDER has a label and a color", () => {
    for (const model of EMBED_MODEL_ORDER) {
      expect(EMBED_MODEL_LABELS[model], `${model} missing a label`).toBeDefined();
      expect(EMBED_MODEL_COLORS[model], `${model} missing a color`).toBeDefined();
    }
  });

  it("LLM_MODEL_ORDER has no duplicate entries", () => {
    expect(new Set(LLM_MODEL_ORDER).size).toBe(LLM_MODEL_ORDER.length);
  });

  it("IMAGE_MODEL_ORDER and EMBED_MODEL_ORDER have no duplicate entries", () => {
    expect(new Set(IMAGE_MODEL_ORDER).size).toBe(IMAGE_MODEL_ORDER.length);
    expect(new Set(EMBED_MODEL_ORDER).size).toBe(EMBED_MODEL_ORDER.length);
  });
});
