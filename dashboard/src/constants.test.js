import { describe, it, expect } from "vitest";
import {
  LLM_MODEL_ORDER, LEGACY_LLM_MODEL_ORDER, LLM_DISPLAY_ORDER,
  LLM_MODEL_LABELS, MODEL_COLORS, MODEL_SIZE_TIER,
  IMAGE_MODEL_ORDER, IMAGE_MODEL_LABELS, IMAGE_MODEL_COLORS,
  EMBED_MODEL_ORDER, EMBED_MODEL_LABELS, EMBED_MODEL_COLORS,
  SIZE_TIER_ORDER, RES_ORDER, RES_COLORS,
  ACCURACY_TESTS, ACCURACY_TEST_LABELS,
} from "./constants";

// These catch the most common maintenance mistake in this file: adding a
// model to one registry (e.g. LLM_MODEL_ORDER, to make it show up at all)
// without also adding it to the others (label, color, size tier) that other
// code assumes are present for every ordered model.
describe("model registry consistency", () => {
  it("contains the complete 12-model LLM catalog in tier order", () => {
    expect(LLM_MODEL_ORDER).toEqual([
      "gemma3-1b", "granite4.1-3b-q4", "qwen3.5-4b-q4",
      "granite4.1-8b-q4", "qwen3.5-9b-q4", "phi4-14b",
      "gemma3-27b-q4", "nemotron3-nano-30b-a3b", "qwen3.6-35b-a3b",
      "llama3.3-70b-q4", "qwen3-coder-next-80b-a3b-q4", "nemotron3-super-120b",
    ]);
  });

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

  it("keeps removed catalog models renderable after the current catalog", () => {
    expect(LLM_DISPLAY_ORDER).toEqual([...LLM_MODEL_ORDER, ...LEGACY_LLM_MODEL_ORDER]);
    expect(new Set(LLM_DISPLAY_ORDER).size).toBe(LLM_DISPLAY_ORDER.length);
    for (const model of LEGACY_LLM_MODEL_ORDER) {
      expect(LLM_MODEL_LABELS[model]).toBeDefined();
      expect(MODEL_COLORS[model]).toBeDefined();
      expect(MODEL_SIZE_TIER[model]).toBeDefined();
    }
  });

  it("IMAGE_MODEL_ORDER and EMBED_MODEL_ORDER have no duplicate entries", () => {
    expect(new Set(IMAGE_MODEL_ORDER).size).toBe(IMAGE_MODEL_ORDER.length);
    expect(new Set(EMBED_MODEL_ORDER).size).toBe(EMBED_MODEL_ORDER.length);
  });
});

describe("image resolution registry", () => {
  it("includes SD 1.5 native resolutions before the larger-model defaults", () => {
    expect(RES_ORDER).toEqual(["512x512", "768x768", "1024x1024", "1536x1536"]);
  });

  it("assigns a color to every ordered resolution", () => {
    for (const resolution of RES_ORDER) expect(RES_COLORS[resolution]).toBeDefined();
  });
});

describe("accuracy registry", () => {
  it("matches the benchmark workload order and labels every test", () => {
    expect(ACCURACY_TESTS).toEqual(["mcq", "math", "reasoning", "code", "tool"]);
    for (const test of ACCURACY_TESTS) expect(ACCURACY_TEST_LABELS[test]).toBeTruthy();
  });
});
