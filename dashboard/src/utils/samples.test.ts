import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { ACCURACY_TESTS } from "../constants";
import type { ResultsFile } from "../types";
import { flattenAccuracyData } from "./accuracy";
import { flattenConcurrencyData } from "./concurrency";
import { flattenEmbedData } from "./embeddings";
import { flattenImageData } from "./images";
import { flattenLlamaBenchData } from "./llamabench";
import { flattenLlamaBenchConcData } from "./llamabenchconc";
import { flattenLLMData } from "./llm";
import { parseResultsJSON } from "./shared";
import { flattenSustainedData } from "./sustained";
import { flattenVllmBenchData } from "./vllmbench";

const samplesDir = resolve(process.cwd(), "../samples");
const sampleNames = readdirSync(samplesDir).filter(name => name.endsWith(".json"));

function loadSample(name: string): ResultsFile {
  const parsed = parseResultsJSON(readFileSync(resolve(samplesDir, name), "utf8"));
  expect(parsed.error).toBeNull();
  return { id: name, hostname: name, data: parsed.data! };
}

describe("sample result compatibility", () => {
  it.each(sampleNames)("renders every populated section in %s", name => {
    const file = loadSample(name);
    const builders: [string, () => unknown[]][] = [
      ["llm", () => flattenLLMData([file])],
      ["llm_conversation", () => flattenLLMData([file], "llm_conversation")],
      ["concurrency_tool", () => flattenConcurrencyData([file], "concurrency_tool")],
      ["concurrency_chat", () => flattenConcurrencyData([file], "concurrency_chat")],
      ["embeddings", () => flattenEmbedData([file])],
      ["images", () => flattenImageData([file])],
      ["llamabench", () => flattenLlamaBenchData([file])],
      ["llamabenchconc", () => flattenLlamaBenchConcData([file])],
      ["vllmbench", () => flattenVllmBenchData([file])],
      ["sustained", () => flattenSustainedData([file])],
      ...ACCURACY_TESTS.map(test => [test, () => flattenAccuracyData([file], test)] as [string, () => unknown[]]),
    ];
    for (const [section, buildRows] of builders) {
      if (Object.keys(file.data[section] || {}).length > 0) expect(buildRows(), section).not.toHaveLength(0);
    }
  });
});
