import {
  buildLlamaBenchDecodeLineConfigsByModel,
  buildLlamaBenchDecodeLineDataByModel,
  buildLlamaBenchPrefillLineConfigsByModel,
  buildLlamaBenchPrefillLineDataByModel,
  llamaBenchHasCombinedOnly,
} from "../../utils/llamabench";
import type { RefObject } from "react";
import { getAllLLMModels } from "../../utils/llm";
import { getModelSizeTier, modelLabel } from "../../utils/shared";
import { SIZE_TIER_ORDER } from "../../constants";
import BySystemPanel from "./BySystemPanel";
import type { ResultsFile } from "../../types";

export default function LlamaBenchBySystemPanel({ containerRef, files, enabledModels, chartWidth, logoSrc, isSplit }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledModels: Set<string>,
  chartWidth: number, logoSrc?: string, isSplit: boolean,
}) {
  const allModels = getAllLLMModels(files)
    .filter(model => enabledModels.has(model) && files.some(file => file.data.llamabench?.[model]));

  const modelGroupSpecs = isSplit
    ? SIZE_TIER_ORDER
        .map(tier => ({ tier, models: allModels.filter(model => getModelSizeTier(model) === tier) }))
        .filter(group => group.models.length > 0)
    : [{ tier: null, models: allModels }];

  const systemGroups = files.map(file => {
    const groups = modelGroupSpecs.map(({ tier, models }) => {
      const decodeData = buildLlamaBenchDecodeLineDataByModel(file, models);
      const decodeConfigs = buildLlamaBenchDecodeLineConfigsByModel(file, models, decodeData);
      const prefillData = buildLlamaBenchPrefillLineDataByModel(file, models);
      const prefillConfigs = buildLlamaBenchPrefillLineConfigsByModel(models, prefillData);
      const metrics = [];
      if (decodeConfigs.length) {
        metrics.push({
          key: "decode", title: "Decode Throughput by Prompt Depth",
          yLabel: "Decode Tokens/sec", unit: "tps", direction: "higher",
          xKey: "promptLabel", xLabel: "Prompt Depth", chartName: "llamabench_decode",
          lineData: decodeData, lineConfigs: decodeConfigs,
        });
      }
      if (prefillConfigs.length) {
        metrics.push({
          key: "prefill", title: "Prompt Processing Throughput",
          yLabel: "Prefill Tokens/sec", unit: "tps", direction: "higher",
          xKey: "promptLabel", xLabel: "Prompt Size", chartName: "llamabench_prefill",
          lineData: prefillData, lineConfigs: prefillConfigs,
        });
      }
      if (!metrics.length) return null;
      return { tier, metrics };
    }).filter(Boolean);

    const skipEntries = allModels.flatMap(model => {
      const modelData = file.data.llamabench?.[model];
      if (modelData?.error)
        return [{ key: model, label: `${modelLabel(model)}: ${modelData.error}` }];
      if (llamaBenchHasCombinedOnly(modelData))
        return [{ key: model, label: `${modelLabel(model)}: combined-only legacy data; rerun llama-bench` }];
      return [];
    });
    if (!groups.length && !skipEntries.length) return null;
    return { file, groups, skipEntries };
  }).filter(Boolean);

  return (
    <BySystemPanel
      containerRef={containerRef} chartWidth={chartWidth} logoSrc={logoSrc}
      isBar={false} emptyLabel="No llama-bench data in the loaded file(s)"
      systemGroups={systemGroups}
    />
  );
}
