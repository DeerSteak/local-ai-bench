import type { RefObject } from "react";
import type { ResultsFile } from "../../types";
import {
  buildLLMCacheComparisonConfigs, buildLLMCacheComparisonData, getAllLLMModels,
} from "../../utils/llm";
import { configsWithValues, modelLabel } from "../../utils/shared";
import { ChartCard } from "../charts/ChartCards";
import { ChartGrid, EmptyState } from "./shared";
import styles from "../ChartPanel.module.css";

export default function LLMCacheComparisonPanel({
  containerRef, files, enabledModels, chartWidth, logoSrc,
}: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[],
  enabledModels: Set<string>, chartWidth: number, logoSrc?: string | null,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const configs = buildLLMCacheComparisonConfigs(files);
  const models = getAllLLMModels(files).filter(model => enabledModels.has(model)).map(model => ({
    model,
    prefill: buildLLMCacheComparisonData(files, model, "prefill"),
    generation: buildLLMCacheComparisonData(files, model, "tps"),
  })).filter(group => group.prefill.length || group.generation.length);

  if (!models.length) {
    return <EmptyState style={containerStyle}>No paired cached and uncached LLM data in the loaded file(s)</EmptyState>;
  }

  return <ChartGrid containerRef={containerRef} style={containerStyle}>
    {models.map(group => <div key={group.model} className={styles.modelGroup}>
      <div className={styles.modelGroupTitle}>{modelLabel(group.model)}</div>
      {group.prefill.length > 0 && <ChartCard
        title="Prompt Processing Speed — Cached vs Uncached" modelName={modelLabel(group.model)}
        data={group.prefill} lineConfigs={configsWithValues(configs, group.prefill)} xKey="ctxLabel" xLabel="Context Length"
        yLabel="Prefill tokens/sec" unit="tps" isMultiFile={files.length > 1}
        chartName="cache_comparison_prefill" chartModel={group.model} logoSrc={logoSrc}
        direction="higher"
      />}
      {group.generation.length > 0 && <ChartCard
        title="Generation Speed — Cached vs Uncached" modelName={modelLabel(group.model)}
        data={group.generation} lineConfigs={configsWithValues(configs, group.generation)} xKey="ctxLabel" xLabel="Context Length"
        yLabel="Tokens/sec" unit="tps" isMultiFile={files.length > 1}
        chartName="cache_comparison_generation" chartModel={group.model} logoSrc={logoSrc}
        direction="higher"
      />}
    </div>)}
  </ChartGrid>;
}
