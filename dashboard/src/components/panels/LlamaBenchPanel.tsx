import type { RefObject } from "react";
import { getAllLLMModels } from "../../utils/llm";
import {
  buildLlamaBenchDecodeLineConfigs,
  buildLlamaBenchDecodeLineData,
  buildLlamaBenchPrefillLineConfigs,
  buildLlamaBenchPrefillLineData,
  llamaBenchHasCombinedOnly,
} from "../../utils/llamabench";
import { modelLabel, isNotNull } from "../../utils/shared";
import { ChartCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import type { ResultsFile } from "../../types";
import styles from "../ChartPanel.module.css";

export default function LlamaBenchPanel({ containerRef, files, enabledModels, chartWidth, logoSrc, isMultiFile }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledModels: Set<string>,
  chartWidth: number, logoSrc?: string | null, isMultiFile: boolean,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllLLMModels(files)
    .filter(model => enabledModels.has(model) && files.some(file => file.data.llamabench?.[model]));

  const modelGroups = allModels.map(model => {
    const prefillData = buildLlamaBenchPrefillLineData(files, model);
    const prefillConfigs = buildLlamaBenchPrefillLineConfigs(files, prefillData);
    const decodeData = buildLlamaBenchDecodeLineData(files, model);
    const decodeConfigs = buildLlamaBenchDecodeLineConfigs(files, model, decodeData);
    const notes = files.flatMap(file => {
      const modelData = file.data.llamabench?.[model];
      if (modelData?.error) return [`${file.hostname}: ${modelData.error}`];
      if (llamaBenchHasCombinedOnly(modelData))
        return [`${file.hostname}: combined-only legacy data; rerun llama-bench for separate prefill/decode charts`];
      return [];
    });
    if (!prefillConfigs.length && !decodeConfigs.length && !notes.length) return null;
    return { model, prefillData, prefillConfigs, decodeData, decodeConfigs, notes };
  }).filter(isNotNull);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No llama-bench data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(({ model, prefillData, prefillConfigs, decodeData, decodeConfigs, notes }) => (
        <div key={model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
          {notes.length > 0 && (
            <div className={styles.skipNote}>
              {notes.map(note => <div key={note}>{note}</div>)}
            </div>
          )}
          {decodeConfigs.length > 0 && (
            <ChartCard
              title="Decode Throughput by Prompt Depth"
              modelName={modelLabel(model)}
              data={decodeData} lineConfigs={decodeConfigs}
              xKey="promptLabel" xLabel="Prompt Depth" yLabel="Decode Tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName="llamabench_decode" chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
          )}
          {prefillConfigs.length > 0 && (
            <ChartCard
              title="Prompt Processing Throughput"
              modelName={modelLabel(model)}
              data={prefillData} lineConfigs={prefillConfigs}
              xKey="promptLabel" xLabel="Prompt Size" yLabel="Prefill Tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName="llamabench_prefill" chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
          )}
        </div>
      ))}
    </ChartGrid>
  );
}
