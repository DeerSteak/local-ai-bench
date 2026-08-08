import type { RefObject } from "react";
import { getAllLLMModels } from "../../utils/llm";
import {
  buildVllmBenchLatencyConfigs,
  buildVllmBenchLatencyData,
  buildVllmBenchThroughputConfigs,
  buildVllmBenchThroughputData,
} from "../../utils/vllmbench";
import { modelLabel } from "../../utils/shared";
import { ChartCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import type { ResultsFile } from "../../types";
import styles from "../ChartPanel.module.css";

export default function VllmBenchPanel({ containerRef, files, enabledModels, chartWidth, logoSrc, isMultiFile }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledModels: Set<string>,
  chartWidth: number, logoSrc?: string, isMultiFile: boolean,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllLLMModels(files)
    .filter(model => enabledModels.has(model) && files.some(file => file.data.vllmbench?.[model]));

  const modelGroups = allModels.map(model => {
    const latencyData = buildVllmBenchLatencyData(files, model);
    const latencyConfigs = buildVllmBenchLatencyConfigs(files, model, latencyData);
    const throughputData = buildVllmBenchThroughputData(files, model);
    const throughputConfigs = buildVllmBenchThroughputConfigs(files, model, throughputData);
    const notes = files.flatMap(file => {
      const modelData = file.data.vllmbench?.[model];
      if (modelData?.error) return [`${file.hostname}: ${modelData.error}`];
      if (modelData?.timed_out)
        return [`${file.hostname}: timed out at ${modelData.timed_out_at ?? "an unrecorded size"}`];
      return [];
    });
    if (!latencyConfigs.length && !throughputConfigs.length && !notes.length) return null;
    return { model, latencyData, latencyConfigs, throughputData, throughputConfigs, notes };
  }).filter(Boolean);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No vllm bench data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(({ model, latencyData, latencyConfigs, throughputData, throughputConfigs, notes }) => (
        <div key={model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
          {notes.length > 0 && (
            <div className={styles.skipNote}>
              {notes.map(note => <div key={note}>{note}</div>)}
            </div>
          )}
          {latencyConfigs.length > 0 && (
            <ChartCard
              title="Batch Latency (vllm bench)"
              modelName={modelLabel(model)}
              data={latencyData} lineConfigs={latencyConfigs}
              xKey="promptLabel" xLabel="Input Length" yLabel="Seconds per batch" unit="s"
              isMultiFile={isMultiFile}
              chartName="vllmbench_latency" chartModel={model}
              logoSrc={logoSrc} direction="lower"
            />
          )}
          {throughputConfigs.length > 0 && (
            <ChartCard
              title="Output Throughput (vllm bench)"
              modelName={modelLabel(model)}
              data={throughputData} lineConfigs={throughputConfigs}
              xKey="promptLabel" xLabel="Input Length" yLabel="Output tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName="vllmbench_throughput" chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
          )}
        </div>
      ))}
    </ChartGrid>
  );
}
