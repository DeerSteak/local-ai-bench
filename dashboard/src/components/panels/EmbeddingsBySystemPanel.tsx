import type { RefObject } from "react";
import { buildEmbedBarDataByModel, buildEmbedBarConfigsByModel, getAllEmbedModels } from "../../utils/embeddings";
import { sortBarData, findMostStrenuousKey, isNotNull } from "../../utils/shared";
import { SECTION_LABELS } from "../../constants";
import { GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import type { ResultsFile, ChartRow } from "../../types";
import styles from "../ChartPanel.module.css";

// Group By: System, Embeddings section — one card per system, a single
// document-ingestion throughput bar per model. No line mode: there's no
// batch-size axis left to plot a line across.
export default function EmbeddingsBySystemPanel({ containerRef, files, enabledEmbedModels, chartWidth, logoSrc }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledEmbedModels: Set<string>,
  chartWidth: number, logoSrc?: string | null,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllEmbedModels(files).filter(m => enabledEmbedModels.has(m));

  const systemGroups = files.map(f => {
    const rawBarData = buildEmbedBarDataByModel(f, allModels);
    const barConfigs = buildEmbedBarConfigsByModel(f, allModels);
    if (!rawBarData.length || !barConfigs.length) return null;
    const strenuousKey = findMostStrenuousKey(rawBarData, barConfigs.map(bc => bc.dataKey));
    const barData = strenuousKey ? sortBarData(rawBarData, [strenuousKey], "desc") : rawBarData;
    return { file: f, barData, barConfigs };
  }).filter(isNotNull);

  if (!systemGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS.embeddings} data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {systemGroups.map(({ file: f, barData, barConfigs }: {
        file: ResultsFile, barData: ChartRow[], barConfigs: { dataKey: string, name: string, fill: string }[],
      }) => (
        <div key={f.id} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{f.hostname}</div>
          <GroupedBarCard
            title="Document Embedding Throughput"
            modelName={f.hostname}
            data={barData}
            barConfigs={barConfigs}
            xKey="modelLabel" yLabel="Chunks/sec" unit="sps"
            chartName="embeddings_by_system" chartModel={f.hostname}
            logoSrc={logoSrc} direction="higher"
          />
        </div>
      ))}
    </ChartGrid>
  );
}
