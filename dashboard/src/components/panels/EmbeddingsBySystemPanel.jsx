import {
  buildEmbedBarDataByModel, buildEmbedBarConfigsByModel,
  buildEmbedLineDataByBatch, buildEmbedLineConfigsByBatch,
  getAllEmbedModels, sortBarData, findMostStrenuousKey,
} from "../../utils";
import { SECTION_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Group By: System, Embeddings section — one card per system, models as
// bars/lines within it.
export default function EmbeddingsBySystemPanel({ containerRef, files, enabledEmbedModels, chartWidth, logoSrc, isBar }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllEmbedModels(files).filter(m => enabledEmbedModels.has(m));

  const systemGroups = files.map(f => {
    const rawBarData = buildEmbedBarDataByModel(f, allModels);
    const barConfigs = buildEmbedBarConfigsByModel(f, allModels);
    const lineData = buildEmbedLineDataByBatch(f, allModels);
    const lineConfigs = buildEmbedLineConfigsByBatch(f, allModels, lineData);
    const hasBar = rawBarData.length > 0 && barConfigs.length > 0;
    const hasLine = lineConfigs.length > 0;
    if (isBar ? !hasBar : !hasLine) return null;
    const strenuousKey = findMostStrenuousKey(rawBarData, barConfigs.map(bc => bc.dataKey));
    const barData = strenuousKey ? sortBarData(rawBarData, [strenuousKey], "desc") : rawBarData;
    return { file: f, barData, barConfigs, lineData, lineConfigs };
  }).filter(Boolean);

  if (!systemGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS.embeddings} data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {systemGroups.map(({ file: f, barData, barConfigs, lineData, lineConfigs }) => (
        <div key={f.id} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{f.hostname}</div>
          {isBar ? (
            <GroupedBarCard
              title="Embedding Throughput"
              modelName={f.hostname}
              data={barData}
              barConfigs={barConfigs}
              xKey="modelLabel" yLabel="Sentences/sec" unit="sps"
              chartName="embeddings_by_system" chartModel={f.hostname}
              logoSrc={logoSrc} direction="higher"
            />
          ) : (
            <ChartCard
              title="Embedding Throughput"
              modelName={f.hostname}
              data={lineData} lineConfigs={lineConfigs}
              xKey="batchLabel" xLabel="Batch Size" yLabel="Sentences/sec" unit="sps"
              isMultiFile={false}
              chartName="embeddings_by_system" chartModel={f.hostname}
              logoSrc={logoSrc} direction="higher"
            />
          )}
        </div>
      ))}
    </ChartGrid>
  );
}
