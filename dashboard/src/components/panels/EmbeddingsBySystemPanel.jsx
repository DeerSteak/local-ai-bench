import {
  buildEmbedBarDataByFile, buildEmbedBarConfigs,
  buildEmbedLineDataByBatch, buildEmbedLineConfigByBatch,
} from "../../utils";
import { SECTION_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Group By: System, Embeddings section — one card per system.
export default function EmbeddingsBySystemPanel({ containerRef, files, chartWidth, logoSrc, isBar }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };

  const systemGroups = files.map(f => {
    const barData = buildEmbedBarDataByFile(f);
    const barConfigs = buildEmbedBarConfigs([f]);
    const lineData = buildEmbedLineDataByBatch(f);
    const lineConfigs = buildEmbedLineConfigByBatch();
    if (isBar ? !barConfigs.length : !lineData.length) return null;
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
