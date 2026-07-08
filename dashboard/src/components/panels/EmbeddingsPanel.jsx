import {
  buildEmbedBarData, buildEmbedBarConfigs,
  buildEmbedData, buildEmbedLineConfigs,
  sortBarData,
} from "../../utils";
import { SECTION_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";

// Group By: Model, Embeddings section — a single combined chart (no
// per-model split, since embeddings has only one "model").
export default function EmbeddingsPanel({ containerRef, files, chartWidth, logoSrc, isBar, isMultiFile }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };

  if (isBar) {
    const rawBarData = buildEmbedBarData(files);
    const barConfigs = buildEmbedBarConfigs(files);
    if (!rawBarData.length || !barConfigs.length) {
      return <EmptyState style={containerStyle}>No {SECTION_LABELS.embeddings} data in the loaded file(s)</EmptyState>;
    }
    const barData = sortBarData(rawBarData, barConfigs.map(bc => bc.dataKey), "desc");
    return (
      <ChartGrid containerRef={containerRef} style={containerStyle}>
        <GroupedBarCard
          title="Embedding Throughput"
          data={barData}
          barConfigs={barConfigs}
          xKey="systemLabel" yLabel="Sentences/sec" unit="sps"
          chartName="embeddings"
          logoSrc={logoSrc} direction="higher"
        />
      </ChartGrid>
    );
  }

  const data = buildEmbedData(files);
  const lineConfigs = buildEmbedLineConfigs(files);

  if (!data.length || !lineConfigs.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS.embeddings} data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      <ChartCard
        title="Embedding Throughput"
        data={data} lineConfigs={lineConfigs}
        xKey="batchLabel" xLabel="Batch Size" yLabel="Sentences/sec" unit="sps"
        isMultiFile={isMultiFile}
        chartName="embeddings"
        logoSrc={logoSrc} direction="higher"
      />
    </ChartGrid>
  );
}
