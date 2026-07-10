import {
  getAllEmbedModels,
  buildEmbedGroupedBarData, buildEmbedGroupedBarConfigs,
  sortBarData, findMostStrenuousKey,
} from "../../utils";
import { GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";

// Group By: Model, Embeddings section — a single chart: one document-ingestion
// throughput value per model, systems as bars. There's no batch-size sweep
// (and so no line-mode axis) anymore — the test embeds one real document's
// chunks in a single call, not an arbitrary batch dial.
export default function EmbeddingsPanel({ containerRef, files, enabledEmbedModels, chartWidth, logoSrc }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllEmbedModels(files).filter(m => enabledEmbedModels.has(m));

  const groupedBarConfigs = buildEmbedGroupedBarConfigs(files, enabledEmbedModels);
  const rawData = buildEmbedGroupedBarData(files, enabledEmbedModels);

  if (!allModels.length || !rawData.length) {
    return <EmptyState style={containerStyle}>No Embeddings data in the loaded file(s)</EmptyState>;
  }

  const modelKeys = groupedBarConfigs.map(bc => bc.dataKey);
  const strenuousKey = findMostStrenuousKey(rawData, modelKeys);
  const data = strenuousKey ? sortBarData(rawData, [strenuousKey], "desc") : rawData;

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      <GroupedBarCard
        title="Document Embedding Throughput"
        modelName="Embeddings"
        data={data}
        barConfigs={groupedBarConfigs}
        xKey="systemLabel" yLabel="Chunks/sec" unit="sps"
        chartName="embeddings"
        logoSrc={logoSrc} direction="higher"
      />
    </ChartGrid>
  );
}
