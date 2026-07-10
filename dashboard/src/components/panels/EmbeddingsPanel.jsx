import {
  getAllEmbedModels,
  buildEmbedGroupedBarDataForBatch, buildEmbedGroupedBarConfigs,
  buildEmbedData, buildEmbedLineConfigs,
  sortBarData, findMostStrenuousKey,
} from "../../utils";
import { EMBED_BATCH_KEYS, EMBED_BATCH_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";

// Group By: Model, Embeddings section — one card per batch size (bar) or a
// single combined chart (line), systems/models as bars/lines within it.
export default function EmbeddingsPanel({ containerRef, files, enabledEmbedModels, chartWidth, logoSrc, isBar, isMultiFile }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllEmbedModels(files).filter(m => enabledEmbedModels.has(m));

  const batchSet = new Set();
  for (const f of files)
    for (const model of allModels)
      for (const bk of Object.keys(f.data.embeddings?.[model] || {})) if (EMBED_BATCH_KEYS.includes(bk)) batchSet.add(bk);
  const batchKeys = EMBED_BATCH_KEYS.filter(bk => batchSet.has(bk));

  if (!batchKeys.length || !allModels.length) {
    return <EmptyState style={containerStyle}>No Embeddings data in the loaded file(s)</EmptyState>;
  }

  const groupedBarConfigs = buildEmbedGroupedBarConfigs(files, enabledEmbedModels);
  const modelKeys = groupedBarConfigs.map(bc => bc.dataKey);
  const lineData = buildEmbedData(files, enabledEmbedModels);
  const lineConfigs = buildEmbedLineConfigs(files, lineData, enabledEmbedModels);

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {isBar ? batchKeys.map(bk => {
        const raw = buildEmbedGroupedBarDataForBatch(files, bk, enabledEmbedModels);
        if (!raw.length) return null;
        const strenuousKey = findMostStrenuousKey(raw, modelKeys);
        const data = strenuousKey ? sortBarData(raw, [strenuousKey], "desc") : raw;
        return (
          <GroupedBarCard
            key={bk}
            title={`Batch ${EMBED_BATCH_LABELS[bk]}`}
            modelName="Embeddings"
            data={data}
            barConfigs={groupedBarConfigs}
            xKey="systemLabel" yLabel="Sentences/sec" unit="sps"
            chartName="embeddings" chartModel={bk}
            logoSrc={logoSrc} direction="higher"
          />
        );
      }) : (
        <ChartCard
          title="Embedding Throughput"
          data={lineData}
          lineConfigs={lineConfigs}
          xKey="batchLabel" xLabel="Batch Size" yLabel="Sentences/sec" unit="sps"
          isMultiFile={isMultiFile}
          chartName="embeddings"
          logoSrc={logoSrc} direction="higher"
        />
      )}
    </ChartGrid>
  );
}
