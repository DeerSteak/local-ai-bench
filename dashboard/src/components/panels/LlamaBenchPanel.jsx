import {
  getAllLLMModels, buildLlamaBenchBarData, buildLlamaBenchBarConfigs,
  modelLabel, sortBarData,
} from "../../utils";
import { GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// One card per model — systems as bars, one bar per (pp, tg) checkpoint actually
// present. No Chart Style/Group By toggle: llama-bench is always this shape.
export default function LlamaBenchPanel({ containerRef, files, enabledModels, chartWidth, logoSrc }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m) && files.some(f => f.data.llamabench?.[m]));

  const modelGroups = allModels.map(model => {
    const barConfigs = buildLlamaBenchBarConfigs(files, model);
    const rawBarData = buildLlamaBenchBarData(files, model);
    const barData = sortBarData(rawBarData, barConfigs.map(bc => bc.dataKey), "desc");
    const errorEntries = files
      .map(f => ({ hostname: f.hostname, error: f.data.llamabench?.[model]?.error }))
      .filter(e => e.error);
    if (!barConfigs.length && !errorEntries.length) return null;
    return { model, barConfigs, barData, errorEntries };
  }).filter(Boolean);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No llama-bench data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(({ model, barConfigs, barData, errorEntries }) => (
        <div key={model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
          {errorEntries.length > 0 && (
            <div className={styles.skipNote}>
              {errorEntries.map(e => <div key={e.hostname}>{e.hostname}: {e.error}</div>)}
            </div>
          )}
          {barConfigs.length > 0 && (
            <GroupedBarCard
              title="llama-bench Throughput"
              modelName={modelLabel(model)}
              data={barData}
              barConfigs={barConfigs}
              xKey="systemLabel" yLabel="Tokens/sec" unit="tps"
              chartName="llamabench" chartModel={model}
              logoSrc={logoSrc} direction="higher" orderedSeries
            />
          )}
        </div>
      ))}
    </ChartGrid>
  );
}
