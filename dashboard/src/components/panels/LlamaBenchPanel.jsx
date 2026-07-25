import {
  getAllLLMModels, llamaBenchTgValues, buildLlamaBenchBarData, buildLlamaBenchBarConfigs, buildLlamaBenchLineData,
  buildFileLineConfigs, modelLabel, sortBarData,
} from "../../utils";
import { GroupedBarCard, ChartCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Group By: Model, llama-bench section — one card per model, one chart per generation
// size (tg) actually swept at that model, so pp size can be the chart's only axis/bar
// dimension instead of a combined "pp2048+tg128" label. Bar: systems as bars, one bar
// per pp size. Line: pp size on the X axis, one line per system. See
// LlamaBenchBySystemPanel for Group By: System.
export default function LlamaBenchPanel({ containerRef, files, enabledModels, chartWidth, logoSrc, isBar, isMultiFile }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m) && files.some(f => f.data.llamabench?.[m]));
  const lineConfigs = buildFileLineConfigs(files);

  const modelGroups = allModels.map(model => {
    const tgValues = llamaBenchTgValues(files, model);
    const charts = tgValues.map(tg => {
      const barConfigs = buildLlamaBenchBarConfigs(files, model, tg);
      const rawBarData = buildLlamaBenchBarData(files, model, tg);
      const barData = sortBarData(rawBarData, barConfigs.map(bc => bc.dataKey), "desc");
      const lineData = buildLlamaBenchLineData(files, model, tg);
      const tgLineConfigs = lineConfigs.filter(lc => lineData.some(r => r[lc.dataKey] != null));
      const hasData = isBar ? barConfigs.length > 0 : tgLineConfigs.length > 0;
      if (!hasData) return null;
      return { tg, barConfigs, barData, lineData, lineConfigs: tgLineConfigs };
    }).filter(Boolean);

    const errorEntries = files
      .map(f => ({ hostname: f.hostname, error: f.data.llamabench?.[model]?.error }))
      .filter(e => e.error);
    if (!charts.length && !errorEntries.length) return null;
    return { model, charts, errorEntries };
  }).filter(Boolean);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No llama-bench data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(({ model, charts, errorEntries }) => (
        <div key={model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
          {errorEntries.length > 0 && (
            <div className={styles.skipNote}>
              {errorEntries.map(e => <div key={e.hostname}>{e.hostname}: {e.error}</div>)}
            </div>
          )}
          {charts.map(({ tg, barConfigs, barData, lineData, lineConfigs: tgLineConfigs }) => isBar ? (
            <GroupedBarCard
              key={tg}
              title={`llama-bench Throughput — tg${tg}`}
              modelName={modelLabel(model)}
              data={barData}
              barConfigs={barConfigs}
              xKey="systemLabel" yLabel="Tokens/sec" unit="tps"
              chartName={`llamabench_tg${tg}`} chartModel={model}
              logoSrc={logoSrc} direction="higher" orderedSeries
            />
          ) : (
            <ChartCard
              key={tg}
              title={`llama-bench Throughput — tg${tg}`}
              modelName={modelLabel(model)}
              data={lineData} lineConfigs={tgLineConfigs}
              xKey="promptLabel" xLabel="Prompt Size" yLabel="Tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName={`llamabench_tg${tg}`} chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
          ))}
        </div>
      ))}
    </ChartGrid>
  );
}
