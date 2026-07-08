import {
  buildLLMDataForModel, buildFileLineConfigs, buildLLMBarConfigs, buildLLMBarData,
  getAllLLMModels, modelLabel, sortBarData,
} from "../../utils";
import { SECTION_LABELS, CTX_ORDER } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Group By: Model, LLM / LLM Conversation section — one card group per model,
// systems as bars/lines within it.
export default function LLMByModelPanel({ containerRef, files, section, enabledModels, chartWidth, logoSrc, isBar, isMultiFile }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m));
  const lineConfigs = buildFileLineConfigs(files);

  const modelGroups = allModels.map(model => {
    const tpsData = buildLLMDataForModel(files, model, "tps", section);
    const ttftData = buildLLMDataForModel(files, model, "ttft", section);
    const tpsLineConfigs = lineConfigs.filter(lc => tpsData.some(r => r[lc.dataKey] != null));
    const ttftLineConfigs = lineConfigs.filter(lc => ttftData.some(r => r[lc.dataKey] != null));
    const rawTpsBarConfigs = buildLLMBarConfigs(files, model, section);
    const rawTtftBarConfigs = buildLLMBarConfigs(files, model, section);
    const rawTpsBarData = buildLLMBarData(files, model, "tps", section);
    const rawTtftBarData = buildLLMBarData(files, model, "ttft", section);
    const byCtxOrder = (a, b) => CTX_ORDER.indexOf(a.dataKey) - CTX_ORDER.indexOf(b.dataKey);
    const tpsBarConfigs = rawTpsBarConfigs.filter(bc => rawTpsBarData.some(r => r[bc.dataKey] != null)).sort(byCtxOrder);
    const ttftBarConfigs = rawTtftBarConfigs.filter(bc => rawTtftBarData.some(r => r[bc.dataKey] != null)).sort(byCtxOrder);
    const tpsBarData = sortBarData(rawTpsBarData, tpsBarConfigs.map(bc => bc.dataKey), "desc");
    const ttftBarData = sortBarData(rawTtftBarData, ttftBarConfigs.map(bc => bc.dataKey), "asc");
    const allTtftVals = ttftData.flatMap(row => lineConfigs.map(lc => row[lc.dataKey])).filter(v => v != null);
    const ttftUnit = allTtftVals.some(v => v >= 60) ? "sec-plain"
      : allTtftVals.length && allTtftVals.every(v => v < 1) ? "ms"
      : "sec";
    const ttftYLabel = ttftUnit === "ms" ? "TTFT (ms)" : "TTFT (sec)";
    const hasTps = isBar ? tpsBarConfigs.length > 0 : tpsLineConfigs.length > 0;
    const hasTtft = isBar ? ttftBarConfigs.length > 0 : ttftLineConfigs.length > 0;
    if (!hasTps && !hasTtft) return null;
    return { model, tpsData, ttftData, tpsLineConfigs, ttftLineConfigs, tpsBarConfigs, ttftBarConfigs, tpsBarData, ttftBarData, ttftUnit, ttftYLabel, hasTps, hasTtft };
  }).filter(Boolean);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS[section]} data in the loaded file(s)</EmptyState>;
  }

  const isConv = section === "llm_conversation";
  const titleSuffix = isConv ? " (Conversation)" : "";
  const chartNamePrefix = isConv ? "conv_" : "";

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(({ model, tpsData, ttftData, tpsLineConfigs, ttftLineConfigs, tpsBarConfigs, ttftBarConfigs, tpsBarData, ttftBarData, ttftUnit, ttftYLabel, hasTps, hasTtft }) => (
        <div key={model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
          {hasTps && (isBar ? (
            <GroupedBarCard
              title={`Tokens/sec${titleSuffix}`}
              modelName={modelLabel(model)}
              data={tpsBarData}
              barConfigs={tpsBarConfigs}
              xKey="systemLabel" yLabel="Tokens/sec" unit="tps"
              chartName={`${chartNamePrefix}tps`} chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
          ) : (
            <ChartCard
              title={`Tokens/sec${titleSuffix}`}
              modelName={modelLabel(model)}
              data={tpsData} lineConfigs={tpsLineConfigs}
              xKey="ctxLabel" xLabel="Context Length" yLabel="Tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName={`${chartNamePrefix}tps`} chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
          ))}
          {hasTtft && (isBar ? (
            <GroupedBarCard
              title={`Time to First Token${titleSuffix}`}
              modelName={modelLabel(model)}
              data={ttftBarData}
              barConfigs={ttftBarConfigs}
              xKey="systemLabel" yLabel={ttftYLabel} unit={ttftUnit}
              chartName={`${chartNamePrefix}ttft`} chartModel={model}
              logoSrc={logoSrc} direction="lower"
            />
          ) : (
            <ChartCard
              title={`Time to First Token${titleSuffix}`}
              modelName={modelLabel(model)}
              data={ttftData} lineConfigs={ttftLineConfigs}
              xKey="ctxLabel" xLabel="Context Length" yLabel={ttftYLabel} unit={ttftUnit}
              isMultiFile={isMultiFile}
              chartName={`${chartNamePrefix}ttft`} chartModel={model}
              logoSrc={logoSrc} direction="lower"
            />
          ))}
        </div>
      ))}
    </ChartGrid>
  );
}
