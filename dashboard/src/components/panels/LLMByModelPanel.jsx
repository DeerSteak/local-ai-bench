import {
  buildLLMDataForModel, buildFileLineConfigs, buildLLMBarConfigs, buildLLMBarData,
  getAllLLMModels, modelLabel, sortBarData, getSkipInfo,
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
  const isConv = section === "llm_conversation";

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
    const hasValueOrStatus = (rows, key) => rows.some(r => r[key] != null || r[`_status_${key}`] != null);
    const tpsBarConfigs = rawTpsBarConfigs.filter(bc => hasValueOrStatus(rawTpsBarData, bc.dataKey)).sort(byCtxOrder);
    const ttftBarConfigs = rawTtftBarConfigs.filter(bc => hasValueOrStatus(rawTtftBarData, bc.dataKey)).sort(byCtxOrder);
    const tpsBarData = sortBarData(rawTpsBarData, tpsBarConfigs.map(bc => bc.dataKey), "desc");
    const ttftBarData = sortBarData(rawTtftBarData, ttftBarConfigs.map(bc => bc.dataKey), "asc");
    const allTtftVals = ttftData.flatMap(row => lineConfigs.map(lc => row[lc.dataKey])).filter(v => v != null);
    const ttftUnit = allTtftVals.some(v => v >= 60) ? "sec-plain"
      : allTtftVals.length && allTtftVals.every(v => v < 1) ? "ms"
      : "sec";
    const ttftYLabel = ttftUnit === "ms" ? "TTFT (ms)" : "TTFT (sec)";
    const hasTps = isBar ? tpsBarConfigs.length > 0 : tpsLineConfigs.length > 0;
    const hasTtft = isBar ? ttftBarConfigs.length > 0 : ttftLineConfigs.length > 0;
    const skipEntries = files
      .map(f => ({ hostname: f.hostname, info: getSkipInfo(f, model, section) }))
      .filter(e => e.info);
    if (!hasTps && !hasTtft && !skipEntries.length) return null;
    return { model, tpsData, ttftData, tpsLineConfigs, ttftLineConfigs, tpsBarConfigs, ttftBarConfigs, tpsBarData, ttftBarData, ttftUnit, ttftYLabel, hasTps, hasTtft, skipEntries };
  }).filter(Boolean);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS[section]} data in the loaded file(s)</EmptyState>;
  }

  const titleSuffix = isConv ? " (Conversation)" : "";
  const chartNamePrefix = isConv ? "conv_" : "";

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(({ model, tpsData, ttftData, tpsLineConfigs, ttftLineConfigs, tpsBarConfigs, ttftBarConfigs, tpsBarData, ttftBarData, ttftUnit, ttftYLabel, hasTps, hasTtft, skipEntries }) => (
        <div key={model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
          {skipEntries.length > 0 && (
            <div className={styles.skipNote}>
              {skipEntries.map(e => (
                <div key={e.hostname}>
                  {isMultiFile ? `${e.hostname}: ` : ""}Skipped — {e.info.detail}
                </div>
              ))}
            </div>
          )}
          {hasTps && (isBar ? (
            <GroupedBarCard
              title={`Tokens/sec${titleSuffix}`}
              modelName={modelLabel(model)}
              data={tpsBarData}
              barConfigs={tpsBarConfigs}
              xKey="systemLabel" yLabel="Tokens/sec" unit="tps"
              chartName={`${chartNamePrefix}tps`} chartModel={model}
              logoSrc={logoSrc} direction="higher" orderedSeries
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
              logoSrc={logoSrc} direction="lower" orderedSeries
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
