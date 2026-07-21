import {
  buildConcurrencyDataForModel, buildFileLineConfigs,
  getAllConcurrencyModels, modelLabel, getSkipInfo, getConcurrencyStopInfo,
} from "../../utils";
import { SECTION_LABELS } from "../../constants";
import { ChartCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Concurrency section — one card group per model, one line per file/system.
// Always rendered as line charts (concurrency levels double each step, same
// reasoning as skipping a bar-chart mode here), and ignores the Group By
// toggle (there's no per-model tier split like LLM's small/medium/large —
// same precedent as AccuracyPanel). `section` is "concurrency_tool" or
// "concurrency_chat" — same layout, different results key and level ladder.
export default function ConcurrencyPanel({ containerRef, files, section, enabledModels, chartWidth, logoSrc, isMultiFile }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllConcurrencyModels(files, section).filter(m => enabledModels.has(m));
  const lineConfigs = buildFileLineConfigs(files);
  const chartPrefix = section === "concurrency_tool" ? "conc_tool" : "conc_chat";

  const modelGroups = allModels.map(model => {
    const tpsData = buildConcurrencyDataForModel(files, section, model, "tps");
    const aggData = buildConcurrencyDataForModel(files, section, model, "aggregate");
    const ttftData = buildConcurrencyDataForModel(files, section, model, "ttft");
    const tpsLineConfigs = lineConfigs.filter(lc => tpsData.some(r => r[lc.dataKey] != null));
    const aggLineConfigs = lineConfigs.filter(lc => aggData.some(r => r[lc.dataKey] != null));
    const ttftLineConfigs = lineConfigs.filter(lc => ttftData.some(r => r[lc.dataKey] != null));

    const allTtftVals = ttftData.flatMap(row => lineConfigs.map(lc => row[lc.dataKey])).filter(v => v != null);
    const ttftUnit = allTtftVals.some(v => v >= 60) ? "sec-plain"
      : allTtftVals.length && allTtftVals.every(v => v < 1) ? "ms"
      : "sec";
    const ttftYLabel = ttftUnit === "ms" ? "TTFT (ms)" : "TTFT (sec)";

    const skipEntries = files
      .map(f => ({ hostname: f.hostname, info: getSkipInfo(f, model, section) }))
      .filter(e => e.info);
    const stopEntries = files
      .map(f => ({ hostname: f.hostname, info: getConcurrencyStopInfo(f, section, model) }))
      .filter(e => e.info);

    const hasAny = tpsLineConfigs.length > 0 || aggLineConfigs.length > 0 || ttftLineConfigs.length > 0;
    if (!hasAny && !skipEntries.length && !stopEntries.length) return null;
    return {
      model, tpsData, aggData, ttftData,
      tpsLineConfigs, aggLineConfigs, ttftLineConfigs,
      ttftUnit, ttftYLabel, skipEntries, stopEntries,
    };
  }).filter(Boolean);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS[section]} data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(g => (
        <div key={g.model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(g.model)}</div>
          {g.skipEntries.length > 0 && (
            <div className={styles.skipNote}>
              {g.skipEntries.map(e => (
                <div key={e.hostname}>
                  {isMultiFile ? `${e.hostname}: ` : ""}Skipped — {e.info.detail}
                </div>
              ))}
            </div>
          )}
          {g.stopEntries.length > 0 && (
            <div className={styles.skipNote}>
              {g.stopEntries.map(e => (
                <div key={e.hostname}>
                  {isMultiFile ? `${e.hostname}: ` : ""}
                  {e.info.nextLevel
                    ? `Stopped before ${e.info.nextLevel}-way — ${e.info.label}`
                    : `Stopped after ${e.info.lastLevel}-way — ${e.info.label}`}
                </div>
              ))}
            </div>
          )}
          {g.tpsLineConfigs.length > 0 && (
            <ChartCard
              title="Per-Request Tokens/sec"
              modelName={modelLabel(g.model)}
              data={g.tpsData} lineConfigs={g.tpsLineConfigs}
              xKey="levelLabel" xLabel="Concurrency Level" yLabel="Tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName={`${chartPrefix}_tps`} chartModel={g.model}
              logoSrc={logoSrc} direction="higher"
            />
          )}
          {g.aggLineConfigs.length > 0 && (
            <ChartCard
              title="Aggregate Tokens/sec"
              modelName={modelLabel(g.model)}
              data={g.aggData} lineConfigs={g.aggLineConfigs}
              xKey="levelLabel" xLabel="Concurrency Level" yLabel="Tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName={`${chartPrefix}_aggregate`} chartModel={g.model}
              logoSrc={logoSrc} direction="higher"
            />
          )}
          {g.ttftLineConfigs.length > 0 && (
            <ChartCard
              title="Time to First Token"
              modelName={modelLabel(g.model)}
              data={g.ttftData} lineConfigs={g.ttftLineConfigs}
              xKey="levelLabel" xLabel="Concurrency Level" yLabel={g.ttftYLabel} unit={g.ttftUnit}
              isMultiFile={isMultiFile}
              chartName={`${chartPrefix}_ttft`} chartModel={g.model}
              logoSrc={logoSrc} direction="lower"
            />
          )}
        </div>
      ))}
    </ChartGrid>
  );
}
