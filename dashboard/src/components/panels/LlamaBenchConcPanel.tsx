import type { RefObject } from "react";
import { getAllLLMModels } from "../../utils/llm";
import {
  llamaBenchConcTgValues, buildLlamaBenchConcLineData, llamaBenchConcPromptDepth,
} from "../../utils/llamabenchconc";
import { buildFileLineConfigs, modelLabel, isNotNull } from "../../utils/shared";
import { ChartCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import type { ResultsFile } from "../../types";
import styles from "../ChartPanel.module.css";

// llama-batched-bench: one card per model, one chart per tg, X = concurrency level.
// Line-only and group-by-agnostic, same reasoning as ConcurrencyPanel.
export default function LlamaBenchConcPanel({ containerRef, files, enabledModels, chartWidth, logoSrc, isMultiFile }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledModels: Set<string>,
  chartWidth: number, logoSrc?: string | null, isMultiFile: boolean,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllLLMModels(files)
    .filter(m => enabledModels.has(m) && files.some(f => f.data.llamabenchconc?.[m]));
  const lineConfigs = buildFileLineConfigs(files);

  const modelGroups = allModels.map(model => {
    const charts = llamaBenchConcTgValues(files, model).map(tg => {
      const data = buildLlamaBenchConcLineData(files, model, tg);
      const tgLineConfigs = lineConfigs.filter(lc => data.some(r => r[lc.dataKey] != null));
      if (!tgLineConfigs.length) return null;
      return { tg, data, lineConfigs: tgLineConfigs };
    }).filter(isNotNull);

    const depths = [...new Set(
      files.map(f => llamaBenchConcPromptDepth(f, model)).filter(d => d != null),
    )];
    const errorEntries = files
      .map(f => ({ hostname: f.hostname, error: f.data.llamabenchconc?.[model]?.error }))
      .filter(e => e.error);
    if (!charts.length && !errorEntries.length) return null;
    return { model, charts, errorEntries, depths };
  }).filter(isNotNull);

  if (!modelGroups.length) {
    return <EmptyState style={containerStyle}>No llama-batched-bench data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {modelGroups.map(({ model, charts, errorEntries, depths }) => (
        <div key={model} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
          {errorEntries.length > 0 && (
            <div className={styles.skipNote}>
              {errorEntries.map(e => <div key={e.hostname}>{e.hostname}: {e.error}</div>)}
            </div>
          )}
          {charts.map(({ tg, data, lineConfigs: tgLineConfigs }) => (
            <ChartCard
              key={tg}
              title={`Aggregate Tokens/sec — tg${tg}${depths.length === 1 ? `, pp${depths[0]}` : ""}`}
              modelName={modelLabel(model)}
              data={data} lineConfigs={tgLineConfigs}
              xKey="levelLabel" xLabel="Concurrency Level" yLabel="Tokens/sec" unit="tps"
              isMultiFile={isMultiFile}
              chartName={`llamabenchconc_tg${tg}`} chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
          ))}
        </div>
      ))}
    </ChartGrid>
  );
}
