import { Fragment } from "react";
import {
  buildLLMBarDataByModel, buildLLMBarConfigsByModel,
  buildLLMLineDataByCtx, buildLLMLineConfigsByCtx,
  getAllLLMModels, sortBarData, getModelSizeTier,
} from "../../utils";
import { SECTION_LABELS, SIZE_TIER_ORDER, SIZE_TIER_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Group By: System, LLM / LLM Conversation section — one card group per
// system, split into small/medium/large model tiers (or combined) per the
// "Model Sizes" toggle.
export default function LLMBySystemPanel({ containerRef, files, section, enabledModels, chartWidth, logoSrc, isBar, isSplit }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m));
  const isConv = section === "llm_conversation";
  const titleSuffix = isConv ? " (Conversation)" : "";
  const chartNamePrefix = isConv ? "conv_" : "";

  // Split into small/medium/large tiers, or a single combined group.
  const modelGroupSpecs = isSplit
    ? SIZE_TIER_ORDER
        .map(tier => ({ tier, models: allModels.filter(m => getModelSizeTier(m) === tier) }))
        .filter(g => g.models.length > 0)
    : [{ tier: null, models: allModels }];

  const systemGroups = files.map(f => {
    const groups = modelGroupSpecs.map(({ tier, models }) => {
      const rawTpsBarData = buildLLMBarDataByModel(f, models, "tps", section);
      const rawTtftBarData = buildLLMBarDataByModel(f, models, "ttft", section);
      const rawTpsBarConfigs = buildLLMBarConfigsByModel(f, models, section);
      const rawTtftBarConfigs = buildLLMBarConfigsByModel(f, models, section);
      const tpsBarConfigs = rawTpsBarConfigs.filter(bc => rawTpsBarData.some(r => r[bc.dataKey] != null));
      const ttftBarConfigs = rawTtftBarConfigs.filter(bc => rawTtftBarData.some(r => r[bc.dataKey] != null));
      const tpsBarData = sortBarData(rawTpsBarData, tpsBarConfigs.map(bc => bc.dataKey), "desc");
      const ttftBarData = sortBarData(rawTtftBarData, ttftBarConfigs.map(bc => bc.dataKey), "asc");

      const tpsLineData = buildLLMLineDataByCtx(f, models, "tps", section);
      const ttftLineData = buildLLMLineDataByCtx(f, models, "ttft", section);
      const tpsLineConfigs = buildLLMLineConfigsByCtx(models, tpsLineData);
      const ttftLineConfigs = buildLLMLineConfigsByCtx(models, ttftLineData);

      const hasTps = isBar ? tpsBarConfigs.length > 0 : tpsLineConfigs.length > 0;
      const hasTtft = isBar ? ttftBarConfigs.length > 0 : ttftLineConfigs.length > 0;
      if (!hasTps && !hasTtft) return null;
      return { tier, tpsBarData, ttftBarData, tpsBarConfigs, ttftBarConfigs, tpsLineData, ttftLineData, tpsLineConfigs, ttftLineConfigs, hasTps, hasTtft };
    }).filter(Boolean);
    if (!groups.length) return null;

    const allTtftVals = (isBar
      ? groups.flatMap(g => g.ttftBarData.flatMap(row => g.ttftBarConfigs.map(bc => row[bc.dataKey])))
      : groups.flatMap(g => g.ttftLineData.flatMap(row => g.ttftLineConfigs.map(lc => row[lc.dataKey])))
    ).filter(v => v != null);
    const ttftUnit = allTtftVals.some(v => v >= 60) ? "sec-plain"
      : allTtftVals.length && allTtftVals.every(v => v < 1) ? "ms"
      : "sec";
    const ttftYLabel = ttftUnit === "ms" ? "TTFT (ms)" : "TTFT (sec)";
    return { file: f, groups, ttftUnit, ttftYLabel };
  }).filter(Boolean);

  if (!systemGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS[section]} data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {systemGroups.map(({ file: f, groups, ttftUnit, ttftYLabel }) => (
        <div key={f.id} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{f.hostname}</div>
          {groups.map(g => {
            const tierSuffix = g.tier ? ` — ${SIZE_TIER_LABELS[g.tier]}` : "";
            const tierKey = g.tier ? `_${g.tier}` : "";
            return (
              <Fragment key={g.tier || "combined"}>
                {g.hasTps && (isBar ? (
                  <GroupedBarCard
                    title={`Tokens/sec${titleSuffix}${tierSuffix}`}
                    modelName={f.hostname}
                    data={g.tpsBarData}
                    barConfigs={g.tpsBarConfigs}
                    xKey="modelLabel" yLabel="Tokens/sec" unit="tps"
                    chartName={`${chartNamePrefix}tps_by_system${tierKey}`} chartModel={f.hostname}
                    logoSrc={logoSrc} direction="higher"
                  />
                ) : (
                  <ChartCard
                    title={`Tokens/sec${titleSuffix}${tierSuffix}`}
                    modelName={f.hostname}
                    data={g.tpsLineData} lineConfigs={g.tpsLineConfigs}
                    xKey="ctxLabel" xLabel="Context Length" yLabel="Tokens/sec" unit="tps"
                    isMultiFile={false}
                    chartName={`${chartNamePrefix}tps_by_system${tierKey}`} chartModel={f.hostname}
                    logoSrc={logoSrc} direction="higher"
                  />
                ))}
                {g.hasTtft && (isBar ? (
                  <GroupedBarCard
                    title={`Time to First Token${titleSuffix}${tierSuffix}`}
                    modelName={f.hostname}
                    data={g.ttftBarData}
                    barConfigs={g.ttftBarConfigs}
                    xKey="modelLabel" yLabel={ttftYLabel} unit={ttftUnit}
                    chartName={`${chartNamePrefix}ttft_by_system${tierKey}`} chartModel={f.hostname}
                    logoSrc={logoSrc} direction="lower"
                  />
                ) : (
                  <ChartCard
                    title={`Time to First Token${titleSuffix}${tierSuffix}`}
                    modelName={f.hostname}
                    data={g.ttftLineData} lineConfigs={g.ttftLineConfigs}
                    xKey="ctxLabel" xLabel="Context Length" yLabel={ttftYLabel} unit={ttftUnit}
                    isMultiFile={false}
                    chartName={`${chartNamePrefix}ttft_by_system${tierKey}`} chartModel={f.hostname}
                    logoSrc={logoSrc} direction="lower"
                  />
                ))}
              </Fragment>
            );
          })}
        </div>
      ))}
    </ChartGrid>
  );
}
