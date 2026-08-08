import type { RefObject, ReactNode } from "react";
import { Fragment } from "react";
import { SIZE_TIER_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import { lookup } from "../../utils/shared";
import type { ResultsFile, ChartRow } from "../../types";
import styles from "../ChartPanel.module.css";

interface Metric {
  key: string, title: string, yLabel: string, unit: string, chartName: string, direction?: string,
  barData?: ChartRow[], barConfigs?: { dataKey: string, name: string, fill: string }[],
  lineData?: ChartRow[], lineConfigs?: { dataKey: string, name: string, stroke?: string }[],
  xKey?: string, xLabel?: string,
}
interface SystemGroup {
  file: ResultsFile, skipEntries: { key: string, label: string }[],
  groups: { tier: string | null, metrics: Metric[] }[],
}

// Generic "Group By: System" renderer, shared by every section that offers by-system
// tier splitting (LLMBySystemPanel, LlamaBenchBySystemPanel) — one card group per
// system, further split into small/medium/large model-size tiers (or combined) per
// the "Model Sizes" toggle, one bar/line chart per metric. Callers own their own data
// shape (LLM's ctx-keyed results vs. llama-bench's dynamic pp/tg checkpoints) and hand
// this component fully-resolved `systemGroups`; this component only lays them out and
// flips Bar/Line — it has no opinion on where the data came from.
export default function BySystemPanel({ containerRef, chartWidth, logoSrc, isBar, emptyLabel, systemGroups }: {
  containerRef?: RefObject<HTMLDivElement | null>, chartWidth: number, logoSrc?: string, isBar: boolean,
  emptyLabel: ReactNode, systemGroups: SystemGroup[],
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };

  if (!systemGroups.length) {
    return <EmptyState style={containerStyle}>{emptyLabel}</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {systemGroups.map(({ file: f, groups, skipEntries }) => (
        <div key={f.id} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{f.hostname}</div>
          {skipEntries.length > 0 && (
            <div className={styles.skipNote}>
              {skipEntries.map(e => <div key={e.key}>{e.label}</div>)}
            </div>
          )}
          {groups.map(({ tier, metrics }) => {
            const tierSuffix = tier ? ` — ${lookup(SIZE_TIER_LABELS, tier)}` : "";
            const tierKey = tier ? `_${tier}` : "";
            return (
              <Fragment key={tier || "combined"}>
                {metrics.map(m => isBar ? (
                  <GroupedBarCard
                    key={m.key}
                    title={`${m.title}${tierSuffix}`}
                    modelName={f.hostname}
                    data={m.barData}
                    barConfigs={m.barConfigs}
                    xKey="modelLabel" yLabel={m.yLabel} unit={m.unit}
                    chartName={`${m.chartName}_by_system${tierKey}`} chartModel={f.hostname}
                    logoSrc={logoSrc} direction={m.direction}
                  />
                ) : (
                  <ChartCard
                    key={m.key}
                    title={`${m.title}${tierSuffix}`}
                    modelName={f.hostname}
                    data={m.lineData} lineConfigs={m.lineConfigs}
                    xKey={m.xKey} xLabel={m.xLabel} yLabel={m.yLabel} unit={m.unit}
                    isMultiFile={false}
                    chartName={`${m.chartName}_by_system${tierKey}`} chartModel={f.hostname}
                    logoSrc={logoSrc} direction={m.direction}
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
