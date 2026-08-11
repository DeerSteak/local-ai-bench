import { LineChart, Line, BarChart, Bar, Cell, LabelList, Rectangle, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useContext, useEffect, useState } from "react";
import { measuredCategoryAxisWidth, prepareOrderedBarGroupData, fmt } from "../../utils/shared";
import type { JsonRecord } from "../../utils/shared";
import { CATEGORY_COLORS } from "../../constants";
import type { ChartRow } from "../../types";
import CustomLegend from "../CustomLegend";
import CustomTooltip from "../CustomTooltip";
import styles from "../ChartPanel.module.css";
import { DeltaModeContext } from "../DeltaModeContext";

interface LineConfig {
  dataKey: string, name: string, stroke?: string, strokeDasharray?: string,
}
interface BarConfig {
  dataKey: string, name: string, fill: string,
}

// The subset of recharts' own render-prop shape actually read here.
interface BarRenderProps {
  x?: number | string, y?: number | string, width?: number | string, height?: number | string,
  payload?: ChartRow, index?: number,
}

function DirectionHint({ direction }: { direction?: string }) {
  if (!direction) return null;
  return (
    <span className={styles.chartDirection}>
      {direction === "higher" ? "↑ higher is better" : "↓ lower is better"}
    </span>
  );
}

export function ChartCard({ title, modelName = null, data, lineConfigs, xKey, xLabel, yLabel, unit, isMultiFile, chartName, chartModel = null, logoSrc, direction }: {
  title: string, modelName?: string | null, data: ChartRow[], lineConfigs: LineConfig[], xKey: string,
  xLabel: string, yLabel: string, unit: string, isMultiFile: boolean, chartName: string,
  chartModel?: string | null, logoSrc?: string | null, direction?: string,
}) {
  const deltaMode = useContext(DeltaModeContext);
  const effectiveUnit = deltaMode ? "pct" : unit;
  const effectiveYLabel = deltaMode ? "Baseline-relative performance (%)" : yLabel;
  const yTickFormatter = (v: number) => fmt(v, effectiveUnit);
  return (
    <div className="card chart-card" style={{ position: "relative" }} data-chart-name={chartName} data-chart-model={chartModel || ""}>
      <div className={styles.chartHeader}>
        {modelName && <div className={styles.chartModelName}>{modelName}</div>}
        <div className={styles.chartTitleRow}>
          <span className={styles.chartTitle}>{title}</span>
          <DirectionHint direction={direction} />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid stroke="#e0e4e8" strokeDasharray="3 3" />
          <XAxis
            dataKey={xKey}
            tick={{ fill: "#57606a", fontSize: 17, dy: 8 }}
            label={{ value: xLabel, position: "insideBottom", offset: -4, fill: "#8c959f", fontSize: 15 }}
            height={60}
          />
          <YAxis
            tick={{ fill: "#57606a", fontSize: 17 }}
            tickFormatter={yTickFormatter}
            width={100}
            label={{ value: effectiveYLabel, angle: -90, position: "insideLeft", offset: 20, fill: "#8c959f", fontSize: 15, dy: 70 }}
          />
          <Tooltip content={<CustomTooltip unit={effectiveUnit} xPrefix={xLabel} />} />
          <Legend content={(props) => (
            <CustomLegend {...props} isMultiFile={isMultiFile} sortOrder={lineConfigs.map(config => config.name)} />
          )} />
          {lineConfigs.map(lc => (
            <Line
              key={lc.dataKey}
              type="monotone"
              dataKey={lc.dataKey}
              name={lc.name}
              stroke={lc.stroke}
              strokeWidth={2}
              dot={{ r: 4, fill: lc.stroke }}
              strokeDasharray={lc.strokeDasharray}
              connectNulls
              activeDot={{ r: 6 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {logoSrc && <img src={logoSrc} className={styles.logoOverlay} alt="" />}
    </div>
  );
}

function MultiLineTick({ x = 0, y = 0, payload = null }: { x?: number, y?: number, payload?: { value: string } | null }) {
  const lines = String(payload?.value ?? '').split('\n');
  const lineH = 15;
  return (
    <g transform={`translate(${x},${y})`}>
      {lines.map((line, i) => (
        <text key={i} x={0} y={(i - (lines.length - 1) / 2) * lineH} dy="0.35em"
          textAnchor="end" fill="#57606a" fontSize={13} fontFamily="'IBM Plex Sans', sans-serif">
          {line}
        </text>
      ))}
    </g>
  );
}

function useCategoryAxisWidth(rows: ChartRow[], key: string): number {
  const [width, setWidth] = useState(60);
  useEffect(() => {
    let active = true;
    document.fonts.ready.then(() => {
      if (!active) return;
      const context = document.createElement("canvas").getContext("2d");
      if (!context) return;
      context.font = "13px 'IBM Plex Sans'";
      setWidth(measuredCategoryAxisWidth(rows, key, label => context.measureText(label).width));
    });
    return () => { active = false; };
  }, [rows, key]);
  return width;
}

function BarLabel({ x = 0, y = 0, width = 0, height = 0, value = null, naKey, statusKey, rowData, formatter }: {
  x?: number | string, y?: number | string, width?: number | string, height?: number | string,
  value?: JsonRecord[string], naKey: string, statusKey: string, rowData?: ChartRow,
  formatter: (v: JsonRecord[string]) => string,
}) {
  const isNa = rowData?.[naKey];
  const status = rowData?.[statusKey];
  const label = status || (isNa ? "N/A" : formatter(value));
  const lx = (status || isNa) ? Number(x ?? 0) + 8 : Number(x ?? 0) + Number(width ?? 0) + 6;
  const ly = Number(y ?? 0) + Number(height ?? 0) / 2;
  return (
    <text x={lx} y={ly} dy="0.35em" fontSize={12} fontFamily="'IBM Plex Mono', monospace"
      fill={status ? "#e36209" : isNa ? "#8c959f" : "#57606a"} fontStyle={(status || isNa) ? "italic" : "normal"}>
      {label}
    </text>
  );
}

function OrderedBarGroup({ x = 0, y = 0, width = 0, height = 0, payload, barConfigs, formatter }: {
  x?: number | string, y?: number | string, width?: number | string, height?: number | string, payload?: ChartRow,
  barConfigs: BarConfig[], formatter: (v: JsonRecord[string]) => string,
}) {
  const numX = Number(x), numY = Number(y), numWidth = Number(width), numHeight = Number(height);
  const slotHeight = numHeight / barConfigs.length;
  const barHeight = Math.max(1, slotHeight - 4);
  const maxValue = payload?._groupMax;
  return (
    <g>
      {barConfigs.map((config, index) => {
        const value = payload?.[config.dataKey];
        const barWidth = value == null || maxValue <= 0 ? 0 : Math.max(1, numWidth * value / maxValue);
        const barY = numY + index * slotHeight + (slotHeight - barHeight) / 2;
        return (
          <g key={config.dataKey}>
            <Rectangle x={numX} y={barY} width={barWidth} height={barHeight} fill={config.fill} radius={[0, 3, 3, 0]} />
            <BarLabel
              x={x} y={barY} width={barWidth} height={barHeight} value={value}
              naKey={`_na_${config.dataKey}`} statusKey={`_status_${config.dataKey}`}
              rowData={payload} formatter={formatter}
            />
          </g>
        );
      })}
    </g>
  );
}

// Reserve enough right margin for the longest bar-end label, including any
// "Timed Out" / "Skipped - ..." status text (which runs longer than a
// formatted value or "N/A").
function computeRightMargin(rows: ChartRow[], barConfigs: BarConfig[]): number {
  let maxChars = 4;
  for (const row of rows) {
    for (const bc of barConfigs) {
      const status = row[`_status_${bc.dataKey}`];
      if (status && status.length > maxChars) maxChars = status.length;
    }
  }
  return Math.min(220, Math.max(60, maxChars * 7 + 20));
}

export function GroupedBarCard({ title, modelName = null, data, barConfigs, xKey, yLabel, unit, chartName, chartModel = null, logoSrc, direction, orderedSeries = false }: {
  title: string, modelName?: string | null, data: ChartRow[], barConfigs: BarConfig[], xKey: string,
  yLabel: string, unit: string, chartName: string, chartModel?: string | null, logoSrc?: string | null,
  direction?: string, orderedSeries?: boolean,
}) {
  const yAxisWidth = useCategoryAxisWidth(data, xKey);
  const deltaMode = useContext(DeltaModeContext);
  const effectiveUnit = deltaMode ? "pct" : unit;
  const effectiveYLabel = deltaMode ? "Baseline-relative performance (%)" : yLabel;
  const valFormatter = (v: number) => fmt(v, effectiveUnit);

  // Replace nulls with 0 so recharts renders the bar slot; track which were null.
  const groupedData = data.map(row => {
    const r: ChartRow = { ...row };
    for (const bc of barConfigs) {
      if (r[bc.dataKey] == null) { r[`_na_${bc.dataKey}`] = true; r[bc.dataKey] = 0; }
    }
    return r;
  });
  const processedData = orderedSeries
    ? prepareOrderedBarGroupData(groupedData, barConfigs)
    : groupedData;

  const maxLabelLines = Math.max(1, ...data.map(row => String(row[xKey] ?? '').split('\n').length));
  const rowH = Math.max(32, maxLabelLines * 16);
  const chartHeight = Math.max(280, data.length * barConfigs.length * rowH + 104);
  const rightMargin = computeRightMargin(data, barConfigs);
  const legendPayload = barConfigs.map(config => ({
    dataKey: config.dataKey, value: config.name, color: config.fill,
  }));
  return (
    <div className="card chart-card" style={{ position: "relative" }} data-chart-name={chartName} data-chart-model={chartModel || ""}>
      <div className={styles.chartHeader}>
        {modelName && <div className={styles.chartModelName}>{modelName}</div>}
        <div className={styles.chartTitleRow}>
          <span className={styles.chartTitle}>{title}</span>
          <DirectionHint direction={direction} />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={chartHeight}>
        {/* Fixed pixel gap, not recharts' default 10%-of-band — with many bars per category
            (e.g. llama-bench's up to 20 checkpoints) the per-category band is tall enough that
            a percentage gap becomes a very visible dead strip between categories. */}
        <BarChart layout="vertical" data={processedData} barCategoryGap={8} margin={{ top: 8, right: rightMargin, bottom: 12, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e4e8" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#57606a", fontSize: 15 }}
            tickFormatter={valFormatter}
            label={{ value: effectiveYLabel, position: "insideBottom", offset: -6, fill: "#8c959f", fontSize: 15 }}
            height={56}
          />
          <YAxis
            type="category"
            dataKey={xKey}
            tick={<MultiLineTick />}
            tickSize={6}
            tickMargin={5}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip unit={effectiveUnit} xPrefix="System" orderedBarConfigs={orderedSeries ? barConfigs : undefined} />} />
          {barConfigs.length > 1 && (
            <Legend content={(props) => <CustomLegend {...props} payload={orderedSeries ? legendPayload : props.payload} isMultiFile={false} sortOrder={barConfigs.map(bc => bc.name)} />} />
          )}
          {orderedSeries ? (
            <Bar
              dataKey="_groupMax" name="Value" fill="#57606a" radius={[0, 3, 3, 0]}
              shape={(props: BarRenderProps) => <OrderedBarGroup {...props} barConfigs={barConfigs} formatter={valFormatter} />}
              isAnimationActive={false}
            />
          ) : barConfigs.map(bc => (
            <Bar key={bc.dataKey} dataKey={bc.dataKey} name={bc.name} fill={bc.fill} maxBarSize={32} minPointSize={1} radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {barConfigs.length === 1 && processedData.map((_, i) => (
                <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
              ))}
              <LabelList dataKey={bc.dataKey} content={(props: BarRenderProps) => (
                <BarLabel {...props} naKey={`_na_${bc.dataKey}`} statusKey={`_status_${bc.dataKey}`} rowData={props.index != null ? processedData[props.index] : undefined} formatter={valFormatter} />
              )} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      {logoSrc && <img src={logoSrc} className={styles.logoOverlay} alt="" />}
    </div>
  );
}
