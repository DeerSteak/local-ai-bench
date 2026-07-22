import { LineChart, Line, BarChart, Bar, Cell, LabelList, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { flattenGroupedBarData, fmt } from "../../utils";
import { CATEGORY_COLORS } from "../../constants";
import CustomLegend from "../CustomLegend";
import CustomTooltip from "../CustomTooltip";
import styles from "../ChartPanel.module.css";

function DirectionHint({ direction }) {
  if (!direction) return null;
  return (
    <span className={styles.chartDirection}>
      {direction === "higher" ? "↑ higher is better" : "↓ lower is better"}
    </span>
  );
}

export function ChartCard({ title, modelName, data, lineConfigs, xKey, xLabel, yLabel, unit, isMultiFile, chartName, chartModel, logoSrc, direction }) {
  const yTickFormatter = v => fmt(v, unit);
  return (
    <div className="card" style={{ position: "relative" }} data-chart-name={chartName} data-chart-model={chartModel || ""}>
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
            label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 20, fill: "#8c959f", fontSize: 15, dy: 70 }}
          />
          <Tooltip content={<CustomTooltip unit={unit} xPrefix={xLabel} />} />
          <Legend content={(props) => <CustomLegend {...props} isMultiFile={isMultiFile} />} />
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

function MultiLineTick({ x, y, payload }) {
  const lines = String(payload?.value ?? '').split('\n');
  const lineH = 15;
  return (
    <g transform={`translate(${x},${y})`}>
      {lines.map((line, i) => (
        <text key={i} x={0} y={(i - (lines.length - 1) / 2) * lineH} dy="0.35em"
          textAnchor="end" fill="#57606a" fontSize={14}>
          {line}
        </text>
      ))}
    </g>
  );
}

function BarLabel({ x, y, width, height, value, naKey, statusKey, rowData, formatter }) {
  const isNa = rowData?.[naKey];
  const status = rowData?.[statusKey];
  const label = status || (isNa ? "N/A" : formatter(value));
  const lx = (status || isNa) ? (x ?? 0) + 8 : (x ?? 0) + (width ?? 0) + 6;
  const ly = (y ?? 0) + (height ?? 0) / 2;
  return (
    <text x={lx} y={ly} dy="0.35em" fontSize={12} fontFamily="'IBM Plex Mono', monospace"
      fill={status ? "#e36209" : isNa ? "#8c959f" : "#57606a"} fontStyle={(status || isNa) ? "italic" : "normal"}>
      {label}
    </text>
  );
}

// Reserve enough Y-axis width for the longest category label (in a vertical-layout chart).
function computeYAxisWidth(rows, key) {
  const lines = rows.flatMap(row => String(row[key] ?? '').split('\n'));
  const maxLabelChars = Math.max(1, ...lines.map(l => l.length));
  return Math.min(260, Math.max(40, maxLabelChars * 7.2 + 26));
}

// Reserve enough right margin for the longest bar-end label, including any
// "Timed Out" / "Skipped - ..." status text (which runs longer than a
// formatted value or "N/A").
function computeRightMargin(rows, barConfigs) {
  let maxChars = 4;
  for (const row of rows) {
    for (const bc of barConfigs) {
      const status = row[`_status_${bc.dataKey}`];
      if (status && status.length > maxChars) maxChars = status.length;
    }
  }
  return Math.min(220, Math.max(60, maxChars * 7 + 20));
}

export function GroupedBarCard({ title, modelName, data, barConfigs, xKey, yLabel, unit, chartName, chartModel, logoSrc, direction, preserveSeriesOrder = false }) {
  const valFormatter = v => fmt(v, unit);

  // Replace nulls with 0 so recharts renders the bar slot; track which were null.
  const groupedData = data.map(row => {
    const r = { ...row };
    for (const bc of barConfigs) {
      if (r[bc.dataKey] == null) { r[`_na_${bc.dataKey}`] = true; r[bc.dataKey] = 0; }
    }
    return r;
  });
  const processedData = preserveSeriesOrder
    ? flattenGroupedBarData(data, barConfigs, xKey)
    : groupedData;

  const maxLabelLines = Math.max(1, ...data.map(row => String(row[xKey] ?? '').split('\n').length));
  const rowH = Math.max(32, maxLabelLines * 16);
  const chartHeight = Math.max(280, data.length * barConfigs.length * rowH + 104);
  const yAxisWidth = computeYAxisWidth(data, xKey);
  const rightMargin = computeRightMargin(data, barConfigs);
  const legendPayload = barConfigs.map(config => ({
    dataKey: config.dataKey, value: config.name, color: config.fill,
  }));
  return (
    <div className="card" style={{ position: "relative" }} data-chart-name={chartName} data-chart-model={chartModel || ""}>
      <div className={styles.chartHeader}>
        {modelName && <div className={styles.chartModelName}>{modelName}</div>}
        <div className={styles.chartTitleRow}>
          <span className={styles.chartTitle}>{title}</span>
          <DirectionHint direction={direction} />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart layout="vertical" data={processedData} margin={{ top: 8, right: rightMargin, bottom: 12, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e4e8" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#57606a", fontSize: 15 }}
            tickFormatter={valFormatter}
            label={{ value: yLabel, position: "insideBottom", offset: -6, fill: "#8c959f", fontSize: 15 }}
            height={56}
          />
          <YAxis
            type="category"
            dataKey={preserveSeriesOrder ? "_axisLabel" : xKey}
            tick={<MultiLineTick />}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip unit={unit} xPrefix="System" />} />
          {barConfigs.length > 1 && (
            <Legend content={(props) => <CustomLegend {...props} payload={preserveSeriesOrder ? legendPayload : props.payload} isMultiFile={false} sortOrder={barConfigs.map(bc => bc.name)} />} />
          )}
          {preserveSeriesOrder ? (
            <Bar dataKey="_value" name="Value" fill="#57606a" maxBarSize={32} minPointSize={1} radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {processedData.map((row, index) => (
                <Cell key={`${row._groupLabel}-${row._seriesKey}`} fill={row._fill || CATEGORY_COLORS[index % CATEGORY_COLORS.length]} />
              ))}
              <LabelList dataKey="_value" content={(props) => (
                <BarLabel {...props} naKey="_na" statusKey="_status" rowData={processedData[props.index]} formatter={valFormatter} />
              )} />
            </Bar>
          ) : barConfigs.map(bc => (
            <Bar key={bc.dataKey} dataKey={bc.dataKey} name={bc.name} fill={bc.fill} maxBarSize={32} minPointSize={1} radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {barConfigs.length === 1 && processedData.map((_, i) => (
                <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
              ))}
              <LabelList dataKey={bc.dataKey} content={(props) => (
                <BarLabel {...props} naKey={`_na_${bc.dataKey}`} statusKey={`_status_${bc.dataKey}`} rowData={processedData[props.index]} formatter={valFormatter} />
              )} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      {logoSrc && <img src={logoSrc} className={styles.logoOverlay} alt="" />}
    </div>
  );
}
