import { LineChart, Line, BarChart, Bar, LabelList, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import {
  buildLLMDataForModel, buildFileLineConfigs,
  buildEmbedData, buildEmbedLineConfigs,
  buildImagesData, buildImagesLineConfigs,
  buildLLMBarData, buildLLMBarConfigs,
  buildEmbedBarData, buildEmbedBarConfigs,
  buildImagesGroupedBarDataForResolution, buildImagesGroupedBarConfigs,
  getAllLLMModels, getAllImageModels, modelLabel, fmt,
  sortBarData, findMostStrenuousKey,
} from "../utils";
import { SECTION_LABELS, FILE_COLORS, RES_ORDER, CTX_ORDER } from "../constants";
import CustomLegend from "./CustomLegend";
import CustomTooltip from "./CustomTooltip";
import styles from "./ChartPanel.module.css";

function DirectionHint({ direction }) {
  if (!direction) return null;
  return (
    <span className={styles.chartDirection}>
      {direction === "higher" ? "↑ higher is better" : "↓ lower is better"}
    </span>
  );
}

function ChartCard({ title, modelName, data, lineConfigs, xKey, xLabel, yLabel, unit, isMultiFile, chartName, chartModel, logoSrc, direction }) {
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

function BarLabel({ x, y, width, height, value, naKey, rowData, formatter }) {
  const isNa = rowData?.[naKey];
  const label = isNa ? "N/A" : formatter(value);
  const lx = isNa ? (x ?? 0) + 8 : (x ?? 0) + (width ?? 0) + 6;
  const ly = (y ?? 0) + (height ?? 0) / 2;
  return (
    <text x={lx} y={ly} dy="0.35em" fontSize={12} fontFamily="'IBM Plex Mono', monospace"
      fill={isNa ? "#8c959f" : "#57606a"} fontStyle={isNa ? "italic" : "normal"}>
      {label}
    </text>
  );
}

function GroupedBarCard({ title, modelName, data, barConfigs, xKey, yLabel, unit, chartName, chartModel, logoSrc, direction }) {
  const valFormatter = v => fmt(v, unit);

  // Replace nulls with 0 so recharts renders the bar slot; track which were null.
  const processedData = data.map(row => {
    const r = { ...row };
    for (const bc of barConfigs) {
      if (r[bc.dataKey] == null) { r[`_na_${bc.dataKey}`] = true; r[bc.dataKey] = 0; }
    }
    return r;
  });

  const maxLabelLines = Math.max(1, ...processedData.map(row => String(row[xKey] ?? '').split('\n').length));
  const rowH = Math.max(32, maxLabelLines * 16);
  const chartHeight = Math.max(280, processedData.length * barConfigs.length * rowH + 100);
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
        <BarChart layout="vertical" data={processedData} margin={{ top: 8, right: 90, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e4e8" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#57606a", fontSize: 15 }}
            tickFormatter={valFormatter}
            label={{ value: yLabel, position: "insideBottom", offset: -16, fill: "#8c959f", fontSize: 15 }}
            height={40}
          />
          <YAxis
            type="category"
            dataKey={xKey}
            tick={<MultiLineTick />}
            width={150}
          />
          <Tooltip content={<CustomTooltip unit={unit} xPrefix="System" />} />
          <Legend content={(props) => <CustomLegend {...props} isMultiFile={false} sortOrder={barConfigs.map(bc => bc.name)} />} />
          {barConfigs.map(bc => (
            <Bar key={bc.dataKey} dataKey={bc.dataKey} name={bc.name} fill={bc.fill} maxBarSize={32} minPointSize={1} radius={[0, 3, 3, 0]}>
              <LabelList dataKey={bc.dataKey} content={(props) => (
                <BarLabel {...props} naKey={`_na_${bc.dataKey}`} rowData={processedData[props.index]} formatter={valFormatter} />
              )} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      {logoSrc && <img src={logoSrc} className={styles.logoOverlay} alt="" />}
    </div>
  );
}

function ImageBarCard({ title, data, files, chartName, chartModel, logoSrc }) {
  const secFormatter = v => fmt(v, "sec");

  const processedData = data.map(row => {
    const r = { ...row };
    for (let fi = 0; fi < files.length; fi++) {
      const k = `f${fi}`;
      if (r[k] == null) { r[`_na_${k}`] = true; r[k] = 0; }
    }
    return r;
  });

  return (
    <div className="card" style={{ position: "relative" }} data-chart-name={chartName} data-chart-model={chartModel || ""}>
      <div className={styles.chartHeader}>
        <div className={styles.chartModelName}>Image Generation</div>
        <div className={styles.chartTitleRow}>
          <span className={styles.chartTitle}>{title}</span>
          <DirectionHint direction="lower" />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(280, processedData.length * files.length * 32 + 100)}>
        <BarChart layout="vertical" data={processedData} margin={{ top: 8, right: 90, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e4e8" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#57606a", fontSize: 15 }}
            tickFormatter={secFormatter}
            label={{ value: "Sec / image", position: "insideBottom", offset: -16, fill: "#8c959f", fontSize: 15 }}
            height={40}
          />
          <YAxis
            type="category"
            dataKey="modelLabel"
            tick={{ fill: "#57606a", fontSize: 14 }}
            width={150}
          />
          <Tooltip content={<CustomTooltip unit="sec" xPrefix="Model" />} />
          <Legend content={(props) => <CustomLegend {...props} isMultiFile={false} />} />
          {files.map((f, fi) => (
            <Bar key={fi} dataKey={`f${fi}`} name={f.hostname} fill={FILE_COLORS[fi % FILE_COLORS.length]} maxBarSize={32} minPointSize={1} radius={[0, 3, 3, 0]}>
              <LabelList dataKey={`f${fi}`} content={(props) => (
                <BarLabel {...props} naKey={`_na_f${fi}`} rowData={processedData[props.index]} formatter={secFormatter} />
              )} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      {logoSrc && <img src={logoSrc} className={styles.logoOverlay} alt="" />}
    </div>
  );
}

export default function ChartPanel({
  containerRef, files, section,
  enabledModels, enabledImageModels, chartWidth, logoSrc, chartStyle,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const isMultiFile = files.length > 1;
  const isBar = chartStyle === "bar";

  if (files.length === 0) {
    return (
      <div className={styles.emptyState} style={containerStyle}>
        Drop a results JSON file above to get started
      </div>
    );
  }

  if (section === "llm") {
    const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m));
    const lineConfigs = buildFileLineConfigs(files);

    const modelGroups = allModels.map(model => {
      const tpsData = buildLLMDataForModel(files, model, "tps");
      const ttftData = buildLLMDataForModel(files, model, "ttft");
      const tpsLineConfigs = lineConfigs.filter(lc => tpsData.some(r => r[lc.dataKey] != null));
      const ttftLineConfigs = lineConfigs.filter(lc => ttftData.some(r => r[lc.dataKey] != null));
      const rawTpsBarConfigs = buildLLMBarConfigs(files, model);
      const rawTtftBarConfigs = buildLLMBarConfigs(files, model);
      const rawTpsBarData = buildLLMBarData(files, model, "tps");
      const rawTtftBarData = buildLLMBarData(files, model, "ttft");
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
      return (
        <div className={styles.emptyState} style={containerStyle}>
          No LLM data in the loaded file(s)
        </div>
      );
    }

    return (
      <div ref={containerRef} className={styles.container} style={containerStyle}>
        {modelGroups.map(({ model, tpsData, ttftData, tpsLineConfigs, ttftLineConfigs, tpsBarConfigs, ttftBarConfigs, tpsBarData, ttftBarData, ttftUnit, ttftYLabel, hasTps, hasTtft }) => (
          <div key={model} className={styles.modelGroup}>
            <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
            {hasTps && (isBar ? (
              <GroupedBarCard
                title="Tokens/sec"
                modelName={modelLabel(model)}
                data={tpsBarData}
                barConfigs={tpsBarConfigs}
                xKey="systemLabel" yLabel="Tokens/sec" unit="tps"
                chartName="tps" chartModel={model}
                logoSrc={logoSrc} direction="higher"
              />
            ) : (
              <ChartCard
                title="Tokens/sec"
                modelName={modelLabel(model)}
                data={tpsData} lineConfigs={tpsLineConfigs}
                xKey="ctxLabel" xLabel="Context Length" yLabel="Tokens/sec" unit="tps"
                isMultiFile={isMultiFile}
                chartName="tps" chartModel={model}
                logoSrc={logoSrc} direction="higher"
              />
            ))}
            {hasTtft && (isBar ? (
              <GroupedBarCard
                title="Time to First Token"
                modelName={modelLabel(model)}
                data={ttftBarData}
                barConfigs={ttftBarConfigs}
                xKey="systemLabel" yLabel={ttftYLabel} unit={ttftUnit}
                chartName="ttft" chartModel={model}
                logoSrc={logoSrc} direction="lower"
              />
            ) : (
              <ChartCard
                title="Time to First Token"
                modelName={modelLabel(model)}
                data={ttftData} lineConfigs={ttftLineConfigs}
                xKey="ctxLabel" xLabel="Context Length" yLabel={ttftYLabel} unit={ttftUnit}
                isMultiFile={isMultiFile}
                chartName="ttft" chartModel={model}
                logoSrc={logoSrc} direction="lower"
              />
            ))}
          </div>
        ))}
      </div>
    );
  }

  if (section === "images") {
    const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));

    const resSet = new Set();
    for (const f of files)
      for (const model of allModels)
        for (const res of Object.keys(f.data.images?.[model]?.resolutions || {})) resSet.add(res);
    const resolutions = RES_ORDER.filter(r => resSet.has(r));

    if (!resolutions.length || !allModels.length) {
      return (
        <div className={styles.emptyState} style={containerStyle}>
          No Images data in the loaded file(s)
        </div>
      );
    }

    const groupedBarConfigs = buildImagesGroupedBarConfigs(files, enabledImageModels);
    const modelKeys = groupedBarConfigs.map(bc => bc.dataKey);
    const lineData = buildImagesData(files, enabledImageModels);
    const lineConfigs = buildImagesLineConfigs(files, lineData, enabledImageModels);
    return (
      <div ref={containerRef} className={styles.container} style={containerStyle}>
        {isBar ? resolutions.map(res => {
          const raw = buildImagesGroupedBarDataForResolution(files, res, enabledImageModels);
          if (!raw.length) return null;
          const strenuousKey = findMostStrenuousKey(raw, modelKeys);
          const data = strenuousKey ? sortBarData(raw, [strenuousKey], "asc") : raw;
          return (
            <GroupedBarCard
              key={res}
              title={res}
              modelName="Image Generation"
              data={data}
              barConfigs={groupedBarConfigs}
              xKey="systemLabel" yLabel="Sec / image" unit="sec"
              chartName="images" chartModel={res}
              logoSrc={logoSrc} direction="lower"
            />
          );
        }) : (
          <ChartCard
            title="Image Generation"
            data={lineData}
            lineConfigs={lineConfigs}
            xKey="resLabel" xLabel="Resolution" yLabel="Sec / image" unit="sec"
            isMultiFile={isMultiFile}
            chartName="images"
            logoSrc={logoSrc} direction="lower"
          />
        )}
      </div>
    );
  }

  if (isBar) {
    const rawBarData = buildEmbedBarData(files);
    const barConfigs = buildEmbedBarConfigs(files);
    if (!rawBarData.length || !barConfigs.length) {
      return (
        <div className={styles.emptyState} style={containerStyle}>
          No {SECTION_LABELS[section]} data in the loaded file(s)
        </div>
      );
    }
    const barData = sortBarData(rawBarData, barConfigs.map(bc => bc.dataKey), "desc");
    return (
      <div ref={containerRef} className={styles.container} style={containerStyle}>
        <GroupedBarCard
          title="Embedding Throughput"
          data={barData}
          barConfigs={barConfigs}
          xKey="systemLabel" yLabel="Sentences/sec" unit="sps"
          chartName="embeddings"
          logoSrc={logoSrc} direction="higher"
        />
      </div>
    );
  }

  const data = buildEmbedData(files);
  const lineConfigs = buildEmbedLineConfigs(files);

  if (!data.length || !lineConfigs.length) {
    return (
      <div className={styles.emptyState} style={containerStyle}>
        No {SECTION_LABELS[section]} data in the loaded file(s)
      </div>
    );
  }

  return (
    <div ref={containerRef} className={styles.container} style={containerStyle}>
      <ChartCard
        title="Embedding Throughput"
        data={data} lineConfigs={lineConfigs}
        xKey="batchLabel" xLabel="Batch Size" yLabel="Sentences/sec" unit="sps"
        isMultiFile={isMultiFile}
        chartName="embeddings"
        logoSrc={logoSrc} direction="higher"
      />
    </div>
  );
}
