import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import {
  buildLLMDataForModel, buildFileLineConfigs,
  buildEmbedData, buildEmbedLineConfigs,
  buildImagesDataForResolution,
  getAllLLMModels, getAllImageModels, modelLabel, fmt,
} from "../utils";
import { SECTION_LABELS, FILE_COLORS, RES_ORDER } from "../constants";
import CustomLegend from "./CustomLegend";
import CustomTooltip from "./CustomTooltip";
import styles from "./ChartPanel.module.css";

function FileSubtitle({ files }) {
  if (!files.length) return null;
  if (files.length === 1) {
    const f = files[0];
    return (
      <div className={styles.subtitleSingle}>
        <span className={styles.subtitleHost}>{f.hostname}</span>
        <span className={styles.subtitleBackend}>· {f.backend}</span>
        {f.os && <span className={styles.subtitleOs}>{f.os}</span>}
      </div>
    );
  }
  return (
    <div className={styles.subtitleMulti}>
      {files.map((f, i) => (
        <div key={f.id} className={styles.subtitleFile}>
          <span
            className={styles.subtitleDot}
            style={{ background: FILE_COLORS[i % FILE_COLORS.length] }}
          />
          <span className={styles.subtitleHost}>{f.hostname}</span>
          <span className={styles.subtitleBackend}>· {f.backend}</span>
        </div>
      ))}
    </div>
  );
}

function DirectionHint({ direction }) {
  if (!direction) return null;
  return (
    <span className={styles.chartDirection}>
      {direction === "higher" ? "↑ higher is better" : "↓ lower is better"}
    </span>
  );
}

function ChartCard({ title, modelName, subtitle, data, lineConfigs, xKey, xLabel, yLabel, unit, isMultiFile, chartName, chartModel, logoSrc, direction }) {
  const yTickFormatter = v => fmt(v, unit);
  return (
    <div className="card" style={{ position: "relative" }} data-chart-name={chartName} data-chart-model={chartModel || ""}>
      <div className={styles.chartHeader}>
        {modelName && <div className={styles.chartModelName}>{modelName}</div>}
        <div className={styles.chartTitleRow}>
          <span className={styles.chartTitle}>{title}</span>
          <DirectionHint direction={direction} />
        </div>
        {subtitle}
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

function ImageBarCard({ title, data, files, chartName, chartModel, logoSrc }) {
  return (
    <div className="card" style={{ position: "relative" }} data-chart-name={chartName} data-chart-model={chartModel || ""}>
      <div className={styles.chartHeader}>
        <div className={styles.chartModelName}>Image Generation</div>
        <div className={styles.chartTitleRow}>
          <span className={styles.chartTitle}>{title}</span>
          <DirectionHint direction="lower" />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e4e8" vertical={false} />
          <XAxis dataKey="modelLabel" tick={{ fill: "#57606a", fontSize: 15 }} />
          <YAxis
            tick={{ fill: "#57606a", fontSize: 15 }}
            tickFormatter={v => fmt(v, "sec")}
            width={100}
            label={{ value: "Sec / image", angle: -90, position: "insideLeft", offset: 20, fill: "#8c959f", fontSize: 15, dy: 70 }}
          />
          <Tooltip content={<CustomTooltip unit="sec" xPrefix="Model" />} />
          <Legend content={(props) => <CustomLegend {...props} isMultiFile={false} />} />
          {files.map((f, fi) => (
            <Bar key={fi} dataKey={`f${fi}`} name={f.hostname} fill={FILE_COLORS[fi % FILE_COLORS.length]} maxBarSize={60} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      {logoSrc && <img src={logoSrc} className={styles.logoOverlay} alt="" />}
    </div>
  );
}

export default function ChartPanel({
  containerRef, files, section,
  enabledModels, enabledImageModels, chartWidth, logoSrc,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const isMultiFile = files.length > 1;

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
      const tpsConfigs = lineConfigs.filter(lc => tpsData.some(r => r[lc.dataKey] != null));
      const ttftConfigs = lineConfigs.filter(lc => ttftData.some(r => r[lc.dataKey] != null));
      if (!tpsConfigs.length && !ttftConfigs.length) return null;
      return { model, tpsData, ttftData, tpsConfigs, ttftConfigs };
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
        <FileSubtitle files={files} />
        {modelGroups.map(({ model, tpsData, ttftData, tpsConfigs, ttftConfigs }) => (
          <div key={model} className={styles.modelGroup}>
            <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
            {tpsConfigs.length > 0 && (
              <ChartCard
                title="Tokens/sec"
                modelName={modelLabel(model)}
                data={tpsData} lineConfigs={tpsConfigs}
                xKey="ctxLabel" xLabel="Context Length" yLabel="Tokens/sec" unit="tps"
                isMultiFile={isMultiFile}
                chartName="tps" chartModel={model}
                logoSrc={logoSrc} direction="higher"
              />
            )}
            {ttftConfigs.length > 0 && (
              <ChartCard
                title="Time to First Token"
                modelName={modelLabel(model)}
                data={ttftData} lineConfigs={ttftConfigs}
                xKey="ctxLabel" xLabel="Context Length" yLabel="TTFT (sec)" unit="sec"
                isMultiFile={isMultiFile}
                chartName="ttft" chartModel={model}
                logoSrc={logoSrc} direction="lower"
              />
            )}
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

    return (
      <div ref={containerRef} className={styles.container} style={containerStyle}>
        <FileSubtitle files={files} />
        {resolutions.map(res => {
          const data = buildImagesDataForResolution(files, res, enabledImageModels);
          if (!data.length) return null;
          return (
            <ImageBarCard
              key={res}
              title={res}
              data={data}
              files={files}
              chartName="images"
              chartModel={res}
              logoSrc={logoSrc}
            />
          );
        })}
      </div>
    );
  }

  const subtitle = <FileSubtitle files={files} />;
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
        subtitle={subtitle}
        data={data} lineConfigs={lineConfigs}
        xKey="batchLabel" xLabel="Batch Size" yLabel="Sentences/sec" unit="sps"
        isMultiFile={isMultiFile}
        chartName="embeddings"
        logoSrc={logoSrc} direction="higher"
      />
    </div>
  );
}
