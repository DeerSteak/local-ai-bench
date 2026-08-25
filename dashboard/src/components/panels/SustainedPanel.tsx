import type { RefObject } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { FILE_COLORS } from "../../constants";
import type { ResultsFile } from "../../types";
import { fmt, modelLabel } from "../../utils/shared";
import { buildSustainedTimeline, preferredTemperatureKey } from "../../utils/sustained";
import { ChartGrid, EmptyState } from "./shared";
import styles from "../ChartPanel.module.css";

const TEMPERATURE_LABELS: Record<string, string> = {
  soc_package_c: "SoC package",
  cpu_package_c: "CPU package",
  gpu_die_c: "GPU die",
  gpu_hotspot_c: "GPU hotspot",
};

const label = (value: unknown) => typeof value === "string"
  ? value.replaceAll("_", " ").replace(/^./, c => c.toUpperCase()) : "Not recorded";

export default function SustainedPanel({ containerRef, files, enabledModels, chartWidth, logoSrc }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledModels: Set<string>,
  chartWidth: number, logoSrc?: string | null,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const runs = files.flatMap((file, fileIndex) => Object.keys(file.data.sustained || {})
    .filter(model => enabledModels.has(model))
    .map(model => ({ file, fileIndex, model, result: file.data.sustained?.[model] })));
  if (!runs.length) return <EmptyState style={containerStyle}>No sustained-load data in the loaded file(s)</EmptyState>;

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {runs.map(({ file, fileIndex, model, result }) => {
        const data = buildSustainedTimeline(result);
        const temperatureKey = preferredTemperatureKey(data);
        const hasPower = data.some(row => typeof row.power_watts === "number");
        const analysis = result?.analysis;
        const error = result?.unexpected_error ?? result?.error ?? result?.crashed;
        const skipped = result?.skipped;
        const title = files.length > 1 ? `${file.hostname ?? "Unknown system"} · ${modelLabel(model)}` : modelLabel(model);
        return (
          <div key={`${file.id ?? fileIndex}:${model}`} className="card chart-card" data-chart-name="sustained_timeline" data-chart-model={model}>
            <div className={styles.chartHeader}>
              <div className={styles.chartModelName}>{title}</div>
              <div className={styles.chartTitleRow}><span className={styles.chartTitle}>Sustained Throughput and Thermals</span></div>
            </div>
            <div className={styles.skipNote}>
              {error ? `Failed — ${String(error)}` : skipped ? `Skipped — ${String(skipped)}` : <>
                Retention {typeof analysis?.retention_ratio === "number" ? fmt(analysis.retention_ratio * 100, "pct") : "not recorded"}
                {` · ${label(analysis?.performance)} · ${label(analysis?.cause)}`}
                {typeof analysis?.throttle_onset_sec === "number" ? ` · onset ${fmt(analysis.throttle_onset_sec, "sec")}` : ""}
                {typeof result?.valid_request_count === "number" ? ` · valid requests ${result.valid_request_count}/${typeof result?.request_count === "number" ? result.request_count : "?"}` : ""}
                {typeof result?.ambient_temp_c === "number" ? ` · ambient ${result.ambient_temp_c.toFixed(1)}°C` : " · ambient not recorded"}
              </>}
            </div>
            {!error && !skipped && data.length > 0 && <ResponsiveContainer width="100%" height={340}>
              <LineChart data={data} margin={{ top: 4, right: hasPower ? 58 : 12, bottom: 4, left: 8 }}>
                <CartesianGrid stroke="#e0e4e8" strokeDasharray="3 3" />
                <XAxis dataKey="elapsed_min" type="number" domain={["dataMin", "dataMax"]}
                  tick={{ fill: "#57606a", fontSize: 15 }} unit="m"
                  label={{ value: "Elapsed time (minutes)", position: "insideBottom", offset: -4, fill: "#8c959f" }} height={55} />
                <YAxis yAxisId="throughput" tick={{ fill: "#57606a", fontSize: 15 }} width={75}
                  label={{ value: "Tokens/sec", angle: -90, position: "insideLeft", fill: "#8c959f" }} />
                {temperatureKey && <YAxis yAxisId="temperature" orientation="right" tick={{ fill: "#d97706", fontSize: 15 }} width={58}
                  label={{ value: "Temperature °C", angle: 90, position: "insideRight", fill: "#d97706" }} />}
                {hasPower && <YAxis yAxisId="power" orientation="right" tick={{ fill: "#7c3aed", fontSize: 15 }} width={58}
                  axisLine={false} tickLine={false} label={{ value: "Power W", angle: 90, position: "outside", fill: "#7c3aed" }} />}
                <Tooltip formatter={(value, name) => [typeof value === "number" ? value.toFixed(1) : value, name]} />
                <Legend />
                <Line yAxisId="throughput" type="monotone" dataKey="tokens_per_sec" name="Throughput" stroke={FILE_COLORS[fileIndex % FILE_COLORS.length]} strokeWidth={3} dot={false} isAnimationActive={false} />
                {temperatureKey && <Line yAxisId="temperature" type="monotone" dataKey={temperatureKey}
                  name={`${TEMPERATURE_LABELS[temperatureKey]} °C`} stroke="#d97706" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />}
                {hasPower && <Line yAxisId="power" type="monotone" dataKey="power_watts" name="Power W"
                  stroke="#7c3aed" strokeWidth={2} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />}
              </LineChart>
            </ResponsiveContainer>}
            {logoSrc && <img src={logoSrc} className={styles.logoOverlay} alt="" />}
          </div>
        );
      })}
    </ChartGrid>
  );
}
