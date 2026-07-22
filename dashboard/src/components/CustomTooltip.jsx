import { fmt } from "../utils";
import styles from "./CustomTooltip.module.css";

export default function CustomTooltip({ active, payload, label, unit, xPrefix, orderedBarConfigs }) {
  if (!active || !payload?.length) return null;
  const groupLabel = payload[0]?.payload?._groupLabel ?? label;
  const row = payload[0]?.payload;
  const entries = orderedBarConfigs
    ? orderedBarConfigs.map(config => ({
        dataKey: config.dataKey,
        name: config.name,
        color: config.fill,
        value: row?.[config.dataKey],
        status: row?.[`_status_${config.dataKey}`],
      }))
    : payload.map(p => ({
        dataKey: p.dataKey,
        name: p.payload?._seriesName ?? p.name,
        color: p.payload?._fill ?? p.color,
        value: p.value,
      }));
  return (
    <div className={styles.tooltip}>
      <div className={styles.xLabel}>{xPrefix ? `${xPrefix}: ` : ""}{groupLabel}</div>
      {entries.map(entry => (
        <div key={`${entry.dataKey}-${entry.name}`} className={styles.row} style={{ color: entry.color }}>
          <span style={{ whiteSpace: "pre-line" }}>{entry.name}</span>: <strong>{entry.status || fmt(entry.value, unit)}</strong>
        </div>
      ))}
    </div>
  );
}
