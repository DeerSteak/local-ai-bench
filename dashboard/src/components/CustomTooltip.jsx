import { fmt } from "../utils";
import styles from "./CustomTooltip.module.css";

export default function CustomTooltip({ active, payload, label, unit, xPrefix }) {
  if (!active || !payload?.length) return null;
  const groupLabel = payload[0]?.payload?._groupLabel ?? label;
  return (
    <div className={styles.tooltip}>
      <div className={styles.xLabel}>{xPrefix ? `${xPrefix}: ` : ""}{groupLabel}</div>
      {payload.map(p => {
        const name = p.payload?._seriesName ?? p.name;
        const color = p.payload?._fill ?? p.color;
        return (
          <div key={`${p.dataKey}-${name}`} className={styles.row} style={{ color }}>
            <span style={{ whiteSpace: "pre-line" }}>{name}</span>: <strong>{fmt(p.value, unit)}</strong>
          </div>
        );
      })}
    </div>
  );
}
