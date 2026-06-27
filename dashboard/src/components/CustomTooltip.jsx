import { fmt } from "../utils";
import styles from "./CustomTooltip.module.css";

export default function CustomTooltip({ active, payload, label, unit, xPrefix }) {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.tooltip}>
      <div className={styles.xLabel}>{xPrefix ? `${xPrefix}: ` : ""}{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className={styles.row} style={{ color: p.color }}>
          <span style={{ whiteSpace: "pre-line" }}>{p.name}</span>: <strong>{fmt(p.value, unit)}</strong>
        </div>
      ))}
    </div>
  );
}
