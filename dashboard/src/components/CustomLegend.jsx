import styles from "./CustomLegend.module.css";

export default function CustomLegend({ payload, isMultiFile }) {
  if (!payload?.length) return null;

  if (isMultiFile) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.items}>
          {payload.map(p => (
            <div key={p.dataKey} className={styles.item}>
              <svg width="24" height="12" style={{ flexShrink: 0 }}>
                <line
                  x1="0" y1="6" x2="24" y2="6"
                  stroke={p.color}
                  strokeWidth="2"
                  strokeDasharray={p.payload?.strokeDasharray || ""}
                />
              </svg>
              {p.value}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.items}>
        {payload.map(p => (
          <div key={p.value} className={styles.item}>
            <span className={styles.swatch} style={{ background: p.color }} />
            {p.value}
          </div>
        ))}
      </div>
    </div>
  );
}
