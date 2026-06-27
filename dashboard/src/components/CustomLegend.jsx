import styles from "./CustomLegend.module.css";

export default function CustomLegend({ payload, isMultiFile, sortOrder }) {
  if (!payload?.length) return null;

  const sorted = sortOrder
    ? [...payload].sort((a, b) => {
        const ai = sortOrder.indexOf(a.value);
        const bi = sortOrder.indexOf(b.value);
        return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
      })
    : payload;

  if (isMultiFile) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.items}>
          {sorted.map(p => (
            <div key={p.dataKey} className={styles.item}>
              <svg width="24" height="12" style={{ flexShrink: 0 }}>
                <line
                  x1="0" y1="6" x2="24" y2="6"
                  stroke={p.color}
                  strokeWidth="2"
                  strokeDasharray={p.payload?.strokeDasharray || ""}
                />
              </svg>
              <span style={{ whiteSpace: "pre-line" }}>{p.value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.items}>
        {sorted.map(p => (
          <div key={p.value} className={styles.item}>
            <span className={styles.swatch} style={{ background: p.color }} />
            <span style={{ whiteSpace: "pre-line" }}>{p.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
