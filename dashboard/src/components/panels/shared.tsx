import styles from "../ChartPanel.module.css";

export function EmptyState({ style, children }) {
  return <div className={styles.emptyState} style={style}>{children}</div>;
}

export function ChartGrid({ containerRef, style, children }) {
  return <div ref={containerRef} className={styles.container} style={style}>{children}</div>;
}
