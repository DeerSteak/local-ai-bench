import type { ReactNode, RefObject, CSSProperties } from "react";
import styles from "../ChartPanel.module.css";

export function EmptyState({ style, children }: { style?: CSSProperties, children: ReactNode }) {
  return <div className={styles.emptyState} style={style}>{children}</div>;
}

export function ChartGrid({ containerRef, style, children }: {
  containerRef?: RefObject<HTMLDivElement | null>, style?: CSSProperties, children: ReactNode,
}) {
  return <div ref={containerRef} className={styles.container} style={style}>{children}</div>;
}
