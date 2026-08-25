import { useMemo, useState } from "react";
import { modelLabel } from "../utils/shared";
import {
  buildPauseSummaries, buildPreflightWarnings, buildValidityRows,
  formatPausedDuration, validitySummary,
} from "../utils/validity";
import type { ResultsFile } from "../types";
import styles from "./ValidityInspector.module.css";

const MAX_VISIBLE_ROWS = 500;

export default function ValidityInspector({ files, section }: { files: ResultsFile[], section: string }) {
  const [filter, setFilter] = useState("all");
  const [manualOpen, setManualOpen] = useState(false);
  const [dismissedSignal, setDismissedSignal] = useState("");
  const rows = useMemo(() => buildValidityRows(files, section), [files, section]);
  const pauses = useMemo(() => buildPauseSummaries(files), [files]);
  const preflightWarnings = useMemo(() => buildPreflightWarnings(files), [files]);
  const summary = useMemo(() => validitySummary(rows), [rows]);
  if (!rows.length && !pauses.length && !preflightWarnings.length) return null;

  const filtered = (filter === "all" ? rows : rows.filter(row => row.status === filter));
  const visible = filtered.slice(0, MAX_VISIBLE_ROWS);
  const attentionSignal = summary.invalid > 0 || pauses.length > 0 || preflightWarnings.length > 0
    ? `${summary.invalid}:${pauses.map(pause => `${pause.fileId}:${pause.count}`).join(",")}`
      + `:${preflightWarnings.length}` : "";
  const open = manualOpen || (attentionSignal !== "" && attentionSignal !== dismissedSignal);
  return (
    <details className={`card ${styles.wrapper}`} open={open}
      onToggle={event => {
        setManualOpen(event.currentTarget.open);
        if (!event.currentTarget.open) setDismissedSignal(attentionSignal);
      }}>
      <summary className={styles.heading}>
        <span>Decision-grade sample review</span>
        <span className={styles.counts}>
          {summary.valid} valid · {summary.invalid} excluded
          {summary.legacy ? ` · ${summary.legacy} legacy` : ""}
        </span>
      </summary>
      <div className={styles.intro}>
        Valid samples contribute to aggregates. Excluded samples do not; their recorded reason remains visible.
        Legacy means the historical file contains an aggregate but no auditable sample payload.
      </div>
      {preflightWarnings.length > 0 && <div className={styles.preflightList}>
        {preflightWarnings.map((warning, index) => (
          <div className={styles.preflightNotice}
            key={`${warning.fileId}-${warning.model}-${warning.check}-${index}`} role="note">
            <span className={styles.preflightBadge}>Preflight</span>
            <strong>{warning.system} · {modelLabel(warning.model)}</strong>
            <span>{warning.check.replaceAll("_", " ")} — {warning.detail}</span>
          </div>
        ))}
      </div>}
      {pauses.length > 0 && <div className={styles.pauseList}>
        {pauses.map(pause => <div className={styles.pauseNotice} key={pause.fileId} role="note">
          <span className={styles.pauseBadge}>Paused run</span>
          <strong>{pause.system}</strong>
          <span>
            {pause.count} pause {pause.count === 1 ? "interval" : "intervals"} · {pause.incomplete
              ? (pause.totalPausedSeconds > 0
                ? `${formatPausedDuration(pause.totalPausedSeconds)} known duration · final interval unavailable`
                : "duration unavailable")
              : `${formatPausedDuration(pause.totalPausedSeconds)} total`}
          </span>
        </div>)}
      </div>}
      {rows.length > 0 && <>
      <label className={styles.filterLabel}>
        Show
        <select value={filter} onChange={event => setFilter(event.target.value)}>
          <option value="all">All recorded samples</option>
          <option value="invalid">Excluded only</option>
          <option value="valid">Valid only</option>
          <option value="legacy">Legacy aggregate-only</option>
        </select>
      </label>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead><tr>
            <th>System</th><th>Model</th><th>Case</th><th>Sample</th><th>Status</th>
            <th>Measurement / reason</th>
          </tr></thead>
          <tbody>{visible.map((row, index) => (
            <tr key={`${row.fileId}-${row.model}-${row.caseLabel}-${row.sample}-${index}`}>
              <td>{row.system}</td>
              <td>{modelLabel(row.model)}</td>
              <td>{row.caseLabel}</td>
              <td>{row.sample}</td>
              <td><span className={`${styles.status} ${styles[row.status]}`}>{row.status}</span></td>
              <td>{row.errors.length ? row.errors.join(", ") : row.summary}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      {filtered.length > visible.length && (
        <div className={styles.truncated}>Showing the first {MAX_VISIBLE_ROWS} of {filtered.length} rows.</div>
      )}
      </>}
    </details>
  );
}
