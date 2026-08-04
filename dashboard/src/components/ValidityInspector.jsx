import { useMemo, useState } from "react";
import { modelLabel } from "../utils/shared";
import { buildValidityRows, validitySummary } from "../utils/validity";
import styles from "./ValidityInspector.module.css";

const MAX_VISIBLE_ROWS = 500;

export default function ValidityInspector({ files, section }) {
  const [filter, setFilter] = useState("all");
  const rows = useMemo(() => buildValidityRows(files, section), [files, section]);
  const summary = useMemo(() => validitySummary(rows), [rows]);
  if (!rows.length) return null;

  const filtered = (filter === "all" ? rows : rows.filter(row => row.status === filter));
  const visible = filtered.slice(0, MAX_VISIBLE_ROWS);
  return (
    <details className={`card ${styles.wrapper}`} defaultOpen={summary.invalid > 0}>
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
    </details>
  );
}
