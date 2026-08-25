import type { JsonRecord } from "../utils/shared";
import { buildTrialDisplayRows } from "../utils/trials";
import styles from "./TrialSetPanel.module.css";

function number(value: number | null, suffix = ""): string {
  return value == null ? "Unavailable" : `${value.toFixed(2)}${suffix}`;
}

function driftLabel(value: string): string {
  return value === "none" ? "No monotonic drift" : value === "insufficient"
    ? "Too few trials for drift" : `${value} drift`;
}

export default function TrialSetPanel({ name, artifact }: { name: string, artifact: JsonRecord }) {
  const rows = buildTrialDisplayRows(artifact);
  return (
    <main className={styles.panel}>
      <div className={styles.eyebrow}>Repeated-trial comparison</div>
      <h2>{name}</h2>
      <p className={styles.summary}>
        {String(artifact.comparison_mode)} comparison · {String(artifact.baseline_trials)} baseline trials · {String(artifact.candidate_trials)} candidate trials
      </p>
      <p className={styles.notice}>Only qualified repeated trials can support a regression verdict. Every interval, drift flag, threshold, and trial count remains visible below.</p>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead><tr>
            <th>Metric</th><th>Baseline distribution</th><th>Candidate distribution</th>
            <th>95% change interval</th><th>Practical threshold</th><th>Verdict</th>
          </tr></thead>
          <tbody>{rows.map(row => (
            <tr key={row.key}>
              <td className={styles.metric}>{row.key}</td>
              <td>Mean {number(row.baselineMean)} · Median {number(row.baselineMedian)} · SD {number(row.baselineStdev)}<br/><span>{driftLabel(row.baselineDrift)}</span></td>
              <td>Mean {number(row.candidateMean)} · Median {number(row.candidateMedian)} · SD {number(row.candidateStdev)}<br/><span>{driftLabel(row.candidateDrift)}</span></td>
              <td>{row.interval ? `${number(row.interval[0], "%")} to ${number(row.interval[1], "%")}` : "Inconclusive"}<br/><span>{row.intervalMethod}</span></td>
              <td>{number(row.threshold, "%")}</td>
              <td><span className={`${styles.verdict} ${styles[row.verdict] || ""}`}>{row.verdict}</span></td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      {!rows.length && <p>No valid trial metrics were recorded in this artifact.</p>}
    </main>
  );
}
