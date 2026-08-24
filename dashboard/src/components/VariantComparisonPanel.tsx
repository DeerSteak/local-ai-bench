import type { JsonRecord } from "../utils/shared";
import { buildVariantDisplayRows } from "../utils/variants";
import styles from "./VariantComparisonPanel.module.css";

export default function VariantComparisonPanel({ name, artifact }: { name: string, artifact: JsonRecord }) {
  const rows = buildVariantDisplayRows(artifact);
  return <main className={styles.panel}>
    <div className={styles.eyebrow}>{String(artifact.base_model)}</div>
    <h2>Quantization tradeoffs</h2>
    <p>Every delta is relative to {String(artifact.reference_variant)}. Quality is shown in percentage points; other metrics use percent change.</p>
    <div className={styles.tableWrap}><table>
      <thead><tr><th>Variant</th><th>Quality</th><th>Trial verdict</th><th>Throughput</th><th>Peak memory</th><th>Energy</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.variant}>
        <th>{row.variant}{row.reference && <span>Reference</span>}</th>
        <td>{row.quality}</td>
        <td className={!row.qualityRanked && !row.reference ? styles.unranked : ""}>{row.qualityVerdict.replaceAll("_", " ")}</td>
        <td>{row.throughput}</td><td>{row.memory}</td><td>{row.energy}</td>
      </tr>)}</tbody>
    </table></div>
    <footer>Unchanged and inconclusive quality differences are deliberately not rankings. Missing measurements are not zero. Source: {name}</footer>
  </main>;
}
