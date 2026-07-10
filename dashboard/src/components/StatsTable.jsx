import { SECTION_LABELS, FILE_COLORS } from "../constants";
import { flattenLLMData, flattenEmbedData, flattenImageData, fmt, modelLabel } from "../utils";
import styles from "./StatsTable.module.css";

function SortTh({ label, sortKey, sortConfig, onCycleSort }) {
  const active = sortConfig.key === sortKey;
  const arrow = active ? (sortConfig.dir === 1 ? " ↑" : " ↓") : " ↕";
  return (
    <th onClick={() => onCycleSort(sortKey)} className={`${styles.th} ${active ? styles.sorted : ""}`}>
      {label}<span className={styles.sortArrow}>{arrow}</span>
    </th>
  );
}

function MachineTd({ fileId, files }) {
  const idx = files.findIndex(f => f.id === fileId);
  if (idx === -1) return null;
  const color = FILE_COLORS[idx % FILE_COLORS.length];
  return (
    <td className={styles.td} style={{ color, fontWeight: 700, fontFamily: "IBM Plex Mono" }}>
      {idx + 1}
    </td>
  );
}

function LLMTable({ files, section, sortConfig, onCycleSort }) {
  const isMulti = files.length > 1;
  const rows = flattenLLMData(files, section).sort((a, b) => {
    const ak = a[sortConfig.key] ?? "";
    const bk = b[sortConfig.key] ?? "";
    return (ak < bk ? -1 : ak > bk ? 1 : 0) * sortConfig.dir;
  });

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Model" sortKey="model" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Context" sortKey="ctx" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="TPS" sortKey="tps_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <SortTh label="TTFT" sortKey="ttft_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <th className={styles.th}>Runs</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => r.skipped ? (
          <tr key={i} className={styles.trSkipped}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={styles.td} colSpan={6}>
              Skipped — {r.skip_detail}
            </td>
          </tr>
        ) : (
          <tr key={i}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={`${styles.td} ${styles.tdCtx}`}>{r.ctx}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.tps_mean, "tps")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.tps_stdev, "tps")}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.ttft_mean, "sec")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.ttft_stdev, "sec")}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.n_runs}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EmbedTable({ files, sortConfig, onCycleSort }) {
  const isMulti = files.length > 1;
  const rows = flattenEmbedData(files).sort((a, b) => {
    const ak = a[sortConfig.key] ?? "";
    const bk = b[sortConfig.key] ?? "";
    return (ak < bk ? -1 : ak > bk ? 1 : 0) * sortConfig.dir;
  });

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Batch Size" sortKey="batchLabel" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Sent/sec" sortKey="sps_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <SortTh label="Peak RAM (MB)" sortKey="peak_ram_mb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>Device</th>
          <th className={styles.th}>Runs</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdCtx}`}>{r.batchLabel}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.sps_mean, "sps")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.sps_stdev, "sps")}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{r.peak_ram_mb != null ? Math.round(r.peak_ram_mb) : "—"}</td>
            <td className={`${styles.td} ${styles.tdDevice}`}>{r.device || "—"}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.n_runs}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ImagesTable({ files, sortConfig, onCycleSort }) {
  const isMulti = files.length > 1;
  const rows = flattenImageData(files).sort((a, b) => {
    const ak = a[sortConfig.key] ?? "";
    const bk = b[sortConfig.key] ?? "";
    return (ak < bk ? -1 : ak > bk ? 1 : 0) * sortConfig.dir;
  });

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Model" sortKey="modelLabel" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>Steps</th>
          <SortTh label="Resolution" sortKey="res" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Sec/image" sortKey="sec_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <th className={styles.th}>Runs</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{r.modelLabel}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.steps ?? "—"}</td>
            <td className={`${styles.td} ${styles.tdCtx}`}>{r.res}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.sec_mean, "sec")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.sec_stdev, "sec")}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.n_runs}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function StatsTable({ files, section, sortConfig, onCycleSort }) {
  if (!files.length) return null;

  return (
    <div className={`card ${styles.wrapper}`}>
      <div className={styles.tableTitle}>Raw Numbers — {SECTION_LABELS[section]}</div>
      {(section === "llm" || section === "llm_conversation") &&
        <LLMTable files={files} section={section} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {section === "embeddings" && <EmbedTable  files={files} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {section === "images"     && <ImagesTable files={files} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
    </div>
  );
}
