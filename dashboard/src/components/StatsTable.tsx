import { SECTION_LABELS, FILE_COLORS, ACCURACY_TEST_LABELS } from "../constants";
import { sortRows, fmt, modelLabel, lookup } from "../utils/shared";
import type { ResultsFile, SortConfig } from "../types";

type CycleSort = (key: string) => void;
import { flattenLLMData } from "../utils/llm";
import { flattenEmbedData } from "../utils/embeddings";
import { flattenImageData } from "../utils/images";
import { flattenAccuracyData } from "../utils/accuracy";
import { flattenConcurrencyData, concurrencySortValue } from "../utils/concurrency";
import { flattenLlamaBenchData } from "../utils/llamabench";
import { flattenLlamaBenchConcData, llamaBenchConcSortValue } from "../utils/llamabenchconc";
import styles from "./StatsTable.module.css";

function SortTh({ label, sortKey, sortConfig, onCycleSort }: {
  label: string, sortKey: string, sortConfig: SortConfig, onCycleSort: CycleSort,
}) {
  const active = sortConfig.key === sortKey;
  const arrow = active ? (sortConfig.dir === 1 ? " ↑" : " ↓") : " ↕";
  return (
    <th onClick={() => onCycleSort(sortKey)} className={`${styles.th} ${active ? styles.sorted : ""}`}>
      {label}<span className={styles.sortArrow}>{arrow}</span>
    </th>
  );
}

function MachineTd({ fileId, files }: { fileId: ResultsFile["id"], files: ResultsFile[] }) {
  const idx = files.findIndex(f => f.id === fileId);
  if (idx === -1) return null;
  const color = FILE_COLORS[idx % FILE_COLORS.length];
  return (
    <td className={styles.td} style={{ color, fontWeight: 700, fontFamily: "IBM Plex Mono" }}>
      {idx + 1}
    </td>
  );
}

function MemoryHeaders({ sortConfig, onCycleSort }: {
  sortConfig: SortConfig, onCycleSort: CycleSort,
}) {
  return <>
    <SortTh label="Host RAM peak" sortKey="host_ram_peak_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
    <SortTh label="Process RSS peak" sortKey="process_rss_peak_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
    <SortTh label="Accelerator peak" sortKey="accelerator_memory_peak_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
    <SortTh label="Headroom" sortKey="headroom_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
  </>;
}

function MemoryCells({ row }: { row: Record<string, unknown> }) {
  const value = (key: string) => typeof row[key] === "number" ? `${fmt(row[key], "gb")} GB` : "Not recorded";
  return <>
    <td className={`${styles.td} ${styles.tdNum}`}>{value("host_ram_peak_gb")}</td>
    <td className={`${styles.td} ${styles.tdNum}`}>{value("process_rss_peak_gb")}</td>
    <td className={`${styles.td} ${styles.tdNum}`}>{value("accelerator_memory_peak_gb")}</td>
    <td className={`${styles.td} ${styles.tdNum}`}>
      {typeof row.headroom_gb === "number" ? `${fmt(row.headroom_gb, "gb")} GB · ${row.headroom_state}` : "Not recorded"}
    </td>
  </>;
}

function LLMTable({  files, section, sortConfig, onCycleSort  }: { files: ResultsFile[], section: string, sortConfig: SortConfig, onCycleSort: CycleSort }) {
  const isMulti = files.length > 1;
  const rows = sortRows(flattenLLMData(files, section), sortConfig);

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
          <SortTh label="Host RAM peak" sortKey="host_ram_peak_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Process RSS peak" sortKey="process_rss_peak_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Accelerator peak" sortKey="accelerator_memory_peak_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Headroom" sortKey="headroom_gb" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>Runs</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => r.skipped ? (
          <tr key={i} className={styles.trSkipped}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={styles.td} colSpan={10}>
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
            <td className={`${styles.td} ${styles.tdNum}`}>{r.host_ram_peak_gb == null ? "Not recorded" : `${fmt(r.host_ram_peak_gb, "gb")} GB`}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{r.process_rss_peak_gb == null ? "Not recorded" : `${fmt(r.process_rss_peak_gb, "gb")} GB`}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{r.accelerator_memory_peak_gb == null ? "Not recorded" : `${fmt(r.accelerator_memory_peak_gb, "gb")} GB`}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{r.headroom_gb == null ? "Not recorded" : `${fmt(r.headroom_gb, "gb")} GB · ${r.headroom_state}`}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.n_runs}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EmbedTable({  files, sortConfig, onCycleSort  }: { files: ResultsFile[], sortConfig: SortConfig, onCycleSort: CycleSort }) {
  const isMulti = files.length > 1;
  const rows = sortRows(flattenEmbedData(files), sortConfig);

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Model" sortKey="modelLabel" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Chunks/sec" sortKey="cps_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <SortTh label="Chunks" sortKey="n_chunks" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>Device</th>
          <MemoryHeaders sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>Runs</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => r.skipped ? (
          <tr key={i} className={styles.trSkipped}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{r.modelLabel}</td>
            <td className={styles.td} colSpan={9}>
              Skipped — {r.skip_detail}
            </td>
          </tr>
        ) : (
          <tr key={i}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{r.modelLabel}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.cps_mean, "sps")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.cps_stdev, "sps")}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{r.n_chunks ?? "—"}</td>
            <td className={`${styles.td} ${styles.tdDevice}`}>{r.device || "—"}</td>
            <MemoryCells row={r} />
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.n_runs}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ImagesTable({  files, sortConfig, onCycleSort  }: { files: ResultsFile[], sortConfig: SortConfig, onCycleSort: CycleSort }) {
  const isMulti = files.length > 1;
  const rows = sortRows(flattenImageData(files), sortConfig);

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
          <MemoryHeaders sortConfig={sortConfig} onCycleSort={onCycleSort} />
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
            <MemoryCells row={r} />
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.n_runs}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ConcurrencyTable({  files, section, sortConfig, onCycleSort  }: { files: ResultsFile[], section: string, sortConfig: SortConfig, onCycleSort: CycleSort }) {
  const isMulti = files.length > 1;
  const rows = sortRows(flattenConcurrencyData(files, section), sortConfig, concurrencySortValue);

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Model" sortKey="model" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Level" sortKey="level" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="TPS" sortKey="tps_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <SortTh label="Prefill TPS" sortKey="prefill_tps_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <SortTh label="Aggregate TPS" sortKey="aggregate_tps" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="TTFT" sortKey="ttft_mean" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <SortTh label="Total Tokens" sortKey="total_tokens" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <MemoryHeaders sortConfig={sortConfig} onCycleSort={onCycleSort} />
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => r.skipped ? (
          <tr key={i} className={styles.trSkipped}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={styles.td} colSpan={13}>
              Skipped — {r.skip_detail}
            </td>
          </tr>
        ) : (
          <tr key={i}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={`${styles.td} ${styles.tdCtx}`}>{r.level}-way</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.tps_mean, "tps")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.tps_stdev, "tps")}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.prefill_tps_mean, "tps")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.prefill_tps_stdev, "tps")}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.aggregate_tps, "tps")}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.ttft_mean, "sec")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.ttft_stdev, "sec")}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.total_tokens}</td>
            <MemoryCells row={r} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function LlamaBenchTable({  files, sortConfig, onCycleSort  }: { files: ResultsFile[], sortConfig: SortConfig, onCycleSort: CycleSort }) {
  const isMulti = files.length > 1;
  const rows = sortRows(flattenLlamaBenchData(files), sortConfig);

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Model" sortKey="model" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Metric" sortKey="metric" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="PP Depth" sortKey="pp" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="TG" sortKey="tg" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Tokens/sec" sortKey="avg_ts" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>± stdev</th>
          <th className={styles.th}>GPU Layers</th>
          <MemoryHeaders sortConfig={sortConfig} onCycleSort={onCycleSort} />
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => r.skipped ? (
          <tr key={i} className={styles.trSkipped}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={styles.td} colSpan={10}>
              Skipped — {r.skip_detail}
            </td>
          </tr>
        ) : (
          <tr key={i}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={`${styles.td} ${styles.tdCtx}`}>{r.metric}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.pp ?? "—"}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.tg ?? "—"}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.avg_ts, "tps")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.stddev_ts, "tps")}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.n_gpu_layers ?? "—"}</td>
            <MemoryCells row={r} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function LlamaBenchConcTable({  files, sortConfig, onCycleSort  }: { files: ResultsFile[], sortConfig: SortConfig, onCycleSort: CycleSort }) {
  const isMulti = files.length > 1;
  const rows = sortRows(flattenLlamaBenchConcData(files), sortConfig, llamaBenchConcSortValue);

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Model" sortKey="model" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Level" sortKey="level" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Prompt" sortKey="pp" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Gen" sortKey="tg" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Aggregate TPS" sortKey="speed_tg" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Prefill TPS" sortKey="speed_pp" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <MemoryHeaders sortConfig={sortConfig} onCycleSort={onCycleSort} />
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => r.skipped ? (
          <tr key={i} className={styles.trSkipped}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={styles.td} colSpan={9}>
              Skipped — {r.skip_detail}
            </td>
          </tr>
        ) : (
          <tr key={i}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={`${styles.td} ${styles.tdCtx}`}>{r.level != null ? `${r.level}-way` : "—"}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.pp ?? "—"}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.tg ?? "—"}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.speed_tg, "tps")}</td>
            <td className={`${styles.td} ${styles.tdStdev}`}>{fmt(r.speed_pp, "tps")}</td>
            <MemoryCells row={r} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function AccuracyTable({ files, testKey, sortConfig, onCycleSort }: {
  files: ResultsFile[], testKey: string, sortConfig: SortConfig, onCycleSort: CycleSort,
}) {
  const isMulti = files.length > 1;
  const rows = sortRows(flattenAccuracyData(files, testKey), sortConfig);

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          {isMulti && <th className={styles.th}>Machine</th>}
          <SortTh label="Model" sortKey="model" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Accuracy" sortKey="accuracy_pct" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Correct" sortKey="correct" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <th className={styles.th}>Total</th>
          <SortTh label="Answered" sortKey="answered" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Timed Out" sortKey="timed_out_count" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Likely Loop" sortKey="likely_loop_count" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Nudged" sortKey="budget_nudged_count" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <SortTh label="Budget Exhausted" sortKey="budget_exceeded_count" sortConfig={sortConfig} onCycleSort={onCycleSort} />
          <MemoryHeaders sortConfig={sortConfig} onCycleSort={onCycleSort} />
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => r.skipped ? (
          <tr key={i} className={styles.trSkipped}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>{modelLabel(r.model)}</td>
            <td className={styles.td} colSpan={12}>
              Skipped — {r.skip_detail}
            </td>
          </tr>
        ) : (
          <tr key={i} className={r.crashed ? styles.trSkipped : undefined}>
            {isMulti && <MachineTd fileId={r._fileId} files={files} />}
            <td className={`${styles.td} ${styles.tdModel}`}>
              {modelLabel(r.model)}{r.crashed ? " (crashed)" : ""}
              {r.preflight_warning ? <span title={String(r.preflight_warning)}> ⚠ template</span> : null}
            </td>
            <td className={`${styles.td} ${styles.tdNum}`}>{fmt(r.accuracy_pct, "pct")}</td>
            <td className={`${styles.td} ${styles.tdNum}`}>{r.correct}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.total}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.answered}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.timed_out_count || "—"}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.likely_loop_count || "—"}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.budget_nudged_count || "—"}</td>
            <td className={`${styles.td} ${styles.tdRuns}`}>{r.budget_exceeded_count || "—"}</td>
            <MemoryCells row={r} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function StatsTable({ files, section, accuracyTest, sortConfig, onCycleSort }: {
  files: ResultsFile[], section: string, accuracyTest: string, sortConfig: SortConfig, onCycleSort: CycleSort,
}) {
  if (!files.length) return null;

  const title = section === "accuracy"
    ? `Raw Numbers — Accuracy (${lookup(ACCURACY_TEST_LABELS, accuracyTest)})`
    : `Raw Numbers — ${lookup(SECTION_LABELS, section)}`;

  return (
    <div className={`card ${styles.wrapper}`}>
      <div className={styles.tableTitle}>{title}</div>
      {(section === "llm" || section === "llm_conversation") &&
        <LLMTable files={files} section={section} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {(section === "concurrency_tool" || section === "concurrency_chat") &&
        <ConcurrencyTable files={files} section={section} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {section === "accuracy"  && <AccuracyTable files={files} testKey={accuracyTest} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {section === "embeddings" && <EmbedTable  files={files} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {section === "images"     && <ImagesTable files={files} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {section === "llamabench" && <LlamaBenchTable files={files} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
      {section === "llamabenchconc" && <LlamaBenchConcTable files={files} sortConfig={sortConfig} onCycleSort={onCycleSort} />}
    </div>
  );
}
