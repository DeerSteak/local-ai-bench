import { entriesOf } from "./shared";
import type { JsonRecord } from "./shared";
import type { ResultsFile } from "../types";

const GENERATION_SECTIONS = new Set([
  "llm", "llm_cached", "llm_conversation", "concurrency_tool", "concurrency_chat",
]);

interface ValidityRow {
  fileId: ResultsFile["id"], system: ResultsFile["hostname"], model: string, caseLabel: string,
  sample: number | string, status: "valid" | "invalid" | "legacy", summary: string, errors: string[],
}

const caseKeys = (modelData: JsonRecord[string], section: string): string[] =>
  Object.keys(modelData || {}).filter(key => {
    const value = modelData[key];
    if (!value || typeof value !== "object" || Array.isArray(value)) return false;
    if (section.startsWith("concurrency_")) return /^\d+$/.test(key);
    return key.endsWith("K");
  });

const measurementSummary = (sample: JsonRecord[string]): string => [
  sample?.tokens_per_sec != null ? `${sample.tokens_per_sec} tok/s` : null,
  sample?.client_ttft_sec != null ? `${sample.client_ttft_sec}s TTFT` : null,
  sample?.client_wall_sec != null ? `${sample.client_wall_sec}s wall` : null,
  sample?.generated_tokens != null ? `${sample.generated_tokens} tokens` : null,
].filter(Boolean).join(" · ") || "Recorded sample";

function generationRows(file: ResultsFile, section: string) {
  const rows: ValidityRow[] = [];
  for (const [model, modelData] of entriesOf(file.data[section])) {
    for (const caseLabel of caseKeys(modelData, section)) {
      const result = modelData[caseLabel];
      (result.valid_samples || []).forEach((sample: JsonRecord[string], index: number) => rows.push({
        fileId: file.id, system: file.hostname, model, caseLabel,
        sample: index + 1, status: "valid", summary: measurementSummary(sample), errors: [],
      }));
      (result.invalid_runs || []).forEach((invalid: JsonRecord[string]) => rows.push({
        fileId: file.id, system: file.hostname, model, caseLabel,
        sample: invalid.run, status: "invalid", summary: "Excluded from aggregates",
        errors: Array.isArray(invalid.errors) ? invalid.errors : ["invalid_measurement"],
      }));
      if (!(result.valid_samples?.length) && !(result.invalid_runs?.length)) {
        const count = result.completed_runs ?? result.n_runs;
        if (count > 0) rows.push({
          fileId: file.id, system: file.hostname, model, caseLabel,
          sample: "—", status: "legacy", summary: `${count} completed; raw samples unavailable`,
          errors: [],
        });
      }
    }
  }
  return rows;
}

function llamaBenchRows(file: ResultsFile) {
  const rows: ValidityRow[] = [];
  for (const [model, modelData] of entriesOf(file.data.llamabench)) {
    for (const entry of [...(modelData.prefill_entries || []), ...(modelData.decode_entries || [])]) {
      const caseLabel = entry.n_gen
        ? `tg${entry.n_gen} @ pp${entry.n_depth || entry.n_prompt || 0}`
        : `pp${entry.n_prompt || 0}`;
      const samples = entry.ts_runs || entry.samples_ts || [];
      samples.forEach((value: number, index: number) => rows.push({
        fileId: file.id, system: file.hostname, model, caseLabel,
        sample: index + 1, status: "valid", summary: `${value} tok/s`, errors: [],
      }));
      if (!samples.length && entry.completed_reps > 0) rows.push({
        fileId: file.id, system: file.hostname, model, caseLabel,
        sample: "—", status: "legacy",
        summary: `${entry.completed_reps} completed; repetition samples unavailable`, errors: [],
      });
    }
  }
  return rows;
}

function scalarRunRows(file: ResultsFile, section: string) {
  const rows: ValidityRow[] = [];
  const sectionData = file.data[section] || {};
  for (const [model, modelData] of entriesOf(sectionData)) {
    if (section === "embeddings") {
      (modelData.runs || []).forEach((value: number, index: number) => rows.push({
        fileId: file.id, system: file.hostname, model, caseLabel: "document",
        sample: index + 1, status: "valid", summary: `${value} chunks/s`, errors: [],
      }));
    } else if (section === "images") {
      for (const [resolution, result] of entriesOf(modelData.resolutions)) {
        (result.runs || []).forEach((value: number, index: number) => rows.push({
          fileId: file.id, system: file.hostname, model, caseLabel: resolution,
          sample: index + 1, status: "valid", summary: `${value}s/image`, errors: [],
        }));
      }
    }
  }
  return rows;
}

export function buildValidityRows(files: ResultsFile[] | null, section: string): ValidityRow[] {
  if (!Array.isArray(files)) return [];
  if (GENERATION_SECTIONS.has(section)) {
    return files.flatMap(file => generationRows(file, section));
  }
  if (section === "llamabench") return files.flatMap(llamaBenchRows);
  if (section === "embeddings" || section === "images") {
    return files.flatMap(file => scalarRunRows(file, section));
  }
  return [];
}

export interface PreflightWarning {
  fileId: ResultsFile["id"], system: ResultsFile["hostname"], model: string,
  check: string, severity: string, detail: string,
}

export function buildPreflightWarnings(files: ResultsFile[] | null): PreflightWarning[] {
  if (!Array.isArray(files)) return [];
  return files.flatMap(file => entriesOf(file.data.preflight?.models).flatMap(([model, report]) =>
    (Array.isArray(report?.checks) ? report.checks : []).flatMap((check: JsonRecord[string]) =>
      ["warning", "workload_blocking", "hard_failure"].includes(String(check?.severity)) ? [{
        fileId: file.id, system: file.hostname, model,
        check: String(check.name || "compatibility"), severity: String(check.severity),
        detail: String(check.detail || "Compatibility check requires attention."),
      }] : []
    )
  ));
}

export function validitySummary(rows: { status: string }[]): Record<string, number> {
  return rows.reduce((summary, row) => {
    summary.total += 1;
    summary[row.status] = (summary[row.status] || 0) + 1;
    return summary;
  }, { total: 0, valid: 0, invalid: 0, legacy: 0 } as Record<string, number>);
}

const timestampMs = (value: unknown): number | null => {
  const parsed = typeof value === "string" ? Date.parse(value) : NaN;
  return Number.isFinite(parsed) ? parsed : null;
};

export function buildPauseSummaries(files: ResultsFile[] | null): { fileId: ResultsFile["id"], system: string | undefined, count: number, totalPausedSeconds: number, incomplete: boolean }[] {
  if (!Array.isArray(files)) return [];
  return files.flatMap(file => {
    const run = file?.data?.run;
    const transitions = Array.isArray(run?.pause?.control_transitions)
      ? run.pause.control_transitions : [];
    let pausedAt: number | null = null;
    let totalMs = 0;
    let incomplete = false;
    let count = 0;
    for (const transition of transitions) {
      const at = timestampMs(transition?.at);
      if (transition?.state === "paused") {
        count += 1;
        if (at == null) incomplete = true;
        else if (pausedAt == null) pausedAt = at;
      } else if (transition?.state === "running" && pausedAt != null && at != null) {
        if (at >= pausedAt) totalMs += at - pausedAt;
        else incomplete = true;
        pausedAt = null;
      }
    }
    if (pausedAt != null) {
      const finishedAt = timestampMs(run?.finished_at);
      if (finishedAt != null && finishedAt >= pausedAt) totalMs += finishedAt - pausedAt;
      else incomplete = true;
    }
    if (!count) return [];
    return [{
      fileId: file.id, system: file.hostname, count,
      totalPausedSeconds: totalMs / 1000, incomplete,
    }];
  });
}

export function formatPausedDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "unknown duration";
  const rounded = Math.round(seconds);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remaining = rounded % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${remaining}s`;
  return `${remaining}s`;
}
