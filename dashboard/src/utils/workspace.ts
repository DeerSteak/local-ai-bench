import type { DisplayFile } from "../types";
import type { JsonRecord } from "./shared";

export interface WorkspaceView {
  section: string;
  accuracy_test: string;
  enabled_models: string[];
  enabled_image_models: string[];
  enabled_embedding_models: string[];
  hostname_overrides: Record<string, string>;
}

export interface WorkspaceSelection {
  artifact_type: "workspace_selection";
  schema_version: 1;
  results: { name: string, sha256: string }[];
  baseline_sha256: string | null;
  view: WorkspaceView;
  acceptance_policy: JsonRecord | null;
  recommendation: JsonRecord | null;
}

export function isWorkspaceSelection(value: unknown): value is WorkspaceSelection {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const selection = value as Partial<WorkspaceSelection>;
  return selection.artifact_type === "workspace_selection" && selection.schema_version === 1
    && Array.isArray(selection.results) && selection.results.length > 0
    && selection.results.every(item => typeof item?.name === "string"
      && typeof item?.sha256 === "string" && /^[0-9a-f]{64}$/.test(item.sha256))
    && selection.view != null && typeof selection.view === "object";
}

export function buildWorkspaceSelection(
  files: DisplayFile[], baselineId: string | null, view: WorkspaceView,
  acceptancePolicy: JsonRecord | null = null, recommendation: JsonRecord | null = null,
): WorkspaceSelection {
  if (!files.length) throw new Error("Workspace selection requires at least one result.");
  const digests = files.map(file => file.sourceSha256 ?? "");
  if (new Set(digests).size !== digests.length || digests.some(digest => !/^[0-9a-f]{64}$/.test(digest))) {
    throw new Error("Workspace selection requires distinct valid result identities.");
  }
  const baseline = baselineId == null ? null : files.find(file => String(file.id) === baselineId);
  if (baselineId != null && !baseline) throw new Error("Workspace baseline is not selected.");
  return {
    artifact_type: "workspace_selection",
    schema_version: 1,
    results: files.map(file => ({ name: file.name, sha256: file.sourceSha256 as string })),
    baseline_sha256: baseline?.sourceSha256 ?? null,
    view,
    acceptance_policy: acceptancePolicy,
    recommendation,
  };
}

export function downloadWorkspaceSelection(selection: WorkspaceSelection, filename = "workspace_selection.json") {
  const blob = new Blob([`${JSON.stringify(selection, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.download = filename;
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
}

type ExportFormat = "html" | "pdf" | "bundle";
type ExportResponse = {
  ok: boolean,
  blob: () => Promise<Blob>,
  json: () => Promise<JsonRecord>,
};
type WorkspaceFetch = (input: string, init?: RequestInit) => Promise<ExportResponse>;

export async function requestWorkspaceExport(
  selection: WorkspaceSelection, files: DisplayFile[], format: ExportFormat,
  fetchFn: WorkspaceFetch = fetch as WorkspaceFetch,
): Promise<{ blob: Blob, filename: string }> {
  if (files.some(file => typeof file.sourceText !== "string")) {
    throw new Error("Original result text is unavailable for workspace export.");
  }
  const config = await fetchFn("/__workspace_config__.json", { cache: "no-store" });
  if (!config.ok) throw new Error("Workspace export service is unavailable.");
  const token = (await config.json()).token;
  if (typeof token !== "string" || !token) throw new Error("Workspace export token is unavailable.");
  const response = await fetchFn("/api/workspace/export", {
    method: "POST",
    headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      format, selection,
      results: files.map(file => ({ name: file.name, text: file.sourceText as string })),
    }),
  });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.error || "Workspace export failed.");
  }
  const filenames: Record<ExportFormat, string> = {
    html: "decision.html", pdf: "decision.pdf", bundle: "workspace.labworkspace",
  };
  return { blob: await response.blob(), filename: filenames[format] };
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.download = filename;
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
}

export async function requestWorkspaceEvaluation(
  selection: WorkspaceSelection, files: DisplayFile[],
  fetchFn: WorkspaceFetch = fetch as WorkspaceFetch,
): Promise<{ acceptance: JsonRecord | null, recommendation: JsonRecord | null }> {
  if (files.some(file => typeof file.sourceText !== "string")) {
    throw new Error("Original result text is unavailable for workspace evaluation.");
  }
  const config = await fetchFn("/__workspace_config__.json", { cache: "no-store" });
  const token = config.ok ? (await config.json()).token : null;
  if (typeof token !== "string" || !token) throw new Error("Workspace evaluation service is unavailable.");
  const response = await fetchFn("/api/workspace/evaluate", {
    method: "POST",
    headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      selection,
      results: files.map(file => ({ name: file.name, text: file.sourceText as string })),
    }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Workspace evaluation failed.");
  return {
    acceptance: body.acceptance ?? null,
    recommendation: body.recommendation ?? null,
  };
}
