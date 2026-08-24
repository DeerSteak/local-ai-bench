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
