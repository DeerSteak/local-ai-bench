import { MAX_FILES } from "../constants";

// The staged-results manifest this reads — see dashboard/stage_selected_results.mjs.
interface SelectedResultsManifest {
  files: { name: string, url: string }[];
}

// Only the subset of Response/fetch this actually calls — narrower than the real
// DOM `fetch` type so tests can pass minimal mock responses without a full Response shape.
type MinimalResponse = {
  ok: boolean,
  json?: () => Promise<SelectedResultsManifest>,
  text?: () => Promise<string>,
};
type MinimalFetch = (url: string, init?: RequestInit) => Promise<MinimalResponse>;

export async function fetchSelectedResultFiles(search: string, fetchFn: MinimalFetch = fetch) {
  if (new URLSearchParams(search).get("autoload") !== "1") return [];
  const manifestResponse = await fetchFn("/__selected_results__.json", { cache: "no-store" });
  if (!manifestResponse.ok) throw new Error("Selected results are no longer available.");
  const manifest = await manifestResponse.json!();
  if (!Array.isArray(manifest.files) || manifest.files.length > MAX_FILES) {
    throw new Error("Selected result manifest is invalid.");
  }
  return Promise.all(manifest.files.map(async entry => {
    if (typeof entry?.name !== "string" || !/^\/__selected_results__\/\d+\.json$/.test(entry?.url)) {
      throw new Error("Selected result manifest is invalid.");
    }
    const response = await fetchFn(entry.url, { cache: "no-store" });
    if (!response.ok) throw new Error(`${entry.name}: Could not read this file.`);
    const text = await response.text!();
    return { name: entry.name, text: async () => text };
  }));
}
