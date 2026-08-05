import { MAX_FILES } from "../constants";

export async function fetchSelectedResultFiles(search, fetchFn = fetch) {
  if (new URLSearchParams(search).get("autoload") !== "1") return [];
  const manifestResponse = await fetchFn("/__selected_results__.json", { cache: "no-store" });
  if (!manifestResponse.ok) throw new Error("Selected results are no longer available.");
  const manifest = await manifestResponse.json();
  if (!Array.isArray(manifest.files) || manifest.files.length > MAX_FILES) {
    throw new Error("Selected result manifest is invalid.");
  }
  return Promise.all(manifest.files.map(async entry => {
    if (typeof entry?.name !== "string" || !/^\/__selected_results__\/\d+\.json$/.test(entry?.url)) {
      throw new Error("Selected result manifest is invalid.");
    }
    const response = await fetchFn(entry.url, { cache: "no-store" });
    if (!response.ok) throw new Error(`${entry.name}: Could not read this file.`);
    const text = await response.text();
    return { name: entry.name, text: async () => text };
  }));
}
