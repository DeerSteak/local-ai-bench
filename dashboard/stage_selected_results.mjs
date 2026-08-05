import { basename, extname, join, resolve } from "node:path";
import { copyFileSync, existsSync, mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const SELECTED_RESULTS_LIMIT = 6;

export function stageSelectedResults(distDirectory, sourcePaths) {
  const dist = resolve(distDirectory);
  const selectedDirectory = join(dist, "__selected_results__");
  const manifestPath = join(dist, "__selected_results__.json");
  if (sourcePaths.length > SELECTED_RESULTS_LIMIT) {
    throw new Error(`Select no more than ${SELECTED_RESULTS_LIMIT} result files.`);
  }
  const sources = sourcePaths.map(sourcePath => {
    const source = resolve(sourcePath);
    if (extname(source).toLowerCase() !== ".json" || !existsSync(source) || !statSync(source).isFile()) {
      throw new Error(`Result file not found: ${sourcePath}`);
    }
    return source;
  });
  rmSync(selectedDirectory, { recursive: true, force: true });
  rmSync(manifestPath, { force: true });
  if (!sources.length) return [];
  mkdirSync(selectedDirectory, { recursive: true });
  const entries = sources.map((source, index) => {
    const filename = `${index}.json`;
    copyFileSync(source, join(selectedDirectory, filename));
    return { name: basename(source), url: `/__selected_results__/${filename}` };
  });
  writeFileSync(manifestPath, JSON.stringify({ files: entries }), "utf8");
  return entries;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  stageSelectedResults(process.argv[2], process.argv.slice(3));
}
