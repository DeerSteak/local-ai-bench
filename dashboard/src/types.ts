// `JsonRecord` lives in utils/shared.ts (not here) so this file stays `any`-free.
import type { JsonRecord } from "./utils/shared";

// A loaded results JSON file, as produced by parseResultsJSON.
export interface ResultsFile {
  id?: string | number;
  hostname?: string;
  engine?: string;
  data: JsonRecord;
}

// A chart-ready row: fixed label column(s) plus dynamic per-file/model keys.
export type ChartRow = Record<string, JsonRecord[string]>;
