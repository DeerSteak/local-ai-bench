// `JsonRecord` lives in utils/shared.ts (not here) so this file stays `any`-free.
import type { JsonRecord } from "./utils/shared";

// A loaded results JSON file, as produced by parseResultsJSON.
export interface ResultsFile {
  id?: string | number;
  hostname?: string;
  backend?: string;
  engine?: string | null;
  engineVersion?: string | null;
  engineVersionRecorded?: boolean;
  data: JsonRecord;
}

// A chart-ready row: fixed label column(s) plus dynamic per-file/model keys.
export type ChartRow = Record<string, JsonRecord[string]>;

// A ResultsFile plus the UI-only metadata benchmark_dashboard.tsx attaches on load.
export interface DisplayFile extends ResultsFile {
  name: string;
  backend: string;
  os: string;
  wsl: boolean;
  ram_gb: number | null;
  version: string | null;
  timestamp: string | null;
  reliabilityWarning: string;
}

// StatsTable's column-sort state, cycled by SortTh/cycleSort.
export interface SortConfig {
  key: string;
  dir: 1 | -1;
}

// One line series. `strokeDasharray` distinguishes models when several files are
// compared, so it is absent on the single-file charts that color by model instead.
export interface LineConfig {
  dataKey: string;
  stroke: string;
  name: string;
  strokeDasharray?: string;
}

// One bar series.
export interface BarConfig {
  dataKey: string;
  name: string;
  fill: string;
}
