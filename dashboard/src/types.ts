// Shared structural types for the dashboard's utils layer.
//
// `JsonRecord` (the one sanctioned dynamic-JSON type this file builds on) lives in
// utils/shared.ts, not here — see the note there. It's imported rather than
// redeclared so this file has no `any` of its own for the dashboard's `any`
// ratchet hook to flag (see docs/release-policy.md#any-ratchet-hook).
import type { JsonRecord } from "./utils/shared";

// A loaded results JSON file, as produced by parseResultsJSON + the file
// picker's own id/engine bookkeeping.
export interface ResultsFile {
  id?: string | number;
  hostname?: string;
  engine?: string;
  data: JsonRecord;
}

// A chart-ready row: fixed label column(s) plus per-file/per-model dynamic
// keys (`f0`, `f0_out128`, `_status_<key>`, ...) assigned after construction.
export type ChartRow = Record<string, JsonRecord[string]>;
