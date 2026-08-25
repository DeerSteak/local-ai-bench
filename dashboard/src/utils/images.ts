import {
  RES_ORDER, FALLBACK_COLORS, FILE_COLORS, MODEL_DASH_PATTERNS,
  IMAGE_DISPLAY_ORDER, IMAGE_BAR_COLORS, RES_COLORS,
} from "../constants";
import { getImageModelColor, imageModelLabel, entriesOf, valuesOf, lookup } from "./shared";
import { memoryFields } from "./memory";
import { powerFields } from "./power";
import type { BarConfig, ChartRow, LineConfig, ResultsFile } from "../types";

const isKnownRes = (res: unknown): res is string =>
  typeof res === "string" && RES_ORDER.includes(res);

// Bar-chart status label for one (file, model, resolution) cell in the
// Images charts, mirroring llm.js's getBarStatusLabel: "{res} - Timed Out" for
// the resolution at which benchmark.py's image generation run itself timed
// out, "{res} - Skipped" for every larger resolution consequently never
// attempted. Returns null for cells with real data.
export function getImageBarStatusLabel(file: ResultsFile, model: string, res: string): string | null {
  const timedOutRes = file.data.images?.[model]?.timed_out;
  if (isKnownRes(timedOutRes)) {
    const timedOutIdx = RES_ORDER.indexOf(timedOutRes);
    const resIdx = RES_ORDER.indexOf(res);
    if (resIdx === timedOutIdx) return `${res} - Timed Out`;
    if (resIdx > timedOutIdx) return `${res} - Skipped`;
  }
  return null;
}

// Return all image model keys from the loaded files, in canonical order
export function getAllImageModels(files: ResultsFile[]): string[] {
  const s = new Set<string>();
  for (const f of files) for (const m of Object.keys(f.data.images || {})) s.add(m);
  const known   = IMAGE_DISPLAY_ORDER.filter(m => s.has(m));
  const unknown = [...s].filter(m => !IMAGE_DISPLAY_ORDER.includes(m));
  return [...known, ...unknown];
}

export function getImageLabel(files: ResultsFile[], model: string): string {
  for (const f of files) {
    const d = f.data.images?.[model];
    if (d?.label) return d.label;
  }
  return imageModelLabel(model);
}

// Images: one chart per model. X = resolution, lines = files.
export function buildImagesDataForModel(files: ResultsFile[], model: string): ChartRow[] {
  const resSet = new Set<string>();
  for (const f of files)
    for (const r of Object.keys(f.data.images?.[model]?.resolutions || {})) resSet.add(r);
  const resLabels = RES_ORDER.filter(r => resSet.has(r));
  return resLabels.map(res => {
    const row: ChartRow = { resLabel: res };
    files.forEach((f, fi) => {
      const s = f.data.images?.[model]?.resolutions?.[res];
      if (s) row[`f${fi}`] = s.sec_per_image_mean;
    });
    return row;
  });
}

// Images: one bar chart per resolution. X = model, bars = files.
export function buildImagesDataForResolution(files: ResultsFile[], resolution: string, enabledImageModels: Set<string>): ChartRow[] {
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  return allModels
    .map(model => {
      const row: ChartRow = { modelLabel: getImageLabel(files, model) };
      files.forEach((f, fi) => {
        const s = f.data.images?.[model]?.resolutions?.[resolution];
        if (s) row[`f${fi}`] = s.sec_per_image_mean;
      });
      return row;
    })
    .filter(row => files.some((_, fi) => row[`f${fi}`] != null));
}

// Legacy: X = resolution, lines = image models (+ file distinction if multi)
export function buildImagesData(files: ResultsFile[], enabledImageModels: Set<string>): ChartRow[] {
  const isSingle = files.length === 1;
  const resSet = new Set<string>();
  for (const f of files)
    for (const md of valuesOf(f.data.images))
      for (const r of Object.keys(md.resolutions || {})) resSet.add(r);
  const resLabels = RES_ORDER.filter(r => resSet.has(r));

  return resLabels
    .map(res => {
      const row: ChartRow = { resLabel: res };
      files.forEach((f, fi) => {
        for (const [model, md] of entriesOf(f.data.images)) {
          if (!enabledImageModels.has(model) || !md.resolutions?.[res]) continue;
          const key = isSingle ? model : `f${fi}_${model}`;
          row[key] = md.resolutions[res].sec_per_image_mean;
        }
      });
      return row;
    })
    .filter(row => Object.keys(row).some(k => k !== "resLabel"));
}

export function buildImagesLineConfigs(files: ResultsFile[], data: ChartRow[], enabledImageModels: Set<string>): LineConfig[] {
  const isSingle = files.length === 1;
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  const configs = [];
  if (isSingle) {
    for (const m of allModels) {
      if (data.some(d => d[m] != null))
        configs.push({ dataKey: m, stroke: getImageModelColor(m), name: getImageLabel(files, m) });
    }
  } else {
    for (let fi = 0; fi < files.length; fi++) {
      const stroke = FILE_COLORS[fi % FILE_COLORS.length];
      allModels.forEach((m, mi) => {
        const dataKey = `f${fi}_${m}`;
        if (data.some(d => d[dataKey] != null))
          configs.push({
            dataKey,
            stroke,
            strokeDasharray: MODEL_DASH_PATTERNS[mi % MODEL_DASH_PATTERNS.length],
            name: `${files[fi].hostname} — ${getImageLabel(files, m)}`,
          });
      });
    }
  }
  return configs;
}

// Images bar chart: rows = files/systems, cols = image models
export function buildImagesGroupedBarDataForResolution(files: ResultsFile[], resolution: string, enabledImageModels: Set<string>): ChartRow[] {
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  return files
    .map(f => {
      const row: ChartRow = { systemLabel: f.hostname };
      for (const model of allModels) {
        const s = f.data.images?.[model]?.resolutions?.[resolution];
        if (s) row[model] = s.sec_per_image_mean;
        const status = getImageBarStatusLabel(f, model, resolution);
        if (status) row[`_status_${model}`] = status;
      }
      return row;
    })
    .filter(row => allModels.some(m => row[m] != null || row[`_status_${m}`] != null));
}

export function buildImagesGroupedBarConfigs(files: ResultsFile[], enabledImageModels: Set<string>): BarConfig[] {
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));
  return allModels.map((m, i) => ({
    dataKey: m,
    name: getImageLabel(files, m),
    fill: lookup(IMAGE_BAR_COLORS, m) || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
  }));
}

// Images bar chart by system: rows = models, cols = resolutions, for one file
export function buildImagesBarDataByModel(file: ResultsFile, models: string[]): ChartRow[] {
  return models
    .map(model => {
      const row: ChartRow = { modelLabel: getImageLabel([file], model) };
      const resData = file.data.images?.[model]?.resolutions || {};
      for (const res of RES_ORDER) {
        const s = resData[res];
        if (s) row[res] = s.sec_per_image_mean;
        const status = getImageBarStatusLabel(file, model, res);
        if (status) row[`_status_${res}`] = status;
      }
      return row;
    })
    .filter(row => RES_ORDER.some(res => row[res] != null || row[`_status_${res}`] != null));
}

export function buildImagesBarConfigsByModel(file: ResultsFile, models: string[]): BarConfig[] {
  const resSet = new Set<string>();
  for (const model of models) {
    for (const res of Object.keys(file.data.images?.[model]?.resolutions || {})) resSet.add(res);
    const timedOutRes = file.data.images?.[model]?.timed_out;
    if (timedOutRes) resSet.add(timedOutRes);
  }
  return RES_ORDER
    .filter(res => resSet.has(res))
    .map((res, i) => ({
      dataKey: res,
      name: res,
      fill: lookup(RES_COLORS, res) || FALLBACK_COLORS[i % FALLBACK_COLORS.length],
    }));
}

// Images line chart by system: rows = resolutions, one line per model, for one file
export function buildImagesLineDataByRes(file: ResultsFile, models: string[]): ChartRow[] {
  const resSet = new Set<string>();
  for (const model of models)
    for (const res of Object.keys(file.data.images?.[model]?.resolutions || {})) resSet.add(res);
  const resLabels = RES_ORDER.filter(r => resSet.has(r));
  return resLabels.map(res => {
    const row: ChartRow = { resLabel: res };
    for (const model of models) {
      const s = file.data.images?.[model]?.resolutions?.[res];
      if (s) row[model] = s.sec_per_image_mean;
    }
    return row;
  });
}

export function buildImagesLineConfigsByRes(file: ResultsFile, models: string[], data: ChartRow[]): LineConfig[] {
  return models
    .filter(m => data.some(row => row[m] != null))
    .map(m => ({ dataKey: m, stroke: getImageModelColor(m), name: getImageLabel([file], m) }));
}

export function flattenImageData(files: ResultsFile[]): ChartRow[] {
  return files.flatMap(f =>
    entriesOf(f.data.images).flatMap(([model, md]) =>
      entriesOf(md.resolutions).map(([res, s]) => ({
        _fileId: f.id, model,
        modelLabel: md.label || model,
        steps: md.steps, res,
        sec_mean: s.sec_per_image_mean,
        sec_stdev: s.sec_per_image_stdev,
        n_runs: s.n_runs,
        ...memoryFields(md),
        ...powerFields(md),
      }))
    )
  );
}
