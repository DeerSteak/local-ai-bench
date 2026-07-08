import {
  getAllImageModels,
  buildImagesGroupedBarDataForResolution, buildImagesGroupedBarConfigs,
  buildImagesData, buildImagesLineConfigs,
  sortBarData, findMostStrenuousKey,
} from "../../utils";
import { RES_ORDER } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";

// Group By: Model, Images section — one card per resolution (bar) or a
// single combined chart (line), systems/models as bars/lines within it.
export default function ImagesPanel({ containerRef, files, enabledImageModels, chartWidth, logoSrc, isBar, isMultiFile }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));

  const resSet = new Set();
  for (const f of files)
    for (const model of allModels)
      for (const res of Object.keys(f.data.images?.[model]?.resolutions || {})) resSet.add(res);
  const resolutions = RES_ORDER.filter(r => resSet.has(r));

  if (!resolutions.length || !allModels.length) {
    return <EmptyState style={containerStyle}>No Images data in the loaded file(s)</EmptyState>;
  }

  const groupedBarConfigs = buildImagesGroupedBarConfigs(files, enabledImageModels);
  const modelKeys = groupedBarConfigs.map(bc => bc.dataKey);
  const lineData = buildImagesData(files, enabledImageModels);
  const lineConfigs = buildImagesLineConfigs(files, lineData, enabledImageModels);

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {isBar ? resolutions.map(res => {
        const raw = buildImagesGroupedBarDataForResolution(files, res, enabledImageModels);
        if (!raw.length) return null;
        const strenuousKey = findMostStrenuousKey(raw, modelKeys);
        const data = strenuousKey ? sortBarData(raw, [strenuousKey], "asc") : raw;
        return (
          <GroupedBarCard
            key={res}
            title={res}
            modelName="Image Generation"
            data={data}
            barConfigs={groupedBarConfigs}
            xKey="systemLabel" yLabel="Sec / image" unit="sec"
            chartName="images" chartModel={res}
            logoSrc={logoSrc} direction="lower"
          />
        );
      }) : (
        <ChartCard
          title="Image Generation"
          data={lineData}
          lineConfigs={lineConfigs}
          xKey="resLabel" xLabel="Resolution" yLabel="Sec / image" unit="sec"
          isMultiFile={isMultiFile}
          chartName="images"
          logoSrc={logoSrc} direction="lower"
        />
      )}
    </ChartGrid>
  );
}
