import type { RefObject } from "react";
import {
  getAllImageModels, getImageResolutions,
  buildImagesGroupedBarDataForResolution, buildImagesGroupedBarConfigs,
  buildImagesData, buildImagesLineConfigs,
} from "../../utils/images";
import { sortBarData, findMostStrenuousKey } from "../../utils/shared";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import type { ResultsFile } from "../../types";

// Group By: Model, Images section — one card per resolution (bar) or a
// single combined chart (line), systems/models as bars/lines within it.
export default function ImagesPanel({ containerRef, files, enabledImageModels, chartWidth, logoSrc, isBar, isMultiFile }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledImageModels: Set<string>,
  chartWidth: number, logoSrc?: string | null, isBar: boolean, isMultiFile: boolean,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));

  const resolutions = getImageResolutions(files, allModels);

  if (!resolutions.length || !allModels.length) {
    return <EmptyState style={containerStyle}>No Images data in the loaded file(s)</EmptyState>;
  }

  const lineData = buildImagesData(files, enabledImageModels);
  const lineConfigs = buildImagesLineConfigs(files, lineData, enabledImageModels);

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {isBar ? resolutions.map(res => {
        const groupedBarConfigs = buildImagesGroupedBarConfigs(files, res, enabledImageModels);
        const modelKeys = groupedBarConfigs.map(bc => bc.dataKey);
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
            colorSingleSeriesByCategory={false}
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
