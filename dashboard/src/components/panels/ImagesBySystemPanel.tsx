import type { RefObject } from "react";
import {
  buildImagesGroupedBarDataForResolution, buildImagesGroupedBarConfigs, getImageResolutions,
  buildImagesLineDataByRes, buildImagesLineConfigsByRes,
  getAllImageModels,
} from "../../utils/images";
import { sortBarData, findMostStrenuousKey, isNotNull } from "../../utils/shared";
import { SECTION_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import type { ResultsFile } from "../../types";
import styles from "../ChartPanel.module.css";

// Keep each system's bar comparisons within one resolution.
export default function ImagesBySystemPanel({ containerRef, files, enabledImageModels, chartWidth, logoSrc, isBar }: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], enabledImageModels: Set<string>,
  chartWidth: number, logoSrc?: string | null, isBar: boolean,
}) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));

  const systemGroups = files.map(f => {
    const resolutionCharts = getImageResolutions([f], allModels).map(resolution => {
      const barConfigs = buildImagesGroupedBarConfigs([f], resolution, enabledImageModels);
      const raw = buildImagesGroupedBarDataForResolution([f], resolution, enabledImageModels);
      const strenuousKey = findMostStrenuousKey(raw, barConfigs.map(bc => bc.dataKey));
      return { resolution, barConfigs, barData: strenuousKey ? sortBarData(raw, [strenuousKey], "asc") : raw };
    }).filter(chart => chart.barData.length > 0 && chart.barConfigs.length > 0);
    const lineData = buildImagesLineDataByRes(f, allModels);
    const lineConfigs = buildImagesLineConfigsByRes(f, allModels, lineData);
    const hasBar = resolutionCharts.length > 0;
    const hasLine = lineConfigs.length > 0;
    if (isBar ? !hasBar : !hasLine) return null;
    return { file: f, resolutionCharts, lineData, lineConfigs };
  }).filter(isNotNull);

  if (!systemGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS.images} data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {systemGroups.map(({ file: f, resolutionCharts, lineData, lineConfigs }) => (
        <div key={f.id} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{f.hostname}</div>
          {isBar ? resolutionCharts.map(({ resolution, barData, barConfigs }) => (
            <GroupedBarCard
              key={resolution}
              title={resolution}
              modelName={f.hostname}
              data={barData}
              barConfigs={barConfigs}
              colorSingleSeriesByCategory={false}
              xKey="systemLabel" yLabel="Sec / image" unit="sec"
              chartName="images_by_system" chartModel={`${f.hostname}_${resolution}`}
              logoSrc={logoSrc} direction="lower"
            />
          )) : (
            <ChartCard
              title="Image Generation"
              modelName={f.hostname}
              data={lineData} lineConfigs={lineConfigs}
              xKey="resLabel" xLabel="Resolution" yLabel="Sec / image" unit="sec"
              isMultiFile={false}
              chartName="images_by_system" chartModel={f.hostname}
              logoSrc={logoSrc} direction="lower"
            />
          )}
        </div>
      ))}
    </ChartGrid>
  );
}
