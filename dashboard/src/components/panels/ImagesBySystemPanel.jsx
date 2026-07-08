import {
  buildImagesBarDataByModel, buildImagesBarConfigsByModel,
  buildImagesLineDataByRes, buildImagesLineConfigsByRes,
  getAllImageModels, sortBarData, findMostStrenuousKey,
} from "../../utils";
import { SECTION_LABELS } from "../../constants";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Group By: System, Images section — one card per system, models as
// bars/lines within it.
export default function ImagesBySystemPanel({ containerRef, files, enabledImageModels, chartWidth, logoSrc, isBar }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const allModels = getAllImageModels(files).filter(m => enabledImageModels.has(m));

  const systemGroups = files.map(f => {
    const rawBarData = buildImagesBarDataByModel(f, allModels);
    const barConfigs = buildImagesBarConfigsByModel(f, allModels);
    const lineData = buildImagesLineDataByRes(f, allModels);
    const lineConfigs = buildImagesLineConfigsByRes(f, allModels, lineData);
    const hasBar = rawBarData.length > 0 && barConfigs.length > 0;
    const hasLine = lineConfigs.length > 0;
    if (isBar ? !hasBar : !hasLine) return null;
    const strenuousKey = findMostStrenuousKey(rawBarData, barConfigs.map(bc => bc.dataKey));
    const barData = strenuousKey ? sortBarData(rawBarData, [strenuousKey], "asc") : rawBarData;
    return { file: f, barData, barConfigs, lineData, lineConfigs };
  }).filter(Boolean);

  if (!systemGroups.length) {
    return <EmptyState style={containerStyle}>No {SECTION_LABELS.images} data in the loaded file(s)</EmptyState>;
  }

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      {systemGroups.map(({ file: f, barData, barConfigs, lineData, lineConfigs }) => (
        <div key={f.id} className={styles.modelGroup}>
          <div className={styles.modelGroupTitle}>{f.hostname}</div>
          {isBar ? (
            <GroupedBarCard
              title="Image Generation"
              modelName={f.hostname}
              data={barData}
              barConfigs={barConfigs}
              xKey="modelLabel" yLabel="Sec / image" unit="sec"
              chartName="images_by_system" chartModel={f.hostname}
              logoSrc={logoSrc} direction="lower"
            />
          ) : (
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
