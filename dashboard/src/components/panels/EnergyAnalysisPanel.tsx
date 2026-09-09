import { buildEnergyAnalysis, energyChartSeries, ENERGY_SECTIONS } from "../../utils/energyAnalysis";
import { ChartCard, GroupedBarCard } from "../charts/ChartCards";
import { ChartGrid } from "./shared";
import type { ResultsFile } from "../../types";

export default function EnergyAnalysisPanel({ files, section, enabledModels, bySystem, chartWidth, logoSrc }: {
  files: ResultsFile[], section: string, enabledModels: Set<string>, bySystem: boolean,
  chartWidth: number, logoSrc?: string | null,
}) {
  if (!ENERGY_SECTIONS.includes(section) || !files.length) return null;
  const { groups, notices } = buildEnergyAnalysis(files, section, enabledModels, bySystem);
  return <ChartGrid style={{ width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth }}>
    <div className="card">
      <h2 id="energy-analysis">Energy analysis</h2>
      <p>Energy per unit of work and measured joules. Figures use absolute units; different power scopes and measurement windows stay separate.</p>
      {section === "images" && <p>Image energy is recorded per model across its measured resolutions, so these figures describe the combined workload.</p>}
      {!groups.length && <p>No usable energy measurements in the selected results.</p>}
      {notices.length > 0 && <details open={!groups.length}>
        <summary>Energy availability ({notices.length})</summary>
        <ul>{notices.map(notice => <li key={notice}>{notice}</li>)}</ul>
      </details>}
    </div>
    {groups.flatMap((group, groupIndex) => [false, true].map(energy => {
      const configs = energyChartSeries(group, energy);
      if (!configs.length) return null;
      const title = energy ? "Measured Energy" : "Energy per Work";
      const unit = "energy";
      const yLabel = energy ? "Joules" : group.unit;
      const chartName = `${section}_${energy ? "energy_joules" : "energy_per_work"}`;
      const direction = "lower";
      const identity = `${group.model}_${groupIndex + 1}`;
      return group.data.length === 1 ? <GroupedBarCard
        key={`${group.id}_${energy}`} title={title} modelName={group.model} caption={group.description}
        data={group.data} barConfigs={configs.map(config => ({
          dataKey: config.dataKey, name: config.name, fill: config.stroke || "#0969da",
        }))} xKey="caseLabel" yLabel={yLabel} unit={unit}
        chartName={chartName} chartModel={identity} logoSrc={logoSrc} direction={direction}
        colorSingleSeriesByCategory={false}
      /> : <ChartCard
        key={`${group.id}_${energy}`} title={title} modelName={group.model} caption={group.description}
        data={group.data} lineConfigs={configs} xKey="caseLabel"
        xLabel={section === "llamabenchconc" ? "Concurrency" : "Context / prompt tokens"}
        yLabel={yLabel} unit={unit} isMultiFile={configs.length > 1}
        chartName={chartName} chartModel={identity} logoSrc={logoSrc} direction={direction}
        connectNulls={false}
      />;
    }))}
  </ChartGrid>;
}
