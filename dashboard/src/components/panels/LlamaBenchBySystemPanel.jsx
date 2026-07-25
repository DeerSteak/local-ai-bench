import {
  buildLlamaBenchBarDataByModel, buildLlamaBenchBarConfigsByModel,
  buildLlamaBenchLineDataByCheckpoint, buildLlamaBenchLineConfigsByCheckpoint,
  llamaBenchTgValuesByModel,
} from "../../utils/llamabench";
import { getAllLLMModels } from "../../utils/llm";
import { sortBarData, getModelSizeTier, modelLabel } from "../../utils/shared";
import { SIZE_TIER_ORDER } from "../../constants";
import BySystemPanel from "./BySystemPanel";

// Group By: System, llama-bench section — resolves llama-bench's dynamic pp/tg
// checkpoint data into BySystemPanel's generic { tier, metrics } shape (one metric per
// generation size actually swept, so pp size can be each chart's only axis/bar
// dimension), reusing the same tier-split renderer (and Model Sizes toggle) as
// LLMBySystemPanel.
export default function LlamaBenchBySystemPanel({ containerRef, files, enabledModels, chartWidth, logoSrc, isBar, isSplit }) {
  const allModels = getAllLLMModels(files).filter(m => enabledModels.has(m) && files.some(f => f.data.llamabench?.[m]));

  const modelGroupSpecs = isSplit
    ? SIZE_TIER_ORDER
        .map(tier => ({ tier, models: allModels.filter(m => getModelSizeTier(m) === tier) }))
        .filter(g => g.models.length > 0)
    : [{ tier: null, models: allModels }];

  const systemGroups = files.map(f => {
    const groups = modelGroupSpecs.map(({ tier, models }) => {
      const tgValues = llamaBenchTgValuesByModel(f, models);
      const metrics = tgValues.map(tg => {
        const rawBarData = buildLlamaBenchBarDataByModel(f, models, tg);
        const barConfigs = buildLlamaBenchBarConfigsByModel(f, models, tg).filter(bc => rawBarData.some(r => r[bc.dataKey] != null));
        const barData = sortBarData(rawBarData, barConfigs.map(bc => bc.dataKey), "desc");
        const lineData = buildLlamaBenchLineDataByCheckpoint(f, models, tg);
        const lineConfigs = buildLlamaBenchLineConfigsByCheckpoint(models, lineData);

        const hasData = isBar ? barConfigs.length > 0 : lineConfigs.length > 0;
        if (!hasData) return null;
        return {
          key: `tg${tg}`, title: `llama-bench Throughput — tg${tg}`, yLabel: "Tokens/sec", unit: "tps", direction: "higher",
          xKey: "promptLabel", xLabel: "Prompt Size", chartName: `llamabench_tg${tg}`,
          barData, barConfigs, lineData, lineConfigs,
        };
      }).filter(Boolean);
      if (!metrics.length) return null;
      return { tier, metrics };
    }).filter(Boolean);

    const skipEntries = allModels
      .filter(m => f.data.llamabench?.[m]?.error)
      .map(m => ({ key: m, label: `${modelLabel(m)}: ${f.data.llamabench[m].error}` }));
    if (!groups.length && !skipEntries.length) return null;
    return { file: f, groups, skipEntries };
  }).filter(Boolean);

  return (
    <BySystemPanel
      containerRef={containerRef} chartWidth={chartWidth} logoSrc={logoSrc}
      isBar={isBar} emptyLabel="No llama-bench data in the loaded file(s)"
      systemGroups={systemGroups}
    />
  );
}
