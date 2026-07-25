import {
  buildLLMBarDataByModel, buildLLMBarConfigsByModel,
  buildLLMLineDataByCtx, buildLLMLineConfigsByCtx,
  getLLMModelsWithSectionResults,
} from "../../utils/llm";
import {
  sortBarData, getModelSizeTier, getSkipInfo, modelLabel, deriveTtftUnit, hasValueOrStatus,
} from "../../utils/shared";
import { SECTION_LABELS, SIZE_TIER_ORDER } from "../../constants";
import BySystemPanel from "./BySystemPanel";

// Group By: System, LLM / LLM Conversation section — resolves this section's own
// ctx-keyed data into BySystemPanel's generic { tier, metrics } shape.
export default function LLMBySystemPanel({ containerRef, files, section, enabledModels, chartWidth, logoSrc, isBar, isSplit }) {
  const allModels = getLLMModelsWithSectionResults(files, section).filter(m => enabledModels.has(m));
  const isConv = section === "llm_conversation";
  const titleSuffix = isConv ? " (Conversation)" : "";
  const chartNamePrefix = isConv ? "conv_" : "";

  const modelGroupSpecs = isSplit
    ? SIZE_TIER_ORDER
        .map(tier => ({ tier, models: allModels.filter(m => getModelSizeTier(m) === tier) }))
        .filter(g => g.models.length > 0)
    : [{ tier: null, models: allModels }];

  const systemGroups = files.map(f => {
    // Resolved once per file from every enabled model, regardless of tier — a tier
    // split shouldn't change ms vs. sec, only which models are shown.
    const allTtftBar = buildLLMBarDataByModel(f, allModels, "ttft", section);
    const allTtftVals = allTtftBar
      .flatMap(row => Object.entries(row).filter(([k]) => k !== "modelLabel" && !k.startsWith("_status_")).map(([, v]) => v))
      .filter(v => v != null);
    const { ttftUnit, ttftYLabel } = deriveTtftUnit(allTtftVals);

    const groups = modelGroupSpecs.map(({ tier, models }) => {
      const rawTpsBarData = buildLLMBarDataByModel(f, models, "tps", section);
      const rawTtftBarData = buildLLMBarDataByModel(f, models, "ttft", section);
      const tpsBarConfigs = buildLLMBarConfigsByModel(f, models, section).filter(bc => hasValueOrStatus(rawTpsBarData, bc.dataKey));
      const ttftBarConfigs = buildLLMBarConfigsByModel(f, models, section).filter(bc => hasValueOrStatus(rawTtftBarData, bc.dataKey));
      const tpsBarData = sortBarData(rawTpsBarData, tpsBarConfigs.map(bc => bc.dataKey), "desc");
      const ttftBarData = sortBarData(rawTtftBarData, ttftBarConfigs.map(bc => bc.dataKey), "asc");

      const tpsLineData = buildLLMLineDataByCtx(f, models, "tps", section);
      const ttftLineData = buildLLMLineDataByCtx(f, models, "ttft", section);
      const tpsLineConfigs = buildLLMLineConfigsByCtx(models, tpsLineData);
      const ttftLineConfigs = buildLLMLineConfigsByCtx(models, ttftLineData);

      const hasTps = isBar ? tpsBarConfigs.length > 0 : tpsLineConfigs.length > 0;
      const hasTtft = isBar ? ttftBarConfigs.length > 0 : ttftLineConfigs.length > 0;

      const metrics = [];
      if (hasTps) metrics.push({
        key: "tps", title: `Tokens/sec${titleSuffix}`, yLabel: "Tokens/sec", unit: "tps", direction: "higher",
        xKey: "ctxLabel", xLabel: "Context Length", chartName: `${chartNamePrefix}tps`,
        barData: tpsBarData, barConfigs: tpsBarConfigs, lineData: tpsLineData, lineConfigs: tpsLineConfigs,
      });
      if (hasTtft) metrics.push({
        key: "ttft", title: `Time to First Token${titleSuffix}`, yLabel: ttftYLabel, unit: ttftUnit, direction: "lower",
        xKey: "ctxLabel", xLabel: "Context Length", chartName: `${chartNamePrefix}ttft`,
        barData: ttftBarData, barConfigs: ttftBarConfigs, lineData: ttftLineData, lineConfigs: ttftLineConfigs,
      });
      if (!metrics.length) return null;
      return { tier, metrics };
    }).filter(Boolean);

    const skipEntries = allModels
      .map(m => ({ model: m, info: getSkipInfo(f, m, section) }))
      .filter(e => e.info)
      .map(e => ({ key: e.model, label: `${modelLabel(e.model)}: Skipped — ${e.info.detail}` }));
    if (!groups.length && !skipEntries.length) return null;
    return { file: f, groups, skipEntries };
  }).filter(Boolean);

  return (
    <BySystemPanel
      containerRef={containerRef} chartWidth={chartWidth} logoSrc={logoSrc}
      isBar={isBar} emptyLabel={`No ${SECTION_LABELS[section]} data in the loaded file(s)`}
      systemGroups={systemGroups}
    />
  );
}
