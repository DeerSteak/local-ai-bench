import {
  getAllAccuracyModels, modelLabel,
  buildAccuracyGroupedBarData, buildAccuracyGroupedBarConfigs,
  buildAccuracyCategoryData, buildAccuracyCategoryConfigs,
  buildAccuracyDifficultyData,
  buildAccuracyTimeoutData,
  sortBarData, findMostStrenuousKey,
} from "../../utils";
import { ACCURACY_TEST_LABELS, ACCURACY_TIMEOUT_BAR_CONFIGS } from "../../constants";
import { GroupedBarCard } from "../charts/ChartCards";
import { EmptyState, ChartGrid } from "./shared";
import styles from "../ChartPanel.module.css";

// Accuracy section (picked via accuracyTest): one overall
// bar chart (accuracy % per model, systems as bars), then one per-category
// breakdown chart per model, then — when at least one timeout or likely loop
// occurs — a diagnostics chart of timed_out_count/likely_loop_count. No
// "Group By: System" or line-chart variant here, unlike LLM/Images/Embeddings —
// accuracy is a single scalar per model rather than a metric swept across
// context lengths or resolutions, so there's no second axis to pivot on.
export default function AccuracyPanel({ containerRef, files, accuracyTest, enabledModels, chartWidth, logoSrc }) {
  const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
  const testLabel = ACCURACY_TEST_LABELS[accuracyTest];
  const allModels = getAllAccuracyModels(files, accuracyTest).filter(m => enabledModels.has(m));

  const overallConfigs = buildAccuracyGroupedBarConfigs(files, accuracyTest, enabledModels);
  const overallRaw = buildAccuracyGroupedBarData(files, accuracyTest, enabledModels);

  if (!allModels.length || !overallRaw.length) {
    return <EmptyState style={containerStyle}>No {testLabel} accuracy data in the loaded file(s)</EmptyState>;
  }

  const modelKeys = overallConfigs.map(bc => bc.dataKey);
  const strenuousKey = findMostStrenuousKey(overallRaw, modelKeys);
  const overallData = strenuousKey ? sortBarData(overallRaw, [strenuousKey], "desc") : overallRaw;

  const categoryConfigs = buildAccuracyCategoryConfigs(files);
  const timeoutData = buildAccuracyTimeoutData(files, accuracyTest, enabledModels);

  return (
    <ChartGrid containerRef={containerRef} style={containerStyle}>
      <GroupedBarCard
        title={`${testLabel} Accuracy`}
        modelName="Overall"
        data={overallData}
        barConfigs={overallConfigs}
        xKey="systemLabel" yLabel="Accuracy (%)" unit="pct"
        chartName={`${accuracyTest}-accuracy`}
        logoSrc={logoSrc} direction="higher"
      />

      {allModels.map(model => {
        const catData = buildAccuracyCategoryData(files, accuracyTest, model);
        const difficultyData = buildAccuracyDifficultyData(files, accuracyTest, model);
        if (!catData.length) return null;
        return (
          <div key={model} className={styles.modelGroup}>
            <div className={styles.modelGroupTitle}>{modelLabel(model)}</div>
            <GroupedBarCard
              title={`${testLabel} Accuracy by Category`}
              modelName={modelLabel(model)}
              data={catData}
              barConfigs={categoryConfigs}
              xKey="categoryLabel" yLabel="Accuracy (%)" unit="pct"
              chartName={`${accuracyTest}-category`} chartModel={model}
              logoSrc={logoSrc} direction="higher"
            />
            {difficultyData.length > 0 && (
              <GroupedBarCard
                title={`${testLabel} Accuracy by Difficulty`}
                modelName={modelLabel(model)}
                data={difficultyData}
                barConfigs={categoryConfigs}
                xKey="difficultyLabel" yLabel="Accuracy (%)" unit="pct"
                chartName={`${accuracyTest}-difficulty`} chartModel={model}
                logoSrc={logoSrc} direction="higher"
              />
            )}
          </div>
        );
      })}

      {timeoutData.length > 0 && (
        <GroupedBarCard
          title={`${testLabel} Timeouts & Likely Loops`}
          modelName="Diagnostics"
          data={timeoutData}
          barConfigs={ACCURACY_TIMEOUT_BAR_CONFIGS}
          xKey="rowLabel" yLabel="Questions" unit="count"
          chartName={`${accuracyTest}-timeouts`}
          logoSrc={logoSrc} direction="lower"
        />
      )}
    </ChartGrid>
  );
}
