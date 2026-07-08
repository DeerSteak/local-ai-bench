import LLMBySystemPanel from "./panels/LLMBySystemPanel";
import ImagesBySystemPanel from "./panels/ImagesBySystemPanel";
import EmbeddingsBySystemPanel from "./panels/EmbeddingsBySystemPanel";
import LLMByModelPanel from "./panels/LLMByModelPanel";
import ImagesPanel from "./panels/ImagesPanel";
import EmbeddingsPanel from "./panels/EmbeddingsPanel";
import { EmptyState } from "./panels/shared";

// Picks the right panel for the current section / Group By / Chart Style
// selection. Each panel owns its own data wiring and empty-state handling —
// see components/panels/*.jsx. Shared chart-rendering primitives (the actual
// recharts wrappers) live in components/charts/ChartCards.jsx.
export default function ChartPanel({
  containerRef, files, section,
  enabledModels, enabledImageModels, chartWidth, logoSrc, chartStyle, groupBy, sizeSplit,
}) {
  const isBar = chartStyle === "bar";
  const isBySystem = groupBy === "system";
  const isSplit = sizeSplit === "tiers";
  const isMultiFile = files.length > 1;

  if (files.length === 0) {
    const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
    return <EmptyState style={containerStyle}>Drop a results JSON file above to get started</EmptyState>;
  }

  if (isBySystem && (section === "llm" || section === "llm_conversation")) {
    return (
      <LLMBySystemPanel
        containerRef={containerRef} files={files} section={section} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isBar={isBar} isSplit={isSplit}
      />
    );
  }

  if (isBySystem && section === "images") {
    return (
      <ImagesBySystemPanel
        containerRef={containerRef} files={files} enabledImageModels={enabledImageModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isBar={isBar}
      />
    );
  }

  if (isBySystem && section === "embeddings") {
    return (
      <EmbeddingsBySystemPanel
        containerRef={containerRef} files={files} chartWidth={chartWidth} logoSrc={logoSrc} isBar={isBar}
      />
    );
  }

  if (section === "llm" || section === "llm_conversation") {
    return (
      <LLMByModelPanel
        containerRef={containerRef} files={files} section={section} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isBar={isBar} isMultiFile={isMultiFile}
      />
    );
  }

  if (section === "images") {
    return (
      <ImagesPanel
        containerRef={containerRef} files={files} enabledImageModels={enabledImageModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isBar={isBar} isMultiFile={isMultiFile}
      />
    );
  }

  return (
    <EmbeddingsPanel
      containerRef={containerRef} files={files} chartWidth={chartWidth} logoSrc={logoSrc}
      isBar={isBar} isMultiFile={isMultiFile}
    />
  );
}
