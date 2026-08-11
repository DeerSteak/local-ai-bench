import type { RefObject } from "react";
import type { ResultsFile } from "../types";
import LLMBySystemPanel from "./panels/LLMBySystemPanel";
import ImagesBySystemPanel from "./panels/ImagesBySystemPanel";
import EmbeddingsBySystemPanel from "./panels/EmbeddingsBySystemPanel";
import LLMByModelPanel from "./panels/LLMByModelPanel";
import ImagesPanel from "./panels/ImagesPanel";
import EmbeddingsPanel from "./panels/EmbeddingsPanel";
import AccuracyPanel from "./panels/AccuracyPanel";
import ConcurrencyPanel from "./panels/ConcurrencyPanel";
import LlamaBenchPanel from "./panels/LlamaBenchPanel";
import VllmBenchPanel from "./panels/VllmBenchPanel";
import LlamaBenchBySystemPanel from "./panels/LlamaBenchBySystemPanel";
import LlamaBenchConcPanel from "./panels/LlamaBenchConcPanel";
import { EmptyState } from "./panels/shared";

// Picks the right panel for the current section / Group By / Chart Style
// selection. Each panel owns its own data wiring and empty-state handling —
// see components/panels/*.jsx. Shared chart-rendering primitives (the actual
// recharts wrappers) live in components/charts/ChartCards.jsx.
export default function ChartPanel({
  containerRef, files, absoluteFiles, section, accuracyTest,
  enabledModels, enabledImageModels, enabledEmbedModels, chartWidth, logoSrc, chartStyle, groupBy, sizeSplit,
}: {
  containerRef?: RefObject<HTMLDivElement | null>, files: ResultsFile[], absoluteFiles: ResultsFile[],
  section: string, accuracyTest: string,
  enabledModels: Set<string>, enabledImageModels: Set<string>, enabledEmbedModels: Set<string>,
  chartWidth: number, logoSrc?: string | null, chartStyle: string, groupBy: string, sizeSplit: string,
}) {
  const isBar = chartStyle === "bar";
  const isBySystem = groupBy === "system";
  const isSplit = sizeSplit === "tiers";
  const isMultiFile = files.length > 1;

  if (files.length === 0) {
    const containerStyle = { width: chartWidth, minWidth: chartWidth, maxWidth: chartWidth };
    return <EmptyState style={containerStyle}>Drop a results JSON file above to get started</EmptyState>;
  }

  if (section === "accuracy") {
    return (
      <AccuracyPanel
        containerRef={containerRef} files={files} accuracyTest={accuracyTest} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc}
      />
    );
  }

  if (isBySystem && section === "llamabench") {
    return (
      <LlamaBenchBySystemPanel
        containerRef={containerRef} files={files} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isSplit={isSplit}
      />
    );
  }

  if (section === "llamabench") {
    return (
      <LlamaBenchPanel
        containerRef={containerRef} files={files} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isMultiFile={isMultiFile}
      />
    );
  }

  if (section === "vllmbench") {
    return (
      <VllmBenchPanel
        containerRef={containerRef} files={files} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isMultiFile={isMultiFile}
      />
    );
  }

  if (section === "llamabenchconc") {
    return (
      <LlamaBenchConcPanel
        containerRef={containerRef} files={files} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isMultiFile={isMultiFile}
      />
    );
  }

  if (section === "concurrency_tool" || section === "concurrency_chat") {
    return (
      <ConcurrencyPanel
        containerRef={containerRef} files={files} sweetSpotFiles={absoluteFiles}
        section={section} enabledModels={enabledModels}
        chartWidth={chartWidth} logoSrc={logoSrc} isMultiFile={isMultiFile}
      />
    );
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
        containerRef={containerRef} files={files} enabledEmbedModels={enabledEmbedModels}
        chartWidth={chartWidth} logoSrc={logoSrc}
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
      containerRef={containerRef} files={files} enabledEmbedModels={enabledEmbedModels}
      chartWidth={chartWidth} logoSrc={logoSrc}
    />
  );
}
