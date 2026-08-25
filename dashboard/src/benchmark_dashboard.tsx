import React, { useState, useCallback, useMemo, useRef, useEffect } from "react";
import html2canvas from "html2canvas";
import { readNamedJSONSource, sanitizeForFilename, filesForSection, getRunReliabilityWarning, getLlamaBenchMethodologyWarning, getConversationTTFTMethodologyWarning, getGpuSplitMethodologyWarning, getNoRepackMethodologyWarning, getCrossEngineWeightsWarning, getMemoryTelemetryMethodologyWarning } from "./utils/shared";
import { isTrialSetArtifact, trialArtifactLoadMode } from "./utils/trials";
import { isRecommendationArtifact, recommendationArtifactLoadMode } from "./utils/recommendations";
import { isVariantComparisonArtifact, variantArtifactLoadMode } from "./utils/variants";
import { getAllLLMModels } from "./utils/llm";
import { getAllImageModels } from "./utils/images";
import { getAllEmbedModels } from "./utils/embeddings";
import { getAccuracySettingsWarning } from "./utils/accuracy";
import { fetchSelectedResultFiles } from "./utils/autoload";
import { applyBaselineDeltas } from "./utils/baseline";
import { MAX_FILES } from "./constants";
import type { DisplayFile, SortConfig } from "./types";
import type { NamedTextSource, ParsedNamedSource } from "./utils/shared";
import Header from "./components/Header";
import Controls from "./components/Controls";
import ChartPanel from "./components/ChartPanel";
import StatsTable from "./components/StatsTable";
import ValidityInspector from "./components/ValidityInspector";
import "./dashboard.css";
import styles from "./benchmark_dashboard.module.css";
import { DeltaModeContext } from "./components/DeltaModeContext";
import RunSummaryCards from "./components/RunSummaryCards";
import TrialSetPanel from "./components/TrialSetPanel";
import RecommendationPanel from "./components/RecommendationPanel";
import VariantComparisonPanel from "./components/VariantComparisonPanel";
import { dashboardHostname } from "./utils/specCard";
import { buildRunCardFilename } from "./utils/specCard";
import { useAutoEnabledSelection } from "./hooks/useAutoEnabledSelection";
import {
  buildWorkspaceSelection, downloadBlob, downloadWorkspaceSelection,
  isWorkspaceSelection, requestWorkspaceEvaluation, requestWorkspaceExport,
} from "./utils/workspace";

export default function Dashboard() {
  const [files, setFiles] = useState<DisplayFile[]>([]);
  const [section, setSection] = useState("llm");
  const [accuracyTest, setAccuracyTest] = useState("mcq");
  const [dragOver, setDragOver] = useState(false);
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: "model", dir: 1 });
  const [chartStyle, setChartStyle] = useState("bar");
  const [groupBy, setGroupBy] = useState("model");
  const [sizeSplit, setSizeSplit] = useState("tiers");
  const [chartWidth, setChartWidth] = useState(708);
  const [hostnameOverrides, setHostnameOverrides] = useState<Record<string, string>>({});
  const [logoSrc, setLogoSrc] = useState<string | null>(null);
  const [logoDragOver, setLogoDragOver] = useState(false);
  const [saving, setSaving] = useState(false);
  const [filenameSuffix, setFilenameSuffix] = useState("");
  const [fileError, setFileError] = useState("");
  const [baselineId, setBaselineId] = useState<string | null>(null);
  const [savingSpecCard, setSavingSpecCard] = useState(false);
  const [trialSet, setTrialSet] = useState<{ name: string, data: import("./utils/shared").JsonRecord } | null>(null);
  const [recommendation, setRecommendation] = useState<{ name: string, data: import("./utils/shared").JsonRecord } | null>(null);
  const [variantComparison, setVariantComparison] = useState<{ name: string, data: import("./utils/shared").JsonRecord } | null>(null);
  const [workspaceExporting, setWorkspaceExporting] = useState<string | null>(null);
  const [workspacePolicyText, setWorkspacePolicyText] = useState("");
  const [workspacePolicy, setWorkspacePolicy] = useState<import("./utils/shared").JsonRecord | null>(null);
  const [workspaceEvaluation, setWorkspaceEvaluation] = useState<import("./utils/shared").JsonRecord | null>(null);
  const [workspaceRecommendation, setWorkspaceRecommendation] = useState<import("./utils/shared").JsonRecord | null>(null);

  const filesRef = useRef(files);
  const sectionRef = useRef(section);
  const autoloadStartedRef = useRef(false);
  useEffect(() => { filesRef.current = files; }, [files]);
  useEffect(() => { sectionRef.current = section; }, [section]);

  const chartRef = useRef<HTMLDivElement>(null);
  const summaryRef = useRef<HTMLDivElement>(null);
  const selectionInputRef = useRef<HTMLInputElement>(null);
  const recommendationInputRef = useRef<HTMLInputElement>(null);

  const allModels = useMemo(() => getAllLLMModels(files), [files]);
  const allImageModels = useMemo(() => getAllImageModels(files), [files]);
  const allEmbedModels = useMemo(() => getAllEmbedModels(files), [files]);
  const {
    enabled: enabledModels, toggle: toggleModel, reset: resetLlmSelection,
  } = useAutoEnabledSelection(allModels);
  const {
    enabled: enabledImageModels, toggle: toggleImageModel, reset: resetImageSelection,
  } = useAutoEnabledSelection(allImageModels);
  const {
    enabled: enabledEmbedModels, toggle: toggleEmbedModel, reset: resetEmbedSelection,
  } = useAutoEnabledSelection(allEmbedModels);

  const displayFiles = useMemo(() => filesForSection(files, section), [files, section]);

  const effectiveFiles = useMemo(() =>
    displayFiles.map(f => {
      const ov = f.id == null ? undefined : hostnameOverrides[f.id];
      return (ov != null && ov !== '') ? { ...f, hostname: ov } : f;
    }), [displayFiles, hostnameOverrides]);

  const effectiveFilesRef = useRef(effectiveFiles);
  useEffect(() => { effectiveFilesRef.current = effectiveFiles; }, [effectiveFiles]);
  const chartFiles = useMemo(
    () => applyBaselineDeltas(effectiveFiles, baselineId), [effectiveFiles, baselineId],
  );
  const accuracySettingsWarning = useMemo(
    () => getAccuracySettingsWarning(effectiveFiles),
    [effectiveFiles],
  );
  const llamaBenchMethodologyWarning = useMemo(
    () => getLlamaBenchMethodologyWarning(effectiveFiles), [effectiveFiles],
  );
  const conversationTTFTMethodologyWarning = useMemo(
    () => getConversationTTFTMethodologyWarning(effectiveFiles), [effectiveFiles],
  );
  const gpuSplitMethodologyWarning = useMemo(
    () => getGpuSplitMethodologyWarning(effectiveFiles), [effectiveFiles],
  );
  const noRepackMethodologyWarning = useMemo(
    () => getNoRepackMethodologyWarning(effectiveFiles, section), [effectiveFiles, section],
  );
  const crossEngineWeightsWarning = useMemo(
    () => getCrossEngineWeightsWarning(effectiveFiles), [effectiveFiles],
  );
  const memoryTelemetryMethodologyWarning = useMemo(
    () => getMemoryTelemetryMethodologyWarning(effectiveFiles), [effectiveFiles],
  );

  const updateHostnameOverride = useCallback((fileId: DisplayFile["id"], value: string) => {
    setHostnameOverrides(prev => ({ ...prev, [fileId as string]: value }));
  }, []);

  const resetModelState = useCallback(() => {
    resetLlmSelection();
    resetImageSelection();
    resetEmbedSelection();
    setHostnameOverrides({});
    setBaselineId(null);
  }, [resetLlmSelection, resetImageSelection, resetEmbedSelection]);

  const parseFile = (file: ParsedNamedSource): { entry: DisplayFile | null, error: string | null } => {
    if (file.error || !file.data) return { entry: null, error: `${file.name}: ${file.error}` };
    const data = file.data;
    const p = data.profile || {};
    const baseHostname = p.hostname || file.name.replace(".json", "");
    const entry: DisplayFile = {
      id: `${file.name}-${Date.now()}`,
      name: file.name,
      sourceSha256: file.sha256 as string,
      sourceText: file.sourceText as string,
      hostname: baseHostname,
      engine:   data.engine || null,
      engineVersion: data.engine_version || null,
      engineVersionRecorded: Object.prototype.hasOwnProperty.call(data, "engine_version"),
      backend:  p.backend  || "cpu",
      os:       p.os       || "",
      wsl:      p.wsl === true,
      ram_gb:   typeof p.ram_gb === "number" ? Math.round(p.ram_gb) : null,
      version:  data.version || null,
      timestamp: p.timestamp || null,
      reliabilityWarning: getRunReliabilityWarning(data),
      data,
    };
    entry.hostname = dashboardHostname(entry);
    return { entry, error: null };
  };

  const processJsonFiles = useCallback(async (jsonFiles: NamedTextSource[]) => {
    const limited = jsonFiles.slice(0, MAX_FILES);
    if (!limited.length) return;
    const candidates = await Promise.all(limited.map(readNamedJSONSource));
    const artifactMode = trialArtifactLoadMode(candidates.map(candidate => candidate.data));
    const recommendationMode = recommendationArtifactLoadMode(candidates.map(candidate => candidate.data));
    const variantMode = variantArtifactLoadMode(candidates.map(candidate => candidate.data));
    const singleArtifacts = [artifactMode, recommendationMode, variantMode].filter(mode => mode === "single").length;
    if ([artifactMode, recommendationMode, variantMode].includes("mixed") || singleArtifacts > 1) {
      setFileError("Load one derived artifact by itself; it cannot be mixed with result files or another artifact.");
      return;
    }
    if (artifactMode === "single" && isTrialSetArtifact(candidates[0].data)) {
      resetModelState();
      setFiles([]);
      setTrialSet({ name: candidates[0].name, data: candidates[0].data });
      setRecommendation(null);
      setVariantComparison(null);
      setFileError("");
      return;
    }
    if (recommendationMode === "single" && isRecommendationArtifact(candidates[0].data)) {
      resetModelState();
      setFiles([]);
      setTrialSet(null);
      setRecommendation({ name: candidates[0].name, data: candidates[0].data });
      setVariantComparison(null);
      setFileError("");
      return;
    }
    if (variantMode === "single" && isVariantComparisonArtifact(candidates[0].data)) {
      resetModelState();
      setFiles([]);
      setTrialSet(null);
      setRecommendation(null);
      setVariantComparison({ name: candidates[0].name, data: candidates[0].data });
      setFileError("");
      return;
    }
    setTrialSet(null);
    setRecommendation(null);
    setVariantComparison(null);
    const parsed = candidates.map(parseFile);
    const entries = parsed.map(result => result.entry).filter((entry): entry is DisplayFile => Boolean(entry));
    setFileError(parsed.map(result => result.error).filter(Boolean).join(" "));
    if (!entries.length) return;

    if (entries.length > 1 || filesRef.current.length >= MAX_FILES) {
      resetModelState();
      setFiles(entries);
    } else {
      setFiles(prev => [...prev, entries[0]]);
    }
  }, [resetModelState]);

  useEffect(() => {
    if (autoloadStartedRef.current) return;
    autoloadStartedRef.current = true;
    fetchSelectedResultFiles(window.location.search)
      .then(selectedFiles => processJsonFiles(selectedFiles))
      .catch(error => setFileError(error.message));
  }, [processJsonFiles]);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const jsonFiles = [...e.dataTransfer.files].filter(f => f.name.endsWith(".json"));
    await processJsonFiles(jsonFiles);
  }, [processJsonFiles]);

  const handleFileInput = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const jsonFiles = [...(e.target.files || [])].filter(f => f.name.endsWith(".json"));
    e.target.value = "";
    await processJsonFiles(jsonFiles);
  }, [processJsonFiles]);

  const removeFile = useCallback((fileId: DisplayFile["id"]) => {
    setFiles(prev => {
      const remaining = prev.filter(f => f.id !== fileId);
      if (remaining.length === 0) resetModelState();
      return remaining;
    });
    setHostnameOverrides(prev => { const n = { ...prev }; delete n[fileId as string]; return n; });
    setBaselineId(current => current === String(fileId) ? null : current);
  }, [resetModelState]);

  const handleLogoDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setLogoDragOver(false);
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (ev) => setLogoSrc(ev.target?.result as string);
    reader.readAsDataURL(file);
  }, []);

  const saveChart = useCallback(async () => {
    if (saving) return;
    setSaving(true);
    try {
      setFileError("");
      await document.fonts.ready;
      const runCards = summaryRef.current
        ? [...summaryRef.current.querySelectorAll<HTMLElement>("[data-spec-card]")] : [];
      const chartCards = chartRef.current
        ? [...chartRef.current.querySelectorAll<HTMLElement>("[data-chart-name]")] : [];
      if (!runCards.length && !chartCards.length) return;
      const runCardNames = runCards.map((card, index) => card.dataset.specName || `run-${index + 1}`);
      const exportCount = runCards.length + chartCards.length;
      let exported = 0;

      for (let index = 0; index < runCards.length; index++) {
        const canvas = await html2canvas(runCards[index], {
          backgroundColor: "#ffffff", scale: 2, useCORS: true, logging: false,
        });
        const link = document.createElement("a");
        link.download = buildRunCardFilename(runCardNames, index, filenameSuffix);
        link.href = canvas.toDataURL("image/png");
        link.click();
        exported++;
        if (exported < exportCount) await new Promise(resolve => setTimeout(resolve, 300));
      }

      for (let index = 0; index < chartCards.length; index++) {
        const canvas = await html2canvas(chartCards[index], {
          backgroundColor: "#ffffff", scale: 2, useCORS: true, logging: false,
        });
        const { chartName, chartModel } = chartCards[index].dataset;
        const rawBase = [chartModel, chartName, filenameSuffix].filter(Boolean).join("_");
        const filename = `${sanitizeForFilename(rawBase)}.png`;
        const link = document.createElement("a");
        link.download = filename;
        link.href = canvas.toDataURL("image/png");
        link.click();
        exported++;
        if (exported < exportCount) await new Promise(resolve => setTimeout(resolve, 300));
      }
    } catch (error) {
      setFileError(`PNG export failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setSaving(false);
    }
  }, [saving, filenameSuffix]);

  const saveSpecCards = useCallback(async () => {
    if (!summaryRef.current || savingSpecCard) return;
    setSavingSpecCard(true);
    try {
      setFileError("");
      await document.fonts.ready;
      const cards = [...summaryRef.current.querySelectorAll<HTMLElement>("[data-spec-card]")];
      if (!cards.length) throw new Error("no run cards are available");
      const names = cards.map((card, index) => card.dataset.specName || `run-${index + 1}`);
      for (let index = 0; index < cards.length; index++) {
        const canvas = await html2canvas(cards[index], {
          backgroundColor: "#ffffff", scale: 2, useCORS: true, logging: false,
        });
        const link = document.createElement("a");
        link.download = buildRunCardFilename(names, index, filenameSuffix);
        link.href = canvas.toDataURL("image/png");
        link.click();
        if (index < cards.length - 1) await new Promise(resolve => setTimeout(resolve, 300));
      }
    } catch (error) {
      setFileError(`Run-card export failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setSavingSpecCard(false);
    }
  }, [savingSpecCard, filenameSuffix]);

  const cycleSort = (key: string) => {
    setSortConfig(prev => prev.key === key ? { key, dir: (prev.dir * -1) as 1 | -1 } : { key, dir: 1 });
  };

  const currentWorkspaceSelection = useCallback(() => buildWorkspaceSelection(
    files, baselineId, {
      section,
      accuracy_test: accuracyTest,
      enabled_models: [...enabledModels].sort(),
      enabled_image_models: [...enabledImageModels].sort(),
      enabled_embedding_models: [...enabledEmbedModels].sort(),
      hostname_overrides: hostnameOverrides,
    }, workspacePolicy, workspaceRecommendation,
  ), [
    files, baselineId, section, accuracyTest, enabledModels, enabledImageModels,
    enabledEmbedModels, hostnameOverrides, workspacePolicy, workspaceRecommendation,
  ]);

  const exportWorkspaceSelection = useCallback(() => {
    try {
      const selection = currentWorkspaceSelection();
      downloadWorkspaceSelection(selection);
      setFileError("");
    } catch (error) {
      setFileError(error instanceof Error ? error.message : String(error));
    }
  }, [currentWorkspaceSelection]);

  const exportWorkspaceArtifact = useCallback(async (format: "html" | "pdf" | "bundle") => {
    if (workspaceExporting) return;
    setWorkspaceExporting(format);
    try {
      const exported = await requestWorkspaceExport(currentWorkspaceSelection(), files, format);
      downloadBlob(exported.blob, exported.filename);
      setFileError("");
    } catch (error) {
      setFileError(error instanceof Error ? error.message : String(error));
    } finally {
      setWorkspaceExporting(null);
    }
  }, [currentWorkspaceSelection, files, workspaceExporting]);

  const updateWorkspacePolicy = useCallback((text: string) => {
    setWorkspacePolicyText(text);
    if (!text.trim()) {
      setWorkspacePolicy(null);
      setWorkspaceEvaluation(null);
      return;
    }
    try {
      const value: unknown = JSON.parse(text);
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("Acceptance policy must be a JSON object.");
      }
      setWorkspacePolicy(value as import("./utils/shared").JsonRecord);
      setFileError("");
    } catch (error) {
      setWorkspacePolicy(null);
      setWorkspaceEvaluation(null);
      setFileError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const loadWorkspaceRecommendation = useCallback(async (file: File | undefined) => {
    if (!file) return;
    try {
      const value: unknown = JSON.parse(await file.text());
      if (!isRecommendationArtifact(value)) throw new Error("Recommendation artifact is invalid.");
      setWorkspaceRecommendation(value);
      setFileError("");
    } catch (error) {
      setWorkspaceRecommendation(null);
      setFileError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const loadWorkspaceSelection = useCallback(async (file: File | undefined) => {
    if (!file) return;
    try {
      const value: unknown = JSON.parse(await file.text());
      if (!isWorkspaceSelection(value)) throw new Error("Workspace selection is invalid.");
      const loaded = files.map(result => result.sourceSha256).sort();
      const selected = value.results.map(result => result.sha256).sort();
      if (JSON.stringify(loaded) !== JSON.stringify(selected)) {
        throw new Error("Load the exact result files recorded by this workspace selection first.");
      }
      setSection(value.view.section);
      setAccuracyTest(value.view.accuracy_test);
      setHostnameOverrides(value.view.hostname_overrides);
      const baseline = files.find(result => result.sourceSha256 === value.baseline_sha256);
      setBaselineId(baseline?.id == null ? null : String(baseline.id));
      for (const [enabled, expected, toggle] of [
        [enabledModels, new Set(value.view.enabled_models), toggleModel],
        [enabledImageModels, new Set(value.view.enabled_image_models), toggleImageModel],
        [enabledEmbedModels, new Set(value.view.enabled_embedding_models), toggleEmbedModel],
      ] as const) {
        for (const item of new Set([...enabled, ...expected])) {
          if (enabled.has(item) !== expected.has(item)) toggle(item);
        }
      }
      setWorkspacePolicy(value.acceptance_policy);
      setWorkspacePolicyText(value.acceptance_policy
        ? JSON.stringify(value.acceptance_policy, null, 2) : "");
      setWorkspaceRecommendation(value.recommendation);
      setWorkspaceEvaluation(null);
      setFileError("");
    } catch (error) {
      setFileError(error instanceof Error ? error.message : String(error));
    }
  }, [
    files, enabledModels, enabledImageModels, enabledEmbedModels,
    toggleModel, toggleImageModel, toggleEmbedModel,
  ]);

  const evaluateWorkspaceDecision = useCallback(async () => {
    try {
      const result = await requestWorkspaceEvaluation(currentWorkspaceSelection(), files);
      setWorkspaceEvaluation(result.acceptance);
      setFileError("");
    } catch (error) {
      setWorkspaceEvaluation(null);
      setFileError(error instanceof Error ? error.message : String(error));
    }
  }, [currentWorkspaceSelection, files]);

  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setDragOver(true); }, []);
  const handleDragLeave = useCallback(() => setDragOver(false), []);
  const handleLogoDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setLogoDragOver(true); }, []);
  const handleLogoDragLeave = useCallback(() => setLogoDragOver(false), []);

  return (
    <div className={styles.root}>
      <Header
        files={displayFiles}
        dragOver={dragOver}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onRemoveFile={removeFile}
        onFileInput={handleFileInput}
        fileError={[
          fileError, accuracySettingsWarning, llamaBenchMethodologyWarning,
          conversationTTFTMethodologyWarning, gpuSplitMethodologyWarning,
          noRepackMethodologyWarning, crossEngineWeightsWarning,
          memoryTelemetryMethodologyWarning,
        ].filter(Boolean).join(" ")}
      />

      {trialSet ? <TrialSetPanel name={trialSet.name} artifact={trialSet.data} />
        : recommendation ? <RecommendationPanel name={recommendation.name} artifact={recommendation.data} />
          : variantComparison ? <VariantComparisonPanel name={variantComparison.name} artifact={variantComparison.data} /> : <>

      {files.length > 0 && <div className="card" style={{ marginBottom: 20 }}>
        <div className={styles.workspaceHeader}>
          <div>
            <div className={styles.workspaceEyebrow}>Decision workspace</div>
            <div className={styles.workspaceNote}>Selection, baseline, filters, labels, and exact result identities are exported together.</div>
          </div>
          <div className={styles.workspaceActions}>
            <button type="button" className="pill inactive"
              onClick={() => selectionInputRef.current?.click()}>Import selection</button>
            <input ref={selectionInputRef} type="file" accept="application/json,.json" hidden
              onChange={event => loadWorkspaceSelection(event.target.files?.[0])} />
            <button className="pill inactive" onClick={exportWorkspaceSelection}>Selection JSON</button>
            <button className="pill active" disabled={workspaceExporting != null}
              onClick={() => exportWorkspaceArtifact("html")}>HTML report</button>
            <button className="pill active" disabled={workspaceExporting != null}
              onClick={() => exportWorkspaceArtifact("pdf")}>PDF report</button>
            <button className="pill active" disabled={workspaceExporting != null}
              onClick={() => exportWorkspaceArtifact("bundle")}>Bundle</button>
          </div>
        </div>
        <div className={styles.workspaceEditors}>
          <label className={styles.workspaceEditor}>
            <span>Acceptance policy JSON</span>
            <textarea value={workspacePolicyText}
              placeholder="Paste or edit a versioned acceptance policy"
              onChange={event => updateWorkspacePolicy(event.target.value)} />
          </label>
          <div className={styles.workspaceDecision}>
            <button className="pill active" disabled={!workspacePolicy && !workspaceRecommendation}
              onClick={evaluateWorkspaceDecision}>Apply decision inputs</button>
            <button type="button" className="pill inactive"
              onClick={() => recommendationInputRef.current?.click()}>Attach recommendation</button>
            <input ref={recommendationInputRef} type="file" accept="application/json,.json" hidden
              onChange={event => loadWorkspaceRecommendation(event.target.files?.[0])} />
            {workspaceRecommendation && <button className="pill inactive"
              onClick={() => setWorkspaceRecommendation(null)}>Remove recommendation</button>}
            <strong>Acceptance: {workspaceEvaluation?.decision
              ? String(workspaceEvaluation.decision).replaceAll("_", " ") : "Not evaluated"}</strong>
            {Array.isArray(workspaceEvaluation?.rules) && <span>
              {workspaceEvaluation.rules.map(rule => `${rule.id}: ${rule.status}`).join(" · ")}
            </span>}
          </div>
        </div>
      </div>}

      {files.length > 0 && workspaceRecommendation && <RecommendationPanel
        name="Workspace recommendation" artifact={workspaceRecommendation}
      />}

      <Controls
        section={section} setSection={setSection}
        accuracyTest={accuracyTest} setAccuracyTest={setAccuracyTest}
        allModels={allModels} enabledModels={enabledModels} onToggleModel={toggleModel}
        allImageModels={allImageModels} enabledImageModels={enabledImageModels} onToggleImageModel={toggleImageModel}
        allEmbedModels={allEmbedModels} enabledEmbedModels={enabledEmbedModels} onToggleEmbedModel={toggleEmbedModel}
        chartStyle={chartStyle} setChartStyle={setChartStyle}
        groupBy={groupBy} setGroupBy={setGroupBy}
        sizeSplit={sizeSplit} setSizeSplit={setSizeSplit}
        chartWidth={chartWidth} setChartWidth={setChartWidth}
        files={displayFiles} hostnameOverrides={hostnameOverrides} onUpdateHostnameOverride={updateHostnameOverride}
        logoSrc={logoSrc} setLogoSrc={setLogoSrc}
        logoDragOver={logoDragOver}
        onLogoDrop={handleLogoDrop}
        onLogoDragOver={handleLogoDragOver}
        onLogoDragLeave={handleLogoDragLeave}
        saving={saving} onSaveChart={saveChart}
        filenameSuffix={filenameSuffix} setFilenameSuffix={setFilenameSuffix}
        baselineId={baselineId} setBaselineId={setBaselineId}
        savingSpecCard={savingSpecCard} onSaveSpecCard={saveSpecCards}
      />

      <RunSummaryCards
        files={effectiveFiles} containerRef={summaryRef} logoSrc={logoSrc} chartWidth={chartWidth}
      />

      <DeltaModeContext.Provider value={baselineId != null && section !== "sustained"}>
        <ChartPanel
          containerRef={chartRef}
          files={section === "sustained" ? effectiveFiles : chartFiles}
          absoluteFiles={effectiveFiles}
          section={section}
          accuracyTest={accuracyTest}
          enabledModels={enabledModels}
          enabledImageModels={enabledImageModels}
          enabledEmbedModels={enabledEmbedModels}
          chartWidth={chartWidth}
          logoSrc={logoSrc}
          chartStyle={chartStyle}
          groupBy={groupBy}
          sizeSplit={sizeSplit}
        />
      </DeltaModeContext.Provider>

      <StatsTable
        files={effectiveFiles}
        section={section}
        accuracyTest={accuracyTest}
        sortConfig={sortConfig}
        onCycleSort={cycleSort}
      />

      <ValidityInspector key={section} files={effectiveFiles} section={section} />
      </>}
    </div>
  );
}
