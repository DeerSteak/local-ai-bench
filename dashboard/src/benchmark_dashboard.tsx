import React, { useState, useCallback, useMemo, useRef, useEffect } from "react";
import html2canvas from "html2canvas";
import { parseResultsJSON, sanitizeForFilename, filesForSection, getRunReliabilityWarning, getLlamaBenchMethodologyWarning, getConversationTTFTMethodologyWarning, getGpuSplitMethodologyWarning, getNoRepackMethodologyWarning, getCrossEngineWeightsWarning } from "./utils/shared";
import { getAllLLMModels } from "./utils/llm";
import { getAllImageModels } from "./utils/images";
import { getAllEmbedModels } from "./utils/embeddings";
import { getAccuracySettingsWarning } from "./utils/accuracy";
import { fetchSelectedResultFiles } from "./utils/autoload";
import { applyBaselineDeltas } from "./utils/baseline";
import { MAX_FILES } from "./constants";
import type { DisplayFile, SortConfig } from "./types";
import Header from "./components/Header";
import Controls from "./components/Controls";
import ChartPanel from "./components/ChartPanel";
import StatsTable from "./components/StatsTable";
import ValidityInspector from "./components/ValidityInspector";
import "./dashboard.css";
import styles from "./benchmark_dashboard.module.css";
import { DeltaModeContext } from "./components/DeltaModeContext";
import RunSummaryCards from "./components/RunSummaryCards";
import { buildRunCardFilename } from "./utils/specCard";

// The minimal shape both real File objects and autoload's staged-result
// entries satisfy — this is all parseFile/processJsonFiles actually need.
type NamedTextSource = { name: string, text: () => Promise<string> };

export default function Dashboard() {
  const [files, setFiles] = useState<DisplayFile[]>([]);
  const [section, setSection] = useState("llm");
  const [accuracyTest, setAccuracyTest] = useState("mcq");
  const [enabledModels, setEnabledModels] = useState<Set<string>>(new Set());
  const [enabledImageModels, setEnabledImageModels] = useState<Set<string>>(new Set());
  const [enabledEmbedModels, setEnabledEmbedModels] = useState<Set<string>>(new Set());
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

  const filesRef = useRef(files);
  const sectionRef = useRef(section);
  const autoloadStartedRef = useRef(false);
  useEffect(() => { filesRef.current = files; }, [files]);
  useEffect(() => { sectionRef.current = section; }, [section]);

  const chartRef = useRef<HTMLDivElement>(null);
  const summaryRef = useRef<HTMLDivElement>(null);

  const allModels = useMemo(() => getAllLLMModels(files), [files]);
  const allImageModels = useMemo(() => getAllImageModels(files), [files]);
  const allEmbedModels = useMemo(() => getAllEmbedModels(files), [files]);

  // Auto-enable newly appearing models
  const prevModelsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const newOnes = allModels.filter(m => !prevModelsRef.current.has(m));
    if (newOnes.length) {
      setEnabledModels(prev => { const n = new Set(prev); newOnes.forEach(m => n.add(m)); return n; });
      newOnes.forEach(m => prevModelsRef.current.add(m));
    }
  }, [allModels]);

  const prevImageModelsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const newOnes = allImageModels.filter(m => !prevImageModelsRef.current.has(m));
    if (newOnes.length) {
      setEnabledImageModels(prev => { const n = new Set(prev); newOnes.forEach(m => n.add(m)); return n; });
      newOnes.forEach(m => prevImageModelsRef.current.add(m));
    }
  }, [allImageModels]);

  const prevEmbedModelsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const newOnes = allEmbedModels.filter(m => !prevEmbedModelsRef.current.has(m));
    if (newOnes.length) {
      setEnabledEmbedModels(prev => { const n = new Set(prev); newOnes.forEach(m => n.add(m)); return n; });
      newOnes.forEach(m => prevEmbedModelsRef.current.add(m));
    }
  }, [allEmbedModels]);

  const toggleModel = useCallback((m: string) => {
    setEnabledModels(prev => {
      const n = new Set(prev);
      if (n.has(m)) n.delete(m); else n.add(m);
      return n;
    });
  }, []);

  const toggleImageModel = useCallback((m: string) => {
    setEnabledImageModels(prev => {
      const n = new Set(prev);
      if (n.has(m)) n.delete(m); else n.add(m);
      return n;
    });
  }, []);

  const toggleEmbedModel = useCallback((m: string) => {
    setEnabledEmbedModels(prev => {
      const n = new Set(prev);
      if (n.has(m)) n.delete(m); else n.add(m);
      return n;
    });
  }, []);

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

  const updateHostnameOverride = useCallback((fileId: DisplayFile["id"], value: string) => {
    setHostnameOverrides(prev => ({ ...prev, [fileId as string]: value }));
  }, []);

  const resetModelState = () => {
    prevModelsRef.current = new Set();
    prevImageModelsRef.current = new Set();
    prevEmbedModelsRef.current = new Set();
    setEnabledModels(new Set());
    setEnabledImageModels(new Set());
    setEnabledEmbedModels(new Set());
    setHostnameOverrides({});
    setBaselineId(null);
  };

  const parseFile = async (file: NamedTextSource): Promise<{ entry: DisplayFile | null, error: string | null }> => {
    let text;
    try {
      text = await file.text();
    } catch {
      return { entry: null, error: `${file.name}: Could not read this file.` };
    }
    const parsed = parseResultsJSON(text);
    if (parsed.error || !parsed.data) return { entry: null, error: `${file.name}: ${parsed.error}` };
    const data = parsed.data;
    const p = data.profile || {};
    const baseHostname = p.hostname || file.name.replace(".json", "");
    return { entry: {
      id: `${file.name}-${Date.now()}`,
      name: file.name,
      hostname: baseHostname,
      engine:   data.engine || null,
      engineVersion: data.engine_version || null,
      engineVersionRecorded: Object.prototype.hasOwnProperty.call(data, "engine_version"),
      backend:  p.backend  || "cpu",
      os:       p.os       || "",
      wsl:      p.wsl === true,
      ram_gb:   p.ram_gb   || null,
      version:  data.version || null,
      timestamp: p.timestamp || null,
      reliabilityWarning: getRunReliabilityWarning(data),
      data,
    }, error: null };
  };

  const processJsonFiles = useCallback(async (jsonFiles: NamedTextSource[]) => {
    const limited = jsonFiles.slice(0, MAX_FILES);
    if (!limited.length) return;
    const parsed = await Promise.all(limited.map(parseFile));
    const entries = parsed.map(result => result.entry).filter((entry): entry is DisplayFile => Boolean(entry));
    setFileError(parsed.map(result => result.error).filter(Boolean).join(" "));
    if (!entries.length) return;

    if (entries.length > 1 || filesRef.current.length >= MAX_FILES) {
      resetModelState();
      setFiles(entries);
    } else {
      setFiles(prev => [...prev, entries[0]]);
    }
  }, []);

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
  }, []);

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
    if (!chartRef.current || saving) return;
    setSaving(true);
    try {
      setFileError("");
      await document.fonts.ready;
      const cards = [...chartRef.current.querySelectorAll<HTMLElement>("[data-chart-name]")];
      if (!cards.length) return;

      for (let i = 0; i < cards.length; i++) {
        const canvas = await html2canvas(cards[i], {
          backgroundColor: "#ffffff", scale: 2, useCORS: true, logging: false,
        });
        const { chartName, chartModel } = cards[i].dataset;
        const rawBase = [chartModel, chartName, filenameSuffix].filter(Boolean).join("_");
        const filename = `${sanitizeForFilename(rawBase)}.png`;
        const link = document.createElement("a");
        link.download = filename;
        link.href = canvas.toDataURL("image/png");
        link.click();
        if (i < cards.length - 1) await new Promise(r => setTimeout(r, 300));
      }
    } catch (error) {
      setFileError(`Chart export failed: ${error instanceof Error ? error.message : String(error)}`);
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
        ].filter(Boolean).join(" ")}
      />

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

      <RunSummaryCards files={effectiveFiles} containerRef={summaryRef} logoSrc={logoSrc} />

      <DeltaModeContext.Provider value={baselineId != null}>
        <ChartPanel
          containerRef={chartRef}
          files={chartFiles}
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
    </div>
  );
}
