import React, { useState, useCallback, useMemo, useRef, useEffect } from "react";
import html2canvas from "html2canvas";
import { parseJSON, getAllLLMModels, getAllImageModels } from "./utils";
import { MAX_FILES } from "./constants";
import Header from "./components/Header";
import Controls from "./components/Controls";
import ChartPanel from "./components/ChartPanel";
import StatsTable from "./components/StatsTable";
import "./dashboard.css";
import styles from "./benchmark_dashboard.module.css";

export default function Dashboard() {
  const [files, setFiles] = useState([]);
  const [section, setSection] = useState("llm");
  const [enabledModels, setEnabledModels] = useState(new Set());
  const [enabledImageModels, setEnabledImageModels] = useState(new Set());
  const [dragOver, setDragOver] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: "model", dir: 1 });
  const [chartWidth, setChartWidth] = useState(708);
  const [logoSrc, setLogoSrc] = useState(null);
  const [logoDragOver, setLogoDragOver] = useState(false);
  const [saving, setSaving] = useState(false);

  const filesRef = useRef(files);
  const sectionRef = useRef(section);
  useEffect(() => { filesRef.current = files; }, [files]);
  useEffect(() => { sectionRef.current = section; }, [section]);

  const chartRef = useRef(null);

  const allModels = useMemo(() => getAllLLMModels(files), [files]);
  const allImageModels = useMemo(() => getAllImageModels(files), [files]);

  // Auto-enable newly appearing models
  const prevModelsRef = useRef(new Set());
  useEffect(() => {
    const newOnes = allModels.filter(m => !prevModelsRef.current.has(m));
    if (newOnes.length) {
      setEnabledModels(prev => { const n = new Set(prev); newOnes.forEach(m => n.add(m)); return n; });
      newOnes.forEach(m => prevModelsRef.current.add(m));
    }
  }, [allModels]);

  const prevImageModelsRef = useRef(new Set());
  useEffect(() => {
    const newOnes = allImageModels.filter(m => !prevImageModelsRef.current.has(m));
    if (newOnes.length) {
      setEnabledImageModels(prev => { const n = new Set(prev); newOnes.forEach(m => n.add(m)); return n; });
      newOnes.forEach(m => prevImageModelsRef.current.add(m));
    }
  }, [allImageModels]);

  const toggleModel = useCallback((m) => {
    setEnabledModels(prev => { const n = new Set(prev); n.has(m) ? n.delete(m) : n.add(m); return n; });
  }, []);

  const toggleImageModel = useCallback((m) => {
    setEnabledImageModels(prev => { const n = new Set(prev); n.has(m) ? n.delete(m) : n.add(m); return n; });
  }, []);

  const resetModelState = () => {
    prevModelsRef.current = new Set();
    prevImageModelsRef.current = new Set();
    setEnabledModels(new Set());
    setEnabledImageModels(new Set());
  };

  const parseFile = async (file) => {
    const text = await file.text();
    const data = parseJSON(text);
    if (!data) return null;
    const p = data.profile || {};
    return {
      id: `${file.name}-${Date.now()}`,
      name: file.name,
      hostname: p.hostname || file.name.replace(".json", ""),
      backend:  p.backend  || "cpu",
      os:       p.os       || "",
      ram_gb:   p.ram_gb   || null,
      timestamp: p.timestamp || null,
      data,
    };
  };

  const processJsonFiles = useCallback(async (jsonFiles) => {
    const limited = jsonFiles.slice(0, MAX_FILES);
    if (!limited.length) return;
    const entries = (await Promise.all(limited.map(parseFile))).filter(Boolean);
    if (!entries.length) return;

    if (entries.length > 1 || filesRef.current.length >= MAX_FILES) {
      resetModelState();
      setFiles(entries);
    } else {
      setFiles(prev => [...prev, entries[0]]);
    }
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    setDragOver(false);
    const jsonFiles = [...e.dataTransfer.files].filter(f => f.name.endsWith(".json"));
    await processJsonFiles(jsonFiles);
  }, [processJsonFiles]);

  const handleFileInput = useCallback(async (e) => {
    const jsonFiles = [...e.target.files].filter(f => f.name.endsWith(".json"));
    e.target.value = "";
    await processJsonFiles(jsonFiles);
  }, [processJsonFiles]);

  const removeFile = useCallback((fileId) => {
    setFiles(prev => {
      const remaining = prev.filter(f => f.id !== fileId);
      if (remaining.length === 0) resetModelState();
      return remaining;
    });
  }, []);

  const handleLogoDrop = useCallback((e) => {
    e.preventDefault();
    setLogoDragOver(false);
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (ev) => setLogoSrc(ev.target.result);
    reader.readAsDataURL(file);
  }, []);

  const saveChart = useCallback(async () => {
    if (!chartRef.current || saving) return;
    setSaving(true);
    try {
      const cards = [...chartRef.current.querySelectorAll("[data-chart-name]")];
      if (!cards.length) return;

      const f = filesRef.current;
      const slug = f.length === 0 ? "machine"
        : f.length === 1 ? f[0].hostname
        : f.map(fi => fi.hostname).join("_vs_");

      for (let i = 0; i < cards.length; i++) {
        const canvas = await html2canvas(cards[i], {
          backgroundColor: "#ffffff", scale: 2, useCORS: true, logging: false,
        });
        const { chartName, chartModel } = cards[i].dataset;
        const filename = chartModel
          ? `${chartModel}_${chartName}_${slug}.png`
          : `${slug}_${chartName}.png`;
        const link = document.createElement("a");
        link.download = filename;
        link.href = canvas.toDataURL("image/png");
        link.click();
        if (i < cards.length - 1) await new Promise(r => setTimeout(r, 300));
      }
    } finally {
      setSaving(false);
    }
  }, [saving]);

  const cycleSort = (key) => {
    setSortConfig(prev => prev.key === key ? { key, dir: prev.dir * -1 } : { key, dir: 1 });
  };

  const handleDragOver = useCallback((e) => { e.preventDefault(); setDragOver(true); }, []);
  const handleDragLeave = useCallback(() => setDragOver(false), []);
  const handleLogoDragOver = useCallback((e) => { e.preventDefault(); setLogoDragOver(true); }, []);
  const handleLogoDragLeave = useCallback(() => setLogoDragOver(false), []);

  return (
    <div className={styles.root}>
      <Header
        files={files}
        dragOver={dragOver}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onRemoveFile={removeFile}
        onFileInput={handleFileInput}
      />

      <Controls
        section={section} setSection={setSection}
        allModels={allModels} enabledModels={enabledModels} onToggleModel={toggleModel}
        allImageModels={allImageModels} enabledImageModels={enabledImageModels} onToggleImageModel={toggleImageModel}
        chartWidth={chartWidth} setChartWidth={setChartWidth}
        logoSrc={logoSrc} setLogoSrc={setLogoSrc}
        logoDragOver={logoDragOver}
        onLogoDrop={handleLogoDrop}
        onLogoDragOver={handleLogoDragOver}
        onLogoDragLeave={handleLogoDragLeave}
        saving={saving} onSaveChart={saveChart}
      />

      <ChartPanel
        containerRef={chartRef}
        files={files}
        section={section}
        enabledModels={enabledModels}
        enabledImageModels={enabledImageModels}
        chartWidth={chartWidth}
        logoSrc={logoSrc}
      />

      <StatsTable
        files={files}
        section={section}
        sortConfig={sortConfig}
        onCycleSort={cycleSort}
      />
    </div>
  );
}
