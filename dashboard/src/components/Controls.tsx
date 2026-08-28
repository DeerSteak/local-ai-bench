import { useRef } from "react";
import { SECTIONS, SECTION_LABELS, FILE_COLORS, ACCURACY_TESTS, ACCURACY_TEST_LABELS } from "../constants";
import {
  modelLabel, imageModelLabel, embedModelLabel,
  getModelColor, getImageModelColor, getEmbedModelColor, sanitizeForFilename, lookup,
} from "../utils/shared";
import type { DisplayFile } from "../types";
import styles from "./Controls.module.css";

export default function Controls({
  section, setSection,
  accuracyTest, setAccuracyTest,
  allModels, enabledModels, onToggleModel,
  allImageModels, enabledImageModels, onToggleImageModel,
  allEmbedModels, enabledEmbedModels, onToggleEmbedModel,
  chartStyle, setChartStyle,
  groupBy, setGroupBy,
  sizeSplit, setSizeSplit,
  chartWidth, setChartWidth,
  files, hostnameOverrides, onUpdateHostnameOverride,
  logoSrc, setLogoSrc,
  logoDragOver, onLogoDrop, onLogoDragOver, onLogoDragLeave,
  saving, onSaveChart,
  filenameSuffix, setFilenameSuffix,
  baselineId, setBaselineId,
  savingSpecCard, onSaveSpecCard,
}: {
  section: string, setSection: (s: string) => void,
  accuracyTest: string, setAccuracyTest: (t: string) => void,
  allModels: string[], enabledModels: Set<string>, onToggleModel: (m: string) => void,
  allImageModels: string[], enabledImageModels: Set<string>, onToggleImageModel: (m: string) => void,
  allEmbedModels: string[], enabledEmbedModels: Set<string>, onToggleEmbedModel: (m: string) => void,
  chartStyle: string, setChartStyle: (s: string) => void,
  groupBy: string, setGroupBy: (s: string) => void,
  sizeSplit: string, setSizeSplit: (s: string) => void,
  chartWidth: number, setChartWidth: (n: number) => void,
  files: DisplayFile[], hostnameOverrides: Record<string, string>,
  onUpdateHostnameOverride: (id: DisplayFile["id"], value: string) => void,
  logoSrc: string | null, setLogoSrc: (s: string | null) => void,
  logoDragOver: boolean, onLogoDrop: (e: React.DragEvent) => void,
  onLogoDragOver: (e: React.DragEvent) => void, onLogoDragLeave: (e: React.DragEvent) => void,
  saving: boolean, onSaveChart: () => void,
  filenameSuffix: string, setFilenameSuffix: (s: string) => void,
  baselineId: string | null, setBaselineId: (id: string | null) => void,
  savingSpecCard: boolean, onSaveSpecCard: () => void,
}) {
  const logoInputRef = useRef<HTMLInputElement>(null);
  const cleanSuffix = sanitizeForFilename(filenameSuffix);
  const isConcurrency = section === "concurrency_tool" || section === "concurrency_chat";
  const isLlamaBench = section === "llamabench";
  // Line-only, no tier split — same treatment as the concurrency sections.
  const isLlamaBenchConc = section === "llamabenchconc";
  const isSustained = section === "sustained";
  const isCacheComparison = section === "llm_cache_comparison";
  return (
    <div className="card" style={{ marginBottom: 20, display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
      <div>
        <div className={styles.controlLabel}>Section</div>
        <div style={{ display: "flex", gap: 6 }}>
          {SECTIONS.map(s => (
            <button type="button" key={s} aria-pressed={section === s}
              className={`pill ${section === s ? "active" : "inactive"}`} onClick={() => setSection(s)}>
              {lookup(SECTION_LABELS, s)}
            </button>
          ))}
        </div>
      </div>

      {section === "accuracy" && (
        <div className={styles.dividerGroup}>
          <div className={styles.controlLabel}>Test</div>
          <div style={{ display: "flex", gap: 6 }}>
            {ACCURACY_TESTS.map(t => (
              <button type="button" key={t} aria-pressed={accuracyTest === t}
                className={`pill ${accuracyTest === t ? "active" : "inactive"}`} onClick={() => setAccuracyTest(t)}>
                {lookup(ACCURACY_TEST_LABELS, t)}
              </button>
            ))}
          </div>
        </div>
      )}

      {section !== "accuracy" && !isConcurrency && !isLlamaBench && !isLlamaBenchConc && !isSustained && !isCacheComparison && (
        <div className={styles.dividerGroup}>
          <div className={styles.controlLabel}>Chart Style</div>
          <div style={{ display: "flex", gap: 6 }}>
            {[["bar", "Bar"], ["line", "Line"]].map(([value, label]) => (
              <button type="button" key={value} aria-pressed={chartStyle === value}
                className={`pill ${chartStyle === value ? "active" : "inactive"}`} onClick={() => setChartStyle(value)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {section !== "accuracy" && !isConcurrency && !isLlamaBenchConc && !isSustained && !isCacheComparison && (
        <div className={styles.dividerGroup}>
          <div className={styles.controlLabel}>Group By</div>
          <div style={{ display: "flex", gap: 6 }}>
            {[["model", "Model"], ["system", "System"]].map(([value, label]) => (
              <button type="button" key={value} aria-pressed={groupBy === value}
                className={`pill ${groupBy === value ? "active" : "inactive"}`} onClick={() => setGroupBy(value)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {groupBy === "system" && (["llm", "llm_cached", "llm_conversation", "llamabench"].includes(section)) && (
        <div className={styles.dividerGroup}>
          <div className={styles.controlLabel}>Model Sizes</div>
          <div style={{ display: "flex", gap: 6 }}>
            {[["tiers", "Split"], ["combined", "Combined"]].map(([value, label]) => (
              <button type="button" key={value} aria-pressed={sizeSplit === value}
                className={`pill ${sizeSplit === value ? "active" : "inactive"}`} onClick={() => setSizeSplit(value)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {files.length > 0 && (
        <div className={styles.rowBreak} />
      )}

      {files.length > 1 && !isSustained && (
        <div className={styles.freshRowGroup}>
          <div className={styles.controlLabel}>Compare As</div>
          <select
            aria-label="Comparison baseline"
            className={styles.baselineSelect}
            value={baselineId ?? ""}
            onChange={event => setBaselineId(event.target.value || null)}
          >
            <option value="">Absolute values</option>
            {files.map(file => <option key={file.id} value={String(file.id)}>% of {file.hostname}</option>)}
          </select>
          {baselineId && <div className={styles.baselineNote}>Baseline is 100%; raw tables stay absolute.</div>}
        </div>
      )}

      {files.length > 0 && (
        <div className={styles.freshRowGroup}>
          <div className={styles.controlLabel}>Labels</div>
          <div className={styles.labelFields}>
            {files.map((f, i) => (
              <div key={f.id} className={styles.labelField}>
                <div className={styles.labelFileName}>
                  <span className={styles.labelDot} style={{ background: FILE_COLORS[i % FILE_COLORS.length] }} />
                  {f.name}
                </div>
                <textarea
                  className={styles.labelTextarea}
                  value={(f.id != null ? hostnameOverrides[f.id] : undefined) ?? f.hostname}
                  onChange={e => onUpdateHostnameOverride(f.id, e.target.value)}
                  rows={4}
                  spellCheck={false}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={styles.rowBreak} />

      {(["llm", "llm_cached", "llm_cache_comparison", "llm_conversation", "accuracy"].includes(section) || isConcurrency || isLlamaBench || isLlamaBenchConc || isSustained) && allModels.length > 0 && (
        <div className={styles.freshRowGroup}>
          <div className={styles.controlLabel}>Models</div>
          <div className={styles.filterGroup}>
            {allModels.map(m => {
              const enabled = enabledModels.has(m);
              const color = getModelColor(m);
              return (
                <label
                  key={m}
                  className={`${styles.filterCheck} ${enabled ? styles.enabled : styles.disabled}`}
                  style={enabled ? { color } : undefined}
                >
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={() => onToggleModel(m)}
                    style={enabled ? { accentColor: color } : undefined}
                  />
                  {modelLabel(m)}
                </label>
              );
            })}
          </div>
        </div>
      )}

      {section === "images" && allImageModels.length > 0 && (
        <div className={styles.freshRowGroup}>
          <div className={styles.controlLabel}>Models</div>
          <div className={styles.filterGroup}>
            {allImageModels.map(m => {
              const enabled = enabledImageModels.has(m);
              const color = getImageModelColor(m);
              return (
                <label
                  key={m}
                  className={`${styles.filterCheck} ${enabled ? styles.enabled : styles.disabled}`}
                  style={enabled ? { color } : undefined}
                >
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={() => onToggleImageModel(m)}
                    style={enabled ? { accentColor: color } : undefined}
                  />
                  {imageModelLabel(m)}
                </label>
              );
            })}
          </div>
        </div>
      )}

      {section === "embeddings" && allEmbedModels.length > 0 && (
        <div className={styles.freshRowGroup}>
          <div className={styles.controlLabel}>Models</div>
          <div className={styles.filterGroup}>
            {allEmbedModels.map(m => {
              const enabled = enabledEmbedModels.has(m);
              const color = getEmbedModelColor(m);
              return (
                <label
                  key={m}
                  className={`${styles.filterCheck} ${enabled ? styles.enabled : styles.disabled}`}
                  style={enabled ? { color } : undefined}
                >
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={() => onToggleEmbedModel(m)}
                    style={enabled ? { accentColor: color } : undefined}
                  />
                  {embedModelLabel(m)}
                </label>
              );
            })}
          </div>
        </div>
      )}

      <div className={`${styles.endGroup} ${styles.dividerGroup}`}>
        <div>
          <div className={styles.controlLabel}>Chart Width</div>
          <div className={styles.widthRow}>
            <input
              aria-label="Chart width in pixels"
              type="number"
              defaultValue={chartWidth}
              key={chartWidth}
              min={400}
              max={2000}
              onBlur={e => setChartWidth(Math.min(2000, Math.max(400, parseInt(e.target.value) || 708)))}
              onKeyDown={e => e.key === "Enter" && e.currentTarget.blur()}
              className={styles.widthInput}
            />
            <span className={styles.widthUnit}>px</span>
          </div>
        </div>

        <div>
          <div className={styles.controlLabel}>Logo</div>
          <input ref={logoInputRef} type="file" accept="image/*" hidden onChange={event => {
            const file = event.target.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = () => setLogoSrc(String(reader.result));
            reader.readAsDataURL(file);
          }} />
          {logoSrc
            ? <div onDrop={onLogoDrop} onDragOver={onLogoDragOver} onDragLeave={onLogoDragLeave}
                className={`${styles.logoDropZone} ${logoDragOver ? styles.over : ""}`}>
                <div className={styles.logoPreview}>
                  <img src={logoSrc} alt="Export logo" className={styles.logoThumb} />
                  <button type="button" aria-label="Remove export logo"
                    onClick={() => setLogoSrc(null)} className={styles.logoClearBtn}>✕</button>
                </div>
              </div>
            : <button type="button" onClick={() => logoInputRef.current?.click()}
                onDrop={onLogoDrop} onDragOver={onLogoDragOver} onDragLeave={onLogoDragLeave}
                className={`${styles.logoDropZone} ${logoDragOver ? styles.over : ""}`}>
                <span className={styles.logoPlaceholder}>↓ logo</span>
              </button>}
        </div>

        <div>
          <div className={styles.controlLabel}>Filename Suffix</div>
          <input
            aria-label="Filename suffix"
            type="text"
            value={filenameSuffix}
            onChange={e => setFilenameSuffix(e.target.value)}
            placeholder="e.g. comparison 2026-07-08"
            className={styles.widthInput}
            style={{ width: 160 }}
          />
          {cleanSuffix && (
            <div className={styles.suffixPreview}>_{cleanSuffix}.png</div>
          )}
        </div>

        <div>
          <div className={styles.controlLabel}>Export</div>
          <button
            onClick={onSaveChart}
            disabled={saving}
            className={`pill inactive ${styles.exportBtn}`}
          >
            {saving ? "Saving…" : "⬇ Save PNGs"}
          </button>
          {files.length > 0 && (
            <button onClick={onSaveSpecCard} disabled={savingSpecCard}
              className={`pill inactive ${styles.exportBtn}`}>
              {savingSpecCard ? "Saving…" : "⬇ Spec Card"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
