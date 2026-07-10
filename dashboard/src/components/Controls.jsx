import { SECTIONS, SECTION_LABELS, FILE_COLORS } from "../constants";
import {
  modelLabel, imageModelLabel, embedModelLabel,
  getModelColor, getImageModelColor, getEmbedModelColor, sanitizeForFilename,
} from "../utils";
import styles from "./Controls.module.css";

export default function Controls({
  section, setSection,
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
}) {
  const cleanSuffix = sanitizeForFilename(filenameSuffix);
  return (
    <div className="card" style={{ marginBottom: 20, display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
      <div>
        <div className={styles.controlLabel}>Section</div>
        <div style={{ display: "flex", gap: 6 }}>
          {SECTIONS.map(s => (
            <button key={s} className={`pill ${section === s ? "active" : "inactive"}`} onClick={() => setSection(s)}>
              {SECTION_LABELS[s]}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.dividerGroup}>
        <div className={styles.controlLabel}>Chart Style</div>
        <div style={{ display: "flex", gap: 6 }}>
          {[["bar", "Bar"], ["line", "Line"]].map(([value, label]) => (
            <button key={value} className={`pill ${chartStyle === value ? "active" : "inactive"}`} onClick={() => setChartStyle(value)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.dividerGroup}>
        <div className={styles.controlLabel}>Group By</div>
        <div style={{ display: "flex", gap: 6 }}>
          {[["model", "Model"], ["system", "System"]].map(([value, label]) => (
            <button key={value} className={`pill ${groupBy === value ? "active" : "inactive"}`} onClick={() => setGroupBy(value)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {groupBy === "system" && (section === "llm" || section === "llm_conversation") && (
        <div className={styles.dividerGroup}>
          <div className={styles.controlLabel}>Model Sizes</div>
          <div style={{ display: "flex", gap: 6 }}>
            {[["tiers", "Split"], ["combined", "Combined"]].map(([value, label]) => (
              <button key={value} className={`pill ${sizeSplit === value ? "active" : "inactive"}`} onClick={() => setSizeSplit(value)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {files.length > 0 && (
        <div className={styles.dividerGroup}>
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
                  value={hostnameOverrides[f.id] ?? f.hostname}
                  onChange={e => onUpdateHostnameOverride(f.id, e.target.value)}
                  rows={2}
                  spellCheck={false}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {(section === "llm" || section === "llm_conversation") && allModels.length > 0 && (
        <div className={styles.dividerGroup}>
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
        <div className={styles.dividerGroup}>
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
        <div className={styles.dividerGroup}>
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

      <div className={styles.endGroup}>
        <div>
          <div className={styles.controlLabel}>Chart Width</div>
          <div className={styles.widthRow}>
            <input
              type="number"
              defaultValue={chartWidth}
              key={chartWidth}
              min={400}
              max={2000}
              onBlur={e => setChartWidth(Math.min(2000, Math.max(400, parseInt(e.target.value) || 708)))}
              onKeyDown={e => e.key === "Enter" && e.target.blur()}
              className={styles.widthInput}
            />
            <span className={styles.widthUnit}>px</span>
          </div>
        </div>

        <div>
          <div className={styles.controlLabel}>Logo</div>
          <div
            onDrop={onLogoDrop}
            onDragOver={onLogoDragOver}
            onDragLeave={onLogoDragLeave}
            className={`${styles.logoDropZone} ${logoDragOver ? styles.over : ""}`}
          >
            {logoSrc
              ? <div className={styles.logoPreview}>
                  <img src={logoSrc} className={styles.logoThumb} />
                  <button onClick={() => setLogoSrc(null)} className={styles.logoClearBtn}>✕</button>
                </div>
              : <span className={styles.logoPlaceholder}>↓ logo</span>
            }
          </div>
        </div>

        <div>
          <div className={styles.controlLabel}>Filename Suffix</div>
          <input
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
            {saving ? "Saving…" : "⬇ Save PNG"}
          </button>
        </div>
      </div>
    </div>
  );
}
