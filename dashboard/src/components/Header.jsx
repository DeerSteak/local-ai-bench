import { useRef } from "react";
import { BACKEND_COLORS, FILE_COLORS, MAX_FILES } from "../constants";
import styles from "./Header.module.css";

function BackendTag({ backend }) {
  const style = BACKEND_COLORS[backend] || BACKEND_COLORS.cpu;
  return (
    <span className={`tag ${styles.tagBackend}`} style={{ background: style.bg, color: style.color, border: `1px solid ${style.border}` }}>
      {backend}
    </span>
  );
}

function formatTimestamp(ts) {
  if (!ts) return null;
  const d = new Date(ts);
  if (isNaN(d)) return null;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export default function Header({ files, dragOver, onDrop, onDragOver, onDragLeave, onRemoveFile, onFileInput, fileError }) {
  const fileInputRef = useRef(null);

  const atMax = files.length >= MAX_FILES;
  const dropText = dragOver
    ? (atMax ? "Drop to replace all" : files.length > 0 ? "Drop to add" : "Drop to load JSON")
    : (atMax ? "↓ Drop or click to replace all" : files.length > 0 ? `↓ Drop or click to add (${files.length}/${MAX_FILES})` : "↓ Drop or click to load JSON");

  return (
    <header className={styles.header}>
      <div className={styles.headerLeft}>
        <div className={styles.brand}>local-ai-bench · Results Explorer</div>
        <h1 className={styles.title}>AI Performance Dashboard</h1>
        {files.map((file, i) => {
          const color = FILE_COLORS[i % FILE_COLORS.length];
          return (
            <div key={file.id} className={styles.fileTagRow}>
              {files.length > 1 && (
                <span
                  className={styles.fileLabel}
                  style={{ color, background: `${color}18`, border: `1px solid ${color}60` }}
                >
                  {i + 1}
                </span>
              )}
              <span className={`tag ${styles.tagHostname}`}>{file.hostname}</span>
              <BackendTag backend={file.backend} />
              {file.os && <span className={`tag ${styles.tagOs}`}>{file.os}</span>}
              {file.ram_gb && (
                <span className={`tag ${styles.tagRam}`}>{file.ram_gb} GB RAM</span>
              )}
              {file.timestamp && (
                <span className={styles.tagTimestamp}>{formatTimestamp(file.timestamp)}</span>
              )}
            </div>
          );
        })}
      </div>

      <div className={styles.dropZoneArea}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          multiple
          onChange={onFileInput}
          style={{ display: "none" }}
        />
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => fileInputRef.current.click()}
          className={`${styles.dropZone} ${dragOver ? styles.over : ""}`}
        >
          <div className={styles.dropZoneText}>{dropText}</div>
        </div>
        {fileError && <div className={styles.fileError} role="alert">{fileError}</div>}
        {files.map((file, i) => {
          const color = FILE_COLORS[i % FILE_COLORS.length];
          return (
            <div key={file.id} className={styles.fileRow}>
              {files.length > 1 && (
                <span
                  className={styles.fileLabel}
                  style={{ color, background: `${color}18`, border: `1px solid ${color}60`, fontSize: 12, padding: "1px 5px" }}
                >
                  {i + 1}
                </span>
              )}
              <span className={styles.fileName} title={file.name}>{file.name}</span>
              <button onClick={() => onRemoveFile(file.id)} title="Remove file" className={styles.removeBtn}>✕</button>
            </div>
          );
        })}
      </div>
    </header>
  );
}
