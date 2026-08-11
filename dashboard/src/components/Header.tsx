import { useRef } from "react";
import { BACKEND_COLORS, FILE_COLORS, MAX_FILES, SUITE_VERSION } from "../constants";
import { backendLabel, lookup } from "../utils/shared";
import type { DisplayFile } from "../types";
import styles from "./Header.module.css";

function BackendTag({ backend }: { backend: string }) {
  const style = lookup(BACKEND_COLORS, backend) || BACKEND_COLORS.cpu;
  return (
    <span className={`tag ${styles.tagBackend}`} style={{ background: style.bg, color: style.color, border: `1px solid ${style.border}` }}>
      {backendLabel(backend)}
    </span>
  );
}

function formatTimestamp(ts: string | null): string | null {
  if (!ts) return null;
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export default function Header({ files, dragOver, onDrop, onDragOver, onDragLeave, onRemoveFile, onFileInput, fileError }: {
  files: DisplayFile[], dragOver: boolean, onDrop: (e: React.DragEvent) => void,
  onDragOver: (e: React.DragEvent) => void, onDragLeave: (e: React.DragEvent) => void,
  onRemoveFile: (id: DisplayFile["id"]) => void, onFileInput: (e: React.ChangeEvent<HTMLInputElement>) => void,
  fileError: string | null,
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const atMax = files.length >= MAX_FILES;
  const dropText = dragOver
    ? (atMax ? "Drop to replace all" : files.length > 0 ? "Drop to add" : "Drop to load JSON")
    : (atMax ? "↓ Drop or click to replace all" : files.length > 0 ? `↓ Drop or click to add (${files.length}/${MAX_FILES})` : "↓ Drop or click to load JSON");

  return (
    <header className={styles.header}>
      <div className={styles.headerLeft}>
        <div className={styles.brand}>
          local-ai-bench · Results Explorer
          {SUITE_VERSION && <span className={styles.suiteVersion}>v{SUITE_VERSION}</span>}
        </div>
        <h1 className={styles.title}>AI Performance Dashboard</h1>
        {files.map((file, i) => {
          const color = FILE_COLORS[i % FILE_COLORS.length];
          const identityLines = String(file.hostname || "").split("\n");
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
              <div className={styles.fileIdentity}>
                <span className={`tag ${styles.tagHostname}`}>{identityLines[0]}</span>
                <BackendTag backend={file.backend} />
                {identityLines.length > 1 && (
                  <span className={styles.tagEngine}>{identityLines.at(-1)}</span>
                )}
              </div>
              {file.os && <span className={`tag ${styles.tagOs}`}>{file.os}</span>}
              {file.wsl && (
                <span
                  className={`tag ${styles.tagWsl}`}
                  title="Ran inside WSL2 — GPU access is virtualized, so results are not directly comparable to bare-metal Linux"
                >
                  WSL2
                </span>
              )}
              {file.ram_gb && (
                <span className={`tag ${styles.tagRam}`}>{file.ram_gb} GB RAM</span>
              )}
              {file.version && (
                <span className={`tag ${styles.tagVersion}`} title="Benchmark suite version that produced this file">
                  v{file.version}
                </span>
              )}
              {file.timestamp && (
                <span className={styles.tagTimestamp}>{formatTimestamp(file.timestamp)}</span>
              )}
              {file.reliabilityWarning && (
                <span className={styles.reliabilityWarning} role="status">{file.reliabilityWarning}</span>
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
          onClick={() => fileInputRef.current?.click()}
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
