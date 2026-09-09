import type { RefObject } from "react";

import { SIZE_TIER_LABELS } from "../constants";
import type { DisplayFile } from "../types";
import { backendLabel, engineRunLabel, lookup } from "../utils/shared";
import { buildSpecCardSummary, runCardGpuLabels, runCardHostname } from "../utils/specCard";
import { runHeadroomSummary } from "../utils/memory";
import { powerScopeLabel, runPowerSummary } from "../utils/power";
import { nativeRunCardSummary } from "../utils/nativeRunCard";
import styles from "./RunSummaryCards.module.css";

export default function RunSummaryCards({ files, containerRef, logoSrc, chartWidth, section }: {
  files: DisplayFile[], containerRef: RefObject<HTMLDivElement | null>, logoSrc?: string | null,
  chartWidth: number, section: string,
}) {
  if (!files.length) return null;
  return (
    <section
      className={styles.section}
      style={{ width: chartWidth, maxWidth: "calc(100vw - 40px)" }}
    >
      <div className={styles.heading}>Shareable Run Cards</div>
      <div ref={containerRef} className={styles.grid}>
        {files.map(file => {
          const native = ["llamabench", "llamabenchconc"].includes(section) ? nativeRunCardSummary(file, section) : null;
          const tiers = native ? [] : buildSpecCardSummary(file);
          const gpuLabels = runCardGpuLabels(file);
          const hostname = runCardHostname(file);
          const headroom = native?.headroom ?? runHeadroomSummary(file);
          const power = native?.power ?? runPowerSummary(file);
          return (
            <article key={file.id} className={styles.card} data-spec-card data-spec-name={hostname}>
              <div className={styles.eyebrow}>LOCAL AI BENCH · RUN CARD{native ? (section === "llamabench" ? " · LLAMA-BENCH" : " · LLAMA-BENCH CONCURRENCY") : ""}</div>
              <div className={styles.hostname}>{hostname}</div>
              <div className={styles.metadata}>
                <span>{backendLabel(file.backend)}</span><span>{file.os}</span>
                {file.ram_gb != null && <span>{file.ram_gb} GB RAM</span>}
                {file.engine && <span>{engineRunLabel(file, section)}{file.engineVersion ? ` ${file.engineVersion}` : ""}</span>}
                {file.version && <span>suite v{file.version}</span>}
              </div>
              {gpuLabels.length > 0 && (
                <div className={`${styles.metadata} ${styles.gpuMetadata}`}>
                  {gpuLabels.map(gpu => <span key={gpu}>{gpu}</span>)}
                </div>
              )}
              <div className={styles.headroom} data-state={headroom.state}>
                <span>{native ? "Tab memory headroom" : "Memory headroom"}</span>
                <strong>{headroom.absoluteGb == null
                  ? "Not recorded"
                  : `${headroom.absoluteGb.toFixed(1)} GB · ${headroom.state}`}</strong>
                {headroom.casePath && <small>{headroom.casePath}</small>}
              </div>
              <div className={styles.headroom} data-state={power.status}>
                <span>{native ? "Tab measured energy" : "Measured energy"} · {powerScopeLabel(power.scope)}</span>
                <strong>{power.energyJoules == null
                  ? (power.reason || "Not recorded")
                  : `${power.energyJoules.toFixed(1)} J`}</strong>
                {power.energyJoules != null && power.reason && <small>{power.reason}</small>}
                {power.idleWatts != null && <small>Idle baseline {power.idleWatts.toFixed(1)} W</small>}
              </div>
              {native ? <>
                <div className={styles.context}>{section === "llamabench" ? "Native throughput leaders by model tier" : "Aggregate decode leaders by model tier"}</div>
                <div className={styles.context}>Matching cases with the most model coverage; ties use {section === "llamabench" ? "the shallowest depth" : "the highest concurrency"}.</div>
                {native.leaders.length ? <div className={styles.tiers}>
                  {native.leaders.map(leader => <div key={`${leader.tier}_${leader.kind}`} className={styles.tier}>
                    <div className={styles.tierName}>{lookup(SIZE_TIER_LABELS, leader.tier) || leader.tier} · {leader.kind}</div>
                    <div className={styles.context}>{leader.checkpoint}</div>
                    <div><span>FASTEST</span><strong>{leader.model}</strong><b>{leader.value.toFixed(1)} tps</b></div>
                  </div>)}
                </div> : <div className={styles.empty}>No comparable measurements for this tab</div>}
              </> : <>
              <div className={styles.context}>Single-shot leaders by model tier</div>
              {tiers.length ? (
                <div className={styles.tiers}>
                  {tiers.map(tier => (
                    <div key={tier.tier} className={styles.tier}>
                      <div className={styles.tierName}>{lookup(SIZE_TIER_LABELS, tier.tier) || tier.tier} · {tier.checkpoint}</div>
                      <div><span>FASTEST</span><strong>{tier.fastest.model}</strong><b>{tier.fastest.value.toFixed(1)} tps</b></div>
                      <div><span>LOWEST TTFT</span><strong>{tier.lowestTtft.model}</strong><b>{tier.lowestTtft.value.toFixed(2)}s</b></div>
                    </div>
                  ))}
                </div>
              ) : <div className={styles.empty}>No comparable LLM measurements</div>}
              </>}
              {logoSrc && <img src={logoSrc} className={styles.logo} alt="" />}
            </article>
          );
        })}
      </div>
    </section>
  );
}
