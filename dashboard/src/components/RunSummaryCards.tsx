import type { RefObject } from "react";

import { SIZE_TIER_LABELS } from "../constants";
import type { DisplayFile } from "../types";
import { lookup } from "../utils/shared";
import { buildSpecCardSummary } from "../utils/specCard";
import styles from "./RunSummaryCards.module.css";

export default function RunSummaryCards({ files, containerRef, logoSrc }: {
  files: DisplayFile[], containerRef: RefObject<HTMLDivElement | null>, logoSrc?: string | null,
}) {
  if (!files.length) return null;
  return (
    <section className={styles.section}>
      <div className={styles.heading}>Shareable Run Cards</div>
      <div ref={containerRef} className={styles.grid}>
        {files.map(file => {
          const tiers = buildSpecCardSummary(file);
          const hostname = String(file.hostname || "").split("\n")[0];
          return (
            <article key={file.id} className={styles.card} data-spec-card data-spec-name={hostname}>
              <div className={styles.eyebrow}>LOCAL AI BENCH · RUN CARD</div>
              <div className={styles.hostname}>{hostname}</div>
              <div className={styles.metadata}>
                <span>{file.backend}</span><span>{file.os}</span>
                {file.ram_gb != null && <span>{file.ram_gb} GB RAM</span>}
                {file.engine && <span>{file.engine}{file.engineVersion ? ` ${file.engineVersion}` : ""}</span>}
                {file.version && <span>suite v{file.version}</span>}
              </div>
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
              {logoSrc && <img src={logoSrc} className={styles.logo} alt="" />}
            </article>
          );
        })}
      </div>
    </section>
  );
}
