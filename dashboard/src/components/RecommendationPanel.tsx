import { entriesOf, type JsonRecord } from "../utils/shared";
import { buildRecommendationDisplayItems, type RecommendationGroup } from "../utils/recommendations";
import styles from "./RecommendationPanel.module.css";

const GROUPS: { key: RecommendationGroup, title: string }[] = [
  { key: "recommended", title: "Recommended" },
  { key: "tied", title: "Tied" },
  { key: "eliminated", title: "Eliminated" },
  { key: "unevaluated", title: "Unevaluated" },
];

const CONSTRAINT_LABELS: Record<string, string> = {
  accuracy_section: "Accuracy test",
  case: "Context / case",
  maximum_memory_gb: "Max memory",
  maximum_ttft_sec: "Max TTFT",
  minimum_accuracy_pct: "Min accuracy",
  minimum_efficiency_per_joule: "Min efficiency",
  minimum_memory_headroom_gb: "Min headroom",
  minimum_throughput: "Min throughput",
  primary_objective: "Optimize for",
  workload: "Workload",
};

const GROUP_DESCRIPTIONS: Record<RecommendationGroup, string> = {
  recommended: "Passed every hard constraint and ranked first on the selected objective.",
  tied: "Qualified trials could not establish an ordering between these candidates.",
  eliminated: "Measured evidence failed at least one hard constraint.",
  unevaluated: "Required evidence is missing; this is not a failed constraint.",
};

function constraintValue(key: string, value: JsonRecord[string]): string {
  if (key === "minimum_accuracy_pct") return `${String(value)}%`;
  if (key === "maximum_ttft_sec") return `${String(value)} sec`;
  if (key === "minimum_throughput") return `${String(value)} tokens/s`;
  if (key === "maximum_memory_gb" || key === "minimum_memory_headroom_gb") return `${String(value)} GB`;
  if (key === "minimum_efficiency_per_joule") return `${String(value)} tokens/J`;
  return String(value).replaceAll("_", " ");
}

export default function RecommendationPanel({ name, artifact }: { name: string, artifact: JsonRecord }) {
  const items = buildRecommendationDisplayItems(artifact);
  const constraints = artifact.constraints as JsonRecord;
  const display = artifact.display as JsonRecord | undefined;
  const activeGroups = GROUPS.filter(group => items.some(item => item.group === group.key));
  return (
    <main className={styles.panel}>
      <div className={styles.headingRow}>
        <div><div className={styles.eyebrow}>Goal-driven recommendation</div>
          <h2>{typeof display?.title === "string" ? display.title : name}</h2></div>
        <div className={styles.badges}>
          {display?.synthetic === true && <span className={styles.synthetic}>Synthetic example</span>}
          <span className={`${styles.verdict} ${styles[`verdict_${String(artifact.verdict)}`] || ""}`}>
            {String(artifact.verdict).replaceAll("_", " ")}
          </span>
        </div>
      </div>
      <p className={styles.intro}>Candidates are filtered by the requirements below before the selected objective is used to rank survivors.</p>
      <dl className={styles.constraints}>{entriesOf(constraints).flatMap(([key, value]) =>
        value == null ? [] : [
          <div key={key}><dt>{CONSTRAINT_LABELS[key] || key.replaceAll("_", " ")}</dt><dd>{constraintValue(key, value)}</dd></div>,
        ],
      )}</dl>
      <div className={styles.groups}>{activeGroups.map(group => {
        const groupItems = items.filter(item => item.group === group.key);
        return <section className={`${styles.group} ${styles[group.key]}`} key={group.key}>
          <h3>{group.title} <span>{groupItems.length}</span></h3>
          <p className={styles.groupDescription}>{GROUP_DESCRIPTIONS[group.key]}</p>
          {groupItems.map(item => <article key={`${group.key}-${item.candidate}`}>
            <strong>{item.candidate}</strong><span>{item.detail}</span>
            {item.evidencePath && <details><summary>Evidence references</summary><code>{item.evidencePath}</code></details>}
          </article>)}
        </section>;
      })}</div>
      <p className={styles.note}>The recommendation was calculated by Local AI Bench from the cited evidence. The dashboard displays that decision without inventing a composite score.</p>
    </main>
  );
}
