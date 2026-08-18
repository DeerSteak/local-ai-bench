import { entriesOf, type JsonRecord } from "../utils/shared";
import { buildRecommendationDisplayItems, type RecommendationGroup } from "../utils/recommendations";
import styles from "./RecommendationPanel.module.css";

const GROUPS: { key: RecommendationGroup, title: string }[] = [
  { key: "recommended", title: "Recommended" },
  { key: "tied", title: "Tied" },
  { key: "eliminated", title: "Eliminated" },
  { key: "unevaluated", title: "Unevaluated" },
];

export default function RecommendationPanel({ name, artifact }: { name: string, artifact: JsonRecord }) {
  const items = buildRecommendationDisplayItems(artifact);
  const constraints = artifact.constraints as JsonRecord;
  return (
    <main className={styles.panel}>
      <div className={styles.eyebrow}>Goal-driven recommendation</div>
      <h2>{name}</h2>
      <div className={`${styles.verdict} ${styles[String(artifact.verdict)] || ""}`}>
        {String(artifact.verdict).replaceAll("_", " ")}
      </div>
      <dl className={styles.constraints}>{entriesOf(constraints).flatMap(([key, value]) =>
        value == null ? [] : [
          <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(value)}</dd></div>,
        ],
      )}</dl>
      <div className={styles.groups}>{GROUPS.map(group => {
        const groupItems = items.filter(item => item.group === group.key);
        return <section className={`${styles.group} ${styles[group.key]}`} key={group.key}>
          <h3>{group.title}</h3>
          {groupItems.length ? groupItems.map(item => <article key={`${group.key}-${item.candidate}`}>
            <strong>{item.candidate}</strong><span>{item.detail}</span>
            {item.evidencePath && <code>{item.evidencePath}</code>}
          </article>) : <p>None</p>}
        </section>;
      })}</div>
      <p className={styles.note}>This view renders the authoritative Python artifact. Missing evidence is not a failed constraint, and no composite score is calculated in the browser.</p>
    </main>
  );
}
