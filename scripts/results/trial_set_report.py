"""Auditable Markdown rendering for repeated-trial artifacts."""


def _number(value, suffix="") -> str:
    return f"{value:.3f}{suffix}" if isinstance(value, (int, float)) else "unavailable"


def render_trial_set_markdown(artifact: dict) -> str:
    raw_rows = artifact.get("rows")
    rows: list = raw_rows if isinstance(raw_rows, list) else []
    lines = [
        "# Repeated-trial comparison",
        "",
        (f"Mode: {artifact.get('comparison_mode', 'unknown')}; "
         f"baseline trials: {artifact.get('baseline_trials', 0)}; "
        f"candidate trials: {artifact.get('candidate_trials', 0)}."),
        "",
    ]
    model_comparison = artifact.get("model_comparison")
    if isinstance(model_comparison, dict):
        lines.extend([
            f"Models: `{model_comparison.get('baseline', 'unknown')}` (baseline) versus "
            f"`{model_comparison.get('candidate', 'unknown')}` (candidate).",
            "",
        ])
    lines.extend([
        "| Metric | Baseline mean ± SD | Candidate mean ± SD | 95% change interval | Threshold | Drift | Verdict |",
        "|---|---:|---:|---:|---:|---|---|",
    ])
    insufficient = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_baseline, raw_candidate = row.get("baseline"), row.get("candidate")
        baseline: dict = raw_baseline if isinstance(raw_baseline, dict) else {}
        candidate: dict = raw_candidate if isinstance(raw_candidate, dict) else {}
        interval = row.get("change_interval_pct")
        interval_text = (f"{_number(interval[0], '%')} to {_number(interval[1], '%')}"
                         if isinstance(interval, list) and len(interval) == 2 else "unavailable")
        insufficient = insufficient or interval_text == "unavailable"
        drift = f"{baseline.get('drift', 'unknown')} / {candidate.get('drift', 'unknown')}"
        lines.append(
            f"| {row.get('key', 'unknown')} | {_number(baseline.get('mean'))} ± "
            f"{_number(baseline.get('stdev'))} | {_number(candidate.get('mean'))} ± "
            f"{_number(candidate.get('stdev'))} | {interval_text} "
            f"({row.get('interval_method') or 'no method'}) | "
            f"{_number(row.get('practical_threshold_pct'), '%')} | {drift} | "
            f"{row.get('verdict', 'inconclusive')} |"
        )
    lines.extend(["", "Repeated trials required for a regression verdict."
                  if insufficient else "Verdicts use compatible independent trials; within-run repetition is descriptive only."])
    return "\n".join(lines) + "\n"
