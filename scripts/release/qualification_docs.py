"""Generate the published platform qualification matrix from evidence."""

import sys
from pathlib import Path

from scripts.release.qualification import qualification_rows
from scripts.runtime import config


START_MARKER = "<!-- qualification-matrix:start -->"
END_MARKER = "<!-- qualification-matrix:end -->"
PUBLISHED_DOCS = ("README.md", "docs/engines.md", "docs/setup.md")


def render_qualification_matrix(current_version: str) -> str:
    rows = qualification_rows(current_version)
    supported = sum(row["support_level"] == "supported" for row in rows)
    lines = [
        START_MARKER,
        f"{supported} of {len(rows)} target runtime combinations are supported by current evidence.",
        "",
        "| Target | Platform | Architecture | Runtime | Backend | Accelerator | Runtime support | ComfyUI images | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        evidence = (
            f"{row['runtime_version']}, {row['qualified_at']}, suite {row['suite_version']}"
            if row["qualified_at"] else "No qualification record"
        )
        support = row["support_level"].capitalize()
        image_support = row["image_support_level"].capitalize() \
            if row["runtime"] == "llamacpp" else "Not applicable"
        if row["stale"]:
            support += " (stale)"
            if row["runtime"] == "llamacpp" and image_support != "Unverified":
                image_support += " (stale)"
        accelerator = " / ".join(str(row["accelerator"]).splitlines())
        lines.append(
            f"| `{row['id']}` | {row['platform']} | {row['architecture']} | {row['runtime']} | "
            f"{row['backend']} | {accelerator} | {support} | {image_support} | {evidence} |"
        )
    lines.append(END_MARKER)
    return "\n".join(lines)


def replace_generated_matrix(text: str, rendered: str) -> str:
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError("qualification document requires exactly one generated matrix section")
    start = text.index(START_MARKER)
    end = text.index(END_MARKER, start) + len(END_MARKER)
    return text[:start] + rendered + text[end:]


def qualification_doc_gaps(repo_root: Path, current_version: str) -> list[str]:
    rendered = render_qualification_matrix(current_version)
    gaps = []
    for relative in PUBLISHED_DOCS:
        path = Path(repo_root) / relative
        try:
            current = path.read_text(encoding="utf-8")
            expected = replace_generated_matrix(current, rendered)
        except (OSError, ValueError):
            gaps.append(relative)
            continue
        if current != expected:
            gaps.append(relative)
    return gaps


def write_qualification_docs(repo_root: Path, current_version: str) -> None:
    rendered = render_qualification_matrix(current_version)
    for relative in PUBLISHED_DOCS:
        path = Path(repo_root) / relative
        current = path.read_text(encoding="utf-8")
        path.write_text(replace_generated_matrix(current, rendered), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    root = Path(__file__).resolve().parents[2]
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--write"):
        raise SystemExit("usage: python -m scripts.release.qualification_docs [--write]")
    if len(sys.argv) == 2:
        write_qualification_docs(root, config.VERSION)
    gaps = qualification_doc_gaps(root, config.VERSION)
    if gaps:
        print("qualification matrix is stale: " + ", ".join(gaps))
        raise SystemExit(1)
