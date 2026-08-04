"""Generate a deterministic third-party notice review document from the SBOM."""

import sys
from pathlib import Path

from sbom import generate_sbom


def generate_notices(sbom):
    """Render dependency identities and keep unresolved licenses visibly blocking."""
    packages = sorted(
        sbom["packages"],
        key=lambda item: (item["ecosystem"], item["name"], item.get("version") or ""),
    )
    unresolved = [item for item in packages if item.get("license") == "NOASSERTION"]
    lines = [
        "# Third-party notices review",
        "",
        "This generated inventory is not legal approval. Review each license and add the required "
        "copyright and license text before release.",
        "",
        f"Unresolved license records: **{len(unresolved)}**",
        "",
        "| Ecosystem | Package | Version | Scope | Declared license | Source |",
        "|---|---|---|---|---|---|",
    ]
    for item in packages:
        lines.append("| " + " | ".join([
            _cell(item.get("ecosystem")), _cell(item.get("name")),
            _cell(item.get("version") or "unversioned"), _cell(item.get("scope")),
            _cell(item.get("license") or "NOASSERTION"), _cell(item.get("resolved") or "not recorded"),
        ]) + " |")
    if unresolved:
        lines += ["", "## Release blockers", ""]
        lines += [
            f"- `{item['ecosystem']}:{item['name']}` has no reviewed license assertion."
            for item in unresolved
        ]
    return "\n".join(lines) + "\n"


def write_notices(repo_root, output_path):
    """Write the canonical notice-review document for committed dependency manifests."""
    Path(output_path).write_text(generate_notices(generate_sbom(repo_root)), encoding="utf-8")


def _cell(value):
    return str(value or "").replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/third_party_notices.py OUTPUT.md")
    write_notices(Path(__file__).resolve().parents[1], sys.argv[1])
