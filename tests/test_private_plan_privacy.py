from pathlib import Path


def test_public_documentation_does_not_name_private_planning_file():
    root = Path(__file__).resolve().parents[1]
    public_docs = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    prohibited = ("COMMERCIAL_PLAN.md", "private commercial roadmap")
    for path in public_docs:
        content = path.read_text(encoding="utf-8")
        assert not any(term in content for term in prohibited), path


def test_private_planning_file_stays_gitignored():
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "COMMERCIAL_PLAN.md" in ignored
