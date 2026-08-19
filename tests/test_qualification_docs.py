from scripts.release.qualification_docs import (
    END_MARKER, START_MARKER, qualification_doc_gaps,
    render_qualification_matrix, replace_generated_matrix,
)


def test_rendered_matrix_includes_reviewed_support_and_defaults_others_to_unverified():
    rendered = render_qualification_matrix("6.0-pre8")
    assert "| macos | arm64 | llamacpp | metal | Supported | b10488, 2026-08-18, suite 6.0-pre8 |" in rendered
    assert "| windows | x86_64 | llamacpp | cuda | Supported | 0.1.2-dev, 2026-08-19, suite 6.0-pre8 |" in rendered
    assert "| linux | aarch64 | llamacpp | cuda | Supported | 0.1.2-dev, 2026-08-19, suite 6.0-pre8 |" in rendered
    assert "| linux | aarch64 | vllm | cuda | Supported | 0.27.1, 2026-08-19, suite 6.0-pre8 |" in rendered
    assert "| wsl2 | x86_64 | vllm | cuda | Unverified | No qualification record |" in rendered


def test_document_drift_is_detected_and_replacement_is_exact(tmp_path, monkeypatch):
    relative = "docs/example.md"
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_text(f"Before\n\n{START_MARKER}\nold\n{END_MARKER}\n\nAfter\n")
    monkeypatch.setattr("scripts.release.qualification_docs.PUBLISHED_DOCS", (relative,))
    rendered = render_qualification_matrix("6.0-pre8")
    assert qualification_doc_gaps(tmp_path, "6.0-pre8") == [relative]
    path.write_text(replace_generated_matrix(path.read_text(), rendered))
    assert qualification_doc_gaps(tmp_path, "6.0-pre8") == []


def test_document_without_exact_markers_is_rejected():
    try:
        replace_generated_matrix("no generated section", "matrix")
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("missing generated markers were accepted")
