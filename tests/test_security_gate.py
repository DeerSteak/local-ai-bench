import pytest

import security_gate
from security_gate import scan_release_tree, security_gate_result


def test_clean_release_tree_passes_without_echoing_content(tmp_path):
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    assert security_gate_result(tmp_path) == {"schema_version": 1, "passed": True, "findings": []}


@pytest.mark.parametrize(("name", "content", "kind"), [
    ("config.txt", "hf_abcdefghijklmnopqrstuvwxyz123456", "Hugging Face token"),
    ("cloud.txt", "AKIAABCDEFGHIJKLMNOP", "AWS access key"),
    ("key.pem", "-----BEGIN PRIVATE KEY-----", "private key"),
    ("request.txt", "Bearer abcdefghijklmnopqrstuvwxyz", "bearer credential"),
])
def test_secret_patterns_are_reported_without_secret_values(tmp_path, name, content, kind):
    (tmp_path / name).write_text(content, encoding="utf-8")
    findings = scan_release_tree(tmp_path)
    assert findings == ({"file": name, "kind": kind, "blocking": True},)
    assert content not in repr(findings)


def test_prohibited_credentials_and_symlinks_are_blocking(tmp_path):
    (tmp_path / "hf.txt").write_text("short", encoding="utf-8")
    target = tmp_path / "target"
    target.write_text("ordinary", encoding="utf-8")
    (tmp_path / "link").symlink_to(target)
    assert scan_release_tree(tmp_path) == (
        {"file": "hf.txt", "kind": "prohibited credential file", "blocking": True},
        {"file": "link", "kind": "symbolic link", "blocking": True},
    )


def test_oversized_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(security_gate, "MAX_SCANNED_FILE_BYTES", 3)
    (tmp_path / "large.bin").write_bytes(b"four")
    assert scan_release_tree(tmp_path) == (
        {"file": "large.bin", "kind": "file exceeds offline scan limit", "blocking": True},
    )


def test_scan_rejects_non_directory(tmp_path):
    file = tmp_path / "file"
    file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="staging directory"):
        scan_release_tree(file)
