import pytest

from scripts.setup.model_inventory import (
    catalog_vllm_repos, delete_non_catalog_vllm_repos, find_non_catalog_vllm_repos,
    hf_cache_repo_id,
)


def make_cached_repo(cache_home, repo, *, weights=True, size=2048):
    directory = cache_home / "hub" / ("models--" + repo.replace("/", "--"))
    snapshot = directory / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (directory / "refs").mkdir(exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"\0" * size)
    return directory


CATALOG_REPO = "cyankiwi/Qwen3.5-9B-AWQ-4bit"


@pytest.mark.parametrize("name,expected", [
    ("models--cyankiwi--Qwen3.5-9B-AWQ-4bit", "cyankiwi/Qwen3.5-9B-AWQ-4bit"),
    ("models--nvidia--Nemotron-Cascade-2-30B-A3B", "nvidia/Nemotron-Cascade-2-30B-A3B"),
    ("datasets--foo--bar", None),
    ("models--onlyowner", None),
    ("models--", None),
    ("models----name", None),
    ("version.txt", None),
])
def test_cache_directory_names_map_to_repo_ids(name, expected):
    assert hf_cache_repo_id(name) == expected


def test_catalog_repos_come_from_both_llm_and_embedding_entries():
    repos = catalog_vllm_repos()
    assert CATALOG_REPO in repos
    assert "nomic-ai/nomic-embed-text-v1.5" in repos


def test_discovery_reports_only_repos_the_catalog_does_not_own(tmp_path):
    make_cached_repo(tmp_path, CATALOG_REPO)
    make_cached_repo(tmp_path, "someone/Old-Model-AWQ")
    make_cached_repo(tmp_path, "another/Stale-Build")
    found = find_non_catalog_vllm_repos(tmp_path)
    assert [entry["repo"] for entry in found] == ["another/Stale-Build", "someone/Old-Model-AWQ"]
    assert all(entry["size"] > 0 for entry in found)


def test_discovery_ignores_cache_entries_holding_no_weights(tmp_path):
    """A tokenizer-only or dataset entry is not a model and must not be offered."""
    make_cached_repo(tmp_path, "someone/Tokenizer-Only", weights=False)
    (tmp_path / "hub" / "datasets--foo--bar" / "snapshots" / "a").mkdir(parents=True)
    assert find_non_catalog_vllm_repos(tmp_path) == []


def test_discovery_survives_a_missing_or_empty_cache(tmp_path):
    assert find_non_catalog_vllm_repos(tmp_path / "absent") == []
    (tmp_path / "hub").mkdir()
    assert find_non_catalog_vllm_repos(tmp_path) == []


def test_deletion_removes_only_the_named_entries(tmp_path):
    keep = make_cached_repo(tmp_path, "someone/Keep-Me")
    drop = make_cached_repo(tmp_path, "someone/Drop-Me")
    removed, failures = delete_non_catalog_vllm_repos(tmp_path, [drop.name])
    assert removed == [drop.name] and failures == {}
    assert not drop.exists() and keep.exists()


def test_deletion_refuses_a_repo_the_catalog_owns(tmp_path):
    """The cleanup list is passed in from the UI; a catalog model must never be
    deletable through it, however it got named."""
    catalog = make_cached_repo(tmp_path, CATALOG_REPO)
    removed, failures = delete_non_catalog_vllm_repos(tmp_path, [catalog.name])
    assert removed == []
    assert "not an eligible non-catalog cache entry" in failures[catalog.name]
    assert catalog.exists()


@pytest.mark.parametrize("name", [
    "../escape", "hub/nested", "models--someone--Absent", "datasets--foo--bar", "notacache",
])
def test_deletion_refuses_paths_that_are_not_eligible_cache_entries(tmp_path, name):
    make_cached_repo(tmp_path, "someone/Real-Model")
    removed, failures = delete_non_catalog_vllm_repos(tmp_path, [name])
    assert removed == [] and name in failures


def test_deletion_refuses_an_entry_that_lost_its_weights(tmp_path):
    empty = make_cached_repo(tmp_path, "someone/Tokenizer-Only", weights=False)
    removed, failures = delete_non_catalog_vllm_repos(tmp_path, [empty.name])
    assert removed == []
    assert failures[empty.name] == "cache entry holds no model weights"
    assert empty.exists()


def test_deletion_does_not_follow_a_symlinked_cache_entry(tmp_path):
    """A symlink could point outside the cache; unlinking the name is not enough
    of a guarantee, so it is refused outright."""
    real = make_cached_repo(tmp_path, "elsewhere/Real")
    link = tmp_path / "hub" / "models--someone--Linked"
    link.symlink_to(real, target_is_directory=True)
    removed, failures = delete_non_catalog_vllm_repos(tmp_path, [link.name])
    assert removed == [] and link.name in failures
    assert real.exists() and link.is_symlink()


# ── selection plumbing ──

def test_selection_separates_the_two_cleanup_kinds():
    """The GGUF folders and the shared HF cache are deleted by different code with
    different guards, so a checked box must not leak from one list into the other."""
    from scripts.setup.setup_selection import selected_cleanup_names
    entries = [
        {"kind": "cleanup", "checked": True, "item": {"directory_names": ["custom-gguf"]}},
        {"kind": "vllm_cleanup", "checked": True,
         "item": {"directory_names": ["models--someone--Old-AWQ"]}},
        {"kind": "vllm_cleanup", "checked": False,
         "item": {"directory_names": ["models--someone--Keep"]}},
        {"kind": "llm", "checked": True, "item": {"tag": "x"}},
    ]
    assert selected_cleanup_names(entries) == ["custom-gguf"]
    assert selected_cleanup_names(entries, "vllm_cleanup") == ["models--someone--Old-AWQ"]


def test_nothing_is_selected_for_cleanup_by_default():
    from scripts.setup.setup_selection import selected_cleanup_names
    entries = [
        {"kind": "cleanup", "checked": False, "item": {"directory_names": ["a"]}},
        {"kind": "vllm_cleanup", "checked": False, "item": {"directory_names": ["b"]}},
    ]
    assert selected_cleanup_names(entries) == []
    assert selected_cleanup_names(entries, "vllm_cleanup") == []
