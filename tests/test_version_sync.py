import re

import pytest

from scripts.release import version_sync as vs
from scripts.release.version_sync import (
    TARGETS,
    VersionTarget,
    apply_version,
    find_versions,
    format_conflicts,
    parse_version,
    plan_sync,
)

README = next(t for t in TARGETS if t.path == "README.md")


@pytest.mark.parametrize(
    "source, expected",
    [
        ('VERSION        = "4.1"\n', "4.1"),
        ("VERSION='4.1'", "4.1"),
        ('VERSION = "4.1.2-rc1"', "4.1.2-rc1"),
        ("# VERSION = \"4.1\"", None),
        ('    VERSION = "4.1"', None),
        ("", None),
        (None, None),
    ],
)
def test_parse_version(source, expected):
    assert parse_version(source) == expected


def test_parse_version_takes_the_real_assignment_among_neighbours():
    source = "\n".join(['IMAGE_SEED = 42', '# VERSION bumps with README', 'VERSION        = "4.12"', "N_RUNS = 3"])
    assert parse_version(source) == "4.12"


def test_readme_target_matches_only_the_title_line():
    text = "# Local AI Bench v4.1\n\nSee Local AI Bench v3.0 notes and `# Local AI Bench v9.9` inline.\n"
    assert find_versions(text, README) == ["4.1"]


def test_apply_version_preserves_prefix_and_trailing_whitespace():
    assert apply_version("# Local AI Bench v4.1\nbody\n", README, "5.0") == "# Local AI Bench v5.0\nbody\n"


def test_apply_version_on_missing_marker_is_a_no_op():
    assert apply_version("no title here\n", README, "5.0") == "no title here\n"


def _sources(readme):
    return {"README.md": readme}


def test_config_bump_rewrites_the_mirror():
    plan = plan_sync(_sources("# Local AI Bench v4.1\n"), _sources("# Local AI Bench v4.1\n"), "5.0", "4.1")
    assert plan.ok
    assert plan.updates == {"README.md": "# Local AI Bench v5.0\n"}


def test_version_edited_in_a_mirror_only_is_a_conflict():
    plan = plan_sync(_sources("# Local AI Bench v5.0\n"), _sources("# Local AI Bench v4.1\n"), "4.1", "4.1")
    assert not plan.ok
    assert plan.updates == {}
    assert plan.conflicts == [("README.md", README.description, "5.0")]


def test_mirror_edit_alongside_a_config_bump_is_rewritten_not_blocked():
    plan = plan_sync(_sources("# Local AI Bench v9.9\n"), _sources("# Local AI Bench v4.1\n"), "5.0", "4.1")
    assert plan.ok
    assert plan.updates == {"README.md": "# Local AI Bench v5.0\n"}


def test_pre_existing_drift_is_repaired_rather_than_blocking_forever():
    stale = "# Local AI Bench v3.0\n"
    plan = plan_sync(_sources(stale), _sources(stale), "4.1", "4.1")
    assert plan.ok
    assert plan.updates == {"README.md": "# Local AI Bench v4.1\n"}


def test_matching_versions_produce_no_work():
    text = "# Local AI Bench v4.1\n"
    plan = plan_sync(_sources(text), _sources(text), "4.1", "4.1")
    assert plan.ok and plan.updates == {}


def test_untracked_mirror_without_head_text_is_synced():
    plan = plan_sync(_sources("# Local AI Bench v3.0\n"), {"README.md": None}, "4.1", "4.1")
    assert plan.updates == {"README.md": "# Local AI Bench v4.1\n"}


def test_missing_file_is_skipped():
    plan = plan_sync({}, {}, "4.1", "4.1")
    assert plan.ok and plan.updates == {}


def test_missing_head_config_is_treated_as_no_bump():
    plan = plan_sync(_sources("# Local AI Bench v5.0\n"), _sources("# Local AI Bench v4.1\n"), "4.1", None)
    assert not plan.ok


def test_multiple_targets_are_planned_independently():
    other = VersionTarget("VERSION.txt", re.compile(r"^(v)(\d[\w.\-]*)(\s*)$", re.MULTILINE), "text marker")
    sources = {"README.md": "# Local AI Bench v9.9\n", "VERSION.txt": "v3.0\n"}
    head = {"README.md": "# Local AI Bench v4.1\n", "VERSION.txt": "v3.0\n"}
    plan = plan_sync(sources, head, "4.1", "4.1", targets=(README, other))
    assert plan.conflicts == [("README.md", README.description, "9.9")]
    assert plan.updates == {"VERSION.txt": "v4.1\n"}


TELEMETRY = next(t for t in TARGETS if t.path == "docs/telemetry.md")


def test_doc_prose_target_matches_only_its_anchored_sentence():
    text = (
        "Local AI Bench 4.1 sends no product telemetry, crash uploads.\n"
        "This defines the Local AI Bench 4.1 neutral methodology.\n"
        "See Local AI Bench 4.1 sends no product telemetry mid-line.\n"
    )
    assert find_versions(text, TELEMETRY) == ["4.1"]
    assert apply_version(text, TELEMETRY, "5.1.1").splitlines()[0].startswith("Local AI Bench 5.1.1 sends")


def test_two_targets_in_one_file_both_apply():
    targets = tuple(t for t in TARGETS if t.path == "docs/product-requirements.md")
    assert len(targets) == 2
    text = "The primary 4.1 product workflow helps.\n\n## Supported 4.1 scope\n\nbody\n"
    sources = {"docs/product-requirements.md": text}
    plan = plan_sync(sources, sources, "5.1.1", "5.1.1", targets=targets)
    assert plan.ok
    assert plan.updates["docs/product-requirements.md"] == (
        "The primary 5.1.1 product workflow helps.\n\n## Supported 5.1.1 scope\n\nbody\n"
    )


@pytest.mark.parametrize("path", sorted({t.path for t in TARGETS}))
def test_every_repo_mirror_agrees_with_config_today(path):
    from pathlib import Path

    root = Path(vs.__file__).resolve().parents[2]
    config_version = parse_version((root / vs.CONFIG_PATH).read_text(encoding="utf-8"))
    text = (root / path).read_text(encoding="utf-8")
    for target in (t for t in TARGETS if t.path == path):
        assert find_versions(text, target) == [config_version], target.description


def test_format_conflicts_names_the_source_of_truth_and_each_offender():
    message = format_conflicts([("README.md", "README title", "5.0")], "4.1")
    assert vs.CONFIG_PATH in message
    assert "README.md (README title) says '5.0'" in message
    assert "'4.1'" in message


def test_repo_readme_and_config_agree_today():
    from pathlib import Path

    root = Path(vs.__file__).resolve().parents[2]
    config_version = parse_version((root / vs.CONFIG_PATH).read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert find_versions(readme, README) == [config_version]
