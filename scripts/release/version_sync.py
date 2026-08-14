"""Propagates config.py's VERSION to every file that mirrors it, for the pre-commit hook."""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = "scripts/runtime/config.py"
VERSION_RE = re.compile(r"""^VERSION\s*=\s*["']([^"']+)["']""", re.MULTILINE)


@dataclass(frozen=True)
class VersionTarget:
    path: str
    pattern: re.Pattern  # group 1 = prefix, group 2 = version, group 3 = suffix
    description: str


def _prose(path: str, prefix: str, suffix: str, description: str) -> VersionTarget:
    """Docs state the app version in prose; anchoring both sides keeps frozen schema/methodology versions out."""
    anchor, text = ("^", prefix[1:]) if prefix.startswith("^") else ("", prefix)
    pattern = re.compile(rf"{anchor}({re.escape(text)})(\d[\w.\-]*)({re.escape(suffix)})", re.MULTILINE)
    return VersionTarget(path=path, pattern=pattern, description=description)


TARGETS = (
    VersionTarget(
        path="README.md",
        pattern=re.compile(r"^(# Local AI Bench v)(\d[\w.\-]*)(\s*)$", re.MULTILINE),
        description="README title",
    ),
    _prose("docs/telemetry.md", "^Local AI Bench ", " sends no product telemetry", "telemetry intro"),
    _prose("docs/security-and-privacy.md", "^Version ", " sends no product telemetry", "telemetry section"),
    _prose("docs/maintenance.md", "^Version ", " has no automatic updater", "updater section"),
    _prose("docs/release-policy.md", "^Local AI Bench ", " is a preview engineering build", "policy intro"),
    _prose("docs/product-requirements.md", "^The primary ", " product workflow", "workflow intro"),
    _prose("docs/product-requirements.md", "^## Supported ", " scope", "supported-scope heading"),
)


@dataclass
class SyncPlan:
    updates: dict = field(default_factory=dict)     # path -> rewritten text
    conflicts: list = field(default_factory=list)   # (path, description, found version)

    @property
    def ok(self) -> bool:
        return not self.conflicts


def parse_version(source: str):
    match = VERSION_RE.search(source or "")
    return match.group(1) if match else None


def find_versions(text: str, target: VersionTarget) -> list:
    return [m.group(2) for m in target.pattern.finditer(text or "")]


def apply_version(text: str, target: VersionTarget, version: str) -> str:
    return target.pattern.sub(lambda m: f"{m.group(1)}{version}{m.group(3)}", text or "")


def plan_sync(sources: dict, head_sources: dict, config_version: str, head_version, targets=TARGETS) -> SyncPlan:
    """Rewrite stale mirrors from config.py's VERSION, but refuse a version edited in a mirror instead."""
    plan = SyncPlan()
    bumped = head_version is not None and head_version != config_version
    for target in targets:
        text = plan.updates.get(target.path, sources.get(target.path))
        if text is None:
            continue
        found = find_versions(text, target)
        stale = [v for v in found if v != config_version]
        if not stale:
            continue
        head_text = (head_sources or {}).get(target.path)
        edited_here = head_text is not None and find_versions(head_text, target) != found
        if edited_here and not bumped:
            plan.conflicts.extend((target.path, target.description, v) for v in stale)
        else:
            plan.updates[target.path] = apply_version(text, target, config_version)
    return plan


def format_conflicts(conflicts, config_version: str) -> str:
    lines = [
        "Version edited outside the single source of truth.",
        f"{CONFIG_PATH} says VERSION = {config_version!r}, but:",
    ]
    lines += [f"  {path} ({description}) says {found!r}" for path, description, found in conflicts]
    lines.append(f"Bump VERSION in {CONFIG_PATH} instead; the pre-commit hook rewrites the rest.")
    return "\n".join(lines)


def _git(root: Path, *args):  # pragma: no cover - subprocess seam
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )


def _head_text(root: Path, rel: str):  # pragma: no cover - subprocess seam
    result = _git(root, "show", f"HEAD:{rel}")
    return result.stdout if result.returncode == 0 else None


def main(argv=None) -> int:  # pragma: no cover - filesystem/git orchestration
    root = Path(__file__).resolve().parents[2]
    config_version = parse_version((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if not config_version:
        print(f"version-sync: could not read VERSION from {CONFIG_PATH}", file=sys.stderr)
        return 1

    sources, head_sources = {}, {}
    for target in TARGETS:
        path = root / target.path
        if path.exists():
            sources[target.path] = path.read_text(encoding="utf-8")
            head_sources[target.path] = _head_text(root, target.path)

    head_config = _head_text(root, CONFIG_PATH)
    plan = plan_sync(sources, head_sources, config_version, parse_version(head_config or ""))
    if not plan.ok:
        print(format_conflicts(plan.conflicts, config_version), file=sys.stderr)
        return 1

    for rel, text in plan.updates.items():
        (root / rel).write_text(text, encoding="utf-8")
        _git(root, "add", "--", rel)
        print(f"version-sync: set {rel} to v{config_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
