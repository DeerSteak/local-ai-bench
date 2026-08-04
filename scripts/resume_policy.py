"""Safe resume identity snapshots and case-boundary decisions."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from run_plan import RunPlan
from result_store import atomic_write_json


IDENTITY_NAME = re.compile(r"^[A-Za-z0-9_.:@+-]+$")


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def build_resume_identity(plan: RunPlan, *, artifacts: dict[str, Path],
                          runtimes: dict[str, Path], methodology: dict[str, str],
                          digest_cache: dict | None = None) -> dict:
    names = [*artifacts, *runtimes, *methodology]
    if any(not isinstance(name, str) or not IDENTITY_NAME.fullmatch(name) for name in names):
        raise ValueError("resume identity names must be stable logical identifiers")
    return {
        "plan_id": plan.plan_id,
        "artifacts": {name: cached_file_identity(path, digest_cache)
                      for name, path in sorted(artifacts.items())},
        "runtimes": {name: cached_file_identity(path, digest_cache)
                     for name, path in sorted(runtimes.items())},
        "methodology": dict(sorted(methodology.items())),
    }


def build_engine_resume_identity(plan: RunPlan, engine, *, model_families,
                                 include_engine_runtime=True, extra_runtimes=None,
                                 digest_cache_path=None) -> dict:
    """Resolve byte identities for every journal-backed model and runtime in a plan."""
    artifacts = {}
    tags = {
        model["tag"] for family in model_families
        for model in plan.models[family] if model.get("tag")
    }
    for tag in sorted(tags):
        paths = engine.resume_artifact_paths(tag)
        for number, path in enumerate(paths, 1):
            artifacts[f"model:{tag}:part{number}"] = path
    runtimes = dict(engine.resume_runtime_paths()) if include_engine_runtime else {}
    runtimes.update(extra_runtimes or {})
    methodology = {
        "execution": hashlib.sha256(json.dumps(
            plan.execution_identity, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")).hexdigest(),
    }
    cache = load_digest_cache(digest_cache_path) if digest_cache_path else None
    identity = build_resume_identity(
        plan, artifacts=artifacts, runtimes=runtimes, methodology=methodology,
        digest_cache=cache,
    )
    if digest_cache_path:
        atomic_write_json(Path(digest_cache_path), {"schema_version": 1, "files": cache})
    return identity


def load_digest_cache(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    files = value.get("files") if isinstance(value, dict) else None
    return files if isinstance(files, dict) else {}


def cached_file_identity(path: Path, cache: dict | None) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    key = str(path)
    cached = cache.get(key) if cache is not None else None
    if isinstance(cached, dict) and cached.get("size") == stat.st_size \
            and cached.get("mtime_ns") == stat.st_mtime_ns \
            and isinstance(cached.get("sha256"), str):
        return {"sha256": cached["sha256"], "size": stat.st_size}
    identity = file_identity(path)
    if cache is not None:
        cache[key] = {**identity, "mtime_ns": stat.st_mtime_ns}
    return identity


@dataclass(frozen=True)
class ResumeDecision:
    can_resume: bool
    reasons: tuple[str, ...]
    completed_cases: tuple[str, ...]
    remaining_cases: tuple[str, ...]
    interrupted_attempts: tuple[str, ...]
    next_attempts: tuple[tuple[str, int], ...]

    @property
    def action(self) -> str:
        return "resume" if self.can_resume else "fork"


def assess_resume(saved_identity: dict, current_identity: dict, projection: dict,
                  planned_case_ids: list[str]) -> ResumeDecision:
    reasons = []
    for key, label in (
        ("plan_id", "plan"), ("artifacts", "model artifacts"),
        ("runtimes", "runtime binaries"), ("methodology", "methodology"),
    ):
        if saved_identity.get(key) != current_identity.get(key):
            reasons.append(f"{label} identity changed")
    jobs = projection.get("jobs", {})
    job_states = {value.get("state") for value in jobs.values()}
    if "complete" in job_states:
        reasons.append("job is already complete")
    cases = projection.get("cases", {})
    unknown_cases = sorted(set(cases) - set(planned_case_ids))
    if unknown_cases:
        reasons.append("journal contains cases outside the current plan")
    completed = tuple(case_id for case_id in planned_case_ids
                      if cases.get(case_id, {}).get("state") == "complete")
    remaining = tuple(case_id for case_id in planned_case_ids if case_id not in completed)
    attempts = projection.get("attempts", {})
    interrupted_attempts = tuple(sorted(
        attempt_id for attempt_id, attempt in attempts.items()
        if attempt.get("state") == "running"
    ))
    next_attempts = []
    for case_id in remaining:
        numbers = [
            attempt.get("number") for attempt in attempts.values()
            if attempt.get("parent_id") == case_id and isinstance(attempt.get("number"), int)
        ]
        next_attempts.append((case_id, max(numbers, default=0) + 1))
    return ResumeDecision(
        not reasons, tuple(reasons), completed, remaining,
        interrupted_attempts, tuple(next_attempts),
    )
