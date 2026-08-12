# Contributor Workflow

This document defines how changes move from local work to a stable Local AI Bench release. It covers branch ownership, pull-request targets, commit and merge conventions, required validation, release stabilization, hotfixes, and the GitHub settings that enforce the process.

## Core principle

`main` is the public default branch and always represents the latest stable release. `develop` is the tested integration branch for the next release. Ordinary work moves forward from `develop` through a versioned release branch to `main`:

```text
feature/* ──→ develop ──→ release/X.Y ──→ main ──→ vX.Y tag
                  ↑              │
                  └──────────────┘
                    stabilization
```

Repository visitors see `main`; contributors explicitly create ordinary branches from `develop` and target pull requests there.

## Branch roles

| Branch | Purpose | Created from | Merges into |
|---|---|---|---|
| `main` | Latest stable release and public default | Completed `release/X.Y` or urgent `hotfix/*` | Nothing during ordinary development |
| `develop` | Tested integration state for the next release | Final stable release initially; then retained | `release/X.Y` when scope freezes |
| `release/X.Y` | Frozen release candidate undergoing versioning, documentation, qualification, and stabilization | `develop` | `main`, then back into `develop` |
| `feature/*`, `fix/*`, `chore/*` | Short-lived implementation or maintenance slice | `develop` | `develop` |
| `fix/X.Y-description` | Release-blocking stabilization slice | Active `release/X.Y` | Active release branch and `develop` |
| `hotfix/X.Y.Z` | Urgent correction to the latest stable release | `main` | `main`, `develop`, and any active newer release branch |

Do not use a release branch as the general integration branch for subsequent work. Once a version is released, ordinary development resumes from `develop`.

## Starting ordinary work

Synchronize `develop` and create a descriptive short-lived branch:

```bash
git switch develop
git pull --ff-only origin develop
git switch -c feature/descriptive-name
```

Use `feature/` for new capability, `fix/` for unreleased defects, and `chore/` for maintenance or technical debt. Keep each branch focused enough to review and revert independently.

Commit coherent slices rather than one large mixed commit. A refactor with separable boundaries should use separate commits for each boundary, its tests, and its documentation when that makes review clearer. Never push secrets, generated benchmark results, `hf.txt`, local environments, or vendored `ComfyUI/` changes.

Before opening a pull request, rebase or merge the current `develop` only when needed to resolve drift, run the applicable validation, and push the branch:

```bash
git push -u origin feature/descriptive-name
```

Open ordinary pull requests against `develop`, not `main` or `release/X.Y`.

## Pull-request expectations

A pull request should explain the problem, the resulting behavior or ownership boundary, important design choices, tests run, documentation changes, and known residual risk. State explicitly when no user-facing behavior or results-schema change is intended.

Reviewable pull requests should:

- keep unrelated cleanup out of the change;
- preserve public entry points and compatibility contracts unless the PR explicitly changes them;
- add meaningful tests for new business logic and adversarial regression cases;
- update documentation and dashboard consumers when behavior or data shape changes;
- avoid executing live setup or benchmark entry points as tests;
- resolve review threads and rerun validation after material changes.

Short-lived feature, fix, and chore pull requests should normally be squash-merged into `develop`. The pull-request title becomes the durable integration commit, so write it as an imperative summary.

## Required validation

Every Python change must pass:

```bash
bash tests.sh
pyright
```

Dashboard changes must additionally pass:

```bash
cd dashboard
npm test
npm run lint
npx tsc --noEmit
```

Tkinter controller tests run without a display. Real screen-construction and application-smoke tests run locally when a display is available and under Xvfb in Linux CI. Pull requests targeting `develop`, `release/**`, or `main` run dependency review, the Python suite, Pyright, and all dashboard gates; keep every `Pull request CI` job green.

Follow the stronger release gates in [Release Policy](release-policy.md) when preparing a stable version. Passing the ordinary unit suite alone is not stable-release approval.

## Preparing a release

When the next version's scope is complete and `develop` is green, cut the release branch:

```bash
git switch develop
git pull --ff-only origin develop
git switch -c release/5.2
git push -u origin release/5.2
```

The release branch accepts only stabilization work: the version bump, release notes, documentation corrections, qualification evidence, and defects that block that release. New unrelated features remain on `develop` for the following version.

Create each release correction from the active release branch:

```bash
git switch release/5.2
git pull --ff-only origin release/5.2
git switch -c fix/5.2-description
```

Merge the reviewed correction into `release/5.2` and also ensure it reaches `develop`. Prefer merging the same focused correction into both destinations while stabilization continues; the final release-back merge reconciles anything remaining.

`scripts/runtime/config.py` is the only application-version source. Keep `develop` at the latest released version during ordinary integration and bump to X.Y on `release/X.Y`; the version-sync hook updates README mirrors.

## Publishing a release

After all release gates pass, merge `release/X.Y` into `main` with a merge commit. Do not squash or rebase the release merge: shared ancestry makes the release boundary visible and keeps reconciliation with `develop` straightforward.

Tag the resulting `main` commit:

```bash
git switch main
git pull --ff-only origin main
git tag -a v5.2 -m "Local AI Bench 5.2"
git push origin v5.2
```

Then merge `release/X.Y` back into `develop` with a merge commit so final stabilization fixes and release metadata remain in future work. Delete the release branch after both merges complete; the immutable tag preserves the released source.

## Emergency hotfixes

Create an urgent stable correction from `main`:

```bash
git switch main
git pull --ff-only origin main
git switch -c hotfix/5.2.1
```

After review and validation, merge the hotfix into `main`, tag the patch release, and merge the same correction into `develop`. If a newer release branch is active, bring the hotfix into that branch as well. Do not wait for the next release-back merge to carry a production security or correctness fix forward.

## Current GitHub settings

As inspected on August 12, 2026, GitHub is configured with:

- `main` as the default branch;
- repository-level support for squash, merge-commit, and rebase merging;
- automatic source-branch deletion disabled;
- an active `protect main, develop, and release` ruleset targeting the default branch, `develop`, and `release/**`;
- deletion and non-fast-forward updates blocked on those branches;
- one approving review, stale-review dismissal, Code Owner review, and approval after the latest push required;
- only squash merges allowed by that ruleset;
- no required status checks in that ruleset.

The squash-only rule conflicts with the release ancestry policy above. Before using this workflow for the next release, update repository enforcement so release pull requests can use merge commits while ordinary feature pull requests remain squash merges.

## Target GitHub protection

Use separate rulesets because normal integration and release publication need different merge semantics.

### `main` and `release/**`

- Block deletion and force pushes.
- Require a pull request, one approval, Code Owner review, approval after the latest push, and resolved review threads.
- Require `Dependency review`, `Python tests (Tk/Xvfb)`, `Pyright`, `Dashboard tests, lint, and types`, release/security gates, and any future applicable checks.
- Allow merge commits for `release/X.Y` into `main` and for release reconciliation.
- Restrict bypass to explicit repository administrators for emergencies, with the bypass reason recorded.

### `develop`

- Block deletion and force pushes.
- Require pull requests and the `Dependency review`, `Python tests (Tk/Xvfb)`, `Pyright`, and `Dashboard tests, lint, and types` checks.
- Require review-thread resolution; require an approval when more than one contributor is active.
- Allow squash merges for ordinary `feature/*`, `fix/*`, and `chore/*` pull requests.

GitHub cannot assign merge methods by source branch within one ordinary branch rule cleanly enough for this workflow. Keep repository-level merge commits enabled, use branch-specific rulesets where possible, and enforce the release merge method through this document and release review.

## Maintainer release checklist

Before declaring a version complete:

1. Confirm the release branch was cut from current `develop`.
2. Confirm only stabilization changes entered after the cut.
3. Complete the required release-policy evidence and automated checks.
4. Merge the release into `main` with a merge commit.
5. Tag the exact `main` merge commit as `vX.Y` or `vX.Y.Z`.
6. Merge the release branch back into `develop` with a merge commit.
7. Confirm `main` renders the released README and remains GitHub's default branch.
8. Delete the completed release and short-lived source branches.

---

[← Back to README](../README.md) · [Testing](testing.md) · [Release Policy →](release-policy.md)
