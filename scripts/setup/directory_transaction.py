"""Transactional replacement of a staged directory with rollback."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class DirectoryValidation(Protocol):
    @property
    def success(self) -> bool: ...
    @property
    def detail(self) -> str: ...
    @property
    def version(self) -> str | None: ...


@dataclass(frozen=True)
class DirectorySwapOutcome:
    validation: DirectoryValidation | None
    backup_cleanup_error: OSError | None = None


@dataclass(frozen=True)
class DirectorySwapSpec:
    target: Path
    staged: Path
    backup: Path
    had_target: bool = True


class DirectorySwapError(RuntimeError):
    def __init__(self, cause: BaseException | str, rollback_error: BaseException | None = None):
        message = str(cause)
        if rollback_error is not None:
            message = f"{message}; rollback: {rollback_error}"
        super().__init__(message)
        self.rollback_error = rollback_error


def swap_staged_directories(specs: list[DirectorySwapSpec], *,
                            replace: Callable[[Path, Path], object],
                            remove: Callable[[Path], object]) -> tuple[OSError, ...]:
    """Commit several staged directories as one rollback unit."""
    backed_up: list[DirectorySwapSpec] = []
    installed: list[DirectorySwapSpec] = []
    try:
        for spec in specs:
            if spec.had_target:
                replace(spec.target, spec.backup)
                backed_up.append(spec)
        for spec in specs:
            replace(spec.staged, spec.target)
            installed.append(spec)
    except Exception as exc:
        rollback_errors = []
        for spec in reversed(installed):
            try:
                if spec.target.exists():
                    remove(spec.target)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        for spec in reversed(backed_up):
            try:
                replace(spec.backup, spec.target)
            except Exception as rollback_exc:
                rollback_errors.append(rollback_exc)
        rollback_error = rollback_errors[0] if rollback_errors else None
        raise DirectorySwapError(exc, rollback_error) from exc
    cleanup_errors = []
    for spec in backed_up:
        try:
            remove(spec.backup)
        except OSError as exc:
            cleanup_errors.append(exc)
    return tuple(cleanup_errors)


def swap_staged_directory(target: Path, staged: Path, backup: Path, *,
                          had_target: bool,
                          replace: Callable[[Path, Path], object],
                          remove: Callable[[Path], object],
                          validate: Callable[[Path], DirectoryValidation] | None = None,
                          ) -> DirectorySwapOutcome:
    if had_target:
        replace(target, backup)
    try:
        replace(staged, target)
        validation = validate(target) if validate is not None else None
        if validation is not None and not validation.success:
            raise RuntimeError(validation.detail)
    except Exception as exc:
        try:
            if target.exists():
                remove(target)
            if had_target:
                replace(backup, target)
        except Exception as rollback_exc:
            raise DirectorySwapError(exc, rollback_exc) from rollback_exc
        raise DirectorySwapError(exc) from exc
    if not had_target:
        return DirectorySwapOutcome(validation)
    try:
        remove(backup)
    except OSError as exc:
        return DirectorySwapOutcome(validation, exc)
    return DirectorySwapOutcome(validation)
