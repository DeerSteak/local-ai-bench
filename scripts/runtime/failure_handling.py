"""Consistent result entries for unexpected per-model failures."""

from datetime import datetime


def unexpected_model_failure(label: str, exc: BaseException, *,
                             crashed: bool | None = None) -> dict:
    try:
        detail = str(exc)
    except Exception:
        detail = "(exception could not be formatted)"
    entry = {
        "label": label, "unexpected_error": True,
        "crashed_at": datetime.now().isoformat(timespec="seconds"),
        "error": f"{type(exc).__name__}: {detail}",
    }
    if crashed is not None:
        entry["crashed"] = crashed
    return entry
