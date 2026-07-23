"""Pure selection-state helpers for the setup model picker."""


def toggle_all_models(entries: list[dict]) -> None:
    """Toggle every install entry without changing destructive cleanup choices."""
    model_entries = [entry for entry in entries if entry["kind"] != "cleanup"]
    checked = not all(entry["checked"] for entry in model_entries)
    for entry in model_entries:
        entry["checked"] = checked


def selected_cleanup_names(entries: list[dict]) -> list[str]:
    """Return names from explicitly selected cleanup entries."""
    return [
        name
        for entry in entries
        if entry["kind"] == "cleanup" and entry["checked"]
        for name in entry["item"]["directory_names"]
    ]
