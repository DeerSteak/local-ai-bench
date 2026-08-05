"""Execute a reviewed saved plan as a new journal-owned run."""

import json
import sys
from pathlib import Path

from scripts.results.recovery_executor import fork_journal_run


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m scripts.results.fork_executor SOURCE.json OUTPUT.json")
    try:
        forked = fork_journal_run(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "error": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({
        "schema_version": 1, "status": forked["run"]["status"],
        "result": str(Path(sys.argv[2]).resolve()),
    }, indent=2))
