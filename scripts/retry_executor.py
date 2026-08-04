"""Retry explicitly selected eligible cases in a stopped journal-owned run."""

import json
import sys
from pathlib import Path

from recovery_executor import retry_selected_cases


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/retry_executor.py RESULT.json CASE_ID [CASE_ID ...]")
    try:
        retried = retry_selected_cases(Path(sys.argv[1]), sys.argv[2:])
    except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "error": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({
        "schema_version": 1, "status": retried["run"]["status"],
        "result": str(Path(sys.argv[1]).resolve()),
    }, indent=2))
