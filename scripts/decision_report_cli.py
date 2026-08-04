#!/usr/bin/env python3
"""Generate deterministic HTML and PDF decision reports."""

import argparse
from pathlib import Path

from acceptance_policy import load_policy
from decision_report import load_result, write_html_report, write_pdf_report
from shared import Shared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Local AI Bench decision report")
    parser.add_argument("result", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args(argv)
    if args.html is None and args.pdf is None:
        parser.error("at least one of --html or --pdf is required")
    try:
        result = load_result(args.result)
        policy = load_policy(args.policy) if args.policy else None
        if args.html:
            write_html_report(result, args.html, policy)
            Shared.ok(f"HTML decision report: {args.html}")
        if args.pdf:
            write_pdf_report(result, args.pdf, policy)
            Shared.ok(f"PDF decision report: {args.pdf}")
    except (OSError, ValueError, KeyError) as exc:
        Shared.err(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
