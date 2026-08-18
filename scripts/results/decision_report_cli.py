#!/usr/bin/env python3
"""Generate deterministic HTML and PDF decision reports."""

import argparse
from pathlib import Path

from scripts.results.acceptance_policy import load_policy
from scripts.results.decision_report import load_result, write_html_report, write_pdf_report
from scripts.results.outbound_metadata import format_outbound_preview, prepare_outbound_result
from scripts.results.recommendation_cli import load_constraints
from scripts.runtime.shared import Shared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Local AI Bench decision report")
    parser.add_argument("result", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--recommendation", type=Path)
    parser.add_argument("--system-alias")
    parser.add_argument("--hardware-alias")
    parser.add_argument("--reviewed-metadata", action="store_true")
    args = parser.parse_args(argv)
    if args.html is None and args.pdf is None:
        parser.error("at least one of --html or --pdf is required")
    try:
        source_result = load_result(args.result)
        if not args.reviewed_metadata:
            Shared.output("Outbound metadata review required:\n" + format_outbound_preview(source_result))
            Shared.err("Review the fields above, then repeat with --reviewed-metadata.")
            return 1
        result = prepare_outbound_result(
            source_result, system_alias=args.system_alias, hardware_alias=args.hardware_alias,
        )
        policy = load_policy(args.policy) if args.policy else None
        recommendation = load_constraints(args.recommendation) if args.recommendation else None
        if args.html:
            write_html_report(result, args.html, policy, recommendation, source_result)
            Shared.ok(f"HTML decision report: {args.html}")
        if args.pdf:
            write_pdf_report(result, args.pdf, policy, recommendation, source_result)
            Shared.ok(f"PDF decision report: {args.pdf}")
    except (OSError, ValueError, KeyError) as exc:
        Shared.err(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
