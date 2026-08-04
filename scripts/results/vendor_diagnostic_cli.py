#!/usr/bin/env python3
"""Create or verify a reviewed vendor discrepancy diagnostic."""

import argparse
import json
from pathlib import Path

from scripts.results.outbound_metadata import format_outbound_preview
from scripts.results.result_history import load_result
from scripts.runtime.shared import Shared
from scripts.results.vendor_diagnostic import verify_vendor_diagnostic, write_vendor_diagnostic


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Vendor discrepancy diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("baseline", type=Path)
    create.add_argument("candidate", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--reviewed-metadata", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("diagnostic", type=Path)
    verify.add_argument("baseline", type=Path)
    verify.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    try:
        baseline = load_result(args.baseline)
        candidate = load_result(args.candidate)
        if args.command == "create":
            if not args.reviewed_metadata:
                Shared.output(
                    "Baseline outbound metadata:\n" + format_outbound_preview(baseline)
                    + "\n\nCandidate outbound metadata:\n" + format_outbound_preview(candidate)
                )
                Shared.err("Review the fields above, then repeat with --reviewed-metadata.")
                return 1
            write_vendor_diagnostic(args.baseline, args.candidate, args.output)
            Shared.ok(f"Vendor diagnostic written: {args.output}")
        else:
            diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
            if not verify_vendor_diagnostic(diagnostic, baseline, candidate):
                raise ValueError("Vendor diagnostic does not match the supplied source results.")
            Shared.ok(f"Vendor diagnostic verified: {args.diagnostic}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        Shared.err(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
