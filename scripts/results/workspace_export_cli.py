"""Export reports and bundles from a saved workspace selection."""

import argparse
from pathlib import Path

from scripts.runtime.shared import Shared
from scripts.results.workspace_export import export_workspace_bundle, write_workspace_reports
from scripts.results.workspace_selection import load_workspace_selection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export one Local AI Bench workspace selection")
    parser.add_argument("selection", type=Path)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args(argv)
    if args.bundle is None and args.html is None and args.pdf is None:
        parser.error("at least one of --bundle, --html, or --pdf is required")
    try:
        selection = load_workspace_selection(args.selection)
        if args.bundle is not None:
            export_workspace_bundle(selection, args.result, args.bundle)
            Shared.ok(f"Workspace bundle: {args.bundle}")
        if args.html is not None or args.pdf is not None:
            write_workspace_reports(
                selection, args.result, html_path=args.html, pdf_path=args.pdf,
            )
            if args.html is not None:
                Shared.ok(f"Workspace HTML report: {args.html}")
            if args.pdf is not None:
                Shared.ok(f"Workspace PDF report: {args.pdf}")
    except (OSError, ValueError) as exc:
        Shared.err(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
