#!/usr/bin/env python3
"""Export, import, or verify portable Local AI Bench result bundles."""

import argparse
from pathlib import Path

from result_bundle import export_result_bundle, import_result_bundle, verify_result_bundle
from shared import Shared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Portable Local AI Bench result bundles")
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="Create a deterministic result bundle")
    export_parser.add_argument("result", type=Path)
    export_parser.add_argument("bundle", type=Path)
    export_parser.add_argument("--artifact", type=Path, action="append", default=[])
    verify_parser = commands.add_parser("verify", help="Verify integrity and reproducible aggregates")
    verify_parser.add_argument("bundle", type=Path)
    import_parser = commands.add_parser("import", help="Verify and import a result bundle")
    import_parser.add_argument("bundle", type=Path)
    import_parser.add_argument("result", type=Path)
    import_parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            export_result_bundle(args.result, args.bundle, args.artifact)
            Shared.ok(f"Exported verified result bundle: {args.bundle}")
        elif args.command == "verify":
            verify_result_bundle(args.bundle)
            Shared.ok(f"Result bundle verified: {args.bundle}")
        else:
            import_result_bundle(args.bundle, args.result, args.artifact_dir)
            Shared.ok(f"Imported verified result: {args.result}")
    except (OSError, ValueError, KeyError) as exc:
        Shared.err(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
