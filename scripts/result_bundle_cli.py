#!/usr/bin/env python3
"""Export, import, or verify portable Local AI Bench result bundles."""

import argparse
import json
from pathlib import Path

from outbound_metadata import format_outbound_preview, verify_source_identity
from result_bundle import export_result_bundle, import_result_bundle, verify_result_bundle
from shared import Shared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Portable Local AI Bench result bundles")
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="Create a deterministic result bundle")
    export_parser.add_argument("result", type=Path)
    export_parser.add_argument("bundle", type=Path)
    export_parser.add_argument("--artifact", type=Path, action="append", default=[])
    export_parser.add_argument("--system-alias")
    export_parser.add_argument("--hardware-alias")
    export_parser.add_argument("--reviewed-metadata", action="store_true")
    verify_parser = commands.add_parser("verify", help="Verify integrity and reproducible aggregates")
    verify_parser.add_argument("bundle", type=Path)
    verify_parser.add_argument("--source-result", type=Path)
    import_parser = commands.add_parser("import", help="Verify and import a result bundle")
    import_parser.add_argument("bundle", type=Path)
    import_parser.add_argument("result", type=Path)
    import_parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            if not args.reviewed_metadata:
                result = json.loads(args.result.read_text(encoding="utf-8"))
                Shared.output("Outbound metadata review required:\n" + format_outbound_preview(result))
                Shared.err("Review the fields above, then repeat with --reviewed-metadata.")
                return 1
            export_result_bundle(
                args.result, args.bundle, args.artifact,
                system_alias=args.system_alias, hardware_alias=args.hardware_alias,
            )
            Shared.ok(f"Exported verified result bundle: {args.bundle}")
        elif args.command == "verify":
            verified = verify_result_bundle(args.bundle)
            if args.source_result:
                source = json.loads(args.source_result.read_text(encoding="utf-8"))
                if not verify_source_identity(verified["result"], source):
                    raise ValueError("Bundle source identity does not match the supplied result.")
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
