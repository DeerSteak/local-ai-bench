"""Isolated runtime lifecycle operations used by qualification recipes."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.release.qualification_install import inspect_install_plan, install_qualification_stack
from scripts.release.qualification_coverage import run_qualification_coverage
from scripts.release.qualification_evidence import (
    archive_generated_artifacts, write_installation_inventory,
)
from scripts.results.decision_report import load_result, write_html_report
from scripts.results.outbound_metadata import prepare_outbound_result
from scripts.results.result_bundle import export_result_bundle, verify_result_bundle
from scripts.runtime import config
from scripts.setup.runtime_identity import parse_runtime_version


RUNTIME_NAMES = {"llamacpp": "llama.cpp", "vllm": "vllm-env"}
PROCESS_MANAGED_NAMES = {*RUNTIME_NAMES.values(), "qualification-comfyui-runtime"}
RUNTIME_VERSION_MARKER = ".qualification-runtime-version"
REMOVABLE_NAMES = {
    "llama.cpp", "vllm-env", "qualification-runtime-baseline",
    "qualification-cache", "qualification-vllm-cache", "qualification-comfyui-runtime",
    "models",
}


def qualification_root(value: Path) -> Path:
    root = Path(value).resolve()
    if not (root / "README.md").is_file() or not (root / "scripts").is_dir():
        raise ValueError("qualification lifecycle requires a Local AI Bench clone")
    return root


def managed_path(root: Path, name: str) -> Path:
    root = qualification_root(root)
    if name not in REMOVABLE_NAMES:
        raise ValueError(f"qualification lifecycle does not manage {name}")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"qualification lifecycle refuses symbolic links: {path}")
    return path


def runtime_path(root: Path, engine: str) -> Path:
    try:
        return managed_path(root, RUNTIME_NAMES[engine])
    except KeyError:
        raise ValueError(f"unsupported qualification engine: {engine}") from None


def remove_managed(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def snapshot_runtime(root: Path, engine: str) -> Path:
    source = runtime_path(root, engine)
    backup = managed_path(root, "qualification-runtime-baseline")
    if not source.is_dir():
        raise ValueError(f"cannot snapshot missing runtime: {source}")
    remove_managed(backup)
    shutil.copytree(source, backup, symlinks=True)
    return backup


def restore_runtime(root: Path, engine: str) -> Path:
    target = runtime_path(root, engine)
    backup = managed_path(root, "qualification-runtime-baseline")
    if not backup.is_dir():
        raise ValueError(f"cannot restore missing baseline snapshot: {backup}")
    remove_managed(target)
    shutil.copytree(backup, target, symlinks=True)
    return target


def install_runtime(root: Path, engine: str, model: str, version: str,
                    *, snapshot: bool = False, inventory: Path | None = None) -> None:  # pragma: no cover
    target = runtime_path(root, engine)
    if target.exists():
        remove_managed(target)
    plan, nvidia, rocm = inspect_install_plan(root, engine, model, version)
    if not install_qualification_stack(plan, nvidia, rocm):
        raise RuntimeError(f"failed to install {engine} {version}")
    if engine == "llamacpp":
        (target / RUNTIME_VERSION_MARKER).write_text(version + "\n", encoding="utf-8")
    if inventory is not None:
        write_installation_inventory(root, engine, version, model, inventory, include_models=True)
    if snapshot:
        snapshot_runtime(root, engine)


def archive_smoke_artifacts(output: Path) -> list[Path]:
    output = Path(output)
    journal = output.with_suffix(".events.sqlite3")
    artifacts = [output, journal, journal.with_suffix(journal.suffix + ".local.json")]
    existing = [path for path in artifacts if path.exists()]
    if not existing:
        return []
    retry = 1
    while any(path.with_name(f"{path.name}.retry-{retry}").exists() for path in existing):
        retry += 1
    archived = []
    for path in existing:
        destination = path.with_name(f"{path.name}.retry-{retry}")
        path.replace(destination)
        archived.append(destination)
    return archived


def smoke_runtime(root: Path, engine: str, model: str, output: Path) -> None:  # pragma: no cover
    archive_smoke_artifacts(output)
    comfyui = Path(root) / "qualification-comfyui-runtime" / "ComfyUI" \
        if engine == "llamacpp" else None
    run_qualification_coverage(engine, model, output, comfyui)


def runtime_version(root: Path, engine: str, *, run=subprocess.run) -> str:
    root = qualification_root(root)
    if engine == "llamacpp":
        source = runtime_path(root, engine)
        executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
        executable = next(
            (path for path in source.rglob(executable_name) if path.is_file()), None,
        )
        if executable is None:
            raise ValueError("llama-server was not discovered in the qualification runtime")
        marker = source / RUNTIME_VERSION_MARKER
        if marker.is_file():
            recorded = marker.read_text(encoding="utf-8").strip()
            if recorded:
                return recorded
        if (source / ".git").is_dir():
            tag = run(
                ["git", "-C", str(source), "describe", "--tags", "--exact-match", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            if tag.returncode == 0 and tag.stdout.strip():
                return tag.stdout.strip()
        command = [str(executable), "--version"]
    elif engine == "vllm":
        binary = "python.exe" if os.name == "nt" else "python"
        command = [str(runtime_path(root, engine) / ("Scripts" if os.name == "nt" else "bin") / binary),
                   "-c", "import vllm; print(vllm.__version__)"]
    else:
        raise ValueError(f"unsupported qualification engine: {engine}")
    timeout = config.VLLM_COLD_IMPORT_TIMEOUT if engine == "vllm" else 30
    result = run(command, capture_output=True, text=True, timeout=timeout)
    output = (result.stdout or result.stderr).strip()
    if result.returncode or not output:
        raise ValueError(output or f"{engine} version discovery failed")
    return parse_runtime_version(output) or output.splitlines()[0]


def require_runtime_version(root: Path, engine: str, expected: str) -> str:
    actual = runtime_version(root, engine)
    if actual.removeprefix("b") != expected.removeprefix("b"):
        raise ValueError(f"expected {engine} {expected}, discovered {actual}")
    return actual


def managed_processes(root: Path, *, run=subprocess.run) -> list[str]:
    root = qualification_root(root)
    if os.name == "nt":
        command = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }",
        ]
    else:
        command = ["ps", "-axo", "pid=,command="]
    result = run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("could not inspect processes before qualification uninstall")
    markers = [str(root / name).casefold() for name in PROCESS_MANAGED_NAMES]
    own_pid = str(os.getpid())
    return [
        line.strip() for line in result.stdout.splitlines()
        if any(marker in line.casefold() for marker in markers)
        and not line.strip().startswith(own_pid + " ")
    ]


def uninstall(root: Path, engine: str) -> None:
    runtime_path(root, engine)
    active = managed_processes(root)
    if active:
        raise ValueError("qualification runtime still has active processes: " + "; ".join(active))
    for name in REMOVABLE_NAMES:
        remove_managed(managed_path(root, name))


def export_verified_bundle(result: Path, bundle: Path, alias: str,
                           artifact_dir: Path | None = None) -> None:
    artifacts = archive_generated_artifacts(result, artifact_dir) if artifact_dir else []
    export_result_bundle(result, bundle, artifacts, system_alias=alias, hardware_alias=alias)
    verify_result_bundle(bundle)


def write_reviewed_report(result: Path, report: Path, alias: str) -> None:
    source = load_result(result)
    outbound = prepare_outbound_result(source, system_alias=alias, hardware_alias=alias)
    write_html_report(outbound, report, recommendation_source=source)


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Manage an isolated qualification runtime")
    parser.add_argument(
        "action", choices=("install", "upgrade", "discover", "rollback", "uninstall", "bundle"),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--engine", required=True, choices=tuple(RUNTIME_NAMES))
    parser.add_argument("--model")
    parser.add_argument("--version")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--alias")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "bundle":
            if not args.result or not args.bundle or not args.alias:
                parser.error("bundle requires --result, --bundle, and --alias")
            export_verified_bundle(args.result, args.bundle, args.alias, args.artifact_dir)
        elif args.action in {"install", "upgrade"}:
            if not args.model or not args.version:
                parser.error(f"{args.action} requires --model and --version")
            install_runtime(
                args.root, args.engine, args.model, args.version,
                snapshot=args.action == "install", inventory=args.inventory,
            )
            print(json.dumps({
                "engine": args.engine,
                "version": require_runtime_version(args.root, args.engine, args.version),
            }))
            if args.action == "upgrade":
                if not args.smoke_output:
                    parser.error("upgrade requires --smoke-output")
                smoke_runtime(args.root, args.engine, args.model, args.smoke_output)
                if not args.report or not args.bundle or not args.alias or not args.artifact_dir:
                    parser.error("upgrade requires --report, --bundle, --alias, and --artifact-dir")
                write_reviewed_report(args.smoke_output, args.report, args.alias)
                export_verified_bundle(
                    args.smoke_output, args.bundle, args.alias, args.artifact_dir,
                )
        elif args.action == "discover":
            actual = (require_runtime_version(args.root, args.engine, args.version)
                      if args.version else runtime_version(args.root, args.engine))
            print(json.dumps({"engine": args.engine, "version": actual}))
        elif args.action == "rollback":
            restore_runtime(args.root, args.engine)
            actual = (require_runtime_version(args.root, args.engine, args.version)
                      if args.version else runtime_version(args.root, args.engine))
            print(json.dumps({"engine": args.engine, "version": actual}))
        else:
            uninstall(args.root, args.engine)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
