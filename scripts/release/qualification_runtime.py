"""Isolated runtime lifecycle operations used by qualification recipes."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.release.qualification_install import inspect_install_plan, install_qualification_stack
from scripts.results.result_bundle import export_result_bundle, verify_result_bundle
from scripts.runtime.llamacpp_tools import find_llamacpp_tool


RUNTIME_NAMES = {"llamacpp": "llama.cpp", "vllm": "vllm-env"}
REMOVABLE_NAMES = {
    "llama.cpp", "vllm-env", "qualification-runtime-baseline",
    "qualification-cache", "qualification-vllm-cache", "models",
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
                    *, snapshot: bool = False) -> None:  # pragma: no cover
    target = runtime_path(root, engine)
    if target.exists():
        remove_managed(target)
    plan, nvidia, rocm = inspect_install_plan(root, engine, model, version)
    if not install_qualification_stack(plan, nvidia, rocm):
        raise RuntimeError(f"failed to install {engine} {version}")
    if snapshot:
        snapshot_runtime(root, engine)


def runtime_version(root: Path, engine: str) -> str:
    root = qualification_root(root)
    if engine == "llamacpp":
        executable = find_llamacpp_tool(
            "llama-server", vendored_dir=runtime_path(root, engine),
            platform_name="Windows" if os.name == "nt" else None,
        )
        if executable is None:
            raise ValueError("llama-server was not discovered in the qualification runtime")
        command = [str(executable), "--version"]
    elif engine == "vllm":
        binary = "python.exe" if os.name == "nt" else "python"
        command = [str(runtime_path(root, engine) / ("Scripts" if os.name == "nt" else "bin") / binary),
                   "-c", "import vllm; print(vllm.__version__)"]
    else:
        raise ValueError(f"unsupported qualification engine: {engine}")
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    output = (result.stdout or result.stderr).strip()
    if result.returncode or not output:
        raise ValueError(output or f"{engine} version discovery failed")
    return output.splitlines()[0]


def uninstall(root: Path, engine: str) -> None:
    runtime_path(root, engine)
    for name in REMOVABLE_NAMES:
        remove_managed(managed_path(root, name))


def export_verified_bundle(result: Path, bundle: Path, alias: str) -> None:
    export_result_bundle(result, bundle, [], system_alias=alias, hardware_alias=alias)
    verify_result_bundle(bundle)


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
    args = parser.parse_args(argv)
    try:
        if args.action == "bundle":
            if not args.result or not args.bundle or not args.alias:
                parser.error("bundle requires --result, --bundle, and --alias")
            export_verified_bundle(args.result, args.bundle, args.alias)
        elif args.action in {"install", "upgrade"}:
            if not args.model or not args.version:
                parser.error(f"{args.action} requires --model and --version")
            install_runtime(
                args.root, args.engine, args.model, args.version,
                snapshot=args.action == "install",
            )
        elif args.action == "discover":
            print(json.dumps({"engine": args.engine, "version": runtime_version(args.root, args.engine)}))
        elif args.action == "rollback":
            restore_runtime(args.root, args.engine)
            print(json.dumps({"engine": args.engine, "version": runtime_version(args.root, args.engine)}))
        else:
            uninstall(args.root, args.engine)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
