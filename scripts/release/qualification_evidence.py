"""Self-contained evidence collection and validation for platform qualification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.release.qualification import QUALIFICATION_LIFECYCLE
from scripts.release.qualification_coverage import qualification_workloads, workload_coverage_errors
from scripts.results.result_bundle import verify_result_bundle
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS
from scripts.setup.comfyui_assets import CHECKPOINT_REPOS


EVIDENCE_SCHEMA = "qualification-evidence-v1"


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": Path(path).stat().st_size}


def _catalog_identity(model: dict, engine: str) -> dict:
    repo_key = "vllm_repo" if engine == "vllm" else "hf_repo"
    files = model.get("hf_file") if engine == "llamacpp" else None
    return {
        "id": model["tag"], "repository": model.get(repo_key), "requested_revision": "default",
        "files": files if isinstance(files, list) else ([files] if files else []),
    }


def _installed_model_files(root: Path, engine: str) -> list[Path]:
    roots = [root / "models"]
    if engine == "vllm":
        roots.append(root / "qualification-vllm-cache" / "hub")
    return sorted(
        path for base in roots if base.is_dir() for path in base.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def installation_inventory(root: Path, engine: str, version: str, model_tag: str,
                           *, include_models: bool) -> dict:
    root = Path(root).resolve()
    runtime = root / ("llama.cpp" if engine == "llamacpp" else "vllm-env")
    runtime_names = {"llama-server", "llama-server.exe", "llama-bench", "llama-bench.exe",
                     "llama-batched-bench", "llama-batched-bench.exe", "python", "python.exe"}
    runtime_files = sorted(path for path in runtime.rglob("*") if path.is_file()
                           and not path.is_symlink() and (
                               path.name in runtime_names or path.name == "pyvenv.cfg"
                               or path.name in {"METADATA", "RECORD"}
                               and ".dist-info" in path.parent.name and path.parent.name.startswith("vllm-")
                               or "vllm" in path.parts and path.suffix.lower() in {".so", ".pyd"}
                           ))
    if not runtime_files:
        raise ValueError("qualification runtime has no identity-bearing executable")
    model_files = _installed_model_files(root, engine) if include_models else []
    if include_models and not model_files:
        raise ValueError("qualification installation has no model artifacts")
    llm = next(model for model in LLM_MODELS if model["tag"] == model_tag)
    catalog = [_catalog_identity(llm, engine), _catalog_identity(EMBED_MODELS[0], engine)]
    if engine == "llamacpp":
        catalog.append({
            "id": IMAGE_MODELS[0]["short"],
            "repository": CHECKPOINT_REPOS[IMAGE_MODELS[0]["short"]],
            "requested_revision": "default", "files": [IMAGE_MODELS[0]["checkpoint"]],
        })
    def records(paths):
        return [
            {"path": path.relative_to(root).as_posix(), **file_identity(path)} for path in paths
        ]
    python = (root / "vllm-env" / ("Scripts" if os.name == "nt" else "bin") /
              ("python.exe" if os.name == "nt" else "python")) if engine == "vllm" else Path(sys.executable)
    packages = _probe([str(python), "-m", "pip", "freeze"])
    dependencies = {"python_packages": packages}
    if engine == "llamacpp":
        comfyui = root / "qualification-comfyui-runtime" / "ComfyUI"
        comfy_files = [path for path in (comfyui / "main.py", comfyui / "requirements.txt")
                       if path.is_file()]
        portable = root / "qualification-comfyui-runtime" / "python_embeded" / "python.exe"
        comfy_python = portable if portable.is_file() else Path(sys.executable)
        dependencies["comfyui"] = {
            "commit": _probe(["git", "-C", str(comfyui), "rev-parse", "HEAD"]),
            "files": records(comfy_files),
            "python_packages": _probe([str(comfy_python), "-m", "pip", "freeze"]),
        }
    return {
        "engine": engine, "version": version, "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime_source": ({
            "repository": "https://github.com/ggml-org/llama.cpp",
            "requested_revision": version,
        } if engine == "llamacpp" else {
            "repository": "https://github.com/vllm-project/vllm",
            "requested_package": f"vllm=={version}",
        }),
        "runtime_files": records(runtime_files), "model_files": records(model_files),
        "model_sources": catalog if include_models else [],
        "dependencies": dependencies,
    }


def write_installation_inventory(root: Path, engine: str, version: str, model_tag: str,
                                 output: Path,
                                 *, include_models: bool) -> None:
    Path(output).write_text(
        json.dumps(installation_inventory(
            root, engine, version, model_tag, include_models=include_models,
        ),
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def archive_generated_artifacts(result: Path, destination: Path) -> list[Path]:
    local = Path(result).with_suffix(".events.sqlite3.local.json")
    if not local.is_file():
        return []
    context = json.loads(local.read_text(encoding="utf-8"))
    images = Path(context.get("images_dir", ""))
    if not images.is_dir():
        return []
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(images.glob("*.png")):
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _probe(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "detail": str(exc)}
    output = "\n".join(
        line for line in (result.stdout or result.stderr).strip().splitlines()
        if not any(secret in line.casefold() for secret in ("serial number", "uuid"))
    ).strip()[:131072]
    return {"status": "captured" if result.returncode == 0 and output else "unavailable",
            "detail": output or f"exit code {result.returncode}"}


def host_inventory() -> dict:
    system = platform.system()
    probes = {"kernel": _probe(["uname", "-a"]) if system != "Windows" else _probe(["cmd", "/c", "ver"])}
    if system == "Darwin":
        probes["hardware_firmware"] = _probe(["system_profiler", "SPHardwareDataType", "SPSoftwareDataType"])
        probes["accelerator_driver"] = _probe(["system_profiler", "SPDisplaysDataType"])
    elif system == "Windows":
        probes["hardware_firmware"] = _probe(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_BIOS | Format-List *"])
        probes["accelerator_driver"] = _probe(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select Name,DriverVersion | Format-List"])
    else:
        probes["hardware_firmware"] = _probe(["sh", "-c", "for f in /sys/class/dmi/id/{bios_version,board_name,product_name}; do test -r \"$f\" && printf '%s: ' \"$f\" && cat \"$f\"; done"])
        if executable := shutil.which("nvidia-smi"):
            probes["accelerator_driver"] = _probe([executable])
        elif executable := shutil.which("rocm-smi"):
            probes["accelerator_driver"] = _probe([
                executable, "--showproductname", "--showdriverversion", "--showvbios", "--json",
            ])
        elif executable := shutil.which("rocminfo"):
            probes["accelerator_driver"] = _probe([executable])
        elif executable := shutil.which("lspci"):
            probes["accelerator_driver"] = _probe([executable, "-nnk"])
        else:
            probes["accelerator_driver"] = {"status": "unavailable", "detail": "no accelerator identity tool"}
    return {
        "system": system, "release": platform.release(), "version": platform.version(),
        "machine": platform.machine(), "python": platform.python_version(),
        "python_executable": Path(sys.executable).name, "probes": probes,
    }


def source_inventory(root: Path) -> dict:
    root = Path(root).resolve()
    commit = _probe(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _probe(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"])
    dirty = status["status"] == "captured" and bool(status["detail"])
    files = {}
    for name in ("requirements.txt", "run_qualification.sh", "run_qualification.bat",
                 "bootstrap_qualification.sh", "bootstrap_qualification.bat"):
        path = root / name
        if path.is_file():
            files[name] = file_identity(path)
    return {"commit": commit, "tracked_worktree_dirty": dirty, "launcher_files": files}


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid required evidence {path.name}: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"invalid required evidence {path.name}")
    return value


def final_evidence_errors(recipe: dict, state: dict, output: Path, *, host: dict | None = None,
                          source: dict | None = None) -> list[str]:
    output = Path(output)
    errors = []
    for step in QUALIFICATION_LIFECYCLE:
        if state.get("steps", {}).get(step, {}).get("status") != "passed":
            errors.append(f"lifecycle step did not pass: {step}")
        log = state.get("steps", {}).get(step, {}).get("log")
        if not log or not (output / log).is_file():
            errors.append(f"lifecycle log is missing: {step}")
    engine = recipe["target"]["runtime"]
    target = recipe["target"]
    workloads = qualification_workloads(engine)
    for label, name in (("baseline", "smoke-result.json"), ("target", "upgraded-smoke-result.json")):
        path = output / name
        if not path.is_file():
            errors.append(f"{label} workload result is missing")
            continue
        result = _load_json(path)
        errors.extend(f"{label}: {item}" for item in workload_coverage_errors(result, workloads))
        profile = result.get("profile", {})
        observed_platform = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
            str(profile.get("os", "")).split()[0]
        )
        if observed_platform != target["platform"] and not (
                target["platform"] == "wsl2" and observed_platform == "linux"):
            errors.append(f"{label} result platform does not match the target")
        observed_arch = {"AMD64": "x86_64", "aarch64": "arm64"}.get(
            profile.get("arch"), profile.get("arch")
        )
        expected_arch = "arm64" if target["architecture"] == "aarch64" else target["architecture"]
        if observed_arch != expected_arch:
            errors.append(f"{label} result architecture does not match the target")
        if profile.get("backend") != target["backend"]:
            errors.append(f"{label} result backend does not match the target")
        if profile.get("hostname") != target["accelerator"]:
            errors.append(f"{label} result accelerator identity does not match the target")
        if result.get("engine") != engine:
            errors.append(f"{label} result engine does not match the target")
        if not path.with_suffix(".events.sqlite3").is_file():
            errors.append(f"{label} recovery journal is missing")
    interrupted = output / "interrupted-result.json"
    if not interrupted.is_file() or _load_json(interrupted).get("run", {}).get("status") != "complete":
        errors.append("cancelled run was not successfully resumed to completion")
    elif not interrupted.with_suffix(".events.sqlite3").is_file():
        errors.append("cancelled-run recovery journal is missing")
    for label in ("baseline", "target"):
        report = output / f"{label}-report.html"
        bundle = output / f"{label}-result.lab.zip"
        if not report.is_file() or not report.stat().st_size:
            errors.append(f"{label} report is missing")
        if not bundle.is_file():
            errors.append(f"{label} bundle is missing")
        else:
            try:
                verify_result_bundle(bundle)
            except (OSError, ValueError) as exc:
                errors.append(f"{label} bundle is invalid: {exc}")
    install_command = recipe["steps"]["install"]["command"]
    baseline_version = install_command[install_command.index("--version") + 1]
    for label, expected_version in (("baseline", baseline_version), ("target", target["runtime_version"])):
        inventory = output / f"{label}-installation.json"
        if not inventory.is_file():
            errors.append(f"{label} installation inventory is missing")
        else:
            value = _load_json(inventory)
            dependencies = value.get("dependencies", {})
            if (value.get("version") != expected_version or not value.get("runtime_files")
                    or not value.get("model_files") or not value.get("model_sources")
                    or dependencies.get("python_packages", {}).get("status") != "captured"
                    or engine == "llamacpp" and (
                        not dependencies.get("comfyui", {}).get("files")
                        or dependencies.get("comfyui", {}).get("python_packages", {}).get("status")
                        != "captured"
                    )):
                errors.append(f"{label} installation inventory is incomplete")
    if engine == "llamacpp":
        for label in ("baseline", "target"):
            if not list((output / "artifacts" / label / "images").glob("*.png")):
                errors.append(f"{label} generated image artifacts are missing")
    root = Path(install_command[install_command.index("--root") + 1])
    source = source or source_inventory(root)
    if source["commit"]["status"] != "captured":
        errors.append("tested source commit could not be captured")
    if source["tracked_worktree_dirty"]:
        errors.append("tested source has uncommitted tracked changes")
    host = host or host_inventory()
    if host["probes"]["kernel"]["status"] != "captured":
        errors.append("host kernel identity could not be captured")
    if host["probes"]["accelerator_driver"]["status"] != "captured":
        errors.append("accelerator driver identity could not be captured")
    if target["platform"] == "wsl2" and "microsoft" not in \
            host["probes"]["kernel"].get("detail", "").casefold():
        errors.append("host kernel identity does not prove WSL2")
    return errors


def build_final_manifest(recipe: dict, state: dict, output: Path) -> dict:
    install_command = recipe["steps"]["install"]["command"]
    root = Path(install_command[install_command.index("--root") + 1])
    source, host = source_inventory(root), host_inventory()
    errors = final_evidence_errors(recipe, state, output, host=host, source=source)
    if errors:
        raise ValueError("qualification evidence is incomplete: " + "; ".join(errors))
    output = Path(output)
    files = {}
    for path in sorted(path for path in output.rglob("*") if path.is_file() and not path.is_symlink()
                       and path.name not in {"qualification-manifest.json", "qualification-entry.json"}):
        files[path.relative_to(output).as_posix()] = file_identity(path)
    return {
        "schema": EVIDENCE_SCHEMA, "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": dict(recipe["target"]), "coverage": dict(recipe["coverage"]),
        "recipe_digest": state["recipe_digest"], "source": source,
        "host": host, "files": files,
    }


def verify_final_manifest(output: Path) -> dict:
    output = Path(output)
    manifest = _load_json(output / "qualification-manifest.json")
    if manifest.get("schema") != EVIDENCE_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("qualification manifest has an unsupported schema or status")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("qualification manifest has no file inventory")
    actual_paths = {
        path.relative_to(output).as_posix() for path in output.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"qualification-manifest.json", "qualification-entry.json"}
    }
    if set(declared) != actual_paths:
        raise ValueError("qualification manifest file inventory does not match the evidence directory")
    for relative, identity in declared.items():
        if identity != file_identity(output / relative):
            raise ValueError(f"qualification evidence integrity check failed: {relative}")
    return manifest
