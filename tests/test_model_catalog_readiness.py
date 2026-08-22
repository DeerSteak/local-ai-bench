import json
from pathlib import Path

from scripts.release.model_catalog_readiness import (
    DEFAULT_INCUMBENT_AUDIT, build_readiness, incumbent_catalog_cost, load_incumbent_audit,
    required_candidate_screens,
    validate_screen_report,
)
from scripts.release.model_catalog_screen import DEFAULT_AUDIT, build_screen_spec, load_source_audit
from scripts.runtime import config
from scripts.runtime.shared import Shared


def candidate(family="llm", status="source_ready"):
    return {
        "id": "candidate", "name": "Candidate", "family": family,
        "status": status, "reasons": ["license review"] if status != "source_ready" else [],
        "sources": {
            "upstream": {
                "repo": "owner/model", "revision": "a" * 40,
                "artifact": {"files": ["model.safetensors"]},
                "configuration": {"context_tokens": 32768},
            },
            "gguf": {
                "repo": "owner/model-gguf", "revision": "b" * 40,
                "artifact": {"files": ["model.gguf"]},
            },
            "vllm": {
                "repo": "quants/model-awq", "revision": "c" * 40,
                "artifact": {"files": ["model.safetensors"]},
            },
        },
    }


def image_candidate():
    record = candidate("image")
    record.update(id="z-image-turbo", name="Z-Image Turbo")
    record["sources"]["pipeline"] = [{
        "repo": "Comfy-Org/z_image_turbo", "revision": "c" * 40,
        "files": [
            {"name": "split_files/text_encoders/qwen_3_4b.safetensors"},
            {"name": "split_files/diffusion_models/z_image_turbo_bf16.safetensors"},
            {"name": "split_files/vae/ae.safetensors"},
        ],
    }]
    return record


def write_screen(tmp_path: Path, record: dict, engine="llamacpp"):
    spec = build_screen_spec(record, engine, tmp_path)
    directory = tmp_path / "screen"
    directory.mkdir(parents=True)
    result_path = directory / "result.json"
    sample = {"valid_runs": 1}
    result = {
        "engine_version": "1.2.3",
        "profile": {"os": "Linux", "arch": "x86_64", "ram_gb": 32,
                    "hardware_backend": "cuda"},
        "run": {
            "status": "complete", "recovery_history": [{"status": "interrupted"}],
            "stages": {"llm": {"status": "complete"}, "conv": {"status": "complete"}},
            "plan": {"effective_config": {
                "methodology_profile": "neutral-v2", "sampling_profile": spec.sampling_profile,
            }},
        },
        "preflight": {"models": {spec.tag: {
            "status": "passed", "checks": [{"name": "formatting_probe", "status": "passed"}],
        }}},
        "llm": {spec.tag: {"2K": sample, "32K": sample}},
        "llm_conversation": {spec.tag: {"2K": sample, "32K": sample}},
    }
    result_path.write_text(json.dumps(result))
    artifacts = []
    for name in ("result.json", "result.events.sqlite3", "initial.log", "resume.log"):
        path = directory / name
        if not path.exists():
            path.write_text(name)
        artifacts.append({"path": name, "size": path.stat().st_size,
                          "sha256": Shared.file_sha256(path)})
    report = {
        "schema_version": 1, "candidate": record["id"], "engine": engine,
        "repo": spec.repo, "revision": spec.revision, "files": list(spec.files),
        "result": "result.json", "status": "passed", "errors": [],
        "evidence_artifacts": artifacts, "image_artifacts": [],
    }
    report_path = directory / "screen-report.json"
    report_path.write_text(json.dumps(report))
    return report_path, report


def write_image_screen(tmp_path: Path):
    record = image_candidate()
    spec = build_screen_spec(record, "llamacpp", tmp_path)
    directory = tmp_path / "screen"
    image_dir = directory / "images_result"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "z-image-turbo_1024x1024.png"
    image_path.write_bytes(b"png")
    result = {
        "engine_version": None,
        "profile": {"os": "Linux", "arch": "x86_64", "ram_gb": 32,
                    "hardware_backend": "cuda"},
        "run": {
            "status": "complete", "recovery_history": [{"status": "interrupted"}],
            "stages": {"img": {"status": "complete"}},
            "plan": {"effective_config": {"methodology_profile": "neutral-v2"}},
        },
        "preflight": {"models": {}},
        "images": {spec.tag: {"resolutions": {"1024x1024": {"n_runs": 1}}}},
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result))
    artifacts = []
    for name in ("result.json", "result.events.sqlite3", "initial.log", "resume.log"):
        path = directory / name
        if not path.exists():
            path.write_text(name)
        artifacts.append({"path": name, "size": path.stat().st_size,
                          "sha256": Shared.file_sha256(path)})
    report = {
        "schema_version": 1, "candidate": record["id"], "engine": "llamacpp",
        "repo": spec.repo, "revision": spec.revision, "files": list(spec.files),
        "result": "result.json", "status": "passed", "errors": [],
        "comfyui_revision": "d" * 40, "evidence_artifacts": artifacts,
        "image_artifacts": [{
            "resolution": "1024x1024", "path": "images_result/z-image-turbo_1024x1024.png",
            "size": image_path.stat().st_size, "sha256": Shared.file_sha256(image_path),
        }],
    }
    path = directory / "screen-report.json"
    path.write_text(json.dumps(report))
    return record, path, report


def test_required_screens_cover_both_engines_and_one_comfyui_lifecycle():
    assert required_candidate_screens(candidate()) == ("llamacpp", "vllm")
    assert required_candidate_screens(candidate("embedding")) == ("llamacpp", "vllm")
    assert required_candidate_screens(candidate("image")) == ("comfyui",)
    assert required_candidate_screens(candidate(status="blocked")) == ()


def test_screen_report_verifies_identity_artifacts_runtime_hardware_and_result(tmp_path):
    record = candidate()
    path, report = write_screen(tmp_path, record)
    assert validate_screen_report(path, report, record) == []
    (path.parent / "resume.log").write_text("tampered")
    report["revision"] = "changed"
    errors = validate_screen_report(path, report, record)
    assert "source identity does not match the pinned audit" in errors
    assert "screen evidence size does not match: resume.log" in errors


def test_screen_report_rejects_path_escape_and_missing_runtime_identity(tmp_path):
    record = candidate()
    path, report = write_screen(tmp_path, record)
    report["result"] = "../result.json"
    assert "result path is unsafe" in validate_screen_report(path, report, record)
    assert "result path does not match the evidence manifest" in \
        validate_screen_report(path, report, record)

    path, report = write_screen(tmp_path / "other", record)
    result_path = path.parent / "result.json"
    result = json.loads(result_path.read_text())
    result["engine_version"] = None
    del result["profile"]["hardware_backend"]
    result_path.write_text(json.dumps(result))
    result_record = next(item for item in report["evidence_artifacts"]
                         if item["path"] == "result.json")
    result_record.update(size=result_path.stat().st_size, sha256=Shared.file_sha256(result_path))
    errors = validate_screen_report(path, report, record)
    assert "runtime version is missing" in errors
    assert "hardware profile is incomplete" in errors


def test_screen_report_requires_the_complete_exact_evidence_manifest(tmp_path):
    record = candidate()
    path, report = write_screen(tmp_path, record)
    report["evidence_artifacts"] = report["evidence_artifacts"][:-1]
    assert "screen evidence manifest does not list the required files" in \
        validate_screen_report(path, report, record)


def test_image_screen_uses_comfyui_revision_and_exact_measured_image_manifest(tmp_path):
    record, path, report = write_image_screen(tmp_path)
    assert validate_screen_report(path, report, record) == []
    report["comfyui_revision"] = None
    report["image_artifacts"] = []
    errors = validate_screen_report(path, report, record)
    assert "ComfyUI revision is missing" in errors
    assert "generated image manifest does not match measured resolutions" in errors


def test_readiness_fails_closed_for_missing_duplicate_and_orphaned_screens(tmp_path):
    record = candidate()
    candidate_audit = {"candidates": [record]}
    incumbent_audit = {"incumbents": []}
    readiness = build_readiness(candidate_audit, incumbent_audit, [])
    assert readiness["status"] == "awaiting_evidence"
    assert len(readiness["blockers"]) == 2

    path, report = write_screen(tmp_path, record)
    readiness = build_readiness(candidate_audit, incumbent_audit,
                                [(path, report), (path, report)])
    assert "multiple screen reports" in readiness["blockers"][0]
    orphan = {**report, "candidate": "unknown"}
    readiness = build_readiness(candidate_audit, incumbent_audit, [(path, orphan)])
    assert any("orphaned screen report" in blocker for blocker in readiness["blockers"])


def test_readiness_passes_only_after_both_required_engine_screens(tmp_path):
    record = candidate()
    llama = write_screen(tmp_path / "llama", record, "llamacpp")
    vllm = write_screen(tmp_path / "vllm", record, "vllm")
    readiness = build_readiness(
        {"candidates": [record]}, {"incumbents": []}, [llama, vllm],
    )
    assert readiness["status"] == "ready_for_decisions"
    assert readiness["blockers"] == []


def test_catalog_cost_uses_exact_runtime_artifact_bytes_and_tracks_unknowns():
    incumbents = [{
        "id": "model", "sources": {
            "llamacpp": {"artifact": {"files": [{"size": 10}, {"size": 20}]}},
            "vllm": {"artifact": {"weight_size": 40}},
            "comfyui": {"artifact": {"files": [{"size": None}]}},
        },
    }]
    assert incumbent_catalog_cost(incumbents) == {
        "llamacpp_bytes": 30, "vllm_bytes": 40, "comfyui_bytes": 0,
        "unknown": {"llamacpp_bytes": [], "vllm_bytes": [], "comfyui_bytes": ["model"]},
    }


def test_tracked_pre_hardware_readiness_snapshot_matches_current_audits():
    expected = build_readiness(
        load_source_audit(DEFAULT_AUDIT), load_incumbent_audit(DEFAULT_INCUMBENT_AUDIT), [],
    )
    tracked = json.loads(
        (config.SCRIPT_DIR / "docs" / "model-catalog-readiness-v6.json").read_text()
    )
    assert tracked == expected
