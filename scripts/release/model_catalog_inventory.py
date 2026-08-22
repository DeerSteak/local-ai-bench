"""Build and validate the shipped catalog inventory used by the Version 6 audit."""

import argparse
import json
from pathlib import Path

from scripts.setup.comfyui_assets import CHECKPOINT_REPOS, GATED_MODELS
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


INVENTORY_SCHEMA_VERSION = 1
DEFAULT_REGISTER = Path(__file__).with_name("model_catalog_incumbents.json")


def load_incumbent_register(path: Path = DEFAULT_REGISTER) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("incumbents"), list):
        raise ValueError("unsupported incumbent register")
    records = value["incumbents"]
    ids = [record.get("id") for record in records]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("incumbent ids must be non-empty and unique")
    for record in records:
        if record.get("family") not in {"llm", "embedding", "image"}:
            raise ValueError(f"invalid incumbent family: {record.get('id')}")
        if not isinstance(record.get("upstream"), str) or not record["upstream"]:
            raise ValueError(f"incumbent requires upstream identity: {record.get('id')}")
        if not isinstance(record.get("role"), str) or not record["role"]:
            raise ValueError(f"incumbent requires a measurable role: {record.get('id')}")
    return records


def _catalog_entries() -> dict[tuple[str, str], dict]:
    return {
        **{("llm", model["tag"]): model for model in LLM_MODELS},
        **{("embedding", model["tag"]): model for model in EMBED_MODELS},
        **{("image", model["short"]): model for model in IMAGE_MODELS},
    }


def _selected_artifacts(family: str, model: dict) -> dict:
    if family == "image":
        return {
            "comfyui": {
                "repo": CHECKPOINT_REPOS[model["short"]],
                "files": [model["checkpoint"]],
                "gated": model["short"] in GATED_MODELS,
            },
        }
    return {
        "llamacpp": {
            "repo": model["hf_repo"],
            "files": ([model["hf_file"]] if isinstance(model["hf_file"], str)
                      else list(model["hf_file"])),
            "download_size": model["download_size"],
        },
        "vllm": {
            "repo": model["vllm_repo"],
            "download_size": model["vllm_download_size"],
        },
    }


def build_incumbent_inventory(register: list[dict]) -> dict:
    catalog = _catalog_entries()
    registered = {(record["family"], record["id"]) for record in register}
    missing = sorted(set(catalog) - registered)
    extra = sorted(registered - set(catalog))
    if missing or extra:
        raise ValueError(f"incumbent register/catalog mismatch: missing={missing}; extra={extra}")
    incumbents = []
    for record in register:
        model = catalog[(record["family"], record["id"])]
        incumbents.append({
            **record,
            "label": model["label"],
            "short": model["short"],
            "tier": model.get("tier"),
            "params_b": model.get("params_b"),
            "selected_artifacts": _selected_artifacts(record["family"], model),
            "decision": "pending_evidence",
        })
    return {"schema_version": INVENTORY_SCHEMA_VERSION, "incumbents": incumbents}


def main(argv=None) -> int:  # pragma: no cover - command entrypoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    inventory = build_incumbent_inventory(load_incumbent_register(args.register))
    rendered = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
