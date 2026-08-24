"""Versioned product catalogs used by recommendation and reporting features."""

from copy import deepcopy

from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS
from scripts.workloads.model_variants import expanded_variant_catalog


CATALOG_VERSION = "3"

HARDWARE_CATALOG = (
    {"id": "apple-mac-mini-m4-pro-2024", "vendor": "Apple", "product": "Mac mini (M4 Pro, 2024)", "kind": "system", "memory_architecture": "unified", "memory_gb_options": [24, 48, 64], "accelerator": "Apple M4 Pro GPU", "source": "https://www.apple.com/mac-mini/specs/", "qualification": "unqualified"},
    {"id": "nvidia-dgx-spark-2025", "vendor": "NVIDIA", "product": "DGX Spark", "kind": "system", "memory_architecture": "coherent_unified", "memory_gb_options": [128], "accelerator": "NVIDIA GB10 Grace Blackwell Superchip", "source": "https://www.nvidia.com/en-us/products/workstations/dgx-spark/", "qualification": "unqualified"},
    {"id": "amd-ryzen-ai-max-plus-395", "vendor": "AMD", "product": "Ryzen AI Max+ 395", "kind": "processor_platform", "memory_architecture": "unified", "memory_gb_options": [], "accelerator": "AMD Radeon 8060S", "source": "https://www.amd.com/en/products/processors/laptop/ryzen.html", "qualification": "unqualified"},
)


def model_catalog():
    """Return catalog records derived from the benchmark's model definitions."""
    records = []
    groups = (
        ("llm", expanded_variant_catalog(LLM_MODELS),
         ["llm", "conversation", "accuracy", "concurrency", "llamabench"]),
        ("embedding", EMBED_MODELS, ["embedding"]),
        ("image", IMAGE_MODELS, ["image"]),
    )
    for family, models, workloads in groups:
        for model in models:
            artifact = model.get("tag") or model.get("checkpoint")
            records.append({
                "id": f"{family}:{artifact}", "family": family, "label": model["label"],
                "artifact": artifact, "quantization": _quantization(model),
                "download_size": model.get("download_size"),
                "memory_requirement": "measured_or_estimated_at_runtime",
                "context_limit": "detected_from_artifact_or_runtime",
                "runtimes": ["llama.cpp"] if family != "image" else ["ComfyUI"],
                "license": {"status": "unverified", "identifier": None, "source": None},
                "workloads": workloads, "source_repository": model.get("hf_repo"),
            })
    return deepcopy(records)


def catalog_bundle():
    """Return the complete versioned catalog without shared mutable records."""
    return {"version": CATALOG_VERSION, "hardware": deepcopy(HARDWARE_CATALOG), "models": model_catalog()}


def recommendation_eligible(model):
    """Require verified licensing before a model can drive a supported recommendation."""
    return model.get("license", {}).get("status") == "verified"


def _quantization(model):
    if isinstance(model.get("quantization"), str):
        return model["quantization"].lower()
    text = f"{model.get('tag', '')} {model.get('hf_file', '')}".upper()
    for name in ("UD-Q4_K_M", "Q4_K_M", "F16", "FP16"):
        if name in text:
            return name.lower()
    return None
