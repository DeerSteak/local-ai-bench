"""Hugging Face custom-model repository inspection."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_PART_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ImportVariant:
    key: str
    label: str
    files: tuple[str, ...]
    size: int | None
    support_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepositoryInspection:
    repo: str
    revision: str
    llama_variants: tuple[ImportVariant, ...]
    vllm_variant: ImportVariant | None
    gated: bool


def normalize_hf_repo(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("only https://huggingface.co model URLs are supported")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("Hugging Face URL must identify an owner and repository")
        value = "/".join(parts[:2])
    if not _REPO_RE.fullmatch(value):
        raise ValueError("repository must be owner/name or a huggingface.co model URL")
    return value


def valid_custom_tag(value: str) -> bool:
    return bool(_TAG_RE.fullmatch(value))


def default_custom_tag(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", repo.rsplit("/", 1)[-1]).strip("-_").lower()


def preferred_variant(variants: tuple[ImportVariant, ...]) -> ImportVariant | None:
    priorities = ("Q4_K_M", "Q4_K_XL", "Q4")
    return next(
        (variant for marker in priorities for variant in variants if marker in variant.label.upper()),
        variants[0] if variants else None,
    )


def _file_size(item) -> int | None:
    size = getattr(item, "size", None)
    lfs = getattr(item, "lfs", None)
    return size if isinstance(size, int) else getattr(lfs, "size", None)


def _llama_variants(files: dict[str, int | None]) -> tuple[ImportVariant, ...]:
    ggufs = {
        name: size for name, size in files.items()
        if name.lower().endswith(".gguf")
        and not Path(name).name.lower().startswith(("mmproj-", "dflash-", "draft-"))
    }
    grouped: dict[str, list[tuple[int, int, str]]] = {}
    singles = []
    for name in ggufs:
        match = _PART_RE.match(Path(name).name)
        if match:
            prefix = str(Path(name).parent / match.group(1))
            grouped.setdefault(prefix, []).append((int(match.group(2)), int(match.group(3)), name))
        else:
            singles.append(name)
    variants = [
        ImportVariant(name, name, (name,), ggufs[name]) for name in sorted(singles)
    ]
    for prefix, parts in sorted(grouped.items()):
        total = parts[0][1]
        if any(item[1] != total for item in parts) or {item[0] for item in parts} != set(range(1, total + 1)):
            continue
        names = tuple(item[2] for item in sorted(parts))
        sizes = [ggufs[name] for name in names]
        size = sum(value for value in sizes if value is not None) \
            if all(value is not None for value in sizes) else None
        variants.append(ImportVariant(prefix, f"{prefix} ({total} parts)", names, size))
    return tuple(sorted(variants, key=lambda item: (item.size is None, item.size or 0, item.label)))


def _indexed_safetensors(index_data: object, files: dict[str, int | None]) -> tuple[str, ...]:
    if not isinstance(index_data, dict) or not isinstance(index_data.get("weight_map"), dict):
        return ()
    names = tuple(sorted(set(index_data["weight_map"].values())))
    if not names or any(
        not isinstance(name, str) or not name.lower().endswith(".safetensors")
        or Path(name).parent != Path(".") or name not in files
        for name in names
    ):
        return ()
    return names


def _vllm_support_files(files: dict[str, int | None], index: str | None) -> tuple[str, ...]:
    exact = {
        "added_tokens.json", "config.json", "generation_config.json", "merges.txt",
        "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
        "quant_config.json", "quantize_config.json", "sentencepiece.bpe.model", "spiece.model",
        "tekken.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json",
        "vocab.txt",
    }
    selected = [name for name in files if Path(name).parent == Path(".") and (
        Path(name).name in exact
        or Path(name).name.startswith("chat_template")
        or Path(name).name.startswith("tokenizer.")
    )]
    if index is not None:
        selected.append(index)
    return tuple(sorted(set(selected)))


def inspect_repository(value: str, revision: str = "main", token: str | None = None,
                       api=None, read_repo_json=None) -> RepositoryInspection:
    repo = normalize_hf_repo(value)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.model_info(repo, revision=revision or "main", files_metadata=True, token=token)
    resolved_revision = str(getattr(info, "sha", None) or revision or "main")
    files = {item.rfilename: _file_size(item) for item in (info.siblings or [])}
    safetensors = [
        name for name in files
        if name.lower().endswith(".safetensors")
        and Path(name).parent == Path(".")
        and Path(name).name.lower() not in {"adapter_model.safetensors"}
    ]
    vllm = None
    indexes = sorted(name for name in files if name.lower().endswith(".safetensors.index.json")
                     and Path(name).parent == Path("."))
    index = ("model.safetensors.index.json" if "model.safetensors.index.json" in indexes
             else indexes[0] if len(indexes) == 1 else None)
    weights: tuple[str, ...] = ()
    if "config.json" in files and index is not None:
        repo_json_reader = read_repo_json
        if read_repo_json is None:
            from huggingface_hub import hf_hub_download

            def default_repo_json_reader(filename):
                path = hf_hub_download(
                    repo_id=repo, filename=filename, revision=resolved_revision, token=token,
                )
                return json.loads(Path(path).read_text(encoding="utf-8"))
            repo_json_reader = default_repo_json_reader
        assert repo_json_reader is not None
        weights = _indexed_safetensors(repo_json_reader(index), files)
    elif "config.json" in files and len(safetensors) == 1:
        weights = tuple(safetensors)
    if weights:
        sizes = [files[name] for name in weights]
        size = sum(value for value in sizes if value is not None) \
            if all(value is not None for value in sizes) else None
        vllm = ImportVariant(
            "snapshot", "Safetensors repository snapshot", weights, size,
            _vllm_support_files(files, index),
        )
    return RepositoryInspection(
        repo=repo,
        revision=resolved_revision,
        llama_variants=_llama_variants(files),
        vllm_variant=vllm,
        gated=bool(getattr(info, "gated", False)),
    )
