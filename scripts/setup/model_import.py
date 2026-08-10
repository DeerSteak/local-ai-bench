"""Hugging Face custom-model repository inspection."""

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


def _file_size(item) -> int | None:
    size = getattr(item, "size", None)
    lfs = getattr(item, "lfs", None)
    return size if isinstance(size, int) else getattr(lfs, "size", None)


def _llama_variants(files: dict[str, int | None]) -> tuple[ImportVariant, ...]:
    ggufs = {
        name: size for name, size in files.items()
        if name.lower().endswith(".gguf")
        and not any(marker in Path(name).name.lower() for marker in ("mmproj", "dflash", "draft"))
    }
    grouped: dict[str, list[tuple[int, int, str]]] = {}
    singles = []
    for name in ggufs:
        match = _PART_RE.match(Path(name).name)
        if match:
            grouped.setdefault(match.group(1), []).append((int(match.group(2)), int(match.group(3)), name))
        else:
            singles.append(name)
    variants = [
        ImportVariant(name, Path(name).name, (name,), ggufs[name]) for name in sorted(singles)
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


def inspect_repository(value: str, revision: str = "main", token: str | None = None,
                       api=None) -> RepositoryInspection:
    repo = normalize_hf_repo(value)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.model_info(repo, revision=revision or "main", files_metadata=True, token=token)
    files = {item.rfilename: _file_size(item) for item in (info.siblings or [])}
    safetensors = [
        name for name in files
        if name.lower().endswith(".safetensors")
        and Path(name).parent == Path(".")
        and Path(name).name.lower() not in {"adapter_model.safetensors"}
    ]
    vllm = None
    if "config.json" in files and safetensors:
        sizes = [files[name] for name in safetensors]
        size = sum(value for value in sizes if value is not None) \
            if all(value is not None for value in sizes) else None
        vllm = ImportVariant("snapshot", "Safetensors repository snapshot", tuple(sorted(safetensors)), size)
    return RepositoryInspection(
        repo=repo,
        revision=str(getattr(info, "sha", None) or revision or "main"),
        llama_variants=_llama_variants(files),
        vllm_variant=vllm,
        gated=bool(getattr(info, "gated", False)),
    )

