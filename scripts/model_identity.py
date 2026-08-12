"""Shared normalization for model identifiers used as directory names."""


def model_tag_slug(tag: str) -> str:
    return tag.replace(":", "_").replace("/", "_")
