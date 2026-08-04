"""Secret and private-home-path redaction for ordinary diagnostic text."""

import os
import re
from pathlib import Path


SECRET_PATTERNS = (
    re.compile(r"(?i)\bhf_[a-z0-9]{12,}\b"),
    re.compile(r"(?i)\bBearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(?<=access_token=)[^&\s]+"),
    re.compile(r"(?i)(?<=token=)[^&\s]+"),
)
GENERIC_HOME = re.compile(r"(?i)(?:[A-Z]:[\\/]Users[\\/][^\\/\s]+|/(?:Users|home)/[^/\s]+)")


def redact_log_text(value, *, home=None):
    """Redact values while retaining non-sensitive error and relative-path context."""
    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("<secret>", text)
    homes = [Path(home).expanduser()] if home is not None else _known_homes()
    for candidate in homes:
        rendered = str(candidate)
        if rendered and rendered not in {"/", "."}:
            text = text.replace(rendered, "<home>")
            text = text.replace(rendered.replace("/", "\\"), "<home>")
    return GENERIC_HOME.sub("<home>", text)


def _known_homes():
    values = [Path.home()]
    for name in ("HOME", "USERPROFILE"):
        if os.environ.get(name):
            values.append(Path(os.environ[name]))
    return values
