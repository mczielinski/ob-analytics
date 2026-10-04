"""The secrets this process has loaded, and their removal from text.

Every credential a :class:`~ob_analytics.config.SourceSettings` holds is
remembered here when the settings are built.  What a capture writes --
``orders.csv``, ``depth.csv``, ``trades.csv``, ``raw.jsonl``, ``meta.json``,
``manifest.json`` -- and what the CLI logs passes through :func:`redact`, so a
key that reaches any of them by mistake (a venue that echoes it in a frame, an
error message that quotes it) is replaced with :data:`MASK`.

Pydantic's ``SecretStr`` already keeps a key out of a ``repr``, a log line that
prints the settings, and ``model_dump``.  This is the second line, for text the
settings never see.
"""

from __future__ import annotations

import json
from typing import Any

#: What a secret is replaced with; the same mask ``SecretStr`` prints.
MASK = "**********"

#: The shortest value treated as a secret.  A shorter one would also match
#: ordinary text (a price, an id) and corrupt every file it is removed from;
#: no venue issues a key that short, and :class:`~ob_analytics.config.Credential`
#: refuses one.
MIN_SECRET_LENGTH = 8

_known: set[str] = set()
# Longest first, so a secret that contains another is replaced whole.
_ordered: tuple[str, ...] = ()


def remember(value: str) -> None:
    """Add *value* to the secrets :func:`redact` removes.

    The JSON-escaped form is remembered too, so a secret holding a newline or a
    quote (a private key file) is also found inside ``raw.jsonl`` and
    ``meta.json``.  A value shorter than :data:`MIN_SECRET_LENGTH` is ignored.
    """
    global _ordered
    if len(value) < MIN_SECRET_LENGTH:
        return
    forms = {value, json.dumps(value)[1:-1]}
    if forms <= _known:
        return
    _known.update(forms)
    _ordered = tuple(sorted(_known, key=len, reverse=True))


def any_known() -> bool:
    """Whether any secret has been remembered, so there is anything to redact."""
    return bool(_ordered)


def redact(text: str) -> str:
    """Return *text* with every remembered secret replaced by :data:`MASK`."""
    for secret in _ordered:
        if secret in text:
            text = text.replace(secret, MASK)
    return text


def redact_log_record(record: Any) -> None:
    """A loguru patcher that removes secrets from each log message."""
    record["message"] = redact(record["message"])


class RedactingFile:
    """A text file whose writes pass through :func:`redact`.

    Wraps a file a capture writes to, so a ``csv`` writer or a plain
    ``write`` both go through it.  Every other attribute is the file's own.
    """

    def __init__(self, fp: Any) -> None:
        self._fp = fp

    def write(self, text: str) -> int:
        return self._fp.write(redact(text))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fp, name)
