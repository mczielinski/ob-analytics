"""Published text in the package carries no issue numbers or plan labels.

Docstrings, ``#:`` attribute comments and string literals (``Field``
descriptions, gallery captions, error messages) reach users through the API
pages, ``help()`` and the gallery.  An issue number such as ``#155`` or a
plan label such as ``WS-4.1`` means nothing to them.  Plain ``#`` comments
are for maintainers and may keep them.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "ob_analytics"

# ``#155`` but not a CSS colour such as ``#333;``, ``#81c784`` or ``#222222``.
_ISSUE = re.compile(r"(?<![\w&:])#\d{2,5}\b(?![0-9a-fA-F;}])")
_ISSUE_WORD = re.compile(r"\b(?:issue|PR)s? #\d+", re.IGNORECASE)
# Plan and audit labels: ``WS-4.1``, ``Phase 3``, ``A7-1``.
_PLAN = re.compile(r"\bWS-\d|\bPhase \d|\bA\d{1,2}-\d{1,2}\b")

_STRING_TOKENS = {tokenize.STRING}
if hasattr(tokenize, "FSTRING_MIDDLE"):  # Python 3.12+
    _STRING_TOKENS.add(tokenize.FSTRING_MIDDLE)


def published_text_refs(source: str) -> list[tuple[int, str]]:
    """Return ``(line, text)`` for each published line with an issue ref."""
    found: list[tuple[int, str]] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _STRING_TOKENS or (
            tok.type == tokenize.COMMENT and tok.string.startswith("#:")
        ):
            text = tok.string[2:] if tok.type == tokenize.COMMENT else tok.string
            for offset, line in enumerate(text.splitlines()):
                if any(p.search(line) for p in (_ISSUE, _ISSUE_WORD, _PLAN)):
                    found.append((tok.start[0] + offset, line.strip()))
    return found


@pytest.mark.parametrize(
    "text",
    [
        '"""Load the file (issue #155)."""',
        '"see #42 for why"',
        "#: Tick size (issue #226).",
        '"""Frame adapters (WS-4.1)."""',
        '"see issue #155; it explains"',
        '"""Done in Phase 3."""',
        '"""Audit finding A7-1."""',
    ],
)
def test_the_check_finds_issue_refs(text: str) -> None:
    assert published_text_refs(text + "\n")


@pytest.mark.parametrize(
    "text",
    [
        '"color:#666;font-style:italic"',
        '"border-left:1px solid #333}"',
        '"#222222"',
        '"#81c784"',
        "# A maintainer comment may name issue #155.",
    ],
)
def test_the_check_ignores_colours_and_comments(text: str) -> None:
    assert not published_text_refs(text + "\n")


def test_no_issue_numbers_in_published_text() -> None:
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{line}: {text}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for line, text in published_text_refs(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, "Issue numbers in published text:\n" + "\n".join(offenders)
