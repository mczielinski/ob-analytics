"""Every exported name has an entry on an API page.

A name in a public module's ``__all__`` is public, so the API reference must
show it.  It counts as shown when a ``::: dotted.path`` entry on any docs page
names the same object under the same name (a root export documented in its
home module counts), or when the entry renders its whole module.  A module in
``__all__`` counts as shown when an entry documents something inside it.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests._public_modules import needs_missing_extra, public_module_params

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_ENTRY = re.compile(r"^::: ([\w.]+)", re.MULTILINE)


def _resolve(path: str) -> Any:
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut])
        try:
            obj: Any = importlib.import_module(candidate)
        except ModuleNotFoundError as error:
            if error.name != candidate:
                raise
            continue
        for part in parts[cut:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(path)


def _can_resolve(path: str) -> bool:
    """False when the entry's module needs an optional extra that is missing."""
    try:
        _resolve(path)
    except ModuleNotFoundError as error:
        if not needs_missing_extra(error):
            raise
        return False
    return True


def _entries() -> list[str]:
    return sorted(
        {
            match.group(1)
            for page in _DOCS.rglob("*.md")
            for match in _ENTRY.finditer(page.read_text(encoding="utf-8"))
            if _can_resolve(match.group(1))
        }
    )


_ENTRIES = _entries()
_DOCUMENTED = {(id(_resolve(path)), path.rsplit(".", 1)[-1]) for path in _ENTRIES}
_WHOLE_MODULES = {path for path in _ENTRIES if isinstance(_resolve(path), ModuleType)}


def _is_documented(module: ModuleType, name: str) -> bool:
    obj = getattr(module, name)
    if isinstance(obj, ModuleType):
        return any(path.startswith(obj.__name__ + ".") for path in _ENTRIES)
    return (id(obj), name) in _DOCUMENTED


@pytest.mark.parametrize("module_name", public_module_params())
def test_every_export_has_an_api_page_entry(module_name: str) -> None:
    if module_name in _WHOLE_MODULES:
        return
    module = importlib.import_module(module_name)
    missing = [
        name
        for name in module.__all__
        if name != "__version__" and not _is_documented(module, name)
    ]
    assert missing == [], (
        f"{module_name} exports names no API page shows; add a "
        f"'::: {module_name}.<name>' entry to a page under docs/api/: {missing}"
    )


def test_every_entry_resolves() -> None:
    for path in _ENTRIES:
        _resolve(path)
