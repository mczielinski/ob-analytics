"""Every exported name has an entry on an API page.

A name in a public module's ``__all__`` is public, so the API reference must
show it.  It counts as shown when a ``::: dotted.path`` entry on any docs page
names the same object under the same name (a root export documented in its
home module counts), or when the entry renders its whole module.  A module in
``__all__`` counts as shown when an entry documents something inside it.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import ob_analytics

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_ENTRY = re.compile(r"^::: ([\w.]+)", re.MULTILINE)


def _public_module_names() -> list[str]:
    names = [ob_analytics.__name__]
    for info in pkgutil.walk_packages(ob_analytics.__path__, "ob_analytics."):
        if not any(part.startswith("_") for part in info.name.split(".")[1:]):
            names.append(info.name)
    return sorted(names)


def _resolve(path: str) -> Any:
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for part in parts[cut:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(path)


def _entries() -> list[str]:
    return sorted(
        {
            match.group(1)
            for page in _DOCS.rglob("*.md")
            for match in _ENTRY.finditer(page.read_text(encoding="utf-8"))
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


@pytest.mark.parametrize("module_name", _public_module_names())
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
