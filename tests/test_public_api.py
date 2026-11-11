"""What the package exports: only what is meant to be public.

Every public module (no part of its dotted name starts with ``_``) lists its
public names in ``__all__``.  The tests check that those lists hold no private
name, that the root package re-exports only names its home module lists, and
that every name the API pages document is exported.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path
from types import ModuleType

import pytest

import ob_analytics

_DOCS = Path(__file__).resolve().parent.parent / "docs"


def _public_modules() -> list[ModuleType]:
    names = [ob_analytics.__name__]
    for info in pkgutil.walk_packages(ob_analytics.__path__, "ob_analytics."):
        if not any(part.startswith("_") for part in info.name.split(".")[1:]):
            names.append(info.name)
    return [importlib.import_module(name) for name in sorted(names)]


_MODULES = _public_modules()


@pytest.mark.parametrize("module", _MODULES, ids=lambda m: m.__name__)
def test_every_public_module_lists_its_exports(module: ModuleType) -> None:
    assert hasattr(module, "__all__"), f"{module.__name__} has no __all__"
    exported = list(module.__all__)
    assert len(exported) == len(set(exported)), "a name is listed twice"
    for name in exported:
        assert hasattr(module, name), f"{module.__name__}.{name} does not exist"


@pytest.mark.parametrize("module", _MODULES, ids=lambda m: m.__name__)
def test_no_private_name_is_exported(module: ModuleType) -> None:
    private = [
        name
        for name in module.__all__
        if name.startswith("_") and name != "__version__"
    ]
    assert private == []


def test_the_root_exports_only_names_their_home_module_exports() -> None:
    """A root export is public where it is defined, not only at the root."""
    exported_by = {
        name
        for module in _MODULES
        if module is not ob_analytics
        for name in module.__all__
    }
    defined_here = {"__version__", "sample_csv_path", "sample_data_dir"}
    stray = [
        name
        for name in ob_analytics.__all__
        if name not in defined_here and name not in exported_by
    ]
    assert stray == []
    # Same name, same object: the root does not export a lookalike.
    for name in set(ob_analytics.__all__) - defined_here:
        homes = [
            module
            for module in _MODULES
            if module is not ob_analytics and name in module.__all__
        ]
        assert any(getattr(m, name) is getattr(ob_analytics, name) for m in homes)


def test_every_documented_name_is_exported() -> None:
    """An API page documents a name only if its module exports it."""
    missing = []
    for page in sorted(_DOCS.rglob("*.md")):
        for match in re.finditer(
            r"^::: (ob_analytics[\w.]*)\.(\w+)\s*$", page.read_text(), re.MULTILINE
        ):
            module_name, name = match.groups()
            try:
                # ``::: ob_analytics.cli`` documents a whole module.
                importlib.import_module(f"{module_name}.{name}")
                continue
            except ModuleNotFoundError:
                module = importlib.import_module(module_name)
            if name not in getattr(module, "__all__", ()):
                missing.append(f"{page.relative_to(_DOCS)}: {module_name}.{name}")
    assert missing == []
