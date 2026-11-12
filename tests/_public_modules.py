"""The package's public modules, and which of them this environment can import.

The core installs without extras, so a public module that imports an optional
extra's package at the top (``ob_analytics.live.bitstamp`` needs
``websockets``) cannot be imported in every test environment.  The Windows CI
job, for one, installs no live extras.  Tests that walk every public module
skip such a module when its extra is missing, and fail on any other import
error.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import ob_analytics

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _optional_packages() -> frozenset[str]:
    """The import names of the packages the extras in pyproject.toml add."""
    extras = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    return frozenset(
        re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("-", "_")
        for requirements in extras.values()
        for requirement in requirements
    )


OPTIONAL_PACKAGES = _optional_packages()


def needs_missing_extra(error: ModuleNotFoundError) -> bool:
    """Whether the import failed only because an optional extra is missing."""
    return (error.name or "").split(".")[0] in OPTIONAL_PACKAGES


def public_module_names() -> list[str]:
    """Every module with no part of its dotted name starting with ``_``."""
    names = [ob_analytics.__name__]
    for info in pkgutil.walk_packages(ob_analytics.__path__, "ob_analytics."):
        if not any(part.startswith("_") for part in info.name.split(".")[1:]):
            names.append(info.name)
    return sorted(names)


def missing_extra(module_name: str) -> str | None:
    """The optional package the module needs and cannot import, if any."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if not needs_missing_extra(error):
            raise
        return error.name
    return None


def public_module_params() -> list[Any]:
    """One ``pytest.param`` per public module name, skipped if an extra is missing."""
    params = []
    for name in public_module_names():
        package = missing_extra(name)
        marks = []
        if package is not None:
            reason = f"{name} needs {package}, which an optional extra installs"
            marks.append(pytest.mark.skip(reason=reason))
        params.append(pytest.param(name, id=name, marks=marks))
    return params


def importable_public_modules() -> list[ModuleType]:
    """The public modules, less those whose optional extra is missing."""
    return [
        importlib.import_module(name)
        for name in public_module_names()
        if missing_extra(name) is None
    ]
