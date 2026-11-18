"""Each layer imports only from the layers ADR 0001 allows (issue #495).

ADR 0001 orders the library into layers: sources at the bottom, then the
engine, analytics, plots and export, and the public API at the top, with the
shared schema and its helpers under all of them.  A layer may import from the
layers below it, never from a layer above.  ``tests/test_engine_boundary.py``
holds the engine to this; this module holds every other layer to it.

The check is static, like the engine's: it parses each module's source with
``ast`` and reads every import, including imports inside functions and under
``TYPE_CHECKING``.  It never imports the modules.

Three tables drive it:

* ``LAYER_OF`` puts every module in a layer.  A new module must be added here,
  or the test fails.
* ``MAY_IMPORT`` says which layers each layer may import from.
* ``ALLOWED_EXCEPTIONS`` lists the few imports that go upward on purpose.  Each
  one is recorded, with its reason, in ADR 0001 under "Allowed upward
  imports".  Add an entry here only together with an entry there.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import ob_analytics

_PACKAGE_DIR = pathlib.Path(ob_analytics.__file__).parent
_PACKAGE = "ob_analytics"

SHARED = "shared"
SOURCES = "sources"
ENGINE = "engine"
ANALYTICS = "analytics"
PLOTS = "plots and export"
API = "public API"

# The layer of each module.  A key names a module or a package; a package key
# covers every module inside it unless a longer key names that module.
LAYER_OF: dict[str, str] = {
    # The shared schema and the small helpers every layer uses.
    "ob_analytics.exceptions": SHARED,
    "ob_analytics.schemas": SHARED,
    "ob_analytics._utils": SHARED,
    "ob_analytics.config": SHARED,
    "ob_analytics.protocols": SHARED,
    "ob_analytics._registry": SHARED,
    "ob_analytics._secrets": SHARED,
    "ob_analytics._quality_checks": SHARED,
    # Sources: files, live feeds, and generated sessions.
    "ob_analytics.sources": SOURCES,
    "ob_analytics.capture_record": SOURCES,
    "ob_analytics.capture_files": SOURCES,
    "ob_analytics.live": SOURCES,
    "ob_analytics.bitstamp": SOURCES,
    "ob_analytics.lobster": SOURCES,
    "ob_analytics.databento": SOURCES,
    "ob_analytics.depth_csv": SOURCES,
    "ob_analytics.datasets": SOURCES,
    "ob_analytics.synth": SOURCES,
    # The engine: numpy only (tests/test_engine_boundary.py checks it closely).
    "ob_analytics.engine": ENGINE,
    # Analytics, including the frame adapter and the frame faces on the engine.
    "ob_analytics._engine_frames": ANALYTICS,
    "ob_analytics.book": ANALYTICS,
    "ob_analytics.queue": ANALYTICS,
    "ob_analytics.depth": ANALYTICS,
    "ob_analytics.quality": ANALYTICS,
    "ob_analytics.trade_sign": ANALYTICS,
    "ob_analytics.analytics": ANALYTICS,
    "ob_analytics.bars": ANALYTICS,
    "ob_analytics.cost": ANALYTICS,
    "ob_analytics.flow_toxicity": ANALYTICS,
    "ob_analytics.hidden_liquidity": ANALYTICS,
    "ob_analytics.features": ANALYTICS,
    "ob_analytics.metrics": ANALYTICS,
    # Plots and export: the renderers, plot data, Parquet I/O and writers.
    "ob_analytics.visualization": PLOTS,
    "ob_analytics.data": PLOTS,
    "ob_analytics.interop": PLOTS,
    # The public API.  ADR 0001 puts plot_result and prepare here, beside the
    # pipeline; the gallery is the same kind of one-call entry point.
    "ob_analytics.__init__": API,
    "ob_analytics.__main__": API,
    "ob_analytics.cli": API,
    "ob_analytics._demos": API,
    "ob_analytics.pipeline": API,
    "ob_analytics._windows": API,
    "ob_analytics.visualization.__init__": API,
    "ob_analytics.visualization.prepare": API,
    "ob_analytics.visualization.gallery": API,
}

# Which layers each layer may import from.  Every layer may import from itself.
MAY_IMPORT: dict[str, frozenset[str]] = {
    SHARED: frozenset({SHARED}),
    ENGINE: frozenset({ENGINE}),
    SOURCES: frozenset({SHARED, SOURCES}),
    ANALYTICS: frozenset({SHARED, ENGINE, ANALYTICS}),
    # Export writes the formats some sources read back, so it may use them.
    PLOTS: frozenset({SHARED, SOURCES, ANALYTICS, PLOTS}),
    API: frozenset({SHARED, SOURCES, ANALYTICS, PLOTS, API}),
}

# The imports that go against MAY_IMPORT on purpose: (importer, imported).
# ADR 0001 records each one and why, under "Allowed upward imports".
ALLOWED_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # Metric.prepare returns the payload a renderer draws.
        ("ob_analytics.metrics", "ob_analytics.visualization._data"),
        # LOBSTER rebuilds the price levels and checks that the book does not cross.
        ("ob_analytics.lobster", "ob_analytics.depth"),
        ("ob_analytics.lobster", "ob_analytics.quality"),
        # A synthetic session comes back with its order types set.
        ("ob_analytics.synth", "ob_analytics.analytics"),
        # Databento rebuilds resting prices with the engine adapter's rule.
        ("ob_analytics.databento", "ob_analytics._engine_frames"),
        # Loading a saved run builds the pipeline's result type.
        ("ob_analytics.data", "ob_analytics.pipeline"),
        # Plot data resolves a level the way plot_result does.
        ("ob_analytics.visualization._data", "ob_analytics.visualization"),
    }
)


def _module_name(path: pathlib.Path) -> str:
    """The dotted module name of *path*, keeping ``__init__`` for packages."""
    return ".".join(path.relative_to(_PACKAGE_DIR.parent).with_suffix("").parts)


def _is_package(name: str) -> bool:
    return (_PACKAGE_DIR.parent / name.replace(".", "/") / "__init__.py").exists()


def _is_module(name: str) -> bool:
    path = _PACKAGE_DIR.parent / name.replace(".", "/")
    return path.with_suffix(".py").exists() or _is_package(name)


def _imported_modules(path: pathlib.Path, importer: str | None = None) -> set[str]:
    """Every package module *path* imports, at any depth in the file.

    ``from pkg import name`` counts as an import of ``pkg.name`` when that is a
    module, and of ``pkg`` otherwise.  A package import is reported as its
    ``__init__`` module, so it is told apart from the package's other modules.
    *importer* is the dotted name relative imports resolve against; it defaults
    to the module name of *path*.
    """
    importer = importer or _module_name(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = importer.split(".")[: -node.level]
                base = ".".join([*parts, node.module] if node.module else parts)
            else:
                base = node.module or ""
            for alias in node.names:
                full = f"{base}.{alias.name}"
                found.add(full if _is_module(full) else base)
    modules = {name for name in found if name.split(".")[0] == _PACKAGE}
    return {f"{name}.__init__" if _is_package(name) else name for name in modules}


def _layer(module: str) -> str | None:
    """The layer of *module*, by the longest key of LAYER_OF that covers it."""
    candidates = [module]
    if module.endswith(".__init__"):
        candidates.append(module.removesuffix(".__init__"))
    parts = module.removesuffix(".__init__").split(".")
    candidates += [".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]
    for name in candidates:
        if name in LAYER_OF:
            return LAYER_OF[name]
    return None


def _display(module: str) -> str:
    return module.removesuffix(".__init__")


def _sources() -> list[pathlib.Path]:
    return sorted(_PACKAGE_DIR.rglob("*.py"))


def _not_allowed(path: pathlib.Path) -> set[str]:
    """The modules *path* imports from layers MAY_IMPORT does not allow it.

    A module with no layer may import nothing; ``test_every_module_has_a_layer``
    says what to do about it.
    """
    layer = _layer(_module_name(path))
    allowed = MAY_IMPORT[layer] if layer else frozenset()
    return {name for name in _imported_modules(path) if _layer(name) not in allowed}


def _upward_imports() -> set[tuple[str, str]]:
    """Every (importer, imported) pair that MAY_IMPORT does not allow."""
    return {
        (_display(_module_name(path)), _display(imported))
        for path in _sources()
        for imported in _not_allowed(path)
    }


def test_there_are_modules_to_check() -> None:
    # Guard the gate: a glob that matched nothing would pass vacuously.
    assert len(_sources()) > 40


def test_every_module_has_a_layer() -> None:
    unplaced = [
        _module_name(path) for path in _sources() if _layer(_module_name(path)) is None
    ]
    assert not unplaced, f"add these modules to LAYER_OF: {unplaced}"


def test_every_layer_has_its_imports_listed() -> None:
    assert set(MAY_IMPORT) == set(LAYER_OF.values())


@pytest.mark.parametrize("source", _sources(), ids=lambda p: _module_name(p))
def test_imports_follow_the_layers(source: pathlib.Path) -> None:
    importer = _module_name(source)
    bad = sorted(
        f"{_display(imported)} ({_layer(imported)})"
        for imported in _not_allowed(source)
        if (_display(importer), _display(imported)) not in ALLOWED_EXCEPTIONS
    )
    assert not bad, (
        f"{_display(importer)} ({_layer(importer)}) imports from a layer ADR 0001 "
        f"does not let it use: {bad}.  Move the helper down, or add the import "
        "to ALLOWED_EXCEPTIONS and to the ADR's allowed upward imports."
    )


def test_every_exception_is_still_used() -> None:
    # An exception for an import that is gone would let it come back unnoticed.
    assert sorted(ALLOWED_EXCEPTIONS - _upward_imports()) == []


def test_function_level_imports_count(tmp_path: pathlib.Path) -> None:
    # A cycle is often broken with an import inside a function; the parser must
    # still see it, and must resolve relative imports against the importer.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n"
        "    from ob_analytics.pipeline import Pipeline\n"
        "    from . import _data\n"
        "    return Pipeline, _data\n"
    )
    found = _imported_modules(probe, importer="ob_analytics.visualization._probe")
    assert found == {"ob_analytics.pipeline", "ob_analytics.visualization._data"}
