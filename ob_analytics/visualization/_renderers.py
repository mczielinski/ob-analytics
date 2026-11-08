"""The renderer registry and the plot-backend modules that fill it.

Each backend module registers its renderers into :data:`RENDERERS` when it is
imported, keyed by ``(concept, level, backend)``.  :data:`_BACKEND_MODULES`
maps a backend name to that module, so :func:`_load_backend` can import a
backend on first use.  Third-party backends advertise themselves through the
``ob_analytics.plot_backends`` entry-point group.

This module imports nothing else from the visualization package, so a backend
can import :data:`RENDERERS` from here at the top of its own module.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator
from importlib.metadata import EntryPoint
from typing import Any

from ob_analytics._registry import Registry, load_plugins
from ob_analytics.exceptions import ConfigError, UnknownNameError
from ob_analytics.protocols import Level

#: The entry-point group a third-party package advertises a plot backend under,
#: e.g. ``[project.entry-points."ob_analytics.plot_backends"]`` with
#: ``altair = "my_pkg.altair_backend"``.  The value names the backend module,
#: which is imported on the first :func:`~ob_analytics.visualization.plot`
#: call that targets the backend.
ENTRY_POINT_GROUP = "ob_analytics.plot_backends"

RendererFn = Callable[..., Any]
RendererKey = tuple[str, Level | None, str]


def _same_kind(a: Level | None, b: Level | None) -> bool:
    """Whether *a* and *b* are one kind: both level-less or both levels."""
    return (a is None) == (b is None)


def _kind_text(levels: Iterable[Level | None]) -> str:
    """Name a kind for a message: ``"level-less"`` or ``"at L2 and L3"``.

    Raises
    ------
    ValueError
        If *levels* is empty: there is no kind to name.
    """
    levels = set(levels)
    if not levels:
        raise ValueError("No levels to describe.")
    if None in levels:
        return "level-less"
    return "at " + " and ".join(sorted(str(lvl) for lvl in levels))


class RendererRegistry(Registry[RendererKey, RendererFn]):
    """The renderer :class:`Registry`, which keeps each concept one kind.

    A concept is either level-less, registered at ``None``, or drawn at a
    level, registered at ``Level.L2`` and/or ``Level.L3``.  It is the same
    kind on every backend: :func:`~ob_analytics.visualization.plot` and the
    gallery call it the same way whichever backend draws it.
    """

    def get(self, key: RendererKey) -> RendererFn:
        """Return the renderer for ``(concept, level, backend)``.

        Raises
        ------
        UnknownNameError
            If no renderer is registered under *key*.  The message says
            which backends or levels do draw the concept, or names the
            concepts closest to it when no backend draws it.
        """
        if self._normalize(key) not in self._items:
            raise self.missing(*key)
        return super().get(key)

    def missing(
        self, concept: str, level: Level | None, backend: str
    ) -> UnknownNameError:
        """The error for a *concept* that *backend* does not draw at *level*.

        It names the backends that draw *concept*, or the levels *backend*
        draws it at, rather than every registered renderer.
        """
        placements = self.placements(concept)
        if not placements:
            concepts = sorted({c for c, _, _ in self._items})
            return UnknownNameError("plot concept", concept, concepts)
        backends = sorted({b for _, b in placements})
        if backend not in backends:
            shown = ", ".join(repr(b) for b in backends)
            return UnknownNameError(
                f"{backend!r} plot",
                concept,
                (),
                hint=f"Backends that draw it: {shown}.",
            )
        levels = [lvl for lvl, b in placements if b == backend]
        return UnknownNameError(
            f"{_kind_text([level]).removeprefix('at ')} {backend!r} plot",
            concept,
            (),
            hint=f"On {backend!r} it is {_kind_text(levels)}.",
        )

    def placements(self, concept: str) -> tuple[tuple[Level | None, str], ...]:
        """Return each ``(level, backend)`` *concept* is registered at, in order.

        A snapshot, so a caller can register while looping over it.
        """
        return tuple(self._placements(concept))

    def _placements(self, concept: str) -> Iterator[tuple[Level | None, str]]:
        return ((lvl, b) for (c, lvl, b) in self._items if c == concept)

    def register(self, key: RendererKey, value: RendererFn) -> None:
        """Register *value* under the key ``(concept, level, backend)``.

        Raises
        ------
        ConfigError
            If *key* is not ``(concept, level, backend)`` with a text
            concept, a :class:`Level` or ``None`` level and a text backend,
            or its concept is already registered, on any backend, as the
            other kind: at a level when *key* is level-less, or level-less
            when *key* has a level.
        """
        if not (
            isinstance(key, tuple)
            and len(key) == 3
            and isinstance(key[0], str)
            and (key[1] is None or key[1] in tuple(Level))
            and isinstance(key[2], str)
        ):
            raise ConfigError(
                f"A renderer key is (concept, level, backend), got {key!r}. "
                "The concept and backend are text; the level is Level.L2, "
                "Level.L3, or None for a level-less plot."
            )
        concept, level, backend = key
        if level is not None:
            level = Level(level)  # store Level.L2, not an equal "L2"
        clash = next(
            (
                (other, b)
                for other, b in self._placements(concept)
                if not _same_kind(other, level)
            ),
            None,
        )
        if clash is not None:
            other, other_backend = clash
            raise ConfigError(
                f"Plot concept {concept!r} is already registered "
                f"{_kind_text([other])} on {other_backend!r}, so it cannot also "
                f"be registered {_kind_text([level])}. "
                "A concept is the same kind on every backend: register it at "
                "None everywhere, or at Level.L2 and/or Level.L3 everywhere."
            )
        super().register((concept, level, backend), value)


#: Registry of ``(concept, level, backend)`` → renderer function, where
#: *level* is a :class:`Level` (``L2``/``L3``) or ``None`` for level-less
#: analytics.  A concept is one kind or the other on every backend.
#: Renderer modules (``_matplotlib``, ``_plotly``) self-register at import
#: time -- see the self-registration block at the bottom of each.
RENDERERS: RendererRegistry = RendererRegistry("renderer")

# Lazy-import bootstrap: backend name → module that self-registers its
# renderers on import.  matplotlib is a hard dep, imported with the
# visualization package for its save and theme helpers, which also fires its
# registration; plotly and bokeh are optional and imported on first use.
_BACKEND_MODULES: dict[str, str] = {
    "matplotlib": "ob_analytics.visualization._matplotlib",
    "plotly": "ob_analytics.visualization._plotly",
    "bokeh": "ob_analytics.visualization._bokeh",
}


def register_plot_backend(name: str, module_path: str) -> None:
    """Register a visualization backend module.

    The module at *module_path* must call ``RENDERERS.register((concept,
    level, name), fn)`` for each plot it supports (typically at import time).
    It is imported lazily on the first
    :func:`~ob_analytics.visualization.plot` call that targets *name*.

    Parameters
    ----------
    name : str
        Backend name used in ``plot(..., backend=name)``.
    module_path : str
        Dotted import path, e.g. ``"my_package._bokeh_backend"``.

    Examples
    --------
    >>> from ob_analytics.visualization import register_plot_backend
    >>> register_plot_backend("bokeh", "my_pkg._bokeh")
    """
    _BACKEND_MODULES[name] = module_path


def _load_backend(backend: str) -> None:
    """Import *backend*'s module so its renderers are in :data:`RENDERERS`.

    The import is cached after the first call.

    Raises
    ------
    UnknownNameError
        If *backend* is not registered.
    """
    if backend not in _BACKEND_MODULES:
        raise UnknownNameError("plot backend", backend, _BACKEND_MODULES)
    importlib.import_module(_BACKEND_MODULES[backend])


def _register_entry_point(ep: EntryPoint) -> str:
    register_plot_backend(ep.name, ep.module)
    return ep.name


def load_plot_backend_plugins(*, force: bool = False) -> list[str]:
    """Discover and register plot backends advertised via entry points.

    Scans the :data:`ENTRY_POINT_GROUP` entry-point group and registers each
    entry's module under the entry-point name, as :func:`register_plot_backend`
    would.  The module is not imported here: it is imported on the first
    :func:`~ob_analytics.visualization.plot` call that targets the backend.

    The scan runs once per process unless *force* is set.  A plug-in that
    fails to register is skipped with a warning, so one broken package cannot
    stop the rest from loading.

    Returns
    -------
    list of str
        The backend names newly registered by this call.
    """
    return load_plugins(ENTRY_POINT_GROUP, _register_entry_point, force=force)
