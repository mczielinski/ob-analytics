"""Visualization functions for limit order book analytics.

The unified :func:`plot` dispatcher renders a plot *concept* at a resolution
*level* on a chosen backend, from already-prepared data::

    from ob_analytics.visualization import plot, _data
    fig = plot("trade_tape", backend="matplotlib",
               **_data.prepare_trades_data(trades))

Backends self-register their renderers into :data:`RENDERERS`, keyed by the
coordinate ``(concept, level, backend)`` where *level* is a :class:`Level`
(``L2``/``L3``) or ``None`` for level-less analytics.  The registry is
extensible via :func:`register_plot_backend`.  ``backend="matplotlib"``
(default) returns a Matplotlib figure; ``backend="plotly"`` returns an
interactive Plotly figure; ``backend="bokeh"`` returns a Bokeh figure
(covering the core concepts -- ``trade_tape``, ``depth_heatmap``,
``book_snapshot``, ``depth_chart`` -- for Bokeh / Panel server dashboards
and streaming views).

A concept registered at a single level resolves it automatically, so callers
pass only the concept name; *comparable* concepts (both L2 and L3 registered)
require an explicit ``level=``.

Plot types: depth heatmaps, event maps, volume maps, order book snapshots,
trade price charts, volume percentiles, and event histograms.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterator
from typing import Any

from loguru import logger
from matplotlib.axes import Axes

from ob_analytics._registry import Registry
from ob_analytics.visualization import _data as _viz_data
from ob_analytics.visualization._model import Level

# `infer_volume_scale` is a stable, user-facing helper that gallery callers
# import from this namespace; keep it as a public re-export. The `prepare_*`
# implementations live in `_data`; their public, friendly-named re-exports are
# the `prepare` namespace (lazily exposed via __getattr__ below).
infer_volume_scale = _viz_data.infer_volume_scale


# ---------------------------------------------------------------------------
# Renderer registry + unified dispatcher
# ---------------------------------------------------------------------------

RendererFn = Callable[..., Any]
RendererKey = tuple[str, Level | None, str]


class RendererRegistry(Registry[RendererKey, RendererFn]):
    """The renderer :class:`Registry`, which keeps each concept one kind.

    A concept is either level-less, registered at ``None``, or drawn at a
    level, registered at ``Level.L2`` and/or ``Level.L3``.  It is the same
    kind on every backend: :func:`plot` and the gallery call it the same way
    whichever backend draws it.
    """

    def placements(self, concept: str) -> Iterator[tuple[Level | None, str]]:
        """Yield each ``(level, backend)`` *concept* is registered at, in order."""
        return ((lvl, b) for (c, lvl, b) in self._items if c == concept)

    def register(self, key: RendererKey, value: RendererFn) -> None:
        """Register *value* under the key ``(concept, level, backend)``.

        Raises
        ------
        ValueError
            If *key* is not a ``(concept, level, backend)`` triple, or its
            concept is already registered, on any backend, as the other kind:
            at a level when *key* is level-less, or level-less when *key* has
            a level.
        """
        if not (isinstance(key, tuple) and len(key) == 3):
            raise ValueError(
                f"A renderer key is (concept, level, backend), got {key!r}. "
                "The level is Level.L2, Level.L3, or None for a level-less "
                "plot."
            )
        concept, level, _backend = key
        clash = next(
            (
                (other, b)
                for other, b in self.placements(concept)
                if (other is None) != (level is None)
            ),
            None,
        )
        if clash is not None:
            other, other_backend = clash
            existing = "level-less" if other is None else f"at {other}"
            wanted = "level-less" if level is None else f"at {level}"
            raise ValueError(
                f"Plot concept {concept!r} is already registered {existing} on "
                f"{other_backend!r}, so it cannot also be registered {wanted}. "
                "A concept is the same kind on every backend: register it at "
                "None everywhere, or at Level.L2 and/or Level.L3 everywhere."
            )
        super().register(key, value)


#: Registry of ``(concept, level, backend)`` → renderer function, where
#: *level* is a :class:`Level` (``L2``/``L3``) or ``None`` for level-less
#: analytics.  A concept is one kind or the other on every backend.
#: Renderer modules (``_matplotlib``, ``_plotly``) self-register at import
#: time -- see the self-registration block at the bottom of each.
RENDERERS: RendererRegistry = RendererRegistry("renderer")

#: Sentinel for ``plot(level=...)`` meaning "resolve the level from the
#: registry" -- distinct from ``None``, which is the explicit level of analytics.
_UNSET: Any = object()

# Lazy-import bootstrap: backend name → module that self-registers its
# renderers on import.  matplotlib is a hard dep (imported just below for its
# theme helpers, which also fires its registration); plotly and bokeh are
# optional and imported on first use.
_BACKEND_MODULES: dict[str, str] = {
    "matplotlib": "ob_analytics.visualization._matplotlib",
    "plotly": "ob_analytics.visualization._plotly",
    "bokeh": "ob_analytics.visualization._bokeh",
}


def register_plot_backend(name: str, module_path: str) -> None:
    """Register a visualization backend module.

    The module at *module_path* must call ``RENDERERS.register((concept,
    level, name), fn)`` for each plot it supports (typically at import time).
    It is imported lazily on the first :func:`plot` call that targets *name*.

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


def _accepts_theme(renderer: RendererFn) -> bool:
    """Whether *renderer* takes a ``theme`` keyword (by name or ``**kwargs``).

    Renderers written before themes applied to every backend take only
    ``(data)``; :func:`plot` must not pass them ``theme=``.
    """
    try:
        params = inspect.signature(renderer).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == "theme" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )


def _resolve_level(concept: str, backend: str) -> Level | None:
    """Resolve the implicit level of *concept* on *backend*.

    A concept registered at exactly one level resolves to it -- this covers
    L2-only plots and level-less analytics (registered at ``None``).  A
    *comparable* concept, registered at both L2 and L3, is ambiguous and needs
    an explicit ``level=`` from the caller.
    """
    levels = [lvl for lvl, b in RENDERERS.placements(concept) if b == backend]
    if not levels:
        raise KeyError(
            f"Unknown plot concept {concept!r} for backend {backend!r}. "
            f"Registered: {RENDERERS.list()}"
        )
    if len(levels) > 1:
        shown = ", ".join(sorted(str(lvl) for lvl in levels))
        raise ValueError(
            f"Plot concept {concept!r} is comparable (registered at levels "
            f"{shown}); pass level=Level.L2 or level=Level.L3 to disambiguate."
        )
    return levels[0]


def plot(
    concept: str,
    level: Level | None = _UNSET,
    *,
    backend: str = "matplotlib",
    ax: Axes | None = None,
    **data: Any,
) -> Any:
    """Render *concept* at *level* on *backend* from already-prepared *data*.

    Prepare data with the matching ``prepare_<concept>_data`` in
    :mod:`ob_analytics.visualization._data` (or a gallery helper) and spread
    it as keyword arguments::

        from ob_analytics.visualization import plot
        from ob_analytics.visualization import _data
        fig = plot("trade_tape", backend="matplotlib",
                   **_data.prepare_trades_data(trades))

    Parameters
    ----------
    concept : str
        Plot concept, e.g. ``"trade_tape"`` or ``"order_activity"``.
    level : Level, optional
        Resolution level (``Level.L2``/``Level.L3``).  Omit to auto-resolve
        when the concept is registered at a single level; required for a
        *comparable* concept registered at both.
    backend : str, optional
        Registered backend name (default ``"matplotlib"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (matplotlib only; ignored by other backends).
    **data
        Prepared plot data, as returned by the matching ``prepare_*`` helper.
        May include ``theme=PlotTheme(...)`` to override :data:`DEFAULT_THEME`
        for this call, on any backend.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure or bokeh.plotting.figure

    Raises
    ------
    ValueError
        If *backend* is not registered, or *concept* is comparable and no
        *level* was given.
    KeyError
        If *concept* (at the resolved *level*) is not registered.
    """
    theme = data.pop("theme", None)
    if backend not in _BACKEND_MODULES:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {sorted(_BACKEND_MODULES)}"
        )
    # Fire the backend's self-registration (cached after first import) so the
    # registry is populated before we resolve the level coordinate.
    importlib.import_module(_BACKEND_MODULES[backend])

    if level is _UNSET:
        level = _resolve_level(concept, backend)
    renderer = RENDERERS.get((concept, level, backend))
    kwargs: dict[str, Any] = {}
    if theme is not None:
        if _accepts_theme(renderer):
            kwargs["theme"] = theme
        else:
            logger.warning(
                "The {!r} renderer for {!r} takes no theme= argument; "
                "ignoring the theme.",
                backend,
                concept,
            )
    if backend == "matplotlib":
        return renderer(data, ax, **kwargs)
    return renderer(data, **kwargs)


# Shared display-window primitive: one mid-anchored clipping
# decision per gallery build instead of per-face ad-hoc clips.
FocusWindow = _viz_data.FocusWindow
focus_window = _viz_data.focus_window

# matplotlib save exports.  Imported *after* RENDERERS is defined: the
# self-registration block at the bottom of _matplotlib imports RENDERERS from
# this (partially initialized) package, so RENDERERS must already exist to
# avoid a circular-import deadlock.
from ob_analytics.visualization._matplotlib import format_time_axis, save_figure
from ob_analytics.visualization._palette import DEFAULT_PALETTE, Palette
from ob_analytics.visualization._theme import DEFAULT_THEME, PlotTheme

__all__ = [
    "DEFAULT_PALETTE",
    "DEFAULT_THEME",
    "RENDERERS",
    "FocusWindow",
    "Level",
    # Themes / persistence
    "Palette",
    "PlotTheme",
    "available_concepts",
    # Ticks -> quote-currency for low-level plotting (issue #155)
    "display_result",
    "focus_window",
    "format_time_axis",
    # Helpers users actually call
    "infer_volume_scale",
    # Dispatcher + registry
    "plot",
    # One-line plotting from a PipelineResult
    "plot_result",
    "prepare",
    "register_plot_backend",
    "save_figure",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the result-level plotting API (PEP 562).

    ``plot_result`` / ``available_concepts`` live in :mod:`.gallery`, which
    imports :func:`plot` from this package; importing them eagerly here would
    create a cycle.  ``prepare`` is re-exported lazily for symmetry.
    """
    if name in ("plot_result", "available_concepts", "display_result"):
        from ob_analytics.visualization import gallery

        return getattr(gallery, name)
    if name == "prepare":
        # importlib (not ``from . import prepare``) so the submodule import does
        # not re-enter this __getattr__ and recurse.
        return importlib.import_module("ob_analytics.visualization.prepare")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
