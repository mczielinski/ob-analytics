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
from typing import Any

from loguru import logger
from matplotlib.axes import Axes

from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import Level
from ob_analytics.visualization import _data as _viz_data
from ob_analytics.visualization._matplotlib import format_time_axis, save_figure
from ob_analytics.visualization._palette import DEFAULT_PALETTE, Palette
from ob_analytics.visualization._renderers import (
    RENDERERS,
    RendererFn,
    RendererRegistry,
    _load_backend,
    load_plot_backend_plugins,
    register_plot_backend,
)
from ob_analytics.visualization._theme import DEFAULT_THEME, PlotTheme

# `infer_volume_scale` is a stable, user-facing helper that gallery callers
# import from this namespace; keep it as a public re-export. The `prepare_*`
# implementations live in `_data`; their public, friendly-named re-exports are
# the `prepare` namespace (lazily exposed via __getattr__ below).
infer_volume_scale = _viz_data.infer_volume_scale


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

#: Sentinel for ``plot(level=...)`` meaning "resolve the level from the
#: registry" -- distinct from ``None``, which is the explicit level of analytics.
_UNSET: Any = object()


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
        raise RENDERERS.missing(concept, None, backend)
    if len(levels) > 1:
        shown = ", ".join(sorted(str(lvl) for lvl in levels))
        raise ConfigError(
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
        Axes to draw on (matplotlib only; ignored by other backends).  The
        figure that holds *ax* is yours: the face draws on *ax* and leaves the
        figure's layout alone, so call ``fig.tight_layout()`` (or use a
        layout engine) once all the panels are drawn.  Without *ax*, the face
        creates its own figure and lays it out.
    **data
        Prepared plot data, as returned by the matching ``prepare_*`` helper.
        May include ``theme=PlotTheme(...)`` to override :data:`DEFAULT_THEME`
        for this call, on any backend.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure or bokeh.plotting.figure

    Raises
    ------
    UnknownNameError
        If *backend* is not registered, or *backend* does not draw
        *concept* at the resolved *level*.
    ConfigError
        If *concept* is comparable and no *level* was given.
    """
    theme = data.pop("theme", None)
    _load_backend(backend)

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

# Plot backends that other packages advertise through entry points.
load_plot_backend_plugins()

__all__ = [
    "DEFAULT_PALETTE",
    "DEFAULT_THEME",
    "RENDERERS",
    "FocusWindow",
    "Level",
    # Themes / persistence
    "Palette",
    "PlotTheme",
    "RendererRegistry",
    "available_concepts",
    # Ticks -> quote-currency for low-level plotting (issue #155)
    "display_result",
    "focus_window",
    "format_time_axis",
    # Helpers users actually call
    "infer_volume_scale",
    "load_plot_backend_plugins",
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
