"""Visualization functions for limit order book analytics.

The unified :func:`plot` dispatcher renders a plot *concept* at a resolution
*level* on a chosen backend, from already-prepared data::

    from ob_analytics.visualization import plot, _data
    fig = plot("trade_tape", backend="matplotlib",
               **_data.prepare_trades_data(trades))

Every plot is a coordinate ``(concept, level, backend)``, where *level* is a
:class:`Level` (``L2``/``L3``) or ``None`` for level-less analytics.  One
concept table declares each plot of the package and the backends that draw
it; the backends register their renderers from it into :data:`RENDERERS`.
Other packages add plots and backends through :data:`RENDERERS` and
:func:`register_plot_backend`.  ``backend="matplotlib"`` (default) returns a
Matplotlib figure; ``backend="plotly"`` returns an interactive Plotly
figure; ``backend="bokeh"`` returns a Bokeh figure (covering the core
concepts -- ``trade_tape``, ``depth_heatmap``, ``book_snapshot``,
``depth_chart`` -- for Bokeh / Panel server dashboards and streaming views).
Asking a backend for a plot it does not draw raises and names the backends
that do: the book replay, for one, is drawn by Plotly only.

A concept drawn at one level resolves it, so callers pass only the concept
name; a concept drawn at both L2 and L3 needs ``level=``.  The same rule holds
for :func:`plot_result` and ``PipelineResult.plot``.

Plot types: depth heatmaps, event maps, volume maps, order book snapshots,
trade price charts, volume percentiles, and event histograms.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
from typing import Any

from loguru import logger
from matplotlib.axes import Axes

from ob_analytics.exceptions import ConfigError
from ob_analytics.pipeline import _UNSET
from ob_analytics.protocols import Level
from ob_analytics.visualization import _concepts, _data as _viz_data
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


def _resolve_level(
    concept: str,
    level: Level | str | None,
    backend: str,
    levels: Iterable[Level | None] | None = None,
) -> Level | None:
    """Return the level to draw *concept* at, for every plot entry point.

    A *level* the caller gave is used as given (``"L2"`` becomes
    ``Level.L2``; ``None`` is the level of a level-less plot).  A *level* left
    out (``_UNSET``) resolves to the only one of *levels*: the levels the
    caller can draw *concept* at for its data, or, when it passes none, the
    levels *backend* draws *concept* at.  More than one is a choice the
    caller must make.

    Raises
    ------
    UnknownNameError
        If *levels* is not given and *backend* does not draw *concept*.
    ConfigError
        If *level* was left out and *concept* can be drawn at more than one
        level.
    """
    if level is not _UNSET:
        return None if level is None else Level(level)
    if levels is None:
        _concepts.require_backend(concept, backend)
        levels = _concepts.levels(concept, backend)
    choices = sorted(set(levels), key=str)
    if len(choices) > 1:
        shown = " or ".join(f'level="{lvl}"' for lvl in choices)
        raise ConfigError(
            f"Plot concept {concept!r} has a plot at each of "
            f"{', '.join(str(lvl) for lvl in choices)}; pass {shown}."
        )
    return choices[0]


def _check_ax(ax: Axes | None, backend: str) -> None:
    """Refuse *ax* for a backend other than Matplotlib.

    Raises
    ------
    ConfigError
        If *ax* is given and *backend* is not ``"matplotlib"``.
    """
    if ax is not None and backend != "matplotlib":
        raise ConfigError(
            f"ax= is a Matplotlib axes, but backend {backend!r} makes its own "
            "figure. Leave ax out, or use backend='matplotlib'."
        )


def plot(
    concept: str,
    level: Level | str | None = _UNSET,
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
    level : Level or str or None, optional
        Resolution level (``Level.L2``/``"L2"`` or ``Level.L3``/``"L3"``).
        Leave it out when *backend* draws the concept at one level; it is
        required when *backend* draws the concept at both L2 and L3.
    backend : str, optional
        Registered backend name (default ``"matplotlib"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (Matplotlib only; another backend refuses it).  The
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
        If *backend* is not registered, or does not draw *concept* (at
        *level*).  The message names the backends that do.
    ConfigError
        If *level* was left out and *backend* draws *concept* at both L2 and
        L3, or *ax* is given for a backend other than Matplotlib.
    """
    theme = data.pop("theme", None)
    _load_backend(backend)

    _check_ax(ax, backend)
    level = _resolve_level(concept, level, backend)
    _concepts.require(concept, level, backend)
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


# The shared display windows: the busiest stretch of time, and the
# mid-anchored price band, each decided once per gallery build.
FocusWindow = _viz_data.FocusWindow
busiest_window = _viz_data.busiest_window
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
    "busiest_window",
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
