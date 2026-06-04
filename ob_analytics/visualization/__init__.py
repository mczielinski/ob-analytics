"""Visualization functions for limit order book analytics.

The unified :func:`plot` dispatcher renders a named plot on a chosen backend
from already-prepared data::

    from ob_analytics.visualization import plot, _data
    fig = plot("trades", backend="matplotlib", **_data.prepare_trades_data(trades))

Backends self-register their renderers into :data:`RENDERERS` (keyed by
``(plot_name, backend)``); the registry is extensible via
:func:`register_plot_backend`.  ``backend="matplotlib"`` (default) returns a
Matplotlib figure; ``backend="plotly"`` returns an interactive Plotly figure.

Plot types: depth heatmaps, event maps, volume maps, order book snapshots,
trade price charts, volume percentiles, and event histograms.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from matplotlib.axes import Axes

from ob_analytics._registry import Registry
from ob_analytics.visualization import _data as _viz_data

# `infer_volume_scale` is a stable, user-facing helper that gallery callers
# import from this namespace; keep it as a public re-export. The `prepare_*`
# helpers are strictly internal — access them via `_viz_data.prepare_*`.
infer_volume_scale = _viz_data.infer_volume_scale


# ---------------------------------------------------------------------------
# Renderer registry + unified dispatcher
# ---------------------------------------------------------------------------

RendererFn = Callable[..., Any]

#: Registry of ``(plot_name, backend)`` → renderer function.  Renderer
#: modules (``_matplotlib``, ``_plotly``) self-register at import time -- see
#: the self-registration block at the bottom of each module.
RENDERERS: Registry[tuple[str, str], RendererFn] = Registry("renderer")

# Lazy-import bootstrap: backend name → module that self-registers its
# renderers on import.  matplotlib is a hard dep (imported just below for its
# theme helpers, which also fires its registration); plotly is optional and
# imported on first use.
_BACKEND_MODULES: dict[str, str] = {
    "matplotlib": "ob_analytics.visualization._matplotlib",
    "plotly": "ob_analytics.visualization._plotly",
}


def register_plot_backend(name: str, module_path: str) -> None:
    """Register a visualization backend module.

    The module at *module_path* must call ``RENDERERS.register((plot_name,
    name), fn)`` for each plot it supports (typically at import time).  It is
    imported lazily on the first :func:`plot` call that targets *name*.

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


def plot(
    name: str,
    *,
    backend: str = "matplotlib",
    ax: Axes | None = None,
    **data: Any,
) -> Any:
    """Render plot *name* on *backend* from already-prepared *data*.

    Prepare data with the matching ``prepare_<name>_data`` in
    :mod:`ob_analytics.visualization._data` (or a gallery helper) and spread
    it as keyword arguments::

        from ob_analytics.visualization import plot
        from ob_analytics.visualization import _data
        fig = plot("trades", backend="matplotlib",
                   **_data.prepare_trades_data(trades))

    Parameters
    ----------
    name : str
        Plot name, e.g. ``"trades"`` or ``"event_map"``.
    backend : str, optional
        Registered backend name (default ``"matplotlib"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (matplotlib only; ignored by other backends).
    **data
        Prepared plot data, as returned by the matching ``prepare_*`` helper.
        May include ``theme=PlotTheme(...)`` to override :data:`DEFAULT_THEME`
        (matplotlib backend only; ignored by other backends).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If *backend* is not registered.
    """
    theme = data.pop("theme", None)
    if (name, backend) not in RENDERERS:
        if backend not in _BACKEND_MODULES:
            raise ValueError(
                f"Unknown backend {backend!r}. Available: {sorted(_BACKEND_MODULES)}"
            )
        importlib.import_module(_BACKEND_MODULES[backend])  # fires registration
    renderer = RENDERERS.get((name, backend))
    if backend == "matplotlib":
        return renderer(data, ax) if theme is None else renderer(data, ax, theme=theme)
    return renderer(data)


# matplotlib theme + save exports.  Imported *after* RENDERERS is defined: the
# self-registration block at the bottom of _matplotlib imports RENDERERS from
# this (partially initialized) package, so RENDERERS must already exist to
# avoid a circular-import deadlock.
from ob_analytics.visualization._matplotlib import (  # noqa: E402
    DEFAULT_THEME,
    PlotTheme,
    save_figure,
)


__all__ = [
    # Dispatcher + registry
    "plot",
    "RENDERERS",
    "register_plot_backend",
    # Themes / persistence
    "PlotTheme",
    "DEFAULT_THEME",
    "save_figure",
    # Helpers users actually call
    "infer_volume_scale",
]
