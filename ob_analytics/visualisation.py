"""Visualization functions for limit order book analytics.

All plot functions return a figure object and accept an optional
``backend`` parameter (default ``"matplotlib"``).  When
``backend="plotly"``, an interactive Plotly figure is returned instead.

The backend registry is extensible — see :func:`register_plot_backend`.

Plot types: depth heatmaps, event maps, volume maps, order book snapshots,
trade price charts, volume percentiles, and event histograms.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ob_analytics._chart_data import (
    prepare_current_depth_data,
    prepare_event_map_data,
    prepare_events_histogram_data,
    prepare_kyle_lambda_data,
    prepare_ofi_data,
    prepare_price_levels_data,
    prepare_time_series_data,
    prepare_trades_data,
    prepare_volume_map_data,
    prepare_volume_percentiles_data,
    prepare_vpin_data,
)

# Re-export theme infrastructure from the matplotlib backend so
# existing ``from ob_analytics.visualisation import PlotTheme`` keeps working.
from ob_analytics._matplotlib import (  # noqa: F401 – re-exports
    PlotTheme,
    _apply_theme,
    _create_axes,
    get_plot_theme,
    save_figure,
    set_plot_theme,
)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, str] = {
    "matplotlib": "ob_analytics._matplotlib",
    "plotly": "ob_analytics._plotly",
}

# Mapping from public function name → backend function name pattern.
# matplotlib: "mpl_{name}", plotly: "plotly_{name}", etc.
_FUNC_PREFIX: dict[str, str] = {
    "matplotlib": "mpl_",
    "plotly": "plotly_",
}


def register_plot_backend(name: str, module_path: str, *, func_prefix: str | None = None) -> None:
    """Register a visualization backend.

    The module at *module_path* must export rendering functions following
    the naming convention ``{prefix}{plot_name}(data, ...)``.  The default
    prefix is ``"{name}_"``.

    Parameters
    ----------
    name : str
        Backend name used in ``plot_*(backend=name)``.
    module_path : str
        Dotted import path, e.g. ``"my_package._bokeh_backend"``.
    func_prefix : str, optional
        Function name prefix. Defaults to ``"{name}_"``.

    Examples
    --------
    >>> from ob_analytics.visualisation import register_plot_backend
    >>> register_plot_backend("bokeh", "my_pkg._bokeh")  # expects bokeh_trades(), etc.
    """
    _BACKENDS[name] = module_path
    _FUNC_PREFIX[name] = func_prefix if func_prefix is not None else f"{name}_"


def _get_renderer(backend: str, plot_name: str) -> Any:
    """Lazy-import the backend module and return the renderer function.

    Parameters
    ----------
    backend : str
        Registered backend name.
    plot_name : str
        Short plot name without prefix (e.g. ``"trades"``).

    Returns
    -------
    Callable
        The backend-specific rendering function.

    Raises
    ------
    ValueError
        If *backend* is not registered.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {sorted(_BACKENDS)}"
        )
    mod = importlib.import_module(_BACKENDS[backend])
    func_name = _FUNC_PREFIX[backend] + plot_name
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Public plot functions (thin dispatchers)
# ---------------------------------------------------------------------------


def plot_time_series(
    timestamp: pd.Series,
    series: pd.Series,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    title: str = "time series",
    y_label: str = "series",
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plots a time series.

    Parameters
    ----------
    timestamp : pandas.Series
        Series of timestamps.
    series : pandas.Series
        Series of values to plot.
    start_time : pandas.Timestamp, optional
        Start time for the plot. Default is None.
    end_time : pandas.Timestamp, optional
        End time for the plot. Default is None.
    title : str, optional
        Title of the plot. Default is 'time series'.
    y_label : str, optional
        Label for the y-axis. Default is 'series'.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_time_series_data(timestamp, series, start_time, end_time, title, y_label)
    renderer = _get_renderer(backend, "time_series")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_trades(
    trades: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plots the trades data as a step plot.

    Parameters
    ----------
    trades : pandas.DataFrame
        DataFrame containing the trades data with columns 'timestamp' and 'price'.
    start_time : pandas.Timestamp, optional
        Start time for the plot. Default is None.
    end_time : pandas.Timestamp, optional
        End time for the plot. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_trades_data(trades, start_time, end_time)
    renderer = _get_renderer(backend, "trades")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_price_levels(
    depth: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    show_mp: bool = True,
    show_all_depth: bool = False,
    col_bias: float = 0.1,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float = 1,
    price_by: float | None = None,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot price levels with depth, spread, and trades data.

    Parameters
    ----------
    depth : pandas.DataFrame
        DataFrame containing depth data.
    spread : pandas.DataFrame, optional
        DataFrame containing spread data. Default is None.
    trades : pandas.DataFrame, optional
        DataFrame containing trades data. Default is None.
    show_mp : bool, optional
        Whether to show midprice. Default is True.
    show_all_depth : bool, optional
        Whether to show all depth levels. Default is False.
    col_bias : float, optional
        Color bias for volume mapping. Default is 0.1.
    start_time : pandas.Timestamp, optional
        Start time for filtering data. Default is None.
    end_time : pandas.Timestamp, optional
        End time for filtering data. Default is None.
    price_from : float, optional
        Minimum price for filtering depth data. Default is None.
    price_to : float, optional
        Maximum price for filtering depth data. Default is None.
    volume_from : float, optional
        Minimum volume for filtering depth data. Default is None.
    volume_to : float, optional
        Maximum volume for filtering depth data. Default is None.
    volume_scale : float, optional
        Scaling factor for volume. Default is 1.
    price_by : float, optional
        Step size for y-axis ticks (price levels). Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_price_levels_data(
        depth, spread, trades, show_mp, show_all_depth, col_bias,
        start_time, end_time, price_from, price_to, volume_from, volume_to,
        volume_scale, price_by,
    )
    renderer = _get_renderer(backend, "price_levels")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


# Keep the fast variant as a thin wrapper for backward compat.
def plot_price_levels_faster(
    depth: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    show_mp: bool = True,
    col_bias: float = 0.1,
    price_by: float | None = None,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Fast plotting of price levels using Matplotlib.

    Parameters
    ----------
    depth : pandas.DataFrame
        Filtered depth DataFrame.
    spread : pandas.DataFrame, optional
        Spread DataFrame. Default is None.
    trades : pandas.DataFrame, optional
        Trades DataFrame. Default is None.
    show_mp : bool, optional
        Whether to show midprice. Default is True.
    col_bias : float, optional
        Color bias for volume mapping. Default is 0.1.
    price_by : float, optional
        Step size for y-axis ticks (price levels). Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    # This function wraps the same renderer as plot_price_levels;
    # the data dict it passes already has pre-filtered depth.
    data = {
        "depth": depth,
        "spread": spread,
        "trades": trades,
        "show_mp": show_mp,
        "col_bias": col_bias,
        "price_by": price_by,
    }
    renderer = _get_renderer(backend, "price_levels")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_event_map(
    events: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float = 1,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot an event map of limit order events.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing event data.
    start_time : pandas.Timestamp, optional
        Start time for the plot. Default is None.
    end_time : pandas.Timestamp, optional
        End time for the plot. Default is None.
    price_from : float, optional
        Minimum price for filtering events. Default is None.
    price_to : float, optional
        Maximum price for filtering events. Default is None.
    volume_from : float, optional
        Minimum volume for filtering events. Default is None.
    volume_to : float, optional
        Maximum volume for filtering events. Default is None.
    volume_scale : float, optional
        Scaling factor for volume. Default is 1.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_event_map_data(
        events, start_time, end_time, price_from, price_to,
        volume_from, volume_to, volume_scale,
    )
    renderer = _get_renderer(backend, "event_map")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_volume_map(
    events: pd.DataFrame,
    action: str = "deleted",
    event_type: list[str] | None = None,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float = 1,
    log_scale: bool = False,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot a volume map of flashed limit orders.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing event data.
    action : str, optional
        The action to filter ('deleted' or 'created'). Default is 'deleted'.
    event_type : list of str, optional
        List of event types to include. Default is ['flashed-limit'].
    start_time : pandas.Timestamp, optional
        Start time for the plot. Default is None.
    end_time : pandas.Timestamp, optional
        End time for the plot. Default is None.
    price_from : float, optional
        Minimum price for filtering events. Default is None.
    price_to : float, optional
        Maximum price for filtering events. Default is None.
    volume_from : float, optional
        Minimum volume for filtering events. Default is None.
    volume_to : float, optional
        Maximum volume for filtering events. Default is None.
    volume_scale : float, optional
        Scaling factor for volume. Default is 1.
    log_scale : bool, optional
        Whether to use a logarithmic scale on the y-axis. Default is False.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if event_type is None:
        event_type = ["flashed-limit"]
    data = prepare_volume_map_data(
        events, action, event_type, start_time, end_time,
        price_from, price_to, volume_from, volume_to,
        volume_scale, log_scale,
    )
    renderer = _get_renderer(backend, "volume_map")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_current_depth(
    order_book: dict,
    volume_scale: float = 1,
    show_quantiles: bool = True,
    show_volume: bool = True,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot the current order book depth.

    Parameters
    ----------
    order_book : dict
        Dictionary containing 'bids', 'asks', and 'timestamp'.
    volume_scale : float, optional
        Scaling factor for volume. Default is 1.
    show_quantiles : bool, optional
        Whether to highlight highest 1%% volume with vertical lines. Default is True.
    show_volume : bool, optional
        Whether to show volume bars. Default is True.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_current_depth_data(order_book, volume_scale, show_quantiles, show_volume)
    renderer = _get_renderer(backend, "current_depth")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_volume_percentiles(
    depth_summary: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    volume_scale: float = 1,
    perc_line: bool = True,
    side_line: bool = True,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot volume percentiles over time.

    Parameters
    ----------
    depth_summary : pandas.DataFrame
        DataFrame containing depth summary statistics.
    start_time : pandas.Timestamp, optional
        Start time for the plot. Default is None.
    end_time : pandas.Timestamp, optional
        End time for the plot. Default is None.
    volume_scale : float, optional
        Scaling factor for volume. Default is 1.
    perc_line : bool, optional
        Whether to draw lines between percentiles. Default is True.
    side_line : bool, optional
        Whether to draw a line separating bids and asks. Default is True.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_volume_percentiles_data(
        depth_summary, start_time, end_time, volume_scale, perc_line, side_line,
    )
    renderer = _get_renderer(backend, "volume_percentiles")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_events_histogram(
    events: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    val: str = "volume",
    bw: float | None = None,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Plot a histogram given event data.

    Convenience function for plotting event price and volume histograms.
    Will plot ask/bid bars side by side.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing event data.
    start_time : pandas.Timestamp, optional
        Include event data >= this time. Default is None.
    end_time : pandas.Timestamp, optional
        Include event data <= this time. Default is None.
    val : str, optional
        'volume' or 'price'. Default is 'volume'.
    bw : float, optional
        Bin width (e.g., for price, 0.5 = 50 cent buckets). Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_events_histogram_data(events, start_time, end_time, val, bw)
    renderer = _get_renderer(backend, "events_histogram")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


# ---------------------------------------------------------------------------
# Flow toxicity plots
# ---------------------------------------------------------------------------


def plot_vpin(
    vpin_df: pd.DataFrame,
    threshold: float = 0.7,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """Plot VPIN time series with a toxicity threshold.

    Parameters
    ----------
    vpin_df : pandas.DataFrame
        Output of :func:`~ob_analytics.flow_toxicity.compute_vpin`
        with columns ``timestamp_end``, ``vpin``, and ``vpin_avg``.
    threshold : float, optional
        Horizontal line indicating "toxic" threshold.  Default 0.7.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_vpin_data(vpin_df, threshold)
    renderer = _get_renderer(backend, "vpin")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_order_flow_imbalance(
    ofi_df: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """Plot order flow imbalance as a bar chart with optional price overlay.

    Parameters
    ----------
    ofi_df : pandas.DataFrame
        Output of :func:`~ob_analytics.flow_toxicity.order_flow_imbalance`
        with columns ``timestamp`` and ``ofi``.
    trades : pandas.DataFrame, optional
        Trades DataFrame with ``timestamp`` and ``price`` columns.
        When provided, the trade price is drawn as a line on a
        secondary y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_ofi_data(ofi_df, trades)
    renderer = _get_renderer(backend, "order_flow_imbalance")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)


def plot_kyle_lambda(
    kyle_result: object,
    ax: Axes | None = None,
    backend: str = "matplotlib",
) -> Any:
    """Plot Kyle's Lambda regression: signed volume vs ΔPrice.

    Parameters
    ----------
    kyle_result : KyleLambdaResult
        Output of :func:`~ob_analytics.flow_toxicity.compute_kyle_lambda`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
        Ignored when *backend* is not ``"matplotlib"``.
    backend : str, optional
        Rendering backend (default ``"matplotlib"``).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    data = prepare_kyle_lambda_data(kyle_result)
    renderer = _get_renderer(backend, "kyle_lambda")
    if backend == "matplotlib":
        return renderer(data, ax)
    return renderer(data)
