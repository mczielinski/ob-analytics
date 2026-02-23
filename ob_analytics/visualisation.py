"""Visualization functions for limit order book analytics.

All plot functions return a :class:`~matplotlib.figure.Figure` and accept
an optional ``ax`` parameter for subplot composition.  Theming is
configurable via :class:`PlotTheme` / :func:`set_plot_theme`.

Plot types: depth heatmaps, event maps, volume maps, order book snapshots,
trade price charts, volume percentiles, and event histograms.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from loguru import logger

from ob_analytics._utils import reverse_matrix
from ob_analytics.depth import filter_depth


# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotTheme:
    """Configurable visual theme for ob-analytics plots.

    Attributes
    ----------
    style : str
        Seaborn style name (e.g. ``"darkgrid"``, ``"whitegrid"``).
    context : str
        Seaborn context name (e.g. ``"notebook"``, ``"talk"``).
    font_scale : float
        Global font scaling factor.
    rc : dict[str, object]
        Matplotlib rc overrides applied on top of the seaborn theme.
    """

    style: str = "darkgrid"
    context: str = "notebook"
    font_scale: float = 1.5
    rc: dict[str, object] = field(
        default_factory=lambda: {"lines.linewidth": 2.5, "axes.facecolor": "darkgray"}
    )


_current_theme: PlotTheme = PlotTheme()


def set_plot_theme(theme: PlotTheme) -> None:
    """Set the global plot theme used by all ob-analytics plot functions.

    Parameters
    ----------
    theme : PlotTheme
        Theme configuration to apply globally.
    """
    global _current_theme
    _current_theme = theme


def get_plot_theme() -> PlotTheme:
    """Return the current global plot theme.

    Returns
    -------
    PlotTheme
        The active theme configuration.
    """
    return _current_theme


def _apply_theme() -> None:
    """Apply the current theme to matplotlib / seaborn."""
    t = _current_theme
    sns.set_theme(style=t.style, context=t.context, font_scale=t.font_scale, rc=dict(t.rc))  # type: ignore


def _create_axes(
    ax: Axes | None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating a new figure only when *ax* is ``None``."""
    if ax is not None:
        return ax.get_figure(), ax  # type: ignore[return-value]
    _apply_theme()
    fig, new_ax = plt.subplots(figsize=figsize)
    return fig, new_ax


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 150,
    **kwargs: object,
) -> None:
    """Save a matplotlib *fig* to *path* with sensible defaults.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to save.
    path : str or Path
        Destination file path (e.g. ``"output/plot.png"``).
    dpi : int, optional
        Resolution in dots per inch (default 150).
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`~matplotlib.figure.Figure.savefig`.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)  # type: ignore


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_time_series(
    timestamp: pd.Series,
    series: pd.Series,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    title: str = "time series",
    y_label: str = "series",
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(timestamp) != len(series):
        raise ValueError("Length of timestamp and series must be the same.")

    df = pd.DataFrame({"ts": timestamp, "val": series})

    if not start_time:
        start_time = df["ts"].min()
    if not end_time:
        end_time = df["ts"].max()

    df = df[(df["ts"] >= start_time) & (df["ts"] <= end_time)]

    fig, ax = _create_axes(ax, figsize=(10, 6))
    sns.lineplot(data=df, x="ts", y="val", drawstyle="steps-post", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_trades(
    trades: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not start_time:
        start_time = trades["timestamp"].min()
    if not end_time:
        end_time = trades["timestamp"].max()

    filtered_trades = trades[
        (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
    ]

    price_range = filtered_trades["price"].max() - filtered_trades["price"].min()
    price_by = 10 ** round(np.log10(price_range) - 1)
    y_breaks = np.arange(
        round(min(filtered_trades["price"]) / price_by) * price_by,
        round(max(filtered_trades["price"]) / price_by) * price_by,
        step=price_by,
    )

    fig, ax = _create_axes(ax, figsize=(10, 6))
    sns.lineplot(
        data=filtered_trades, x="timestamp", y="price",
        drawstyle="steps-post", ax=ax,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_yticks(y_breaks)
    ax.grid(True)
    fig.tight_layout()
    return fig


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
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    depth_local = depth.copy()
    depth_local["volume"] = depth_local["volume"] * volume_scale

    if start_time is None:
        start_time = depth_local["timestamp"].iloc[0]
    if end_time is None:
        end_time = depth_local["timestamp"].iloc[-1]

    if spread is not None:
        spread = spread[
            (spread["timestamp"] >= start_time) & (spread["timestamp"] <= end_time)
        ]
        if price_from is None:
            price_from = 0.995 * spread["best_bid_price"].min()
        if price_to is None:
            price_to = 1.005 * spread["best_ask_price"].max()

    if trades is not None:
        trades = trades[
            (trades["timestamp"] >= start_time) & (trades["timestamp"] <= end_time)
        ]
        if price_from is None:
            price_from = 0.995 * trades["price"].min()
        else:
            trades = trades[trades["price"] >= price_from]
        if price_to is None:
            price_to = 1.005 * trades["price"].max()
        else:
            trades = trades[trades["price"] <= price_to]

    if price_from is not None:
        depth_local = depth_local[depth_local["price"] >= price_from]
    if price_to is not None:
        depth_local = depth_local[depth_local["price"] <= price_to]
    if volume_from is not None:
        depth_local = depth_local[(depth_local["volume"] >= volume_from) | (depth_local["volume"] == 0)]
    if volume_to is not None:
        depth_local = depth_local[depth_local["volume"] <= volume_to]

    depth_filtered = filter_depth(depth_local, start_time, end_time)

    if not show_all_depth:

        counts = depth_filtered.groupby("price", as_index=False)["timestamp"].agg(
            count="size",
            first_ts="min",
            last_ts="max"
        )
        unchanged = counts[
            (counts["count"] == 2) &
            (counts["first_ts"] == start_time) &
            (counts["last_ts"] == end_time)
        ]
        depth_filtered = depth_filtered[~depth_filtered["price"].isin(unchanged["price"])]

    depth_filtered.loc[depth_filtered["volume"] == 0, "volume"] = np.nan

    return plot_price_levels_faster(
        depth_filtered, spread, trades, show_mp, col_bias, price_by, ax=ax,
    )


def plot_price_levels_faster(
    depth: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    show_mp: bool = True,
    col_bias: float = 0.1,
    price_by: float | None = None,
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    depth = depth.copy()
    depth.sort_values(by="timestamp", inplace=True, kind="stable")
    if depth.empty or depth.groupby("price").size().min() < 2:
        logger.warning("Not enough data for any price level")
        fig, ax = _create_axes(ax, figsize=(12, 7))
        return fig

    depth["alpha"] = np.where(
        depth["volume"].isna(), 0, np.where(depth["volume"] < 1, 0.1, 1)
    )

    log_10 = False
    if col_bias <= 0:
        col_bias = 1
        log_10 = True

    cmap = plt.get_cmap("viridis")

    vmin = depth["volume"].min()
    vmax = depth["volume"].max()
    if log_10:
        if vmin <= 0:
            vmin = depth["volume"][depth["volume"] > 0].min()
        norm: mcolors.Normalize = mcolors.LogNorm(vmin=vmin, vmax=vmax)  # type: ignore
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = _create_axes(ax, figsize=(12, 7))

    depth["timestamp_numeric"] = mdates.date2num(depth["timestamp"])

    for price, group in depth.groupby("price"):
        group = group.sort_values("timestamp", kind="stable")
        x = group["timestamp_numeric"].values
        y = group["price"].values
        v = group["volume"].values
        a = group["alpha"].values
        if len(x) < 2:
            continue
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = cmap(norm(np.asarray(v[:-1])))
        seg_colors[:, -1] = a[:-1]
        lc = collections.LineCollection(segments, colors=seg_colors, linewidths=2)
        ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Volume")

    if spread is not None:
        spread = spread.copy()
        spread.sort_values(by="timestamp", inplace=True, kind="stable")
        if show_mp and "best_bid_price" in spread and "best_ask_price" in spread:
            spread["midprice"] = (
                spread["best_bid_price"] + spread["best_ask_price"]
            ) / 2
            ax.plot(
                spread["timestamp"],
                spread["midprice"],
                color="#ffffff",
                linewidth=1.1,
                label="Midprice",
            )
        else:
            if "best_ask_price" in spread:
                ax.step(
                    spread["timestamp"],
                    spread["best_ask_price"],
                    color="#ff0000",
                    linewidth=1.5,
                    where="post",
                    label="Best Ask",
                )
            if "best_bid_price" in spread:
                ax.step(
                    spread["timestamp"],
                    spread["best_bid_price"],
                    color="#00ff00",
                    linewidth=1.5,
                    where="post",
                    label="Best Bid",
                )

    if trades is not None:
        buys = trades[trades["direction"] == "buy"]
        sells = trades[trades["direction"] == "sell"]

        if not sells.empty:
            ax.scatter(
                sells["timestamp"],
                sells["price"],
                s=50,
                facecolors="none",
                edgecolors="#ff0000",
                linewidths=1.5,
                zorder=5,
                marker="v",
                label="Sell Trades",
            )

        if not buys.empty:
            ax.scatter(
                buys["timestamp"],
                buys["price"],
                s=50,
                facecolors="none",
                edgecolors="#00ff00",
                linewidths=1.5,
                zorder=5,
                marker="^",
                label="Buy Trades",
            )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_title("Price Levels Over Time")

    ymin = depth["price"].min()
    ymax = depth["price"].max()
    ax.set_ylim((ymin, ymax))

    if price_by is not None:
        y_ticks = np.arange(round(ymin), round(ymax) + price_by, price_by)
        ax.set_yticks(y_ticks)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    fig.tight_layout()
    return fig


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
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    events = events[
        (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
        & ((events["type"] == "flashed-limit") | (events["type"] == "resting-limit"))
    ].copy()

    events["volume"] *= volume_scale

    if volume_from is not None:
        events = events[events["volume"] >= volume_from]
    if volume_to is not None:
        events = events[events["volume"] <= volume_to]

    if price_from is None:
        price_from = events["price"].quantile(0.01)
    if price_to is None:
        price_to = events["price"].quantile(0.99)

    events = events[(events["price"] >= price_from) & (events["price"] <= price_to)]

    created = events[events["action"] == "created"]
    deleted = events[events["action"] == "deleted"]

    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    price_by = 10 ** round(np.log10(events["price"].max() - events["price"].min()) - 1)

    fig, ax = _create_axes(ax, figsize=(10, 6))

    sns.scatterplot(
        data=created,
        x="timestamp",
        y="price",
        size="volume",
        sizes=(20, 200),
        color="#333333",
        ax=ax,
        legend=False,
        marker="o",
    )

    sns.scatterplot(
        data=deleted,
        x="timestamp",
        y="price",
        size="volume",
        sizes=(20, 200),
        color="#333333",
        ax=ax,
        legend=False,
        marker="o",
        edgecolor="black",
        alpha=0.5,
    )

    sns.scatterplot(
        data=events,
        x="timestamp",
        y="price",
        hue="direction",
        size=0.1,
        palette=col_pal,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_yticks(
        np.arange(
            round(events["price"].min() / price_by) * price_by,
            round(events["price"].max() / price_by) * price_by,
            price_by,
        )
    )

    fig.tight_layout()
    return fig


def plot_volume_map(
    events: pd.DataFrame,
    action: str = "deleted",
    event_type: list[str] = ["flashed-limit"],  # noqa: B006
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    volume_from: float | None = None,
    volume_to: float | None = None,
    volume_scale: float = 1,
    log_scale: bool = False,
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    if action not in ("deleted", "created"):
        raise ValueError(f"action must be 'deleted' or 'created', got {action!r}")

    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    events["volume"] *= volume_scale

    mask = (
        (events["action"] == action)
        & events["type"].isin(event_type)
        & (events["timestamp"] >= start_time)
        & (events["timestamp"] <= end_time)
    )
    events = events[mask]

    if price_from:
        events = events[events["price"] >= price_from]
    if price_to:
        events = events[events["price"] <= price_to]
    if volume_from is None:
        volume_from = events["volume"].quantile(0.0001)
    events = events[events["volume"] >= volume_from]
    if volume_to is None:
        volume_to = events["volume"].quantile(0.9999)
    events = events[events["volume"] <= volume_to]

    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    fig, ax = _create_axes(ax, figsize=(10, 6))
    if log_scale:
        ax.set_yscale("log")
    sns.scatterplot(
        data=events,
        x="timestamp",
        y="volume",
        hue="direction",
        palette=col_pal,
        size=0.5,
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Volume")
    ax.set_title("Volume Map of Flashed Limit Orders")
    fig.tight_layout()
    return fig


def plot_current_depth(
    order_book: dict,
    volume_scale: float = 1,
    show_quantiles: bool = True,
    show_volume: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """
    Plot the current order book depth.

    Parameters
    ----------
    order_book : dict
        Dictionary containing 'bids', 'asks', and 'timestamp'.
    volume_scale : float, optional
        Scaling factor for volume. Default is 1.
    show_quantiles : bool, optional
        Whether to highlight highest 1% volume with vertical lines. Default is True.
    show_volume : bool, optional
        Whether to show volume bars. Default is True.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bids = reverse_matrix(order_book["bids"])
    asks = reverse_matrix(order_book["asks"])
    assert isinstance(bids, pd.DataFrame)
    assert isinstance(asks, pd.DataFrame)

    x = np.concatenate(
        [
            bids["price"].values,
            [bids["price"].values[-1]],
            [asks["price"].values[0]],
            asks["price"].values,
        ]
    )
    y1 = (
        np.concatenate([bids["liquidity"].values, [0], [0], asks["liquidity"].values])
        * volume_scale
    )
    y2 = (
        np.concatenate([bids["volume"].values, [0], [0], asks["volume"].values])
        * volume_scale
    )
    side = ["bid"] * (len(bids) + 1) + ["ask"] * (len(asks) + 1)

    depth = pd.DataFrame({"price": x, "liquidity": y1, "volume": y2, "side": side})

    fig, ax = _create_axes(ax, figsize=(12, 7))

    if show_volume:
        unique_prices = np.sort(np.unique(depth["price"]))
        price_diffs = np.diff(unique_prices)
        price_diffs = price_diffs[price_diffs > 0]
        if len(price_diffs) > 0:
            resolution = np.min(price_diffs)
            bar_width = resolution
        else:
            bar_width = 1

        ax.bar(
            depth["price"],
            depth["volume"],
            width=bar_width,
            color="white",
            align="center",
            edgecolor=None,
        )

    col_pal = {"ask": "#ff0000", "bid": "#0000ff"}
    for side_value in ["bid", "ask"]:
        side_data = depth[depth["side"] == side_value]
        ax.step(
            side_data["price"],
            side_data["liquidity"],
            where="pre",
            color=col_pal[side_value],
            label=side_value,
            linewidth=2,
        )

    if show_quantiles:
        bid_quantile = bids["volume"].quantile(0.99)
        bid_quantiles = bids.loc[bids["volume"] >= bid_quantile, "price"]
        ask_quantile = asks["volume"].quantile(0.99)
        ask_quantiles = asks.loc[asks["volume"] >= ask_quantile, "price"]
        for x_value in bid_quantiles:
            ax.axvline(x=x_value, color="#222222", linestyle="--")
        for x_value in ask_quantiles:
            ax.axvline(x=x_value, color="#222222", linestyle="--")

    xmin = round(bids["price"].min())
    xmax = round(asks["price"].max())
    xticks = np.arange(xmin, xmax + 1, 1)
    ax.set_xticks(xticks)

    timestamp = pd.to_datetime(order_book["timestamp"], unit="s", utc=True)
    ax.set_title(timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
    ax.set_xlabel("Price")
    ax.set_ylabel("Liquidity")

    ax.legend()
    fig.tight_layout()
    return fig


def plot_volume_percentiles(
    depth_summary: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    volume_scale: float = 1,
    perc_line: bool = True,
    side_line: bool = True,
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    if start_time is None:
        start_time = depth_summary["timestamp"].iloc[0]
    if end_time is None:
        end_time = depth_summary["timestamp"].iloc[-1]

    bid_names = [f"bid_vol{i}bps" for i in range(25, 501, 25)]
    ask_names = [f"ask_vol{i}bps" for i in range(25, 501, 25)]

    td = (end_time - start_time).total_seconds()
    td = round(td)

    frequency = "mins" if td > 900 else "secs"

    delta = timedelta(seconds=60 if frequency == "mins" else 1)
    mask = (depth_summary["timestamp"] >= (start_time - delta)) & (
        depth_summary["timestamp"] <= end_time
    )
    ob_percentiles = depth_summary.loc[mask, ["timestamp"] + bid_names + ask_names]

    ob_percentiles = ob_percentiles.drop_duplicates(subset="timestamp", keep="last")

    ob_percentiles.set_index("timestamp", inplace=True)

    if frequency == "mins":
        intervals = pd.DatetimeIndex(ob_percentiles.index).floor("T")
    else:
        intervals = pd.DatetimeIndex(ob_percentiles.index).floor("S")

    aggregated = ob_percentiles.groupby(intervals).mean()

    aggregated.index = aggregated.index + delta
    aggregated.reset_index(inplace=True)
    aggregated.rename(columns={"index": "timestamp"}, inplace=True)
    ob_percentiles = aggregated

    bid_names = [f"bid_vol{int(i):03d}bps" for i in range(25, 501, 25)]
    ask_names = [f"ask_vol{int(i):03d}bps" for i in range(25, 501, 25)]
    ob_percentiles.columns = pd.Index(["timestamp"] + bid_names + ask_names)

    max_ask = ob_percentiles[ask_names].sum(axis=1).max()
    max_bid = ob_percentiles[bid_names].sum(axis=1).max()

    melted_asks = ob_percentiles.melt(
        id_vars="timestamp",
        value_vars=ask_names,
        var_name="percentile",
        value_name="liquidity",
    )
    melted_asks["percentile"] = pd.Categorical(
        melted_asks["percentile"], categories=ask_names[::-1], ordered=True
    )
    melted_asks["liquidity"] *= volume_scale

    melted_bids = ob_percentiles.melt(
        id_vars="timestamp",
        value_vars=bid_names,
        var_name="percentile",
        value_name="liquidity",
    )
    melted_bids["percentile"] = pd.Categorical(
        melted_bids["percentile"], categories=bid_names[::-1], ordered=True
    )
    melted_bids["liquidity"] *= volume_scale

    colors_list = [
        "#f92b20",
        "#fe701b",
        "#facd1f",
        "#d6fd1c",
        "#65fe1b",
        "#1bfe42",
        "#1cfdb4",
        "#1fb9fa",
        "#1e71fb",
        "#261cfd",
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_list, N=20)
    col_pal = [cmap(i / 19) for i in range(20)]
    col_pal *= 2

    legend_names = [f"+{int(i):03d}bps" for i in range(500, 49, -50)] + [
        f"-{int(i):03d}bps" for i in range(50, 501, 50)
    ]

    pl = 0.1 if perc_line else 0

    asks_pivot = melted_asks.pivot(
        index="timestamp", columns="percentile", values="liquidity"
    )
    bids_pivot = melted_bids.pivot(
        index="timestamp", columns="percentile", values="liquidity"
    )

    asks_pivot = asks_pivot[ask_names[::-1]]
    bids_pivot = bids_pivot[bid_names[::-1]]

    asks_cumsum = asks_pivot.cumsum(axis=1)
    bids_cumsum = bids_pivot.cumsum(axis=1)

    bids_cumsum_neg = -bids_cumsum

    asks_cols = asks_cumsum.columns.tolist()
    bids_cols = bids_cumsum.columns.tolist()
    all_cols = asks_cols + bids_cols
    colors_dict = dict(zip(all_cols, col_pal))

    fig, ax = _create_axes(ax, figsize=(12, 8))

    prev = np.zeros(len(asks_cumsum))
    x = asks_cumsum.index
    for percentile in asks_cols:
        current = asks_cumsum[percentile].values
        ax.fill_between(
            x,
            prev,
            current,
            facecolor=colors_dict[percentile],
            edgecolor="black" if perc_line else None,
            linewidth=pl,
        )
        prev = current

    prev = np.zeros(len(bids_cumsum_neg))
    x = bids_cumsum_neg.index
    for percentile in bids_cols:
        current = bids_cumsum_neg[percentile].values
        ax.fill_between(
            x,
            prev,
            current,
            facecolor=colors_dict[percentile],
            edgecolor="black" if perc_line else None,
            linewidth=pl,
        )
        prev = current

    if side_line:
        ax.axhline(y=0, color="#000000", linewidth=0.1)

    y_range = volume_scale * max(max_ask, max_bid)
    ax.set_ylim(-y_range, y_range)

    ax.set_xlabel("time")

    fig.autofmt_xdate()

    legend_elements = []
    for col, label in zip(all_cols, legend_names):
        patch = Patch(
            facecolor=colors_dict[col],
            edgecolor="black" if perc_line else None,
            label=label,
        )
        legend_elements.append(patch)

    ax.legend(
        handles=legend_elements,
        title="depth         \n",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    return fig


def plot_events_histogram(
    events: pd.DataFrame,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    val: str = "volume",
    bw: float | None = None,
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    if val not in ("volume", "price"):
        raise ValueError(f"val must be 'volume' or 'price', got {val!r}")

    if start_time is None:
        start_time = events["timestamp"].min()
    if end_time is None:
        end_time = events["timestamp"].max()

    events_filtered = events[
        (events["timestamp"] >= start_time) & (events["timestamp"] <= end_time)
    ]

    fig, ax = _create_axes(ax, figsize=(12, 7))

    sns.histplot(
        data=events_filtered,
        x=val,
        hue="direction",
        multiple="dodge",
        binwidth=bw,
        palette={"bid": "#0000ff", "ask": "#ff0000"},
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_title(f"Events {val} distribution")
    ax.set_xlabel(val.capitalize())
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Flow toxicity plots
# ---------------------------------------------------------------------------


def plot_vpin(
    vpin_df: pd.DataFrame,
    threshold: float = 0.7,
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _create_axes(ax, figsize=(12, 5))

    ax.bar(
        vpin_df["timestamp_end"],
        vpin_df["vpin"],
        width=0.6 * (
            (vpin_df["timestamp_end"].iloc[-1] - vpin_df["timestamp_end"].iloc[0])
            / max(len(vpin_df) - 1, 1)
        ) if len(vpin_df) > 1 else 0.001,
        color="#5dade2",
        alpha=0.35,
        label="Per-bucket VPIN",
    )

    if "vpin_avg" in vpin_df.columns:
        ax.plot(
            vpin_df["timestamp_end"],
            vpin_df["vpin_avg"],
            color="#e74c3c",
            linewidth=2,
            label="VPIN (rolling avg)",
        )

    ax.axhline(
        y=threshold,
        color="#f39c12",
        linewidth=1.5,
        linestyle="--",
        label=f"Threshold ({threshold})",
    )

    # Shade regions above threshold
    if "vpin_avg" in vpin_df.columns:
        above = vpin_df["vpin_avg"] >= threshold
        ax.fill_between(
            vpin_df["timestamp_end"],
            threshold,
            vpin_df["vpin_avg"],
            where=above,
            color="#e74c3c",
            alpha=0.15,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("VPIN")
    ax.set_title("Volume-Synchronized Probability of Informed Trading")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_order_flow_imbalance(
    ofi_df: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    ax: Axes | None = None,
) -> Figure:
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

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _create_axes(ax, figsize=(12, 5))

    colors = [
        "#27ae60" if v >= 0 else "#e74c3c" for v in ofi_df["ofi"]
    ]
    ax.bar(ofi_df["timestamp"], ofi_df["ofi"], color=colors, alpha=0.7)

    ax.axhline(y=0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("OFI")
    ax.set_title("Order Flow Imbalance")

    if trades is not None and "price" in trades.columns:
        ax2 = ax.twinx()
        ax2.plot(
            trades["timestamp"],
            trades["price"],
            color="#f1c40f",
            linewidth=1.2,
            alpha=0.8,
            label="Price",
        )
        ax2.set_ylabel("Price", color="#f1c40f")
        ax2.tick_params(axis="y", labelcolor="#f1c40f")

    fig.tight_layout()
    return fig


def plot_kyle_lambda(
    kyle_result: object,
    ax: Axes | None = None,
) -> Figure:
    """Plot Kyle's Lambda regression: signed volume vs ΔPrice.

    Parameters
    ----------
    kyle_result : KyleLambdaResult
        Output of :func:`~ob_analytics.flow_toxicity.compute_kyle_lambda`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    reg_df = kyle_result.regression_df  # type: ignore[union-attr]
    lambda_ = kyle_result.lambda_  # type: ignore[union-attr]
    r_squared = kyle_result.r_squared  # type: ignore[union-attr]
    t_stat = kyle_result.t_stat  # type: ignore[union-attr]

    fig, ax = _create_axes(ax, figsize=(8, 6))

    ax.scatter(
        reg_df["signed_volume"],
        reg_df["delta_price"],
        color="#5dade2",
        alpha=0.6,
        edgecolors="white",
        linewidths=0.5,
        s=50,
        zorder=3,
    )

    # Regression line
    if not np.isnan(lambda_):
        x_range = np.linspace(
            reg_df["signed_volume"].min(),
            reg_df["signed_volume"].max(),
            100,
        )
        intercept = reg_df["delta_price"].mean() - lambda_ * reg_df["signed_volume"].mean()
        ax.plot(
            x_range,
            intercept + lambda_ * x_range,
            color="#e74c3c",
            linewidth=2,
            label=f"λ = {lambda_:.6f}",
            zorder=4,
        )

    ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="white", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Signed Order Flow (net volume)")
    ax.set_ylabel("ΔPrice")
    title = "Kyle's Lambda — Price Impact Regression"
    if not np.isnan(r_squared):
        title += f"\nR² = {r_squared:.3f}, t = {t_stat:.2f}"
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

