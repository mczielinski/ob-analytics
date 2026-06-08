"""Matplotlib / seaborn rendering backend for ob-analytics.

This module contains all matplotlib-specific rendering logic.  Each
``mpl_*()`` function takes a prepared data dict (from
:mod:`~ob_analytics.visualization._data`) and an optional *ax* parameter, and
returns a :class:`~matplotlib.figure.Figure`.

The :class:`PlotTheme` value object and :data:`DEFAULT_THEME` also live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from loguru import logger


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


#: Default theme applied when a renderer creates its own figure.  Pass a
#: ``theme=`` kwarg to :func:`~ob_analytics.visualization.plot` (or directly to
#: a renderer) to override it per call; there is no global mutable theme.
DEFAULT_THEME: PlotTheme = PlotTheme()


def _apply_theme(theme: PlotTheme = DEFAULT_THEME) -> None:
    """Apply *theme* to matplotlib / seaborn."""
    sns.set_theme(
        style=cast(Any, theme.style),
        context=cast(Any, theme.context),
        font_scale=theme.font_scale,
        rc=dict(theme.rc),
    )


def _create_axes(
    ax: Axes | None,
    figsize: tuple[float, float] = (10, 6),
    theme: PlotTheme = DEFAULT_THEME,
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, creating a new figure only when *ax* is ``None``."""
    if ax is not None:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)
        return fig, ax
    _apply_theme(theme)
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
# Rendering functions
# ---------------------------------------------------------------------------


def mpl_time_series(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render a time-series step plot."""
    df = data["df"]
    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)
    sns.lineplot(data=df, x="ts", y="val", drawstyle="steps-post", ax=ax)
    ax.set_title(data["title"])
    ax.set_xlabel("time")
    ax.set_ylabel(data["y_label"])
    ax.grid(True)
    fig.tight_layout()
    return fig


def mpl_trades(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render a trade-price step plot."""
    filtered = data["filtered_trades"]
    y_breaks = data["y_breaks"]
    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)
    sns.lineplot(
        data=filtered,
        x="timestamp",
        y="price",
        drawstyle="steps-post",
        ax=ax,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_yticks(y_breaks)
    ax.grid(True)
    fig.tight_layout()
    return fig


def mpl_price_levels(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render the price-level depth heatmap."""
    depth = data["depth"]
    spread = data["spread"]
    trades = data["trades"]
    show_mp = data["show_mp"]
    col_bias = data["col_bias"]
    price_by = data["price_by"]

    depth = depth.copy()
    depth.sort_values(by="timestamp", inplace=True, kind="stable")
    if depth.empty or depth.groupby("price").size().min() < 2:
        logger.warning("Not enough data for any price level")
        fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)
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
        norm: mcolors.Normalize = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)

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
        lc = collections.LineCollection(
            segments.tolist(), colors=seg_colors, linewidths=2
        )
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

    y_range = data.get("y_range")
    if y_range is not None:
        ax.set_ylim(y_range)
    else:
        ymin = depth["price"].min()
        ymax = depth["price"].max()
        ax.set_ylim((ymin, ymax))

    if price_by is not None and y_range is not None:
        ymin, ymax = y_range
        y_ticks = np.arange(round(ymin), round(ymax) + price_by, price_by)
        ax.set_yticks(y_ticks)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    fig.tight_layout()
    return fig


def mpl_event_map(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render a limit-order event map."""
    events = data["events"]
    created = data["created"]
    deleted = data["deleted"]
    price_by = data["price_by"]

    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)

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


def mpl_volume_map(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render a volume map of flashed limit orders."""
    events = data["events"]
    log_scale = data["log_scale"]
    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)
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


# Order-book snapshot palette, shared by the book_snapshot + depth_chart faces
# (kept identical to the plotly backend for cross-backend parity).
_BID_COLOR = "#4477dd"
_ASK_COLOR = "#dd4444"

# Order-lifecycle fate palette for the order_activity L3 Gantt (cancelled vs
# filled/resting); identical to the plotly backend for cross-backend parity.
_FLASHED_COLOR = "#e09f3e"  # flashed-limit: placed and pulled (cancelled)
_RESTING_COLOR = "#2a9d8f"  # resting-limit: rested / filled

# Trade-tape aggressor palette (taker side) for the L3 tape maker-bars;
# identical to the plotly backend for cross-backend parity.
_BUY_COLOR = "#2e9e5b"  # buyer-initiated execution (lifts the ask)
_SELL_COLOR = "#dd4444"  # seller-initiated execution (hits the bid)

# Competing-risks outcome palette for the order_outcome L3 scatter; identical to
# the plotly backend for cross-backend parity.
_FILLED_COLOR = "#2a9d8f"  # fully executed
_PARTIAL_COLOR = "#8c8cd8"  # partially executed, remainder removed
_CANCELLED_COLOR = "#e09f3e"  # removed without any execution


def _book_bar_width(*sides: pd.DataFrame) -> float:
    """Smallest positive gap between distinct prices across *sides*."""
    arrays = [s["price"].to_numpy() for s in sides if not s.empty]
    if not arrays:
        return 1.0
    uniq = np.unique(np.concatenate(arrays))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    return float(np.min(diffs)) if diffs.size else 1.0


def _mpl_book_bars(
    data: dict, ax: Axes | None, theme: PlotTheme, *, per_order: bool
) -> Figure:
    """Book-snapshot bars: one bar per level (L2) or stacked per order (L3)."""
    bids = data["bids"]
    asks = data["asks"]
    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)

    width = _book_bar_width(bids, asks) * 0.9
    # Thin white separators delineate per-order segments without swamping the
    # bars (dark edges previously blacked out a dense L3 book).
    edge = "white" if per_order else "none"
    for side, color, label in ((bids, _BID_COLOR, "bid"), (asks, _ASK_COLOR, "ask")):
        if side.empty:
            continue
        ax.bar(
            side["price"],
            side["seg_hi"] - side["seg_lo"],
            bottom=side["seg_lo"],
            width=width,
            color=color,
            edgecolor=edge,
            linewidth=0.6,
            align="center",
            label=label,
        )

    if data["show_quantiles"]:
        for x_value in data["bid_quantiles"]:
            ax.axvline(x=x_value, color="#222222", linestyle="--", linewidth=1)
        for x_value in data["ask_quantiles"]:
            ax.axvline(x=x_value, color="#222222", linestyle="--", linewidth=1)

    ax.set_title(data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"))
    ax.set_xlabel("Price")
    ax.set_ylabel("Size")
    ax.legend(loc="upper center")
    fig.tight_layout()
    return fig


def mpl_book_snapshot_aggregate(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L2 (MBP) book snapshot: aggregate size per price level."""
    return _mpl_book_bars(data, ax, theme, per_order=False)


def mpl_book_snapshot_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) book snapshot: each order a stacked segment within its level."""
    return _mpl_book_bars(data, ax, theme, per_order=True)


def _mpl_depth_curve(
    data: dict, ax: Axes | None, theme: PlotTheme, *, per_order: bool
) -> Figure:
    """Cumulative-depth curve: stepped per level (L2) or per order (L3)."""
    # DEFERRED (lower priority). The L3 depth curve is visually identical to L2.
    # FUTURE(--density): segment each riser by per-order composition (whale vs
    # crowd) so the per-order resolution becomes legible. Left as-is for now.
    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)
    for side, color, label in (
        (data["bids"], _BID_COLOR, "bid"),
        (data["asks"], _ASK_COLOR, "ask"),
    ):
        if side.empty:
            continue
        s = side.sort_values("price")
        ax.step(
            s["price"],
            s["liquidity"],
            where="pre",
            color=color,
            linewidth=2,
            label=label,
        )
        ax.fill_between(s["price"], s["liquidity"], step="pre", color=color, alpha=0.15)
        if per_order:
            ax.scatter(s["price"], s["liquidity"], s=18, color=color, zorder=3)

    ax.set_title(data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"))
    ax.set_xlabel("Price")
    ax.set_ylabel("Cumulative liquidity")
    ax.legend(loc="upper center")
    fig.tight_layout()
    return fig


def mpl_depth_chart_aggregate(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L2 (MBP) depth chart: cumulative liquidity stepped per price level."""
    return _mpl_depth_curve(data, ax, theme, per_order=False)


def mpl_depth_chart_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) depth chart: cumulative liquidity stepped per individual order."""
    return _mpl_depth_curve(data, ax, theme, per_order=True)


def mpl_cancellations_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) cancellations: each cancelled order as an age x distance point."""
    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)
    # FUTURE(--density): high-cardinality 2D cloud -> log-log hexbin (count as
    # saturation). For now rasterize the marker layer so the vector file does
    # not balloon (~140k points was ~839 KB) and fade overlapping points.
    for side, color, label in (
        (data["bids"], _BID_COLOR, "bid"),
        (data["asks"], _ASK_COLOR, "ask"),
    ):
        if side.empty:
            continue
        ax.scatter(
            side["age_s"],
            side["distance_bps"],
            s=side["marker_area"],
            color=color,
            alpha=0.25,
            edgecolors="none",
            rasterized=True,
            label=label,
        )
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=1)
    ax.set_title("Cancelled orders by age and distance from touch")
    ax.set_xlabel("Order age (s)")
    ax.set_ylabel("Placement distance from touch (bps)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def mpl_order_activity_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) order activity: each order one lifecycle bar, coloured by fate."""
    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)
    flashed = data["flashed"]
    resting = data["resting"]
    for side, color, label in (
        (flashed, _FLASHED_COLOR, "flashed-limit (cancelled)"),
        (resting, _RESTING_COLOR, "resting-limit (filled)"),
    ):
        if side.empty:
            continue
        ax.hlines(
            side["price"],
            mdates.date2num(side["start_ts"]),
            mdates.date2num(side["end_ts"]),
            colors=color,
            linewidth=1.2,
            alpha=0.5,
            label=label,
        )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    y_range = data.get("y_range")
    price_by = data["price_by"]
    if y_range is not None:
        ax.set_ylim(y_range)
        ax.set_yticks(
            np.arange(
                round(y_range[0] / price_by) * price_by,
                round(y_range[1] / price_by) * price_by,
                price_by,
            )
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_title("Order lifecycles (place → outcome)")
    if not (flashed.empty and resting.empty):
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def mpl_liquidity_at_touch(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L2 (MBP) liquidity at the touch: best bid/ask resting size over time."""
    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)
    ts = data["timestamp"]
    ax.plot(
        ts,
        data["bid_vol"],
        color=_BID_COLOR,
        linewidth=1.2,
        drawstyle="steps-post",
        label="Best bid size",
    )
    ax.plot(
        ts,
        data["ask_vol"],
        color=_ASK_COLOR,
        linewidth=1.2,
        drawstyle="steps-post",
        label="Best ask size",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Size at touch")
    ax.set_title("Liquidity at the touch")
    if len(ts) > 0:
        ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    return fig


def mpl_order_outcome_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) order outcome: each order as placement distance x size, by fate."""
    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)
    any_pts = False
    # Draw the dominant 'cancelled' class first (underneath) so the rarer
    # filled/partial outcomes land on top instead of being buried, and fade it.
    # FUTURE(--density): at high cancelled cardinality, degrade this raw scatter
    # to a distance-binned stacked composition + a common-baseline fill-rate dot
    # plot (parts-of-whole AND a position-encoded rate).
    for frame, color, label, pt_alpha in (
        (data["cancelled"], _CANCELLED_COLOR, "cancelled", 0.18),
        (data["partial"], _PARTIAL_COLOR, "partial", 0.6),
        (data["filled"], _FILLED_COLOR, "filled", 0.85),
    ):
        if frame.empty:
            continue
        any_pts = True
        ax.scatter(
            frame["distance_bps"],
            frame["placed"],
            s=frame["marker_area"],
            color=color,
            alpha=pt_alpha,
            edgecolors="none",
            label=label,
        )
    ax.axvline(x=0, color="#888888", linestyle="--", linewidth=1)
    ax.set_title("Order outcome by placement distance and size")
    ax.set_xlabel("Placement distance from touch (bps)")
    ax.set_ylabel("Order size")
    if any_pts:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def mpl_trade_tape_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) trade tape: executions plus each maker order's resting bar."""
    # DEFERRED. trade_tape.L3 currently sizes execution markers by bubble AREA
    # (rank-5) and draws full maker-rest hlines. FUTURE(--color-by) + encoding
    # rethink: encode size by length (lollipop), keep side as hue, and reserve
    # the long maker spans for an explicit "time resting" read. Left as-is for
    # the simple pass.
    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)
    any_pts = False
    for side, color, label in (
        (data["buys"], _BUY_COLOR, "buy (lifts ask)"),
        (data["sells"], _SELL_COLOR, "sell (hits bid)"),
    ):
        if side.empty:
            continue
        any_pts = True
        ax.hlines(
            side["price"],
            mdates.date2num(side["created_ts"]),
            mdates.date2num(side["timestamp"]),
            colors=color,
            linewidth=1.0,
            alpha=0.35,
        )
        ax.scatter(
            mdates.date2num(side["timestamp"]),
            side["price"],
            s=side["marker_area"],
            color=color,
            alpha=0.7,
            edgecolors="none",
            label=label,
        )
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    y_range = data.get("y_range")
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.set_xlabel("Time")
    ax.set_ylabel("Execution Price")
    ax.set_title("Trade tape with maker order lifecycles")
    if any_pts:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def mpl_volume_percentiles(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render volume-percentile stacked area chart."""
    asks_cumsum = data["asks_cumsum"]
    bids_cumsum_neg = data["bids_cumsum_neg"]
    asks_cols = data["asks_cols"]
    bids_cols = data["bids_cols"]
    all_cols = data["all_cols"]
    colors_dict = data["colors_dict"]
    legend_names = data["legend_names"]
    max_ask = data["max_ask"]
    max_bid = data["max_bid"]
    volume_scale = data["volume_scale"]
    perc_line = data["perc_line"]
    side_line = data["side_line"]

    pl = 0.1 if perc_line else 0

    fig, ax = _create_axes(ax, figsize=(12, 8), theme=theme)

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


def mpl_events_histogram(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render an events price/volume histogram."""
    events = data["events"]
    val = data["val"]
    bw = data["bw"]

    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)
    sns.histplot(
        data=events,
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


def mpl_vpin(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render VPIN time series."""
    vpin_df = data["vpin_df"]
    threshold = data["threshold"]
    bar_width = data["bar_width"]

    fig, ax = _create_axes(ax, figsize=(12, 5), theme=theme)

    ax.bar(
        vpin_df["timestamp_end"],
        vpin_df["vpin"],
        width=bar_width,
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


def mpl_order_flow_imbalance(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render order flow imbalance bar chart."""
    ofi_df = data["ofi_df"]
    trades = data["trades"]
    colors = data["colors"]

    # Bar width in matplotlib date-number units (days): 80% of the median
    # inter-bar gap. Computed here because it is backend-specific to the
    # date-number x-axis.
    if len(ofi_df) > 1:
        median_gap = ofi_df["timestamp"].diff().median()
        bar_width = mdates.date2num(
            ofi_df["timestamp"].iloc[0] + median_gap * 0.8
        ) - mdates.date2num(ofi_df["timestamp"].iloc[0])
    else:
        bar_width = 0.001

    fig, ax = _create_axes(ax, figsize=(12, 5), theme=theme)

    ax.bar(
        ofi_df["timestamp"],
        ofi_df["ofi"],
        width=bar_width,
        color=colors,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
    )

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


def mpl_kyle_lambda(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render Kyle's Lambda regression scatter."""
    reg_df = data["reg_df"]
    lambda_ = data["lambda_"]
    r_squared = data["r_squared"]
    t_stat = data["t_stat"]

    fig, ax = _create_axes(ax, figsize=(8, 6), theme=theme)

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

    if not np.isnan(lambda_):
        x_range = np.linspace(
            reg_df["signed_volume"].min(),
            reg_df["signed_volume"].max(),
            100,
        )
        intercept = (
            reg_df["delta_price"].mean() - lambda_ * reg_df["signed_volume"].mean()
        )
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


# ---------------------------------------------------------------------------
# LOBSTER-enriched plots
# ---------------------------------------------------------------------------


def mpl_hidden_executions(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render hidden execution volume overlaid on the trade price."""
    trades = data["trades"]
    hidden = data["hidden"]
    has_hidden = data["has_hidden"]

    fig, ax = _create_axes(ax, figsize=(12, 6), theme=theme)

    if not trades.empty:
        ax.step(
            trades["timestamp"],
            trades["price"],
            where="post",
            color="#5dade2",
            linewidth=1,
            alpha=0.7,
            label="Trade price",
        )

    if has_hidden and not hidden.empty:
        vol = hidden["volume"]
        vol_max = float(vol.max()) if vol.max() > 0 else 1.0
        ax.scatter(
            hidden["timestamp"],
            hidden["price"],
            s=data["marker_area"],
            c=vol,
            cmap="Reds",
            vmin=0,
            vmax=vol_max,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.3,
            label="Hidden executions",
            zorder=3,
        )
        ax.set_title("Hidden Order Executions")
    else:
        ax.set_title("Hidden Order Executions (no hidden execution data)")
        ax.text(
            0.5,
            0.5,
            "No hidden execution events\n(raw_event_type == 5)\nin this dataset",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#888",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    y_range = data.get("y_range")
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def mpl_trading_halts(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render trade price with shaded halt periods."""
    trades = data["trades"]
    halt_periods = data["halt_periods"]
    has_halts = data["has_halts"]

    fig, ax = _create_axes(ax, figsize=(12, 6), theme=theme)

    if not trades.empty:
        ax.step(
            trades["timestamp"],
            trades["price"],
            where="post",
            color="#5dade2",
            linewidth=1,
            alpha=0.8,
            label="Trade price",
        )

    if has_halts and halt_periods:
        for i, (h_start, h_end) in enumerate(halt_periods):
            ax.axvspan(
                h_start,
                h_end,
                color="#e74c3c",
                alpha=0.2,
                label="Trading halt" if i == 0 else None,
            )
        ax.set_title("Trading Halts")
    else:
        ax.set_title("Trading Halts (no halt data)")
        ax.text(
            0.5,
            0.5,
            "No trading halt events\n(raw_event_type == 7)\nin this dataset",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#888",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Renderer self-registration
# ---------------------------------------------------------------------------
# Imported here (not at module top) so RENDERERS -- defined in the package
# __init__ -- already exists when this module is imported during package init.
from ob_analytics.visualization import RENDERERS, Level  # noqa: E402

# (concept, level, renderer).  L2 = aggregate per price level; analytics carry
# no level and register at ``None``.  The level is a registry *coordinate*, not
# a name suffix -- L3 faces register the same concept at ``Level.L3``.
_L2 = Level.L2
_L3 = Level.L3
for _concept, _level, _fn in [
    ("time_series", _L2, mpl_time_series),
    ("trade_tape", _L2, mpl_trades),
    ("trade_tape", _L3, mpl_trade_tape_per_order),
    ("depth_heatmap", _L2, mpl_price_levels),
    ("order_activity", _L2, mpl_event_map),
    ("order_activity", _L3, mpl_order_activity_per_order),
    ("order_outcome", _L3, mpl_order_outcome_per_order),
    ("liquidity_at_touch", _L2, mpl_liquidity_at_touch),
    ("cancellations", _L2, mpl_volume_map),
    ("cancellations", _L3, mpl_cancellations_per_order),
    ("book_snapshot", _L2, mpl_book_snapshot_aggregate),
    ("book_snapshot", _L3, mpl_book_snapshot_per_order),
    ("depth_chart", _L2, mpl_depth_chart_aggregate),
    ("depth_chart", _L3, mpl_depth_chart_per_order),
    ("volume_percentiles", _L2, mpl_volume_percentiles),
    ("events_histogram", _L2, mpl_events_histogram),
    ("hidden_executions", _L2, mpl_hidden_executions),
    ("vpin", None, mpl_vpin),
    ("order_flow_imbalance", None, mpl_order_flow_imbalance),
    ("kyle_lambda", None, mpl_kyle_lambda),
    ("trading_halts", None, mpl_trading_halts),
]:
    RENDERERS.register((_concept, _level, "matplotlib"), _fn)
