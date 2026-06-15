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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from loguru import logger

from ob_analytics.visualization._data import book_mid
from ob_analytics.visualization._palette import (
    _ASK_COLOR,
    _BID_COLOR,
    _BUY_COLOR,
    _CANCELLED_COLOR,
    _FILLED_COLOR,
    _PARTIAL_COLOR,
    _SELL_COLOR,
)


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

    style: str = "white"
    context: str = "notebook"
    font_scale: float = 1.05
    rc: dict[str, object] = field(
        # The reference style (Cleveland–McGill bundle): white background,
        # dotted light grid, no top/right spines, bold left-aligned titles.
        # Built on top of seaborn (style + context still apply).
        default_factory=lambda: {
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "lines.linewidth": 2.0,
        }
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


def format_time_axis(ax: Axes) -> None:
    """Consistent intraday ticks for a date-valued x-axis.

    Sets an explicit ``AutoDateLocator`` + ``ConciseDateFormatter`` pair.
    Renderers that plot ``date2num`` floats stay off matplotlib's date
    *units converter*, whose registration forces the slow collection-draw
    path (one ``Path`` rebuilt per segment on every draw — the WS-2.4
    profile); for unit-converted axes only the tick style changes (the
    default formatter rendered "02 02:45"-style day prefixes).
    """
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


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
        :meth:`~matplotlib.figure.Figure.savefig`.  Pass
        ``bbox_inches="tight"`` explicitly to crop to artist extents —
        it is not the default because it forces a second full draw
        (roughly doubling save time on dense figures), and every
        renderer already applies ``tight_layout``.
    """
    fig.savefig(path, dpi=dpi, **kwargs)  # type: ignore


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
    format_time_axis(ax)
    ax.grid(True)
    fig.tight_layout()
    return fig


def _draw_lollipops(
    ax: Axes,
    side: pd.DataFrame,
    color: str,
    label: str,
    *,
    stem_alpha: float = 0.55,
    marker_alpha: float = 0.9,
) -> None:
    """Stems from the mid line to each trade price, tipped by sized markers.

    The stem carries direction (buys reach up above the mid, sells down below)
    and the price excursion; the marker area carries the trade size.
    """
    x = mdates.date2num(side["timestamp"])
    ax.vlines(
        x,
        side["mid"].to_numpy(),
        side["price"].to_numpy(),
        colors=color,
        linewidth=0.9,
        alpha=stem_alpha,
    )
    ax.scatter(
        x,
        side["price"],
        s=side["marker_area"],
        color=color,
        alpha=marker_alpha,
        edgecolors="none",
        label=label,
        zorder=3,
    )


def mpl_trades(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render the L2 signed-lollipop trade tape.

    Each trade is a stem from the mid line to its execution price, tipped by a
    volume-sized marker and coloured by aggressor side.  The price axis spans
    the full data extent (no quantile clip), so spike prints stay visible.
    """
    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)

    mid_line = data.get("mid_line")
    if mid_line is not None and not mid_line.empty:
        ax.plot(
            mdates.date2num(mid_line["timestamp"]),
            mid_line["mid"].to_numpy(),
            color="#888888",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )

    any_pts = False
    for side, color, label in (
        (data["buys"], _BUY_COLOR, "buy (lifts ask)"),
        (data["sells"], _SELL_COLOR, "sell (hits bid)"),
    ):
        if side.empty:
            continue
        any_pts = True
        _draw_lollipops(ax, side, color, label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    y_range = data.get("y_range")
    if y_range is not None:
        lo, hi = y_range
        pad = (hi - lo) * 0.04 or 1.0
        ax.set_ylim(lo - pad, hi + pad)
    format_time_axis(ax)
    ax.grid(True)
    if any_pts:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _volume_norm(volume: pd.Series, col_bias: float) -> mcolors.Normalize:
    """Color normalization for the depth heatmap.

    ``col_bias`` is the gamma of a power-law norm: ``1.0`` is linear so
    high-volume walls stand out against a dark field, ``0 < col_bias < 1``
    brightens low-volume levels to reveal near-touch structure (``0.1``
    matches the R package's palette bias), and ``col_bias <= 0`` selects a
    log10 scale.
    """
    vmin = volume.min()
    vmax = volume.max()
    if col_bias <= 0:
        if vmin <= 0:
            vmin = volume[volume > 0].min()
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if col_bias != 1:
        return mcolors.PowerNorm(gamma=col_bias, vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


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

    cmap = plt.get_cmap("viridis")
    norm = _volume_norm(depth["volume"], col_bias)

    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)

    depth["timestamp_numeric"] = mdates.date2num(depth["timestamp"])

    # All within-price segments go into one LineCollection: sort globally by
    # (price, timestamp) — identical ordering to the per-price groupby this
    # replaces — and mask out the segments that would bridge two prices.
    # One collection draws an order of magnitude faster than thousands.
    dd = depth.sort_values(["price", "timestamp"], kind="stable")
    tn = dd["timestamp_numeric"].to_numpy()
    pr = dd["price"].to_numpy()
    vv = dd["volume"].to_numpy()
    aa = dd["alpha"].to_numpy()
    same_price = pr[:-1] == pr[1:] if len(pr) > 1 else np.empty(0, dtype=bool)
    if same_price.any():
        starts = np.column_stack([tn[:-1], pr[:-1]])[same_price]
        ends = np.column_stack([tn[1:], pr[1:]])[same_price]
        segments = np.stack([starts, ends], axis=1)
        seg_colors = cmap(norm(vv[:-1][same_price]))
        seg_colors[:, -1] = aa[:-1][same_price]
        ax.add_collection(
            collections.LineCollection(
                segments,  # ty: ignore[invalid-argument-type] -- (N,2,2) ndarray is accepted at runtime
                colors=seg_colors,
                linewidths=2,
            )
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Volume")

    if spread is not None:
        spread = spread.copy()
        spread.sort_values(by="timestamp", inplace=True, kind="stable")
        spread_x = mdates.date2num(spread["timestamp"])
        if show_mp and "best_bid_price" in spread and "best_ask_price" in spread:
            spread["midprice"] = (
                spread["best_bid_price"] + spread["best_ask_price"]
            ) / 2
            ax.plot(
                spread_x,
                spread["midprice"],
                color="#222222",
                linewidth=1.1,
                label="Midprice",
            )
        else:
            if "best_ask_price" in spread:
                ax.step(
                    spread_x,
                    spread["best_ask_price"],
                    color=_ASK_COLOR,
                    linewidth=1.5,
                    where="post",
                    label="Best Ask",
                )
            if "best_bid_price" in spread:
                ax.step(
                    spread_x,
                    spread["best_bid_price"],
                    color=_BID_COLOR,
                    linewidth=1.5,
                    where="post",
                    label="Best Bid",
                )

    if trades is not None:
        buys = trades[trades["direction"] == "buy"]
        sells = trades[trades["direction"] == "sell"]

        if not sells.empty:
            ax.scatter(
                mdates.date2num(sells["timestamp"]),
                sells["price"],
                s=50,
                facecolors="none",
                edgecolors=_SELL_COLOR,
                linewidths=1.5,
                zorder=5,
                marker="v",
                label="Sell Trades",
            )
        if not buys.empty:
            ax.scatter(
                mdates.date2num(buys["timestamp"]),
                buys["price"],
                s=50,
                facecolors="none",
                edgecolors=_BUY_COLOR,
                linewidths=1.5,
                zorder=5,
                marker="^",
                label="Buy Trades",
            )

    # All artists above plot date2num floats; format_time_axis keeps the
    # axis off the date units converter (see its docstring).
    format_time_axis(ax)

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

    col_pal = {"bid": _BID_COLOR, "ask": _ASK_COLOR}

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
    # Direction-coloured dots: a fixed small marker (explicit ``s=``; the old
    # ``size=0.1`` mis-fed seaborn's size *semantic*, which adds a spurious
    # size legend and shrinks every dot to a speck).
    sns.scatterplot(
        data=events,
        x="timestamp",
        y="price",
        hue="direction",
        s=14,
        palette=col_pal,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_yticks(
        _rounded_price_ticks(events["price"].min(), events["price"].max(), price_by)
    )
    # The map overlays two unlabelled mark families; a legend disambiguates the
    # gray volume-sized create/delete circles from the bid/ask direction dots.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=_BID_COLOR,
            markeredgecolor="none",
            markersize=7,
            label="bid",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=_ASK_COLOR,
            markeredgecolor="none",
            markersize=7,
            label="ask",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#333333",
            markeredgecolor="none",
            markersize=8,
            label="created",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#333333",
            markeredgecolor="black",
            alpha=0.5,
            markersize=8,
            label="deleted",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    format_time_axis(ax)
    fig.tight_layout()
    return fig


def mpl_volume_map(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """Render a volume map of flashed limit orders."""
    events = data["events"]
    log_scale = data["log_scale"]
    col_pal = {"bid": _BID_COLOR, "ask": _ASK_COLOR}

    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)
    if log_scale:
        ax.set_yscale("log")
    sns.scatterplot(
        data=events,
        x="timestamp",
        y="volume",
        hue="direction",
        palette=col_pal,
        s=14,  # fixed marker size (was size=0.5, a misuse of the size semantic)
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Volume")
    ax.set_title("Volume Map of Flashed Limit Orders")
    format_time_axis(ax)
    fig.tight_layout()
    return fig


def _book_bar_thickness(*sides: pd.DataFrame) -> float:
    """Smallest positive gap between distinct prices across *sides*.

    Used as the ladder bar thickness (price units); windowing to the touch
    keeps this gap roughly the tick size, so bars stay tall and contiguous.
    """
    arrays = [s["price"].to_numpy() for s in sides if not s.empty]
    if not arrays:
        return 1.0
    uniq = np.unique(np.concatenate(arrays))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    return float(np.min(diffs)) if diffs.size else 1.0


def _rounded_price_ticks(
    lo: float, hi: float, price_by: float, *, max_ticks: int = 12
) -> np.ndarray:
    """Round-number price ticks spanning ``[lo, hi]``, thinned to <= ``max_ticks``.

    The base grid steps by ``price_by``; when that would place more than
    ``max_ticks`` ticks the step is widened to the next integer multiple of
    ``price_by`` so the labels stay legible (the dense L3 Gantt and the event
    map otherwise stacked one label per level into an unreadable smear).
    """
    if price_by <= 0:
        return np.array([lo])
    start = round(lo / price_by) * price_by
    stop = round(hi / price_by) * price_by
    n_steps = max(1, round((stop - start) / price_by))
    factor = max(1, int(np.ceil(n_steps / max_ticks)))
    return np.arange(start, stop, price_by * factor)


def _mpl_book_bars(
    data: dict, ax: Axes | None, theme: PlotTheme, *, per_order: bool
) -> Figure:
    """Horizontal book ladder: price on y, size on x, bids below / asks above.

    L2 draws one bar per price level; L3 segments each level into its individual
    orders (biggest-first from the axis) with white separators, so a whale and a
    crowd of small orders that look identical on L2 read differently here.
    """
    bids = data["bids"]
    asks = data["asks"]
    fig, ax = _create_axes(ax, figsize=(11, 8), theme=theme)

    thickness = _book_bar_thickness(bids, asks) * 0.9
    # Windowing to the touch keeps bars tall, so L3 separators are always on.
    edgecolor = "white" if per_order else "none"
    linewidth = 1.3 if per_order else 0.0
    for side, color, label in ((bids, _BID_COLOR, "bid"), (asks, _ASK_COLOR, "ask")):
        if side.empty:
            continue
        ax.barh(
            side["price"],
            side["seg_hi"] - side["seg_lo"],
            left=side["seg_lo"],
            height=thickness,
            color=color,
            align="center",
            label=label,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    mid = book_mid(bids, asks)
    if mid is not None:
        ax.axhline(mid, color="#444444", linestyle="--", linewidth=1, zorder=1)

    if data["show_quantiles"]:
        for y_value in (*data["bid_quantiles"], *data["ask_quantiles"]):
            ax.axhline(
                y=y_value,
                color="#888888",
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
                zorder=1,
            )

    ax.set_title(data["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"))
    ax.set_xlabel("Size (per order)" if per_order else "Size (aggregate per level)")
    ax.set_ylabel("Price")
    ax.set_xlim(left=0)
    ax.legend(loc="best")
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
    """Cumulative-depth curve: stepped per level (L2) or per order (L3).

    The L3 face currently differs from L2 only by per-order markers; making the
    per-order resolution legible is roadmap §3.x (docs/plans/, --density).
    """
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


def _annotate_cancel_populations(ax: Axes) -> None:
    """Faint hints for the three latent populations on a log-log cancel panel."""

    def label(x: float, y: float, text: str) -> None:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            color="#555555",
            fontstyle="italic",
            fontsize=9,
            zorder=5,
        )

    label(0.03, 0.06, "fleeting / fishing\n(<100ms, at touch)")
    label(0.40, 0.45, "patient / human\n(seconds, few bps)")
    label(0.62, 0.88, "deep resting\n(pulled later)")


def mpl_cancellations_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) cancellations: per-side log-log *density* of age x distance.

    The latent populations span orders of magnitude, so a linear scatter
    crushes everything against the origin.  Each side gets its own log-log
    hexbin (small multiples, bid | ask) where saturation encodes cancelled
    orders per bin -- density, not magnitude.
    """
    cmap = plt.get_cmap("Blues")

    if ax is None:
        _apply_theme(theme)
        # constrained layout: tight_layout cannot place a colorbar spanning two
        # axes (it warns and mis-sizes), so let the constrained engine do it.
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 6), sharex=True, sharey=True, layout="constrained"
        )
        panels: list[tuple[Axes, pd.DataFrame, str]] = [
            (axes[0], data["bids"], "bid"),
            (axes[1], data["asks"], "ask"),
        ]
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)
        # Composition into a caller's single ax: both sides as one density.
        combined = pd.concat([data["bids"], data["asks"]], ignore_index=True)
        panels = [(ax, combined, "orders")]

    last_hb = None
    for panel_ax, side, label in panels:
        if not side.empty:
            hb = panel_ax.hexbin(
                side["age_s"],
                side["distance_from_touch"],
                gridsize=25,
                xscale="log",
                yscale="log",
                cmap=cmap,
                # Log color: the instant-cancel spike is orders of magnitude
                # denser than the patient/deep populations, which a linear ramp
                # washes out entirely.
                norm=mcolors.LogNorm(),
                mincnt=1,
                edgecolors="white",
                linewidths=0.2,
            )
            hb.set_rasterized(True)
            last_hb = hb
            _annotate_cancel_populations(panel_ax)
        else:
            panel_ax.set_xscale("log")
            panel_ax.set_yscale("log")
        panel_ax.set_title(f"Cancelled {label}")
        panel_ax.set_xlabel("Age at cancel (s, log)")
        panel_ax.set_ylabel("Distance from touch (bps, log)")

    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=[p[0] for p in panels])
        cbar.set_label("Cancelled orders per bin")
    # Layout is managed by the constrained engine (ax is None) or the caller
    # (ax provided); tight_layout cannot handle the multi-axes colorbar.
    return fig


def mpl_order_activity_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) order activity: each order one lifeline, coloured by outcome.

    Width encodes order size; the terminal marker (x filled, o cancelled) is
    drawn when few enough spans survive.  Dense books are degraded upstream
    (see :func:`prepare_order_activity_l3_data`) and annotated "showing n of N".
    """
    fig, ax = _create_axes(ax, figsize=(11, 7), theme=theme)
    show_markers = data.get("show_markers", False)
    drew_any = False
    for side, color, label, marker in (
        (data["filled"], _FILLED_COLOR, "filled", "x"),
        (data["cancelled"], _CANCELLED_COLOR, "cancelled", "o"),
        (data["resting"], _PARTIAL_COLOR, "still resting", None),
    ):
        if side.empty:
            continue
        drew_any = True
        ax.hlines(
            side["price"],
            mdates.date2num(side["start_ts"]),
            mdates.date2num(side["end_ts"]),
            colors=color,
            linewidth=side["linewidth"].to_numpy(),
            alpha=0.6,
            label=label,
        )
        if show_markers and marker is not None:
            ends = mdates.date2num(side["end_ts"])
            if marker == "o":  # cancelled: open circle
                ax.scatter(
                    ends,
                    side["price"],
                    marker="o",
                    s=22,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.8,
                    zorder=4,
                )
            else:  # filled: cross
                ax.scatter(
                    ends, side["price"], marker=marker, s=22, color=color, zorder=4
                )

    format_time_axis(ax)

    y_range = data.get("y_range")
    price_by = data["price_by"]
    if y_range is not None:
        ax.set_ylim(y_range)
        ax.set_yticks(_rounded_price_ticks(y_range[0], y_range[1], price_by))

    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_title("Order lifecycles (place → outcome)")

    shown_of = data.get("shown_of")
    if shown_of is not None:
        ax.text(
            0.99,
            0.01,
            f"showing {shown_of[0]:,} of {shown_of[1]:,} orders",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
            fontstyle="italic",
        )
    if drew_any:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def mpl_queue_position_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) queue position: each touch order's FIFO rank over time, by fate.

    Each line is one order's march toward the front (rank 1, at the top of the
    inverted y-axis) as orders ahead leave; colour = terminal outcome, with a
    × (filled) / ○ (cancelled) at the order's last seen rank when sparse.
    """
    fig, ax = _create_axes(ax, figsize=(11, 7), theme=theme)
    show_markers = data.get("show_markers", False)
    drew_any = False
    for side, color, label, marker in (
        (data["filled"], _FILLED_COLOR, "filled", "x"),
        (data["cancelled"], _CANCELLED_COLOR, "cancelled", "o"),
        (data["resting"], _PARTIAL_COLOR, "still resting", None),
    ):
        if side.empty:
            continue
        drew_any = True
        for _, g in side.groupby("id", sort=False):
            g = g.sort_values("timestamp")
            ax.plot(
                mdates.date2num(g["timestamp"]),
                g["rank"].to_numpy(),
                color=color,
                alpha=0.5,
                linewidth=1.0,
                drawstyle="steps-post",
            )
        if show_markers and marker is not None:
            ends = side.sort_values("timestamp").groupby("id", sort=False).tail(1)
            x = mdates.date2num(ends["timestamp"])
            if marker == "o":  # cancelled: open circle
                ax.scatter(
                    x,
                    ends["rank"],
                    marker="o",
                    s=22,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.8,
                    zorder=4,
                )
            else:  # filled: cross
                ax.scatter(x, ends["rank"], marker=marker, s=22, color=color, zorder=4)

    format_time_axis(ax)
    ax.set_ylim(bottom=0.5, top=data.get("max_rank", 1) + 0.5)
    ax.invert_yaxis()  # rank 1 (front of queue) at the top
    ax.set_xlabel("Time")
    ax.set_ylabel("Queue rank (1 = front)")
    ax.set_title("Queue position at the touch")
    if drew_any:
        handles = [
            Line2D([0], [0], color=_FILLED_COLOR, label="filled"),
            Line2D([0], [0], color=_CANCELLED_COLOR, label="cancelled"),
            Line2D([0], [0], color=_PARTIAL_COLOR, label="still resting"),
        ]
        ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return fig


def mpl_liquidity_at_touch(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L2 (MBP) liquidity at the touch: best bid/ask resting size over time."""
    fig, ax = _create_axes(ax, figsize=(10, 6), theme=theme)
    ts = data["timestamp"]
    # Thin, semi-transparent step lines so the bid and ask series stay legible
    # where they overplot in the dense band near the touch.
    ax.plot(
        ts,
        data["bid_vol"],
        color=_BID_COLOR,
        linewidth=0.9,
        alpha=0.7,
        drawstyle="steps-post",
        label="Best bid size",
    )
    ax.plot(
        ts,
        data["ask_vol"],
        color=_ASK_COLOR,
        linewidth=0.9,
        alpha=0.7,
        drawstyle="steps-post",
        label="Best ask size",
    )
    format_time_axis(ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Size at touch")
    ax.set_title("Liquidity at the touch")
    if len(ts) > 0:
        ax.legend(loc="upper right")
    format_time_axis(ax)
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
    # A distance-binned fate variant is roadmap §3.8 (docs/plans/).
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
    # The touch: points right of it improved the best quote (the aggressive
    # tail the asymmetric clip in the prepare fn keeps visible — roadmap §3.8).
    ax.axvline(x=0, color="#888888", linestyle="--", linewidth=1, label="touch")
    ax.set_title("Order outcome by placement distance and size")
    ax.set_xlabel("Placement distance from touch (bps)  -  >0 improved the touch")
    ax.set_ylabel("Order size")
    if any_pts:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def mpl_trade_tape_per_order(
    data: dict, ax: Axes | None = None, *, theme: PlotTheme = DEFAULT_THEME
) -> Figure:
    """L3 (MBO) signed-lollipop trade tape with maker resting spans.

    Same signed lollipops as the L2 tape, plus the L3 differentiator: a thin
    span from each consumed maker order's creation to its fill.  Trade prices
    are never clipped, so spike prints stay visible.  Above the density
    threshold the lollipops are per-second VWAPs (roadmap §3.4).
    """
    fig, ax = _create_axes(ax, figsize=(12, 7), theme=theme)
    dense = data.get("dense", False)
    # Maker resting spans: faint underneath the lollipops.  Thinner/fainter when
    # dense so the span cloud reads as texture rather than a solid block.
    span_alpha = 0.12 if dense else 0.35
    span_lw = 0.5 if dense else 1.0
    for side, color in (
        (data["buys"], _BUY_COLOR),
        (data["sells"], _SELL_COLOR),
    ):
        if side.empty:
            continue
        ax.hlines(
            side["price"],
            mdates.date2num(side["created_ts"]),
            mdates.date2num(side["timestamp"]),
            colors=color,
            linewidth=span_lw,
            alpha=span_alpha,
            zorder=1,
        )

    mid_line = data.get("mid_line")
    if mid_line is not None and not mid_line.empty:
        ax.plot(
            mdates.date2num(mid_line["timestamp"]),
            mid_line["mid"].to_numpy(),
            color="#888888",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )

    any_pts = False
    suffix = ", per-s VWAP" if dense else ""
    for side, color, label in (
        (data["lolli_buys"], _BUY_COLOR, f"buy (lifts ask){suffix}"),
        (data["lolli_sells"], _SELL_COLOR, f"sell (hits bid){suffix}"),
    ):
        if side.empty:
            continue
        any_pts = True
        _draw_lollipops(ax, side, color, label, marker_alpha=0.85)

    format_time_axis(ax)
    y_range = data.get("y_range")
    if y_range is not None:
        lo, hi = y_range
        pad = (hi - lo) * 0.04 or 1.0
        ax.set_ylim(lo - pad, hi + pad)
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
    colors_dict = data["colors_dict"]
    legend_entries = data["legend_entries"]
    max_ask = data["max_ask"]
    max_bid = data["max_bid"]
    volume_scale = data["volume_scale"]
    perc_line = data["perc_line"]
    side_line = data["side_line"]

    pl = 0.1 if perc_line else 0

    fig, ax = _create_axes(ax, figsize=(12, 8), theme=theme)

    # Plot date2num floats (not the DatetimeIndex) so the axis stays off
    # matplotlib's slow date unit-converter; format_time_axis sets the ticks.
    x = mdates.date2num(asks_cumsum.index)
    prev = np.zeros(len(asks_cumsum))
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

    x = mdates.date2num(bids_cumsum_neg.index)
    prev = np.zeros(len(bids_cumsum_neg))
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
        ax.axhline(y=0, color="#000000", linewidth=0.6)

    y_range = volume_scale * max(max_ask, max_bid)
    ax.set_ylim(-y_range, y_range)
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative volume  (bid ↓ / ask ↑)")
    format_time_axis(ax)

    # Collapsed legend: a handful of representative depths per side (touch ->
    # far) instead of 2N swatches.  Hue = side, luminance = distance to touch.
    legend_elements = [
        Patch(
            facecolor=color,
            edgecolor="black" if perc_line else None,
            label=label,
        )
        for label, color in legend_entries
    ]
    ax.legend(
        handles=legend_elements,
        title="depth from touch",
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
    # Overlaid step outlines instead of dodged bars: dodging splits each bin
    # into side-by-side combs that misread as a finer x-resolution; layered
    # steps compare the bid and ask distributions bin-for-bin.
    sns.histplot(
        data=events,
        x=val,
        hue="direction",
        element="step",
        multiple="layer",
        fill=True,
        alpha=0.35,
        binwidth=bw,
        palette={"bid": _BID_COLOR, "ask": _ASK_COLOR},
        linewidth=1.2,
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
    format_time_axis(ax)
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

    ax.axhline(y=0, color="#444444", linewidth=0.8, alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("OFI")
    ax.set_title("Order Flow Imbalance")
    format_time_axis(ax)

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

    ax.axhline(y=0, color="#444444", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="#444444", linewidth=0.5, alpha=0.5)

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

    if has_hidden and not hidden.empty:
        # Hue by aggressor side (bid/ask) -- size already encodes volume, so a
        # Reds-by-volume ramp double-encoded it and washed typical prints out to
        # near-white (roadmap §3.6).  Fall back to a single neutral hue when no
        # direction is available.  A thin contrasting edge keeps overlapping
        # prints readable as discrete events.
        neutral = "#7f8c8d"
        direction = data.get("direction")
        if direction is not None:
            colors = direction.map({"bid": _BID_COLOR, "ask": _ASK_COLOR})
            colors = colors.where(colors.notna(), neutral).to_numpy()
        else:
            colors = neutral
        ax.scatter(
            mdates.date2num(hidden["timestamp"]),
            hidden["price"],
            s=data["marker_area"],
            c=colors,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )

    # Price line drawn above the markers so it stays legible against dense
    # overlapping prints.
    if not trades.empty:
        ax.step(
            mdates.date2num(trades["timestamp"]),
            trades["price"],
            where="post",
            color="#222222",
            linewidth=1.0,
            alpha=0.9,
            label="Trade price",
            zorder=3,
        )

    if has_hidden and not hidden.empty:
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
    format_time_axis(ax)
    y_range = data.get("y_range")
    if y_range is not None:
        ax.set_ylim(y_range)

    handles, labels = ax.get_legend_handles_labels()
    if has_hidden and not hidden.empty:
        if data.get("direction") is not None:
            handles += [
                Patch(facecolor=_BID_COLOR, edgecolor="white", label="Hidden (bid)"),
                Patch(facecolor=_ASK_COLOR, edgecolor="white", label="Hidden (ask)"),
            ]
        else:
            handles.append(
                Patch(facecolor="#7f8c8d", edgecolor="white", label="Hidden executions")
            )
    if handles:
        ax.legend(handles=handles, loc="upper left")
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
    format_time_axis(ax)
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
    ("queue_position", _L3, mpl_queue_position_per_order),
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
