"""Matplotlib / seaborn rendering backend for ob-analytics.

This module contains all matplotlib-specific rendering logic.  Each
``mpl_*()`` function takes a prepared data dict (from
:mod:`~ob_analytics._chart_data`) and an optional *ax* parameter, and
returns a :class:`~matplotlib.figure.Figure`.

The :class:`PlotTheme` dataclass and related helpers also live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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
# Rendering functions
# ---------------------------------------------------------------------------


def mpl_time_series(data: dict, ax: Axes | None = None) -> Figure:
    """Render a time-series step plot."""
    df = data["df"]
    fig, ax = _create_axes(ax, figsize=(10, 6))
    sns.lineplot(data=df, x="ts", y="val", drawstyle="steps-post", ax=ax)
    ax.set_title(data["title"])
    ax.set_xlabel("time")
    ax.set_ylabel(data["y_label"])
    ax.grid(True)
    fig.tight_layout()
    return fig


def mpl_trades(data: dict, ax: Axes | None = None) -> Figure:
    """Render a trade-price step plot."""
    filtered = data["filtered_trades"]
    y_breaks = data["y_breaks"]
    fig, ax = _create_axes(ax, figsize=(10, 6))
    sns.lineplot(
        data=filtered, x="timestamp", y="price",
        drawstyle="steps-post", ax=ax,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Limit Price")
    ax.set_yticks(y_breaks)
    ax.grid(True)
    fig.tight_layout()
    return fig


def mpl_price_levels(data: dict, ax: Axes | None = None) -> Figure:
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
                spread["timestamp"], spread["midprice"],
                color="#ffffff", linewidth=1.1, label="Midprice",
            )
        else:
            if "best_ask_price" in spread:
                ax.step(
                    spread["timestamp"], spread["best_ask_price"],
                    color="#ff0000", linewidth=1.5, where="post", label="Best Ask",
                )
            if "best_bid_price" in spread:
                ax.step(
                    spread["timestamp"], spread["best_bid_price"],
                    color="#00ff00", linewidth=1.5, where="post", label="Best Bid",
                )

    if trades is not None:
        buys = trades[trades["direction"] == "buy"]
        sells = trades[trades["direction"] == "sell"]

        if not sells.empty:
            ax.scatter(
                sells["timestamp"], sells["price"],
                s=50, facecolors="none", edgecolors="#ff0000",
                linewidths=1.5, zorder=5, marker="v", label="Sell Trades",
            )
        if not buys.empty:
            ax.scatter(
                buys["timestamp"], buys["price"],
                s=50, facecolors="none", edgecolors="#00ff00",
                linewidths=1.5, zorder=5, marker="^", label="Buy Trades",
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


def mpl_event_map(data: dict, ax: Axes | None = None) -> Figure:
    """Render a limit-order event map."""
    events = data["events"]
    created = data["created"]
    deleted = data["deleted"]
    price_by = data["price_by"]

    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    fig, ax = _create_axes(ax, figsize=(10, 6))

    sns.scatterplot(
        data=created, x="timestamp", y="price",
        size="volume", sizes=(20, 200), color="#333333",
        ax=ax, legend=False, marker="o",
    )
    sns.scatterplot(
        data=deleted, x="timestamp", y="price",
        size="volume", sizes=(20, 200), color="#333333",
        ax=ax, legend=False, marker="o", edgecolor="black", alpha=0.5,
    )
    sns.scatterplot(
        data=events, x="timestamp", y="price",
        hue="direction", size=0.1, palette=col_pal,
        ax=ax, legend=False,
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


def mpl_volume_map(data: dict, ax: Axes | None = None) -> Figure:
    """Render a volume map of flashed limit orders."""
    events = data["events"]
    log_scale = data["log_scale"]
    col_pal = {"bid": "#0000ff", "ask": "#ff0000"}

    fig, ax = _create_axes(ax, figsize=(10, 6))
    if log_scale:
        ax.set_yscale("log")
    sns.scatterplot(
        data=events, x="timestamp", y="volume",
        hue="direction", palette=col_pal, size=0.5, marker="o", ax=ax,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Volume")
    ax.set_title("Volume Map of Flashed Limit Orders")
    fig.tight_layout()
    return fig


def mpl_current_depth(data: dict, ax: Axes | None = None) -> Figure:
    """Render order book depth snapshot."""
    depth_df = data["depth_df"]
    bids = data["bids"]
    asks = data["asks"]
    show_volume = data["show_volume"]
    show_quantiles = data["show_quantiles"]
    bid_quantiles = data["bid_quantiles"]
    ask_quantiles = data["ask_quantiles"]
    timestamp = data["timestamp"]

    fig, ax = _create_axes(ax, figsize=(12, 7))

    if show_volume:
        unique_prices = np.sort(np.unique(depth_df["price"]))
        price_diffs = np.diff(unique_prices)
        price_diffs = price_diffs[price_diffs > 0]
        bar_width = np.min(price_diffs) if len(price_diffs) > 0 else 1

        ax.bar(
            depth_df["price"], depth_df["volume"],
            width=bar_width, color="white", align="center", edgecolor=None,
        )

    col_pal = {"ask": "#ff0000", "bid": "#0000ff"}
    for side_value in ["bid", "ask"]:
        side_data = depth_df[depth_df["side"] == side_value]
        ax.step(
            side_data["price"], side_data["liquidity"],
            where="pre", color=col_pal[side_value], label=side_value, linewidth=2,
        )

    if show_quantiles:
        for x_value in bid_quantiles:
            ax.axvline(x=x_value, color="#222222", linestyle="--")
        for x_value in ask_quantiles:
            ax.axvline(x=x_value, color="#222222", linestyle="--")

    xmin = round(bids["price"].min())
    xmax = round(asks["price"].max())
    xticks = np.arange(xmin, xmax + 1, 1)
    ax.set_xticks(xticks)

    ax.set_title(timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
    ax.set_xlabel("Price")
    ax.set_ylabel("Liquidity")
    ax.legend()
    fig.tight_layout()
    return fig


def mpl_volume_percentiles(data: dict, ax: Axes | None = None) -> Figure:
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

    fig, ax = _create_axes(ax, figsize=(12, 8))

    prev = np.zeros(len(asks_cumsum))
    x = asks_cumsum.index
    for percentile in asks_cols:
        current = asks_cumsum[percentile].values
        ax.fill_between(
            x, prev, current,
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
            x, prev, current,
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
        handles=legend_elements, title="depth         \n",
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        ncol=1, borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    return fig


def mpl_events_histogram(data: dict, ax: Axes | None = None) -> Figure:
    """Render an events price/volume histogram."""
    events = data["events"]
    val = data["val"]
    bw = data["bw"]

    fig, ax = _create_axes(ax, figsize=(12, 7))
    sns.histplot(
        data=events, x=val, hue="direction",
        multiple="dodge", binwidth=bw,
        palette={"bid": "#0000ff", "ask": "#ff0000"},
        edgecolor="white", linewidth=0.5, ax=ax,
    )
    ax.set_title(f"Events {val} distribution")
    ax.set_xlabel(val.capitalize())
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def mpl_vpin(data: dict, ax: Axes | None = None) -> Figure:
    """Render VPIN time series."""
    vpin_df = data["vpin_df"]
    threshold = data["threshold"]
    bar_width = data["bar_width"]

    fig, ax = _create_axes(ax, figsize=(12, 5))

    ax.bar(
        vpin_df["timestamp_end"], vpin_df["vpin"],
        width=bar_width, color="#5dade2", alpha=0.35, label="Per-bucket VPIN",
    )

    if "vpin_avg" in vpin_df.columns:
        ax.plot(
            vpin_df["timestamp_end"], vpin_df["vpin_avg"],
            color="#e74c3c", linewidth=2, label="VPIN (rolling avg)",
        )

    ax.axhline(
        y=threshold, color="#f39c12", linewidth=1.5, linestyle="--",
        label=f"Threshold ({threshold})",
    )

    if "vpin_avg" in vpin_df.columns:
        above = vpin_df["vpin_avg"] >= threshold
        ax.fill_between(
            vpin_df["timestamp_end"], threshold, vpin_df["vpin_avg"],
            where=above, color="#e74c3c", alpha=0.15,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("VPIN")
    ax.set_title("Volume-Synchronized Probability of Informed Trading")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def mpl_order_flow_imbalance(data: dict, ax: Axes | None = None) -> Figure:
    """Render order flow imbalance bar chart."""
    ofi_df = data["ofi_df"]
    trades = data["trades"]
    colors = data["colors"]
    bar_width = data["bar_width"]

    fig, ax = _create_axes(ax, figsize=(12, 5))

    ax.bar(
        ofi_df["timestamp"], ofi_df["ofi"],
        width=bar_width, color=colors, alpha=0.7,
        edgecolor="white", linewidth=0.3,
    )

    ax.axhline(y=0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("OFI")
    ax.set_title("Order Flow Imbalance")

    if trades is not None and "price" in trades.columns:
        ax2 = ax.twinx()
        ax2.plot(
            trades["timestamp"], trades["price"],
            color="#f1c40f", linewidth=1.2, alpha=0.8, label="Price",
        )
        ax2.set_ylabel("Price", color="#f1c40f")
        ax2.tick_params(axis="y", labelcolor="#f1c40f")

    fig.tight_layout()
    return fig


def mpl_kyle_lambda(data: dict, ax: Axes | None = None) -> Figure:
    """Render Kyle's Lambda regression scatter."""
    reg_df = data["reg_df"]
    lambda_ = data["lambda_"]
    r_squared = data["r_squared"]
    t_stat = data["t_stat"]

    fig, ax = _create_axes(ax, figsize=(8, 6))

    ax.scatter(
        reg_df["signed_volume"], reg_df["delta_price"],
        color="#5dade2", alpha=0.6, edgecolors="white",
        linewidths=0.5, s=50, zorder=3,
    )

    if not np.isnan(lambda_):
        x_range = np.linspace(
            reg_df["signed_volume"].min(),
            reg_df["signed_volume"].max(),
            100,
        )
        intercept = reg_df["delta_price"].mean() - lambda_ * reg_df["signed_volume"].mean()
        ax.plot(
            x_range, intercept + lambda_ * x_range,
            color="#e74c3c", linewidth=2, label=f"λ = {lambda_:.6f}", zorder=4,
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
