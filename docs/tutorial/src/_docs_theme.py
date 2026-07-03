"""Shared styling and teaching figures for the tutorial chapters.

Every tutorial figure goes through :data:`DOCS_THEME` so the whole site
has one visual identity, and recurring teaching figures live here so
chapters stay short. This module is docs-build machinery, not part of
the installed package.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ob_analytics.analytics import order_book
from ob_analytics.visualization import PlotTheme

DOCS_THEME = PlotTheme(
    style="whitegrid",
    context="notebook",
    font_scale=1.0,
    rc={"figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff"},
)

# Match the bid/ask hues of the built-in faces.
BID_COLOR = "#1170aa"
ASK_COLOR = "#d1610d"
TRADE_MARK_COLOR = "#c8102e"


def plot_book_keyframes(
    events: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    every_s: int = 5,
    last_s: int = 60,
    actor_col: str = "actor",
    ax_row: list[plt.Axes] | None = None,
    fig_width: float = 17.0,
) -> plt.Figure:
    """Render the book as a strip of mini-ladders, one every *every_s* seconds.

    Within each price level, per-order segments stack in **arrival order**
    (first in queue nearest the axis) and are labeled from *actor_col*, so
    price–time priority is visible frame by frame. A star on a frame's edge
    marks any trade in the window ending at that frame. Designed for the
    toy session (`ob_analytics.datasets`); any small classified events
    frame works.
    """
    t0 = events["timestamp"].iloc[0]
    ticks = list(range(0, last_s + 1, every_s))
    label_of = (
        dict(zip(events["id"], events[actor_col].astype(str).str[:2]))
        if actor_col in events.columns
        else {}
    )

    if ax_row is None:
        fig, axes = plt.subplots(1, len(ticks), figsize=(fig_width, 2.6), sharey=True)
    else:
        axes = ax_row
        fig = axes[0].figure

    prices = events.loc[events["price"] > 0, "price"]
    y_lo, y_hi = prices.min() - 0.6, prices.max() + 0.6

    for axk, k in zip(axes, ticks):
        tp = t0 + pd.Timedelta(seconds=k)
        snap = order_book(events, tp=tp)
        x_max = 0.0
        for side, color in ((snap["bids"], BID_COLOR), (snap["asks"], ASK_COLOR)):
            lefts: dict[float, float] = {}
            for row in side.itertuples():
                lo = lefts.get(row.price, 0.0)
                axk.barh(
                    row.price,
                    row.volume,
                    left=lo,
                    height=0.8,
                    color=color,
                    edgecolor="white",
                    linewidth=1.2,
                )
                label = label_of.get(row.id)
                if label and row.volume >= 0.75:
                    axk.text(
                        lo + row.volume / 2,
                        row.price,
                        label,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7.5,
                        fontweight="bold",
                    )
                lefts[row.price] = lo + row.volume
            if lefts:
                x_max = max(x_max, max(lefts.values()))
        bids, asks = snap["bids"], snap["asks"]
        if len(bids) and len(asks):
            mid = (bids["price"].max() + asks["price"].min()) / 2
            axk.axhline(mid, color="#222222", ls="--", lw=1.1)
        if trades is not None:
            window = trades[
                (trades["timestamp"] > tp - pd.Timedelta(seconds=every_s))
                & (trades["timestamp"] <= tp)
            ]
            for trd in window.itertuples():
                axk.plot(
                    x_max + 0.45,
                    trd.price,
                    marker="*",
                    color=TRADE_MARK_COLOR,
                    markersize=9,
                    clip_on=False,
                )
        axk.set_xlim(0, max(x_max + 0.6, 1.0))
        axk.set_ylim(y_lo, y_hi)
        axk.set_xticks([])
        axk.tick_params(labelsize=7, length=0)
        axk.set_title(f"t={k}s", fontsize=8.5)
        for spine in axk.spines.values():
            spine.set_color("#dddddd")

    axes[0].set_ylabel("Price", fontsize=8)
    return fig
