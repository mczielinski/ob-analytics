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


def plot_l1_ticker(
    bid: float,
    ask: float,
    last: float | None = None,
    symbol: str = "TOY",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Draw a brokerage-app-style Level-1 quote card: best bid / best ask / last.

    The tutorial's scene-2 prop — the familiar 'price widget' decomposed into
    the three numbers it actually contains. Pass *ax* to compose the card
    into a multi-panel figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 1.9))
    else:
        fig = ax.figure
    ax.set_axis_off()
    card = plt.Rectangle(
        (0.02, 0.06),
        0.96,
        0.88,
        transform=ax.transAxes,
        facecolor="#f7f7f9",
        edgecolor="#c9c9d1",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(card)
    ax.text(
        0.08,
        0.76,
        symbol,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        0.08, 0.58, "Level 1 quote", transform=ax.transAxes, fontsize=8, color="#888888"
    )
    cols = [
        ("BID", f"{bid:g}", BID_COLOR),
        ("ASK", f"{ask:g}", ASK_COLOR),
        ("LAST", "—" if last is None else f"{last:g}", "#222222"),
    ]
    for x, (label, value, color) in zip((0.42, 0.62, 0.82), cols):
        ax.text(
            x,
            0.62,
            label,
            transform=ax.transAxes,
            fontsize=9,
            color="#888888",
            ha="center",
        )
        ax.text(
            x,
            0.30,
            value,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            color=color,
            ha="center",
        )
    return fig


def plot_queue_story(
    events: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    at_s: list[float],
    actor_col: str = "actor",
) -> plt.Figure:
    """The queue_position face, actor-labeled, anchored to book keyframes.

    Top: the ``queue_position`` L3 face with each trajectory's terminal
    marker labeled by actor (from *actor_col*), so 'walk the lines' is
    literal. Bottom: mini book ladders at the *at_s* storyboard instants.
    Toy-scale teaching figure — real data has too many trajectories to
    label.
    """
    from ob_analytics.visualization import plot, prepare

    payload = prepare.queue_position_l3(events)
    actor_of = dict(zip(events["id"], events[actor_col].astype(str)))

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(
        2, len(at_s), height_ratios=[1.7, 1.0], hspace=0.34, wspace=0.16
    )
    ax_q = fig.add_subplot(gs[0, :])
    plot("queue_position", level="L3", ax=ax_q, theme=DOCS_THEME, **payload)
    ax_q.set_title("")  # clear the face's centered title; we set our own
    ax_q.margins(x=0.07)  # room for terminal labels at the session's end
    legend = ax_q.get_legend()
    if legend is not None:  # move the face's legend out of the label zone
        legend.set_loc("lower left")

    # Label each trajectory at its terminal point; stagger ends that land
    # within a few seconds of each other so late-session labels stay apart.
    seen: dict[pd.Timestamp, int] = {}
    for fate in ("filled", "cancelled", "resting"):
        frame = payload[fate]
        if frame is None or frame.empty:
            continue
        for _, last in frame.groupby("id").tail(1).iterrows():
            bucket = last["timestamp"].floor("5s")
            bump = seen.get(bucket, 0)
            seen[bucket] = bump + 1
            ax_q.annotate(
                actor_of.get(last["id"], str(last["id"])),
                (last["timestamp"], last["rank"]),
                textcoords="offset points",
                xytext=(6, 10 + 13 * bump),
                fontsize=8.5,
                fontweight="bold",
                color="#333333",
            )
    ax_q.set_title(
        "Queue rank at the touch — every trajectory labeled by its owner",
        fontsize=10,
        loc="left",
    )

    key_axes = [fig.add_subplot(gs[1, i]) for i in range(len(at_s))]
    for axk in key_axes[1:]:
        axk.sharey(key_axes[0])
    plot_book_keyframes(events, trades, at_s=at_s, ax_row=key_axes)
    for axk in key_axes[1:]:
        axk.tick_params(labelleft=False)
    return fig


def plot_book_keyframes(
    events: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    every_s: int = 5,
    last_s: int = 60,
    at_s: list[float] | None = None,
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

    Pass *at_s* to render chosen storyboard instants instead of the regular
    grid; each frame's trade-star window then reaches back to the previous
    listed instant (or *every_s* seconds for the first).
    """
    t0 = events["timestamp"].iloc[0]
    ticks = list(at_s) if at_s is not None else list(range(0, last_s + 1, every_s))
    label_of = (
        dict(zip(events["id"], events[actor_col].astype(str).str[:2]))
        if actor_col in events.columns
        else {}
    )

    if ax_row is None:
        width = fig_width if len(ticks) > 6 else 1.35 * len(ticks) + 0.8
        fig, axes = plt.subplots(1, len(ticks), figsize=(width, 2.6), sharey=True)
        if len(ticks) == 1:
            axes = [axes]
    else:
        axes = ax_row
        fig = axes[0].figure

    prices = events.loc[events["price"] > 0, "price"]
    y_lo, y_hi = prices.min() - 0.6, prices.max() + 0.6

    for i, (axk, k) in enumerate(zip(axes, ticks)):
        window_start_s = ticks[i - 1] if i > 0 else k - every_s
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
                (trades["timestamp"] > t0 + pd.Timedelta(seconds=window_start_s))
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


def plot_toy_depth_heatmap(
    depth: pd.DataFrame,
    spread: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    *,
    ax: plt.Axes | None = None,
    col_bias: float = 1.0,
    band_pt: float = 22.0,
    **prepare_kwargs,
) -> plt.Figure:
    """The depth_heatmap face with fat price bands for toy-scale reading.

    The face draws each price level as a 2-pt line — right for thousands
    of levels, hairline-thin for five. This boosts the LineCollection
    width so every level reads as a solid band whose colour changes are
    checkable against the keyframe strip, and pins the y-range to the
    toy's full price grid.
    """
    from matplotlib.collections import LineCollection

    from ob_analytics.visualization import plot, prepare

    lo, hi = depth["price"].min(), depth["price"].max()
    fig_or_ax = ax
    if fig_or_ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = fig_or_ax.figure
    plot(
        "depth_heatmap",
        level="L2",
        ax=ax,
        theme=DOCS_THEME,
        **prepare.price_levels(
            depth,
            spread=spread,
            trades=trades,
            col_bias=col_bias,
            price_from=lo - 0.4,
            price_to=hi + 0.4,
            **prepare_kwargs,
        ),
    )
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            coll.set_linewidth(band_pt)
    ax.set_yticks(sorted(depth["price"].unique()))
    # widen the frame so the fat edge bands render whole, not half-clipped
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)
    return fig
