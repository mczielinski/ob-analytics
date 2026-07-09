"""Example specs for the docs gallery.

Each entry in :data:`GALLERY` is one figure: a slug, a title, a category,
a one-line caption, and a ``render(result)`` function that returns a
Matplotlib figure. ``scripts/build_gallery.py`` executes each, saves the
image, and shows the function's source as the copy-pasteable recipe — so
every gallery figure is produced by exactly the code displayed beneath it.
The functions deliberately import what they need at the top of their body
so each reads as a standalone snippet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from ob_analytics.analytics import order_book
from ob_analytics.depth import get_spread
from ob_analytics.flow_toxicity import (
    compute_kyle_lambda,
    compute_vpin,
    order_flow_imbalance,
)
from ob_analytics.visualization import plot, prepare


@dataclass(frozen=True)
class Example:
    name: str
    title: str
    category: str
    caption: str
    render: Callable[[Any], Any]


def _window(result: Any, minutes: int = 10) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = result.trades["timestamp"].min()
    return t0, t0 + pd.Timedelta(minutes=minutes)


# ── Book structure ───────────────────────────────────────────────────


def depth_heatmap(result):
    """Standing volume at every price over time — the book's whole history."""
    start, end = _window(result)
    return plot(
        "depth_heatmap",
        level="L2",
        **prepare.price_levels(
            result.depth,
            spread=get_spread(result.depth_summary),
            trades=result.trades,
            col_bias=0.4,
            start_time=start,
            end_time=end,
        ),
    )


def book_snapshot(result):
    """The bid/ask ladder at one instant, per resting order (L3)."""
    tp = result.events["timestamp"].iloc[0] + pd.Timedelta(minutes=10)
    snap = order_book(result.events, tp=tp)
    return plot(
        "book_snapshot", level="L3", **prepare.book_snapshot(snap, per_order=True)
    )


def depth_chart(result):
    """The classic cumulative depth curve, bids left, asks right."""
    tp = result.events["timestamp"].iloc[0] + pd.Timedelta(minutes=10)
    snap = order_book(result.events, tp=tp)
    return plot("depth_chart", level="L2", **prepare.book_snapshot(snap))


# ── Trades ───────────────────────────────────────────────────────────


def trade_tape(result):
    """Every execution as a signed lollipop against the mid price."""
    start, end = _window(result)
    return plot(
        "trade_tape",
        level="L2",
        **prepare.trades(
            result.trades,
            spread=get_spread(result.depth_summary),
            start_time=start,
            end_time=end,
        ),
    )


def trade_size(result):
    """Trade sizes over time — spotting the large prints."""
    start, end = _window(result)
    return plot(
        "trade_size",
        level="L2",
        **prepare.trade_size(result.trades, start_time=start, end_time=end),
    )


# ── Liquidity ────────────────────────────────────────────────────────


def volume_percentiles(result):
    """Cumulative resting liquidity in basis-point rings around the mid."""
    start, end = _window(result)
    return plot(
        "volume_percentiles",
        level="L2",
        **prepare.volume_percentiles(
            result.depth_summary, start_time=start, end_time=end
        ),
    )


def liquidity_at_touch(result):
    """How much size sits right at the best bid and ask, over time."""
    start, end = _window(result)
    return plot(
        "liquidity_at_touch",
        level="L2",
        **prepare.liquidity_at_touch(
            result.depth_summary, start_time=start, end_time=end
        ),
    )


# ── Order lifecycles (L3) ────────────────────────────────────────────


def order_outcome(result):
    """Every order by placement distance and size, coloured by its fate."""
    return plot("order_outcome", level="L3", **prepare.order_outcome_l3(result.events))


def queue_position(result):
    """FIFO rank at the touch over time — the race to the front of the line."""
    start, end = _window(result)
    return plot(
        "queue_position",
        level="L3",
        **prepare.queue_position_l3(result.events, start_time=start, end_time=end),
    )


def cancellations(result):
    """Where and how fast orders are cancelled — age vs distance from touch."""
    return plot("cancellations", level="L3", **prepare.cancellations_l3(result.events))


# ── Flow toxicity ────────────────────────────────────────────────────


def vpin(result):
    """Volume-synchronised probability of informed trading over the session."""
    v = compute_vpin(result.trades, bucket_volume=result.trades["volume"].sum() / 20)
    return plot("vpin", **prepare.vpin(v, threshold=0.7))


def kyle_lambda(result):
    """Price impact per unit of signed order flow — Kyle's λ regression."""
    return plot(
        "kyle_lambda",
        **prepare.kyle_lambda(compute_kyle_lambda(result.trades, window="5min")),
    )


def flow_imbalance(result):
    """Net buy/sell pressure per minute, tape above, imbalance below."""
    ofi = order_flow_imbalance(result.trades, window="1min")
    return plot("order_flow_imbalance", **prepare.ofi(ofi, trades=result.trades))


GALLERY: list[Example] = [
    Example(
        "depth_heatmap",
        "Depth heatmap",
        "The book",
        "Standing volume at every price over time.",
        depth_heatmap,
    ),
    Example(
        "book_snapshot",
        "Book snapshot (L3)",
        "The book",
        "The ladder at one instant, per resting order.",
        book_snapshot,
    ),
    Example(
        "depth_chart",
        "Depth chart",
        "The book",
        "Cumulative depth, bids left, asks right.",
        depth_chart,
    ),
    Example(
        "trade_tape",
        "Trade tape",
        "Trades",
        "Each execution as a signed lollipop against the mid.",
        trade_tape,
    ),
    Example(
        "trade_size",
        "Trade sizes",
        "Trades",
        "Trade sizes over time — spotting the large prints.",
        trade_size,
    ),
    Example(
        "volume_percentiles",
        "Volume percentiles",
        "Liquidity",
        "Resting liquidity in basis-point rings around the mid.",
        volume_percentiles,
    ),
    Example(
        "liquidity_at_touch",
        "Liquidity at touch",
        "Liquidity",
        "Size at the best bid and ask over time.",
        liquidity_at_touch,
    ),
    Example(
        "order_outcome",
        "Order outcomes (L3)",
        "Order lifecycles",
        "Orders by placement and size, coloured by fate.",
        order_outcome,
    ),
    Example(
        "queue_position",
        "Queue position (L3)",
        "Order lifecycles",
        "FIFO rank at the touch — the race to the front.",
        queue_position,
    ),
    Example(
        "cancellations",
        "Cancellations (L3)",
        "Order lifecycles",
        "Where and how fast orders are cancelled.",
        cancellations,
    ),
    Example(
        "vpin",
        "VPIN",
        "Flow toxicity",
        "Volume-synchronised probability of informed trading.",
        vpin,
    ),
    Example(
        "kyle_lambda",
        "Kyle's lambda",
        "Flow toxicity",
        "Price impact per unit of signed order flow.",
        kyle_lambda,
    ),
    Example(
        "flow_imbalance",
        "Order flow imbalance",
        "Flow toxicity",
        "Net buy/sell pressure per minute.",
        flow_imbalance,
    ),
]
