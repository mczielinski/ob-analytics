"""The colours every rendering backend draws with.

:class:`Palette` is the single source of truth for plot colours.  A
:class:`~ob_analytics.visualization.PlotTheme` carries one, and the
matplotlib, plotly, and bokeh renderers read every colour from it, so one
concept looks the same in each backend and a theme recolours all of them.

Three categorical vocabularies, one hue pair/triple each:

* **Side** (resting order's side): bid / ask — book snapshot, depth chart,
  cancellations, best-quote lines.
* **Fate** (order lifecycle outcome): flashed (cancelled) / resting (filled),
  plus the competing-risks split filled / partial / cancelled.
* **Aggressor** (taker side of an execution): buy / sell — trade tape, trade
  markers, and order flow imbalance.

The rest are neutral marks (price and reference lines, zero rules, labels)
and the accent colours of the analytics faces.

The default hues derive from the Okabe–Ito colorblind-safe palette and were
verified under Machado-matrix deuteranopia simulation (the most common
color-vision deficiency): every pair keeps a normalized RGB distance ≥ 0.26
(dominant pairs ≥ 0.5), where the previous buy-green/sell-red pair collapsed
to 0.13.  Luminances are spread (0.37–0.64) so the classes also survive
grayscale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """Named plot colours, shared by every rendering backend.

    Every field is a hex colour string.  Override single fields with
    :func:`dataclasses.replace` or by passing them to the constructor::

        Palette(bid="#1f77b4", ask="#d62728")

    Attributes
    ----------
    bid, ask : str
        Side of a resting order (Okabe–Ito blue / vermillion).
    buy, sell : str
        Aggressor side of an execution: buyer-initiated (lifts the ask) and
        seller-initiated (hits the bid).
    flashed, resting : str
        Order-activity fate: placed and pulled / rested or filled.
    filled, partial, cancelled : str
        Order-outcome fate: fully executed / partly executed with the
        remainder removed / removed without any execution.
    hidden_trade : str
        A confirmed hidden-order trade.
    check_trade : str
        A trade flagged as hidden whose maker order was visible (a diff
        feed's stale depth summary), which needs checking rather than
        trusting.
    price_line : str
        The main price line (trade price, microprice) and dark markers.
    reference_line : str
        Secondary reference marks: the mid line, dotted guides, "created"
        markers, and "no data" messages.
    rule : str
        Zero lines, the book's mid rule, and axis edges.
    label : str
        Annotation text.
    spread_fill : str
        The band between best bid and best ask.
    neutral : str
        Points with no side (for example hidden executions of unknown
        direction).
    series : str
        A single unclassified data series (time series, VPIN buckets,
        Kyle's-lambda scatter).
    emphasis : str
        A summary drawn over a series: rolling average, fitted line, halt
        span.
    threshold : str
        A threshold line.
    secondary_axis : str
        A series on a secondary y-axis, and that axis's labels.
    imbalance : str
        The order book imbalance line.
    effective_spread, realized_spread : str
        The two transaction-cost series.
    """

    # Side (bid / ask) — Okabe–Ito blue / vermillion
    bid: str = "#0072B2"
    ask: str = "#D55E00"

    # Aggressor (taker side) — bluish green / vermillion
    buy: str = "#009E73"
    sell: str = "#D55E00"

    # Fate: order_activity L3 Gantt
    flashed: str = "#E69F00"
    resting: str = "#009E73"

    # Fate: order_outcome L3 competing-risks scatter
    filled: str = "#009E73"
    partial: str = "#CC79A7"
    cancelled: str = "#E69F00"

    # Hidden liquidity overlay (depth heatmap / order activity map). Icebergs
    # take their side colour from bid/ask.
    hidden_trade: str = "#F0E442"  # Okabe–Ito yellow
    check_trade: str = "#999999"

    # Neutral marks
    price_line: str = "#222222"
    reference_line: str = "#888888"
    rule: str = "#444444"
    label: str = "#555555"
    spread_fill: str = "#9aa0a6"
    neutral: str = "#7f8c8d"

    # Accents of the analytics faces
    series: str = "#5dade2"
    emphasis: str = "#e74c3c"
    threshold: str = "#f39c12"
    secondary_axis: str = "#f1c40f"
    imbalance: str = "#6d28d9"
    effective_spread: str = "#0072B2"
    realized_spread: str = "#CC79A7"


#: The palette :data:`~ob_analytics.visualization.DEFAULT_THEME` uses.
DEFAULT_PALETTE: Palette = Palette()
