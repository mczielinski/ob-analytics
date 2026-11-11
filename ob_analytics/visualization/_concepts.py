"""Every plot the package draws, declared once.

A plot is a coordinate ``(concept, level, backend)``: what it shows, at which
resolution (``Level.L2``, ``Level.L3``, or ``None`` for a level-less
analytic), and which library draws it.  :data:`CONCEPTS` declares each
concept once: its gallery title and note, and at each level the renderer
that draws it, the backends that have that renderer, and how the gallery
binds its data from a run.  :data:`PLOTS` is the same table keyed by the
coordinate.

The gallery, :func:`~ob_analytics.visualization.plot_result` and
:func:`~ob_analytics.visualization.plot` read this table to know where a plot
is drawn, so a backend a plot does not have is a declared fact (the book
replay is Plotly only) rather than a missing registration.  The backend
modules only draw: each calls :func:`register_backend` once, which registers
the backend's function for every coordinate the table gives it.

A renderer is named after its backend and its face: the L2 trade tape face,
``"trade_tape_aggregate"``, is ``mpl_trade_tape_aggregate`` in the matplotlib
module, ``plotly_trade_tape_aggregate`` in the plotly module and
``bokeh_trade_tape_aggregate`` in the bokeh module.  A concept drawn at both
levels names its faces ``<concept>_aggregate`` (L2) and ``<concept>_per_order``
(L3); a concept drawn at one level names its face after the concept.

A plot registered straight into :data:`~ob_analytics.visualization.RENDERERS`
by another package (a plug-in plot or backend) is not in this table.
:func:`placements` adds those registrations to the table's, so a plug-in
plot is found the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import pandas as pd

from ob_analytics.book import order_book
from ob_analytics.depth import price_level_snapshots
from ob_analytics.protocols import Level
from ob_analytics.visualization import _data
from ob_analytics.visualization._renderers import (
    RENDERERS,
    RendererKey,
    missing_plot,
)

L2, L3 = Level.L2, Level.L3

#: The prefix of a renderer's function name in each built-in backend module.
BACKEND_PREFIXES: dict[str, str] = {
    "matplotlib": "mpl",
    "plotly": "plotly",
    "bokeh": "bokeh",
}

_ALL = ("matplotlib", "plotly", "bokeh")
_MPL_PLOTLY = ("matplotlib", "plotly")
_PLOTLY = ("plotly",)


# ---------------------------------------------------------------------------
# The run a gallery is drawn from
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """What every gallery face of one run is drawn from.

    The gallery builds one scene per run, in display units, and each face
    takes its data from it.  So faces that show the same window, or the same
    price band, get the same bounds.

    Attributes
    ----------
    events, trades, depth : pandas.DataFrame
        The run's tables.  ``events`` is empty on a price-level (L2) run.
    depth_summary : pandas.DataFrame
        The depth summary without its first minute, while the book fills.
    spread : pandas.DataFrame
        The best bid and ask through time.
    volume_scale : float
        The volume display scale.
    zoom_start, zoom_end : pandas.Timestamp
        The busiest stretch of the run (:func:`~ob_analytics.visualization.busiest_window`).
        The book faces show the book at ``zoom_end``.
    summary_start, summary_end : pandas.Timestamp
        The window of the depth-summary faces: the zoom window when it holds
        a summary row, else the whole summary.
    focus : FocusWindow
        The price band the clipped faces share.
    overlay : dict
        The hidden-liquidity overlay for the depth heatmap and the per-order
        activity map; empty when there is none.
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    depth: pd.DataFrame
    depth_summary: pd.DataFrame
    spread: pd.DataFrame
    volume_scale: float
    zoom_start: pd.Timestamp
    zoom_end: pd.Timestamp
    summary_start: pd.Timestamp
    summary_end: pd.Timestamp
    focus: _data.FocusWindow
    overlay: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def per_order(self) -> bool:
        """Whether the run has per-order events (an L3 run)."""
        return not self.events.empty

    @property
    def classified(self) -> bool:
        """Whether the events carry the ``type`` the book rebuild reads."""
        return self.per_order and "type" in self.events.columns

    @property
    def price_band(self) -> dict[str, float | None]:
        """The focus band as the ``price_from``/``price_to`` keywords."""
        return {"price_from": self.focus.price_from, "price_to": self.focus.price_to}

    @cached_property
    def price_level_book(self) -> Any:
        """The book at ``zoom_end`` from the depth table, or ``None``."""
        if self.depth.empty:
            return None
        return price_level_snapshots(self.depth, [self.zoom_end])[0]

    @cached_property
    def per_order_book(self) -> Any:
        """The book at ``zoom_end`` rebuilt from the events, or ``None``."""
        if not self.classified:
            return None
        return order_book(self.events, at=self.zoom_end)

    @property
    def snapshot_label(self) -> str:
        """The time of the book faces, for their titles."""
        return self.zoom_end.strftime("%H:%M")


@dataclass(frozen=True)
class Binding:
    """A face's data for one run: the prepare function and its keywords."""

    prepare: Callable[..., dict]
    kwargs: dict[str, Any]


#: Binds a face's data from a scene, or returns ``None`` when the run lacks it.
Binder = Callable[[Scene], Binding | None]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Face:
    """One concept at one level.

    Attributes
    ----------
    renderer : str
        The renderer's name without its backend prefix (see
        :data:`BACKEND_PREFIXES`), e.g. ``"trade_tape_aggregate"`` for ``mpl_trade_tape_aggregate``.
    backends : tuple of str
        The backends that draw this face.
    bind : callable or None
        How the gallery binds the face's data from a :class:`Scene`.
        ``None`` for a face the gallery does not build itself: a metric
        (it has its own panel) or a plot drawn only by
        :func:`~ob_analytics.visualization.plot`.
    """

    renderer: str
    backends: tuple[str, ...]
    bind: Binder | None = None


@dataclass(frozen=True)
class Concept:
    """One plot concept: its gallery card and its face at each level.

    *title* may hold ``{snapshot}``, the time of the book faces.
    """

    key: str
    faces: Mapping[Level | None, Face]
    title: str = ""
    note: str = ""


def _bound(prepare: Callable[..., dict], **kwargs: Any) -> Binding:
    return Binding(prepare, kwargs)


def _trade_tape_l2(s: Scene) -> Binding | None:
    if s.trades.empty:
        return None
    return _bound(_data.prepare_trade_tape_l2_data, trades=s.trades)


def _trade_tape_l3(s: Scene) -> Binding | None:
    if not s.per_order:
        return None
    return _bound(
        _data.prepare_trade_tape_l3_data,
        events=s.events,
        trades=s.trades,
        volume_scale=s.volume_scale,
    )


def _depth_heatmap(s: Scene) -> Binding:
    return _bound(
        _data.prepare_depth_heatmap_data,
        depth=s.depth,
        spread=s.spread,
        trades=s.trades,
        volume_scale=s.volume_scale,
        **s.price_band,
        **s.overlay,
    )


def _order_activity_l2(s: Scene) -> Binding | None:
    if not s.per_order:
        return None
    # The depth heatmap's price band, so the event map clips around the
    # touch: far flashed orders would otherwise squash the activity.
    return _bound(
        _data.prepare_order_activity_l2_data,
        events=s.events,
        volume_scale=s.volume_scale,
        **s.price_band,
    )


def _order_activity_l3(s: Scene) -> Binding | None:
    if not s.per_order:
        return None
    return _bound(
        _data.prepare_order_activity_l3_data,
        events=s.events,
        volume_scale=s.volume_scale,
        **s.price_band,
        **s.overlay,
    )


def _cancellations_l2(s: Scene) -> Binding | None:
    if not s.per_order:
        return None
    return _bound(
        _data.prepare_cancellations_l2_data,
        events=s.events,
        volume_scale=s.volume_scale,
        log_scale=True,
    )


def _cancellations_l3(s: Scene) -> Binding | None:
    if not s.per_order:
        return None
    return _bound(
        _data.prepare_cancellations_l3_data,
        events=s.events,
        volume_scale=s.volume_scale,
    )


def _book(level: Level) -> Binder:
    """The book faces read the book of their own level (see ``book_snapshot``)."""

    def bind(s: Scene) -> Binding | None:
        book = s.price_level_book if level is L2 else s.per_order_book
        if book is None:
            return None
        return _bound(
            _data.prepare_book_snapshot_data,
            order_book=book,
            level=level,
            volume_scale=s.volume_scale,
        )

    return bind


#: The most frames the gallery's book replay holds: enough to scrub the zoom
#: window smoothly, few enough to keep the page small and quick to build.
REPLAY_FRAMES = 300


def _book_replay(level: Level) -> Binder:
    def bind(s: Scene) -> Binding | None:
        if s.depth.empty or (level is L3 and not s.classified):
            return None
        interval = max(pd.Timedelta("1s"), (s.zoom_end - s.zoom_start) / REPLAY_FRAMES)
        events = {"events": s.events} if s.classified else {}
        return _bound(
            _data.prepare_book_replay_data,
            depth=s.depth,
            trades=s.trades,
            level=level,
            start_time=s.zoom_start,
            end_time=s.zoom_end,
            interval=interval,
            volume_scale=s.volume_scale,
            **events,
        )

    return bind


def _order_outcome(s: Scene) -> Binding | None:
    # It needs the distance from the touch at placement, the per-event bps.
    if not s.per_order or "aggressiveness_bps" not in s.events.columns:
        return None
    return _bound(
        _data.prepare_order_outcome_data,
        events=s.events,
        volume_scale=s.volume_scale,
    )


def _queue_position(s: Scene) -> Binding | None:
    if not s.per_order:
        return None
    # The zoom window: a whole session of touch orders smears into a band at
    # the front, while a few minutes show single orders climbing the queue.
    return _bound(
        _data.prepare_queue_position_data,
        events=s.events,
        start_time=s.zoom_start,
        end_time=s.zoom_end,
    )


def _liquidity_at_touch_l2(s: Scene) -> Binding | None:
    if s.depth_summary.empty or not s.per_order:
        return None
    return _bound(
        _data.prepare_liquidity_at_touch_l2_data,
        depth_summary=s.depth_summary,
        events=s.events,
        start_time=s.summary_start,
        end_time=s.summary_end,
        volume_scale=s.volume_scale,
    )


def _liquidity_at_touch_l3(s: Scene) -> Binding | None:
    if s.depth_summary.empty or not s.per_order:
        return None
    return _bound(
        _data.prepare_liquidity_at_touch_l3_data,
        events=s.events,
        start_time=s.zoom_start,
        end_time=s.zoom_end,
    )


def _volume_percentiles(s: Scene) -> Binding | None:
    if s.depth_summary.empty:
        return None
    return _bound(
        _data.prepare_volume_percentiles_data,
        depth_summary=s.depth_summary,
        start_time=s.summary_start,
        end_time=s.summary_end,
        volume_scale=s.volume_scale,
    )


def _price_view(s: Scene) -> Binding | None:
    if s.depth_summary.empty:
        return None
    return _bound(
        _data.prepare_price_view_data,
        depth_summary=s.depth_summary,
        trades=s.trades,
        start_time=s.summary_start,
        end_time=s.summary_end,
    )


def _events_histogram(s: Scene) -> Binding | None:
    if not s.per_order:
        return None
    events = s.events[["timestamp", "direction", "price", "volume"]].copy()
    events["volume"] = events["volume"] * s.volume_scale
    # The bin width is a hundredth of the focus band, so the near-touch
    # distribution stays readable; the prepare step clips to the band.
    if s.focus.price_from is not None and s.focus.price_to is not None:
        price_window = s.focus.price_to - s.focus.price_from
    else:  # no focus band (a degenerate tape): use the tail span
        price_window = events["price"].quantile(0.99) - events["price"].quantile(0.01)
    return _bound(
        _data.prepare_events_histogram_data,
        events=events,
        val="price",
        bw=max(0.01, round(price_window / 100, 2)),
        **s.price_band,
    )


def _trade_size(s: Scene) -> Binding | None:
    if s.trades.empty:
        return None
    return _bound(
        _data.prepare_trade_size_data, trades=s.trades, volume_scale=s.volume_scale
    )


def _hidden_executions(s: Scene) -> Binding | None:
    # LOBSTER marks executions against hidden orders as event type 5.
    if "raw_event_type" not in s.events.columns:
        return None
    if not (s.events["raw_event_type"] == 5).any():
        return None
    return _bound(
        _data.prepare_hidden_executions_data, events=s.events, trades=s.trades
    )


# How the depth heatmap picks its levels, said in its note.
_HIDDEN_LEVELS_NOTE = (
    "Levels that do not change in the window are left out unless no level "
    'changes; plot_result(result, "depth_heatmap", show_all_depth=True) '
    "keeps them all."
)

#: Every plot concept, in gallery order.
CONCEPTS: tuple[Concept, ...] = (
    Concept(
        "trade_tape",
        {
            L2: Face("trade_tape_aggregate", _ALL, _trade_tape_l2),
            L3: Face("trade_tape_per_order", _ALL, _trade_tape_l3),
        },
        "Trade Tape",
        "Executions over time. L2: price as a step line. L3: each execution "
        "dot sits at the end of a bar tracing how long the consumed maker "
        "order had rested; color = aggressor side.",
    ),
    Concept(
        "depth_heatmap",
        {L2: Face("depth_heatmap", _ALL, _depth_heatmap)},
        "Depth Heatmap",
        "Resting liquidity through time: one horizontal line per price level, "
        "colored by available volume; the pale line is the midprice. "
        "Triangles mark executions (aggressor side). Gaps mean the level "
        "emptied. Pass col_bias<1 to brighten thin levels and reveal "
        "near-touch structure. On a per-order run, diamonds mark suspected "
        "iceberg refills (joined by a line per iceberg) and stars mark trades "
        f"against hidden orders, inside the zoom window. {_HIDDEN_LEVELS_NOTE}",
    ),
    Concept(
        "order_activity",
        {
            L2: Face("order_activity_aggregate", _MPL_PLOTLY, _order_activity_l2),
            L3: Face("order_activity_per_order", _MPL_PLOTLY, _order_activity_l3),
        },
        "Order Activity",
        "Order placement and removal. L2: created/deleted events scattered at "
        "their price. L3: each order is one lifespan from placement to "
        "outcome - orange = pulled (flashed), green = rested/filled. L3 also "
        "overlays iceberg refills and trades against hidden orders.",
    ),
    Concept(
        "cancellations",
        {
            L2: Face("cancellations_aggregate", _MPL_PLOTLY, _cancellations_l2),
            L3: Face("cancellations_per_order", _MPL_PLOTLY, _cancellations_l3),
        },
        "Cancellations",
        "Pulled orders. L2: cancelled volume over time (log scale). L3: every "
        "cancellation as how-long-it-rested vs how-far-from-the-touch-it-sat, "
        "sized by volume.",
    ),
    Concept(
        "book_snapshot",
        {
            L2: Face("book_snapshot_aggregate", _ALL, _book(L2)),
            L3: Face("book_snapshot_per_order", _ALL, _book(L3)),
        },
        "Book Snapshot ({snapshot})",
        "The resting book at one instant. L2: total size per price level, "
        "from the depth table. L3: one segment per order, from the book "
        "rebuilt from the events. One large order and many small ones of the "
        "same total size look the same on L2 and different on L3.",
    ),
    Concept(
        "depth_chart",
        {
            L2: Face("depth_chart_aggregate", _ALL, _book(L2)),
            L3: Face("depth_chart_per_order", _ALL, _book(L3)),
        },
        "Depth Chart ({snapshot})",
        "Cumulative size from the touch outwards: how far the price moves to "
        "fill X units. L2 reads the depth table; L3 marks each order along "
        "the curve.",
    ),
    Concept(
        "book_replay",
        {
            L2: Face("book_replay_aggregate", _PLOTLY, _book_replay(L2)),
            L3: Face("book_replay_per_order", _PLOTLY, _book_replay(L3)),
        },
        "Book Replay",
        "The book through the zoom window. Drag the slider or press a play "
        "button (1x is real time); the line on the trades panel marks the "
        "time shown. Click a trade to move there: its price level is marked "
        "in the trade's colour. L2: one bar per price level, from the depth "
        "table. L3: one segment per order. Both remove stale crossed orders, "
        "so the trades sit around the mid. Drawn by Plotly only.",
    ),
    Concept(
        "order_outcome",
        {L3: Face("order_outcome_per_order", _MPL_PLOTLY, _order_outcome)},
        "Order Outcome",
        "Where orders were placed vs how they ended. x = signed distance from "
        "the best price at placement (right of 0 improved the touch), y = "
        "size; green filled, purple partial, orange cancelled. Still-resting "
        "orders are not shown.",
    ),
    Concept(
        "queue_position",
        {L3: Face("queue_position_per_order", _MPL_PLOTLY, _queue_position)},
        "Queue Position",
        "Each line is a resting order's FIFO rank at the touch over time, "
        "marching toward the front (rank 1, at the top) as the orders ahead "
        "fill or cancel; colour = how it ended.",
    ),
    Concept(
        "liquidity_at_touch",
        {
            L2: Face(
                "liquidity_at_touch_aggregate", _MPL_PLOTLY, _liquidity_at_touch_l2
            ),
            L3: Face(
                "liquidity_at_touch_per_order", _MPL_PLOTLY, _liquidity_at_touch_l3
            ),
        },
        "Liquidity at the Touch",
        "Liquidity a marketable order meets first. L2: resting size at the "
        "best bid/ask, with an event rug beneath (gray = created, amber = "
        "cancelled, green = filled) marking when the book churned. L3: the "
        "touch queue's composition - each cell is the age of the order at "
        "that rank; pale = recent churn, dark = sticky liquidity.",
    ),
    Concept(
        "volume_percentiles",
        {L2: Face("volume_percentiles", _MPL_PLOTLY, _volume_percentiles)},
        "Volume Percentiles",
        "Book depth through time, stacked by distance from the touch in bps "
        "bands: asks above zero, bids below; band thickness = resting volume "
        "in that bps ring.",
    ),
    Concept(
        "price_view",
        {L2: Face("price_view", _MPL_PLOTLY, _price_view)},
        "Price View",
        "The spread as a ribbon (best bid to best ask) with the "
        "volume-weighted microprice through it - it leans toward the side "
        "with the heavier opposite book (where price is likely to go). Dots "
        "are executions by aggressor side.",
    ),
    Concept(
        "events_histogram",
        {L2: Face("events_histogram", _MPL_PLOTLY, _events_histogram)},
        "Events Histogram (price)",
        "Where order activity concentrated: event counts binned by price "
        "within the focus window, split by side.",
    ),
    Concept(
        "trade_size",
        {L2: Face("trade_size", _MPL_PLOTLY, _trade_size)},
        "Trade Size",
        "How big trades are: each execution a jittered dot on a log size "
        "axis, buys and sells in separate bands. Trade size is heavy-tailed, "
        "so the log axis spreads the bulk while keeping the rare block trades "
        "on-scale.",
    ),
    Concept(
        "hidden_executions",
        {L2: Face("hidden_executions", _MPL_PLOTLY, _hidden_executions)},
        "Hidden Executions",
        "LOBSTER-only: executions against hidden orders (type 5) scattered "
        "over the trade price line - liquidity that never showed in the "
        "visible book.",
    ),
    # Drawn by plot() only: no gallery card.
    Concept("time_series", {L2: Face("time_series", _MPL_PLOTLY)}),
    # Level-less analytics.  The gallery draws a metric from its own panel
    # and the others from the *_panel helpers, so none binds from a scene.
    *(
        Concept(key, {None: Face(key, _MPL_PLOTLY)})
        for key in (
            "l1_ticker",
            "book_signals",
            "bars",
            "vpin",
            "order_flow_imbalance",
            "transaction_costs",
            "ofi_by_window",
            "kyle_lambda",
            "trading_halts",
        )
    ),
    # One value per window, drawn by the shared window-measure renderer.
    *(
        Concept(key, {None: Face("window_measure", _MPL_PLOTLY)})
        for key in ("amihud", "roll_spread")
    ),
)

#: The same table keyed by the coordinate ``(concept, level, backend)``.
PLOTS: dict[RendererKey, Face] = {
    (concept.key, level, backend): face
    for concept in CONCEPTS
    for level, face in concept.faces.items()
    for backend in face.backends
}

_BY_KEY: dict[str, Concept] = {concept.key: concept for concept in CONCEPTS}


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------


def concept(key: str) -> Concept | None:
    """The table's entry for *key*, or ``None`` when the table has none."""
    return _BY_KEY.get(key)


def placements(key: str) -> tuple[tuple[Level | None, str], ...]:
    """Each ``(level, backend)`` *key* is drawn at, in order, without repeats.

    The table's coordinates first, then any a plug-in registered straight
    into :data:`~ob_analytics.visualization.RENDERERS`.
    """
    declared = [(lvl, b) for (c, lvl, b) in PLOTS if c == key]
    return tuple(dict.fromkeys([*declared, *RENDERERS.placements(key)]))


def backends(key: str, level: Level | None) -> tuple[str, ...]:
    """The backends that draw *key* at *level*."""
    return tuple(b for lvl, b in placements(key) if lvl == level)


def levels(key: str, backend: str) -> tuple[Level | None, ...]:
    """The levels *backend* draws *key* at."""
    return tuple(lvl for lvl, b in placements(key) if b == backend)


def _missing(key: str, level: Level | None, backend: str, *, any_level: bool):
    concepts = {c for c, _, _ in [*PLOTS, *RENDERERS.list()]}
    return missing_plot(
        key, level, backend, placements(key), sorted(concepts), any_level=any_level
    )


def require_backend(key: str, backend: str) -> None:
    """Raise unless *backend* draws *key* at some level.

    Raises
    ------
    UnknownNameError
        If *backend* does not draw *key*.  The message names the backends
        that do, or the plots closest to *key* when it is not a plot.
    """
    if not levels(key, backend):
        raise _missing(key, None, backend, any_level=True)


def require(key: str, level: Level | None, backend: str) -> None:
    """Raise unless *backend* draws *key* at *level*.

    Raises
    ------
    UnknownNameError
        If *backend* does not draw *key* at *level*.  The message names the
        backends that do, or, when none does, the levels *key* is drawn at.
    """
    if backend not in backends(key, level):
        raise _missing(key, level, backend, any_level=False)


def register_backend(backend: str, namespace: Mapping[str, Any]) -> None:
    """Register *backend*'s renderer for every coordinate the table gives it.

    A built-in backend module calls this once, with its own ``globals()``,
    after it defines its renderers.

    Raises
    ------
    KeyError
        If *namespace* lacks a renderer the table names for *backend*.
    """
    prefix = BACKEND_PREFIXES[backend]
    for (key, level, b), face in PLOTS.items():
        if b == backend:
            RENDERERS.register((key, level, b), namespace[f"{prefix}_{face.renderer}"])
