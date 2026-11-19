"""Every plot the package draws, declared once.

A plot is a coordinate ``(concept, level, backend)``: what it shows, at which
resolution (``Level.L2``, ``Level.L3``, or ``None`` for a level-less
analytic), and which library draws it.  :data:`CONCEPTS` declares each
concept once: its gallery title and note, and at each level the backends
that draw it, the data it needs, and how the gallery binds that data from a
run.  :data:`PLOTS` is the same table keyed by the coordinate.

An L2 face needs only aggregated data: the depth table, the depth summary
and the trades.  So every L2 face draws on a price-level run, and on a
per-order run it shows the aggregated view of the same book.  A face that
needs per-order events is an L3 face.

The gallery, :func:`~ob_analytics.visualization.plot_result` and
:func:`~ob_analytics.visualization.plot` read this table to know where a plot
is drawn, so a backend a plot does not have is a declared fact (the book
replay is Plotly only) rather than a missing registration.  The backend
modules only draw: each marks its renderers with
:func:`~ob_analytics.visualization._renderers.renders`, which registers a
function at its coordinate when the module is imported.  A test checks that
the built-in backends register exactly the coordinates of this table.

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

from ob_analytics.book import order_book, order_lifecycles
from ob_analytics.depth import price_level_snapshots
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import Level
from ob_analytics.queue import queue_positions
from ob_analytics.visualization import _data
from ob_analytics.visualization._renderers import (
    RENDERERS,
    RendererKey,
    missing_plot,
)

L2, L3 = Level.L2, Level.L3

#: The backends this package ships.
BUILTIN_BACKENDS: tuple[str, ...] = ("matplotlib", "plotly", "bokeh")

_ALL = BUILTIN_BACKENDS
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

    @cached_property
    def lifecycles(self) -> pd.DataFrame:
        """One row per order (:func:`~ob_analytics.book.order_lifecycles`)."""
        return order_lifecycles(self.events)

    @cached_property
    def queue(self) -> pd.DataFrame:
        """The touch queue (:func:`~ob_analytics.queue.queue_positions`)."""
        return queue_positions(self.events, levels="touch")

    @property
    def snapshot_label(self) -> str:
        """The time of the book faces, for their titles."""
        return self.zoom_end.strftime("%H:%M")


@dataclass(frozen=True)
class Binding:
    """A face's data for one run: the prepare function and its keywords."""

    prepare: Callable[..., dict]
    kwargs: dict[str, Any]


#: Binds a face's data from a scene that has what the face needs.
Binder = Callable[[Scene], Binding]


#: What a table is called in a sentence that says a run lacks it.
_TABLE_TEXT: dict[str, str] = {
    "events": "per-order events",
    "trades": "trades",
    "depth": "a depth table",
    "depth_summary": "a depth summary",
}


@dataclass(frozen=True)
class Needs:
    """The data a face reads from a run.

    The counterpart of a metric's ``levels`` and ``needs`` and a feature's
    ``requires``: what the face needs is declared here, not found in its
    binder's code.  The gallery, :func:`~ob_analytics.visualization.available_concepts`
    and :func:`~ob_analytics.visualization.plot_result` read it to know
    whether a run can draw the face, and the gallery card says what is
    missing when it cannot.

    Attributes
    ----------
    tables : tuple of str
        The tables that must have rows: ``"events"``, ``"trades"``,
        ``"depth"`` or ``"depth_summary"``.  ``"events"`` means per-order
        events, which only an L3 run has.
    uses : tuple of str
        Tables the face also reads when they have rows, and draws without
        when they do not (an event strip, a trade overlay).
    classified : bool
        Whether the events must carry the order ``type`` that
        :func:`~ob_analytics.analytics.set_order_types` adds.
    columns : tuple of str
        Event columns the face reads, e.g. ``"raw_event_type"``, which only
        LOBSTER data fills.  A column that holds no value counts as missing.
    """

    tables: tuple[str, ...]
    uses: tuple[str, ...] = ()
    classified: bool = False
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = (set(self.tables) | set(self.uses)) - _TABLE_TEXT.keys()
        if unknown:
            raise ConfigError(
                f"Unknown tables {sorted(unknown)}; use {list(_TABLE_TEXT)}"
            )
        if (self.classified or self.columns) and "events" not in self.tables:
            raise ConfigError("A face that reads event columns needs the events table.")

    @property
    def per_order(self) -> bool:
        """Whether the face needs per-order events (an L3 run)."""
        return "events" in self.tables

    def unmet(self, run: Any) -> str | None:
        """Say what *run* lacks for this face, or return ``None`` when it has it all.

        The text is a clause, e.g. ``"it needs per-order events, and this
        run has none"``, like :func:`~ob_analytics.metrics.skip_reason`.

        *run* is anything with the four tables as attributes: a
        :class:`~ob_analytics.pipeline.PipelineResult` or a :class:`Scene`.
        """
        empty = [_TABLE_TEXT[t] for t in self.tables if getattr(run, t).empty]
        if empty:
            return f"it needs {' and '.join(empty)}, and this run has none"
        events = run.events
        if self.classified and "type" not in events.columns:
            return (
                "it needs events with order types (set_order_types adds them), "
                "and this run's events have none"
            )
        missing = [
            c
            for c in self.columns
            if c not in events.columns or not events[c].notna().any()
        ]
        if missing:
            names = " and ".join(missing)
            return f"it needs the events' {names}, and this run's events lack it"
        return None


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Face:
    """One concept at one level.

    Attributes
    ----------
    backends : tuple of str
        The backends that draw this face.
    needs : Needs or None
        The data the face reads from a run.  ``None`` exactly when *bind* is.
    bind : callable or None
        How the gallery binds the face's data from a :class:`Scene` that has
        what *needs* says.  ``None`` for a face the gallery does not build
        itself: a metric (it has its own card) or a plot drawn only by
        :func:`~ob_analytics.visualization.plot`.
    """

    backends: tuple[str, ...]
    needs: Needs | None = None
    bind: Binder | None = None

    def __post_init__(self) -> None:
        if (self.needs is None) != (self.bind is None):
            raise ConfigError(
                "A face declares needs without a binder, or a binder without needs."
            )


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


def _with_scene_tables(
    prepare: Callable[..., dict], s: Scene, *tables: str
) -> Callable[..., dict]:
    """*prepare*, handed the scene's per-order *tables* when the card is drawn.

    The tables are read from *s* only then, so a model that never draws the
    card does not build them, and every card of one scene shares one copy.
    """

    def call(**kwargs: Any) -> dict:
        return prepare(**{**{name: getattr(s, name) for name in tables}, **kwargs})

    return call


# Each binder is called only for a scene that has what its face needs (see
# Needs.unmet), so none checks the run itself.


def _trade_tape_l2(s: Scene) -> Binding:
    return _bound(_data.prepare_trade_tape_l2_data, trades=s.trades)


def _trade_tape_l3(s: Scene) -> Binding:
    return _bound(
        _data.prepare_trade_tape_l3_data,
        events=s.events,
        trades=s.trades,
    )


def _depth_heatmap(s: Scene) -> Binding:
    return _bound(
        _data.prepare_depth_heatmap_data,
        depth=s.depth,
        spread=s.spread,
        trades=s.trades,
        **s.price_band,
        **s.overlay,
    )


def _order_activity_l2(s: Scene) -> Binding:
    # The depth heatmap's price band, so the map clips around the touch:
    # changes at far levels would otherwise squash the activity.
    return _bound(
        _data.prepare_order_activity_l2_data,
        depth=s.depth,
        **s.price_band,
    )


def _order_activity_l3(s: Scene) -> Binding:
    return _bound(
        _with_scene_tables(_data.prepare_order_activity_l3_data, s, "lifecycles"),
        events=s.events,
        **s.price_band,
        **s.overlay,
    )


def _cancellations_l3(s: Scene) -> Binding:
    return _bound(
        _with_scene_tables(_data.prepare_cancellations_l3_data, s, "lifecycles"),
        events=s.events,
    )


def _book(level: Level) -> Binder:
    """The book faces read the book of their own level (see ``book_snapshot``)."""

    def bind(s: Scene) -> Binding:
        book = s.price_level_book if level is L2 else s.per_order_book
        return _bound(
            _data.prepare_book_snapshot_data,
            order_book=book,
            level=level,
        )

    return bind


#: The most frames the gallery's book replay holds: enough to scrub the zoom
#: window smoothly, few enough to keep the page small and quick to build.
REPLAY_FRAMES = 300


def _book_replay(level: Level) -> Binder:
    def bind(s: Scene) -> Binding:
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
            **events,
        )

    return bind


def _order_outcome(s: Scene) -> Binding:
    return _bound(
        _with_scene_tables(_data.prepare_order_outcome_data, s, "lifecycles"),
        events=s.events,
    )


def _queue_position(s: Scene) -> Binding:
    # The zoom window: a whole session of touch orders smears into a band at
    # the front, while a few minutes show single orders climbing the queue.
    return _bound(
        _with_scene_tables(_data.prepare_queue_position_data, s, "queue", "lifecycles"),
        events=s.events,
        start_time=s.zoom_start,
        end_time=s.zoom_end,
    )


def _liquidity_at_touch_l2(s: Scene) -> Binding:
    # The event strip beneath the series is drawn on a per-order run only.
    events = {"events": s.events} if s.per_order else {}
    return _bound(
        _data.prepare_liquidity_at_touch_l2_data,
        depth_summary=s.depth_summary,
        start_time=s.summary_start,
        end_time=s.summary_end,
        **events,
    )


def _liquidity_at_touch_l3(s: Scene) -> Binding:
    return _bound(
        _data.prepare_liquidity_at_touch_l3_data,
        events=s.events,
        start_time=s.zoom_start,
        end_time=s.zoom_end,
    )


def _depth_bands(s: Scene) -> Binding:
    return _bound(
        _data.prepare_depth_bands_data,
        depth_summary=s.depth_summary,
        start_time=s.summary_start,
        end_time=s.summary_end,
    )


def _price_view(s: Scene) -> Binding:
    return _bound(
        _data.prepare_price_view_data,
        depth_summary=s.depth_summary,
        trades=s.trades,
        start_time=s.summary_start,
        end_time=s.summary_end,
    )


def _prepare_level_change_histogram(
    depth: pd.DataFrame,
    *,
    val: str = "price",
    bw: float | None = None,
    price_from: float | None = None,
    price_to: float | None = None,
    **kwargs: Any,
) -> dict:
    """The events histogram of the price-level changes: where the book changed.

    The changes are found here, when the card is drawn, not when the gallery
    model is built, so a model that never draws this card costs nothing.
    """
    changes = _data.price_level_changes(depth)
    changes = changes[["timestamp", "direction", "price", "volume"]]
    if bw is None:
        # The bin width is a hundredth of the focus band, so the near-touch
        # distribution stays readable; the prepare step clips to the band.
        if price_from is not None and price_to is not None:
            price_window = price_to - price_from
        else:  # no focus band (a degenerate tape): use the tail span
            prices = changes["price"]
            price_window = prices.quantile(0.99) - prices.quantile(0.01)
        bw = max(0.01, round(float(price_window) / 100, 2))
    return _data.prepare_events_histogram_data(
        changes,
        val=val,
        bw=bw,
        price_from=price_from,
        price_to=price_to,
        **kwargs,
    )


def _events_histogram(s: Scene) -> Binding:
    return _bound(_prepare_level_change_histogram, depth=s.depth, **s.price_band)


def _trade_size(s: Scene) -> Binding:
    return _bound(_data.prepare_trade_size_data, trades=s.trades)


def _hidden_executions(s: Scene) -> Binding:
    # LOBSTER marks executions against hidden orders as event type 5; a run
    # with none draws the trade price only.
    return _bound(
        _data.prepare_hidden_executions_data, events=s.events, trades=s.trades
    )


# How the depth heatmap picks its levels, said in its note.
_HIDDEN_LEVELS_NOTE = (
    "Levels that do not change in the window are left out unless no level "
    'changes; plot_result(result, "depth_heatmap", show_all_depth=True) '
    "keeps them all."
)

# What the book faces read: the depth table at L2, the book rebuilt from the
# classified events at L3.
_BOOK_L2 = Needs(("depth",))
_BOOK_L3 = Needs(("events",), classified=True)

#: Every plot concept, in gallery order.
CONCEPTS: tuple[Concept, ...] = (
    Concept(
        "trade_tape",
        {
            L2: Face(_ALL, Needs(("trades",)), _trade_tape_l2),
            L3: Face(
                _ALL,
                Needs(("events", "trades")),
                _trade_tape_l3,
            ),
        },
        "Trade Tape",
        "Executions over time. L2: price as a step line. L3: each execution "
        "dot sits at the end of a bar tracing how long the consumed maker "
        "order had rested; color = aggressor side.",
    ),
    Concept(
        "depth_heatmap",
        {
            L2: Face(
                _ALL,
                Needs(("depth",), uses=("trades",)),
                _depth_heatmap,
            )
        },
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
            L2: Face(
                _MPL_PLOTLY,
                Needs(("depth",)),
                _order_activity_l2,
            ),
            L3: Face(
                _MPL_PLOTLY,
                Needs(("events",), classified=True),
                _order_activity_l3,
            ),
        },
        "Order Activity",
        "Order placement and removal. L2: size added to (filled dot) and "
        "removed from (outlined dot) each price level, at its price, from the "
        "depth table; a removal can be a cancel or a fill. L3: each order is "
        "one lifespan from placement to "
        "outcome - orange = pulled (cancelled), green = rested/filled. L3 also "
        "overlays iceberg refills and trades against hidden orders.",
    ),
    Concept(
        "cancellations",
        {
            L3: Face(
                _MPL_PLOTLY,
                Needs(("events",), classified=True, columns=("aggressiveness_bps",)),
                _cancellations_l3,
            )
        },
        "Cancellations",
        "Pulled orders: every cancellation as how-long-it-rested vs "
        "how-far-from-the-touch-it-sat, sized by volume. A price level has no "
        "order identity, so this needs per-order events.",
    ),
    Concept(
        "book_snapshot",
        {
            L2: Face(_ALL, _BOOK_L2, _book(L2)),
            L3: Face(_ALL, _BOOK_L3, _book(L3)),
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
            L2: Face(_ALL, _BOOK_L2, _book(L2)),
            L3: Face(_ALL, _BOOK_L3, _book(L3)),
        },
        "Depth Chart ({snapshot})",
        "Cumulative size from the touch outwards: how far the price moves to "
        "fill X units. L2 reads the depth table; L3 marks each order along "
        "the curve.",
    ),
    Concept(
        "book_replay",
        {
            L2: Face(
                _PLOTLY,
                Needs(("depth",), uses=("trades", "events")),
                _book_replay(L2),
            ),
            L3: Face(
                _PLOTLY,
                Needs(("depth", "events"), uses=("trades",), classified=True),
                _book_replay(L3),
            ),
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
        {
            L3: Face(
                _MPL_PLOTLY,
                Needs(("events",), columns=("aggressiveness_bps",)),
                _order_outcome,
            )
        },
        "Order Outcome",
        "Where orders were placed vs how they ended. x = signed distance from "
        "the best price at placement (right of 0 improved the touch), y = "
        "size; green filled, purple partial, orange cancelled. Still-resting "
        "orders are not shown.",
    ),
    Concept(
        "queue_position",
        {
            L3: Face(
                _MPL_PLOTLY,
                Needs(("events",), classified=True),
                _queue_position,
            )
        },
        "Queue Position",
        "Each line is a resting order's FIFO rank at the touch over time, "
        "marching toward the front (rank 1, at the top) as the orders ahead "
        "fill or cancel; colour = how it ended.",
    ),
    Concept(
        "liquidity_at_touch",
        {
            L2: Face(
                _MPL_PLOTLY,
                Needs(("depth_summary",), uses=("events",)),
                _liquidity_at_touch_l2,
            ),
            L3: Face(
                _MPL_PLOTLY,
                Needs(("events",), classified=True),
                _liquidity_at_touch_l3,
            ),
        },
        "Liquidity at the Touch",
        "Liquidity a marketable order meets first. L2: resting size at the "
        "best bid/ask. On a per-order run, an event rug beneath (gray = "
        "created, amber = cancelled, green = filled) marks when the book "
        "churned. L3: the "
        "touch queue's composition - each cell is the age of the order at "
        "that rank; pale = recent churn, dark = sticky liquidity.",
    ),
    Concept(
        "depth_bands",
        {
            L2: Face(
                _MPL_PLOTLY,
                Needs(("depth_summary",)),
                _depth_bands,
            )
        },
        "Depth Bands",
        "Book depth through time, stacked by distance from the touch in bps "
        "bands: asks above zero, bids below; band thickness = resting volume "
        "in that bps ring.",
    ),
    Concept(
        "price_view",
        {
            L2: Face(
                _MPL_PLOTLY,
                Needs(("depth_summary",), uses=("trades",)),
                _price_view,
            )
        },
        "Price View",
        "The spread as a ribbon (best bid to best ask) with the "
        "volume-weighted microprice through it - it leans toward the side "
        "with the heavier opposite book (where price is likely to go). Dots "
        "are executions by aggressor side.",
    ),
    Concept(
        "events_histogram",
        {L2: Face(_MPL_PLOTLY, Needs(("depth",)), _events_histogram)},
        "Events Histogram (price)",
        "Where the book changed: the number of price-level changes in the "
        "depth table, binned by price within the focus window, split by side.",
    ),
    Concept(
        "trade_size",
        {L2: Face(_MPL_PLOTLY, Needs(("trades",)), _trade_size)},
        "Trade Size",
        "How big trades are: each execution a jittered dot on a log size "
        "axis, buys and sells in separate bands. Trade size is heavy-tailed, "
        "so the log axis spreads the bulk while keeping the rare block trades "
        "on-scale.",
    ),
    Concept(
        "hidden_executions",
        {
            L3: Face(
                _MPL_PLOTLY,
                Needs(("events",), uses=("trades",), columns=("raw_event_type",)),
                _hidden_executions,
            )
        },
        "Hidden Executions",
        "LOBSTER-only: executions against hidden orders (type 5) scattered "
        "over the trade price line - liquidity that never showed in the "
        "visible book.",
    ),
    # Level-less analytics, and time_series, which plot() alone draws.  The
    # gallery draws a metric from its own card and the others from the
    # *_card helpers, so none binds from a scene.
    *(
        Concept(key, {None: Face(_MPL_PLOTLY)})
        for key in (
            "time_series",
            "l1_ticker",
            "book_signals",
            "bars",
            "vpin",
            "trade_imbalance",
            "transaction_costs",
            "trade_imbalance_by_window",
            "kyle_lambda",
            "trading_halts",
            "amihud",
            "roll_spread",
        )
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
