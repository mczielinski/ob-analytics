"""Tiny synthetic datasets for teaching and testing.

This module ships one hand-written order-book session small enough to
verify with mental arithmetic: 24 events, 12 orders, 5 trades, prices
98–103 around a mid of 100, sizes 1–3, spanning one synthetic minute.
The tutorial builds every microstructure concept on this stream before
touching real data; the test suite uses it as a readable fixture.

The frames follow the canonical schemas (see :mod:`ob_analytics.schemas`)
and the exact conventions of :class:`~ob_analytics.bitstamp.BitstampLoader`
output, so they flow through the real pipeline stages —
:func:`~ob_analytics.analytics.set_order_types`,
:func:`~ob_analytics.depth.price_level_volume`,
:func:`~ob_analytics.analytics.order_book` — and the real plot faces.
An extra ``actor`` column (and ``maker_actor`` / ``taker_actor`` on
trades) names each order for annotation; extra columns are permitted by
every schema validator.

The script
----------

======  =======  ======  =========  ====  ======  =======  ====
event   t (s)    actor   action     side  price   volume   fill
======  =======  ======  =========  ====  ======  =======  ====
1       0        Alice   created    bid   99      2        0
2       2        Bob     created    ask   101     3        0
3       5        Chen    created    bid   98      1        0
4       6        Ivy     created    bid   99      2        0
5       8        Dana    created    bid   98      3        0
6       12       Erin    created    ask   102     2        0
7       20       Frank   created    bid   101     1        0
8       20       Bob     changed    ask   101     2        1
9       20       Frank   deleted    bid   101     0        1
10      35       Gus     created    ask   103     2        0
11      40       Dana    deleted    bid   98      3        0
12      45.0     Eve     created    bid   100     1        0
13      45.8     Eve     deleted    bid   100     1        0
14      48       Hana    created    bid   101     3        0
15      48       Bob     deleted    ask   101     0        2
16      48       Hana    changed    bid   101     1        2
17      52       Iris    created    ask   101     1        0
18      52       Hana    deleted    bid   101     0        1
19      52       Iris    deleted    ask   101     0        1
20      56       Sam     created    ask   99      3        0
21      56       Alice   deleted    bid   99      0        2
22      56       Sam     changed    ask   99      1        2
23      57       Ivy     changed    bid   99      1        1
24      57       Sam     deleted    ask   99      0        1
======  =======  ======  =========  ====  ======  =======  ====

What it contains, by design:

* a **queue** at 99 (Alice before Ivy — price–time priority pays off at
  t=56/57, when Sam's sweep fills Alice fully and Ivy only partially);
* a **market buy** (Frank crosses the spread at t=20, partially filling
  Bob) and a **market sell sweep** (Sam, two fills at t=56–57);
* a **market-limit** order (Hana crosses for 2 at t=48, rests 1 at 101,
  and is later filled by Iris at t=52);
* a **flash** (Eve posts and pulls within 800 ms) and a **plain
  cancellation** (Dana at t=40 — note the classifier labels any
  unfilled create-then-cancel ``flashed-limit`` regardless of resting
  time, so Dana and Eve classify identically);
* **resting limits** that never trade (Chen, Erin, Gus survive to the
  end of the stream).

:func:`toy_orders` returns the same session as a script of twelve
:class:`ToyOrder` entries, one per actor, and :func:`match_toy_orders` matches a
script by price–time priority.  Edit the script — add an order, remove one,
change when one is cancelled — and match it to get the events and trades of
the session you wrote.

Under :func:`~ob_analytics.analytics.set_order_types` the twelve orders
classify with no ``unknown`` leftovers: Alice, Bob, Chen, Ivy, Erin, Gus
→ ``resting-limit``; Frank, Iris, Sam → ``market``; Hana →
``market-limit``; Dana, Eve → ``flashed-limit``.

At t=30 the book is: bids 4 @ 99 (Alice 2, Ivy 2) and 4 @ 98 (Chen 1,
Dana 3); asks 2 @ 101 (Bob) and 2 @ 102 (Erin). Best bid 99, best ask
101, spread 2, mid 100.

Volume/fill semantics match the canonical contract: ``volume`` is the
outstanding size after the event (``created``/``changed``) — a full fill
therefore ends in a ``deleted`` row with ``volume == 0`` and the executed
quantity in ``fill``, while a cancellation's ``deleted`` row carries the
cancelled size with ``fill == 0``.

Timestamps are tz-aware UTC nanoseconds (the schema's canonical time model)
starting from an arbitrary Monday morning; ``exchange_timestamp`` equals
``timestamp`` (as in LOBSTER sessions, where only exchange time exists).

Prices are stored as integer ticks and sizes as integer lots (the schema's
canonical price and size models).  The toy book has a tick size of ``1.0`` and a
lot size of ``1.0`` — the prices 98-103 and the sizes 1-3 are already whole
ticks and whole lots — so the stored integers read as the same numbers the
script above lists; multiply by ``TICK_SIZE`` (``1.0``) for the quote currency
and by ``LOT_SIZE`` (``1.0``) for the base asset.  Both constants apply to the
L2 samples below as well.  Anything that reads the toy frames back as floats —
an export to a backtesting engine, say — needs both: a
``PipelineConfig(tick_size=TICK_SIZE, lot_size=LOT_SIZE)``, not the config
defaults, whose ``lot_size`` of ``1e-8`` would scale a size of 2 to ``2e-08``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd

from ob_analytics.exceptions import ConfigError

__all__ = [
    "LOT_SIZE",
    "TICK_SIZE",
    "ToyOrder",
    "match_toy_orders",
    "toy_events",
    "toy_l2_depth",
    "toy_l2_trades",
    "toy_orders",
    "toy_trades",
]

#: Tick size of the toy book (issue #155).  ``1.0`` so the integer-tick prices
#: equal the whole-number prices in the script; the display price is
#: ``ticks * TICK_SIZE``.
TICK_SIZE = 1.0

#: Lot size of the toy book (issue #226).  ``1.0`` so the integer-lot sizes
#: equal the whole-number sizes in the script; the base-asset size is
#: ``lots * LOT_SIZE``.
LOT_SIZE = 1.0

_BASE = pd.Timestamp("2026-01-05 10:00:00")

_ACTOR_IDS: dict[str, int] = {
    "Alice": 1,
    "Bob": 2,
    "Chen": 3,
    "Ivy": 4,
    "Dana": 5,
    "Erin": 6,
    "Frank": 7,
    "Gus": 8,
    "Eve": 9,
    "Hana": 10,
    "Iris": 11,
    "Sam": 12,
}

# (event_id, seconds, actor, action, direction, price, volume, fill)
_EVENTS: tuple[tuple[int, float, str, str, str, float, float, float], ...] = (
    (1, 0.0, "Alice", "created", "bid", 99.0, 2.0, 0.0),
    (2, 2.0, "Bob", "created", "ask", 101.0, 3.0, 0.0),
    (3, 5.0, "Chen", "created", "bid", 98.0, 1.0, 0.0),
    (4, 6.0, "Ivy", "created", "bid", 99.0, 2.0, 0.0),
    (5, 8.0, "Dana", "created", "bid", 98.0, 3.0, 0.0),
    (6, 12.0, "Erin", "created", "ask", 102.0, 2.0, 0.0),
    (7, 20.0, "Frank", "created", "bid", 101.0, 1.0, 0.0),
    (8, 20.0, "Bob", "changed", "ask", 101.0, 2.0, 1.0),
    (9, 20.0, "Frank", "deleted", "bid", 101.0, 0.0, 1.0),
    (10, 35.0, "Gus", "created", "ask", 103.0, 2.0, 0.0),
    (11, 40.0, "Dana", "deleted", "bid", 98.0, 3.0, 0.0),
    (12, 45.0, "Eve", "created", "bid", 100.0, 1.0, 0.0),
    (13, 45.8, "Eve", "deleted", "bid", 100.0, 1.0, 0.0),
    (14, 48.0, "Hana", "created", "bid", 101.0, 3.0, 0.0),
    (15, 48.0, "Bob", "deleted", "ask", 101.0, 0.0, 2.0),
    (16, 48.0, "Hana", "changed", "bid", 101.0, 1.0, 2.0),
    (17, 52.0, "Iris", "created", "ask", 101.0, 1.0, 0.0),
    (18, 52.0, "Hana", "deleted", "bid", 101.0, 0.0, 1.0),
    (19, 52.0, "Iris", "deleted", "ask", 101.0, 0.0, 1.0),
    (20, 56.0, "Sam", "created", "ask", 99.0, 3.0, 0.0),
    (21, 56.0, "Alice", "deleted", "bid", 99.0, 0.0, 2.0),
    (22, 56.0, "Sam", "changed", "ask", 99.0, 1.0, 2.0),
    (23, 57.0, "Ivy", "changed", "bid", 99.0, 1.0, 1.0),
    (24, 57.0, "Sam", "deleted", "ask", 99.0, 0.0, 1.0),
)

# (seconds, price, volume, taker side, maker event, taker event)
_TRADES: tuple[tuple[float, float, float, str, int, int], ...] = (
    (20.0, 101.0, 1.0, "buy", 8, 9),  # Frank market-buys 1 from Bob
    (48.0, 101.0, 2.0, "buy", 15, 16),  # Hana crosses for 2 against Bob
    (52.0, 101.0, 1.0, "sell", 18, 19),  # Iris hits Hana's resting 1
    (56.0, 99.0, 2.0, "sell", 21, 22),  # Sam's sweep: Alice filled fully
    (57.0, 99.0, 1.0, "sell", 23, 24),  # Sam's sweep: Ivy filled partially
)


def toy_events() -> pd.DataFrame:
    """Return the toy session's canonical events DataFrame.

    24 events over one synthetic minute, in the exact column layout and
    dtypes of :class:`~ob_analytics.bitstamp.BitstampLoader` output
    (plus a non-canonical ``actor`` column naming each order). Rows are
    in chronological ``event_id`` order.

    Returns
    -------
    pandas.DataFrame
        Columns ``original_number``, ``id``, ``timestamp``,
        ``exchange_timestamp``, ``price``, ``volume``, ``action``,
        ``direction``, ``event_id``, ``fill``, ``raw_event_type``,
        ``actor``.

    Examples
    --------
    >>> from ob_analytics.datasets import toy_events, toy_trades
    >>> from ob_analytics.analytics import set_order_types
    >>> events = set_order_types(toy_events(), toy_trades())
    >>> sorted(events["type"].unique().dropna().astype(str))  # doctest: +SKIP
    ['flashed-limit', 'market', 'market-limit', 'resting-limit']
    """
    return _events_frame(_EVENTS, _ACTOR_IDS)


def _timestamps(seconds: list[float]) -> pd.Series:
    """Seconds from the start of the toy session, as tz-aware UTC timestamps."""
    return (
        pd.Series([_BASE + pd.Timedelta(milliseconds=round(s * 1000)) for s in seconds])
        .astype("datetime64[ns]")
        .dt.tz_localize("UTC")
    )


def _events_frame(
    rows: tuple[tuple[int, float, str, str, str, float, float, float], ...],
    actor_ids: dict[str, int],
) -> pd.DataFrame:
    """Build the canonical events frame from ``_EVENTS``-shaped *rows*."""
    event_id = np.array([e[0] for e in rows], dtype=np.int64)
    actors = [e[2] for e in rows]
    ts = _timestamps([e[1] for e in rows])

    return pd.DataFrame(
        {
            "original_number": event_id.copy(),
            "id": np.array([actor_ids[a] for a in actors], dtype=np.int64),
            "timestamp": ts,
            "exchange_timestamp": ts.copy(),
            # Integer ticks (issue #155); TICK_SIZE is 1.0, so ticks == price.
            "price": np.array([e[5] for e in rows], dtype=np.int64),
            # Integer lots (issue #226); LOT_SIZE is 1.0, so lots == size.
            "volume": np.array([e[6] for e in rows], dtype=np.int64),
            "action": pd.Categorical(
                [e[3] for e in rows],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "direction": pd.Categorical(
                [e[4] for e in rows],
                categories=["bid", "ask"],
                ordered=True,
            ),
            "event_id": event_id,
            "fill": np.array([e[7] for e in rows], dtype=np.int64),
            "raw_event_type": pd.NA,
            "actor": actors,
        }
    )


def toy_trades() -> pd.DataFrame:
    """Return the toy session's canonical trades DataFrame.

    Five trades consistent with :func:`toy_events`: each trade's
    ``maker_event_id`` / ``taker_event_id`` points at the event row
    carrying that fill, in the exact column layout of
    :class:`~ob_analytics.bitstamp.BitstampTradeReader` output (plus
    non-canonical ``maker_actor`` / ``taker_actor`` columns).

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp``, ``price``, ``volume``, ``direction``
        (taker side, ``buy``/``sell``), ``maker_event_id``,
        ``taker_event_id``, ``maker``, ``taker``, ``maker_og``,
        ``taker_og``, ``maker_actor``, ``taker_actor``.
    """
    return _trades_frame(_TRADES, toy_events())


def _trades_frame(
    rows: tuple[tuple[float, float, float, str, int, int], ...],
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical trades frame from ``_TRADES``-shaped *rows*.

    *events* is the events frame the rows' maker and taker event ids point
    into; it supplies the order ids and actors.
    """
    eid_to_oid = dict(zip(events["event_id"], events["id"]))
    eid_to_og = dict(zip(events["event_id"], events["original_number"]))
    oid_to_actor = dict(zip(events["id"], events["actor"]))

    maker_eid = [t[4] for t in rows]
    taker_eid = [t[5] for t in rows]
    maker = [eid_to_oid[e] for e in maker_eid]
    taker = [eid_to_oid[e] for e in taker_eid]

    return pd.DataFrame(
        {
            "timestamp": _timestamps([t[0] for t in rows]),
            "price": np.array([t[1] for t in rows], dtype=np.int64),
            "volume": np.array([t[2] for t in rows], dtype=np.int64),
            "direction": pd.Categorical(
                [t[3] for t in rows], categories=["buy", "sell"], ordered=True
            ),
            "maker_event_id": np.array(maker_eid, dtype=object),
            "taker_event_id": np.array(taker_eid, dtype=object),
            "maker": np.array(maker, dtype=np.int64),
            "taker": np.array(taker, dtype=np.int64),
            "maker_og": np.array([eid_to_og[e] for e in maker_eid], dtype=np.int64),
            "taker_og": np.array([eid_to_og[e] for e in taker_eid], dtype=np.int64),
            "maker_actor": [oid_to_actor[o] for o in maker],
            "taker_actor": [oid_to_actor[o] for o in taker],
        }
    )


# ---------------------------------------------------------------------------
# A toy book you can edit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyOrder:
    """One order in a toy session script, and when it is cancelled, if it is.

    An order whose price reaches the other side of the book is matched on
    arrival, so a market order is written as a limit order priced through the
    touch (Frank's bid at 101 is the toy session's market buy).

    Attributes
    ----------
    at : float
        Seconds from the start of the session when the order arrives.
    direction : {"bid", "ask"}
        The side of the book.
    price : int
        The limit price, in ticks (the toy book's tick size is ``1.0``).
    volume : int
        The size, in lots, at least 1.
    cancel_at : float, optional
        Seconds from the start when the owner cancels whatever is still
        resting.  Nothing happens at that time if the order has already
        filled.  ``None`` (the default) leaves the order in the book.
    """

    at: float
    direction: Literal["bid", "ask"]
    price: int
    volume: int
    cancel_at: float | None = None


# The toy session's twelve orders, in arrival order.  match_toy_orders turns
# these back into _EVENTS and _TRADES (see match_toy_orders for the one
# difference).
_ORDERS: dict[str, ToyOrder] = {
    "Alice": ToyOrder(0.0, "bid", 99, 2),
    "Bob": ToyOrder(2.0, "ask", 101, 3),
    "Chen": ToyOrder(5.0, "bid", 98, 1),
    "Ivy": ToyOrder(6.0, "bid", 99, 2),
    "Dana": ToyOrder(8.0, "bid", 98, 3, cancel_at=40.0),
    "Erin": ToyOrder(12.0, "ask", 102, 2),
    "Frank": ToyOrder(20.0, "bid", 101, 1),
    "Gus": ToyOrder(35.0, "ask", 103, 2),
    "Eve": ToyOrder(45.0, "bid", 100, 1, cancel_at=45.8),
    "Hana": ToyOrder(48.0, "bid", 101, 3),
    "Iris": ToyOrder(52.0, "ask", 101, 1),
    "Sam": ToyOrder(56.0, "ask", 99, 3),
}


def toy_orders() -> dict[str, ToyOrder]:
    """Return the toy session's twelve orders, keyed by actor, in arrival order.

    Change, add or remove entries, then pass the result to
    :func:`match_toy_orders` to get the events and trades of the session you
    wrote.  Each call returns a new dictionary, so editing it never changes
    the toy session itself.

    Returns
    -------
    dict of str to ToyOrder
        ``{"Alice": ToyOrder(...), "Bob": ..., ...}``.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from ob_analytics.datasets import ToyOrder, match_toy_orders, toy_orders
    >>> orders = toy_orders()
    >>> orders["Jo"] = ToyOrder(at=10, direction="bid", price=99, volume=1)
    >>> orders["Eve"] = replace(orders["Eve"], cancel_at=None)  # Eve stays
    >>> events, trades = match_toy_orders(orders)
    """
    return dict(_ORDERS)


def _check_order(actor: str, order: ToyOrder) -> None:
    """Raise ConfigError if *order* cannot be placed in a toy book."""

    def whole(value: object) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    problems = []
    if order.direction not in ("bid", "ask"):
        problems.append(f"direction must be 'bid' or 'ask', got {order.direction!r}")
    if not whole(order.price):
        problems.append(f"price must be a whole number of ticks, got {order.price!r}")
    if not whole(order.volume) or order.volume < 1:
        problems.append(
            f"volume must be a whole number of lots, at least 1, got {order.volume!r}"
        )
    if not math.isfinite(order.at) or order.at < 0:
        problems.append(f"at must be a finite, non-negative time, got {order.at!r}")
    elif order.cancel_at is not None and not (
        math.isfinite(order.cancel_at) and order.cancel_at > order.at
    ):
        problems.append(
            f"cancel_at ({order.cancel_at!r}) must be a finite time after at "
            f"({order.at!r})"
        )
    if problems:
        raise ConfigError(f"Toy order {actor!r}: " + "; ".join(problems) + ".")


def match_toy_orders(
    orders: Mapping[str, ToyOrder],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match a toy session script and return its events and trades.

    The orders arrive in time order and are matched by price–time priority:
    an arriving order trades with the best-priced resting order on the other
    side, the earliest one first, for as long as the prices cross; whatever
    is left rests at its limit price.  Each trade is at the resting order's
    price.  A cancellation removes what is still resting.  At any one time,
    every arrival comes before every cancellation, and arrivals (or
    cancellations) at the same time are taken in the order of *orders*.

    Each maker fill writes two events, the maker's and then the taker's, as
    the toy session does.  ``match_toy_orders(toy_orders())`` returns
    :func:`toy_events` and :func:`toy_trades` exactly, with one difference:
    here Sam's sell fills Alice and Ivy at the same instant (t=56), as a
    matching engine would.  The hand-written session puts Ivy's fill one
    second later, so the tutorial's pictures can show the book between the
    two fills.

    Parameters
    ----------
    orders : mapping of str to ToyOrder
        The script, keyed by actor.  Start from :func:`toy_orders` to edit the
        toy session, or write a new one.  The twelve toy actors keep their
        order ids (Alice is 1, Sam is 12); any other actor gets the next free
        id, in the order of *orders*.

    Returns
    -------
    events : pandas.DataFrame
        In the layout of :func:`toy_events`, including the ``actor`` column.
    trades : pandas.DataFrame
        In the layout of :func:`toy_trades`, including ``maker_actor`` and
        ``taker_actor``.

    Raises
    ------
    ConfigError
        If an order's direction is not ``"bid"`` or ``"ask"``, its price or
        volume is not a whole number, its volume is less than 1, a time is
        not finite, it arrives before the start, or it is cancelled before it
        arrives.
    """
    for actor, order in orders.items():
        _check_order(actor, order)

    actor_ids: dict[str, int] = {}
    next_id = max(_ACTOR_IDS.values()) + 1
    for actor in orders:
        if actor in _ACTOR_IDS:
            actor_ids[actor] = _ACTOR_IDS[actor]
        else:
            actor_ids[actor] = next_id
            next_id += 1

    # (time, 0 to place or 1 to cancel, position in the script, actor)
    steps = sorted(
        [(o.at, 0, i, a) for i, (a, o) in enumerate(orders.items())]
        + [
            (o.cancel_at, 1, i, a)
            for i, (a, o) in enumerate(orders.items())
            if o.cancel_at is not None
        ]
    )

    events: list[tuple[int, float, str, str, str, float, float, float]] = []
    trades: list[tuple[float, float, float, str, int, int]] = []
    # The outstanding volume of each resting order, by actor.  A dict keeps
    # insertion order, which is arrival order and breaks price ties.
    resting: dict[str, int] = {}

    def emit(t: float, actor: str, action: str, volume: int, fill: int) -> int:
        order = orders[actor]
        event_id = len(events) + 1
        events.append(
            (event_id, t, actor, action, order.direction, order.price, volume, fill)
        )
        return event_id

    for t, kind, _, actor in steps:
        if kind == 1:
            outstanding = resting.pop(actor, None)
            if outstanding is not None:
                emit(t, actor, "deleted", outstanding, 0)
            continue

        order = orders[actor]
        emit(t, actor, "created", order.volume, 0)
        remaining = order.volume
        buying = order.direction == "bid"
        while remaining:
            crossing = [
                other
                for other in resting
                if orders[other].direction != order.direction
                and (
                    orders[other].price <= order.price
                    if buying
                    else orders[other].price >= order.price
                )
            ]
            if not crossing:
                break
            # The best price first; min() keeps the earliest of equal prices.
            maker = min(
                crossing,
                key=lambda a: orders[a].price if buying else -orders[a].price,
            )
            size = min(remaining, resting[maker])
            resting[maker] -= size
            remaining -= size
            maker_left = resting[maker]
            maker_event = emit(
                t, maker, "deleted" if maker_left == 0 else "changed", maker_left, size
            )
            taker_event = emit(
                t, actor, "deleted" if remaining == 0 else "changed", remaining, size
            )
            trades.append(
                (
                    t,
                    orders[maker].price,
                    size,
                    "buy" if buying else "sell",
                    maker_event,
                    taker_event,
                )
            )
            if maker_left == 0:
                del resting[maker]
        if remaining:
            resting[actor] = remaining

    events_frame = _events_frame(tuple(events), actor_ids)
    return events_frame, _trades_frame(tuple(trades), events_frame)


# ---------------------------------------------------------------------------
# L2 (price-level) counterpart
# ---------------------------------------------------------------------------

# A tiny price-level (L2 / market-by-price) session — the aggregate counterpart
# to the per-order stream above.  No order identity: each row is one price
# level's **new absolute** resting size (0 removes the level), the shape
# :class:`~ob_analytics.depth.DepthMetricsEngine` consumes directly.  The first
# four rows (t=0) are the opening *snapshot*; the rest are price-level deltas.
#
#   time  side  price  volume   best bid / best ask / mid   note
#   ----  ----  -----  ------   -------------------------   ----------------
#   0     bid   99      5       99 / 101 / 100  (spread 2)  snapshot
#   0     bid   98      8
#   0     ask   101     4
#   0     ask   102     7
#   5     ask   101     2       99 / 101 / 100              ask 101 shrinks
#   10    bid   100     3      100 / 101 / 100.5 (spread 1) new best bid
#   15    ask   101     0      100 / 102 / 101   (spread 2) best ask cleared
#   20    bid   100     0       99 / 102 / 100.5 (spread 3) best bid cleared
#   25    ask   100     4       99 / 100 / 99.5  (spread 1) new best ask
#   30    bid   99      7       99 / 100 / 99.5             best bid grows
#
# (seconds, side, price, new_absolute_volume)
_L2_DEPTH: tuple[tuple[float, str, float, float], ...] = (
    (0.0, "bid", 99.0, 5.0),
    (0.0, "bid", 98.0, 8.0),
    (0.0, "ask", 101.0, 4.0),
    (0.0, "ask", 102.0, 7.0),
    (5.0, "ask", 101.0, 2.0),
    (10.0, "bid", 100.0, 3.0),
    (15.0, "ask", 101.0, 0.0),
    (20.0, "bid", 100.0, 0.0),
    (25.0, "ask", 100.0, 4.0),
    (30.0, "bid", 99.0, 7.0),
)

# Trade prints, on a separate channel from the book (as on a real aggregated
# feed).  ``direction`` is the taker's aggressor side, and equals what
# Lee–Ready recovers from the prevailing mid above — so a test can null it out
# and check the classifier round-trips it (buy, sell, buy, buy).
#
#   time  price  volume  prevailing mid   side
#   ----  -----  ------  --------------   ----
#   7     101      1     100  (bid99/ask101)  buy   (print above mid)
#   12    100      2     100.5 (bid100/ask101) sell (print below mid)
#   22    102      1     100.5 (bid99/ask102)  buy  (print above mid)
#   27    100      3      99.5 (bid99/ask100)  buy  (print above mid)
#
# (seconds, price, volume, taker side)
_L2_TRADES: tuple[tuple[float, float, float, str], ...] = (
    (7.0, 101.0, 1.0, "buy"),
    (12.0, 100.0, 2.0, "sell"),
    (22.0, 102.0, 1.0, "buy"),
    (27.0, 100.0, 3.0, "buy"),
)


def toy_l2_depth() -> pd.DataFrame:
    """Return the toy session's canonical **L2 depth** DataFrame.

    Ten price-level updates over 30 synthetic seconds (a four-level opening
    snapshot at ``t=0`` followed by six deltas), in the exact column layout
    :class:`~ob_analytics.depth.DepthMetricsEngine` /
    :func:`~ob_analytics.depth.depth_metrics` consume — the L2 counterpart to
    :func:`toy_events`.  ``volume`` is each level's **new absolute** resting
    size (``0`` removes it), *not* a signed delta.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp``, ``price``, ``volume``, ``direction``
        (categorical ``bid``/``ask``), in timestamp order — a
        :func:`~ob_analytics.schemas.validate_depth_df` frame.

    Examples
    --------
    >>> from ob_analytics.datasets import toy_l2_depth
    >>> from ob_analytics.depth import depth_metrics, get_spread
    >>> summary = depth_metrics(toy_l2_depth())
    >>> summary[["best_bid_price", "best_ask_price"]].iloc[-1].tolist()
    [99, 100]
    """
    ts = (
        pd.Series(
            [_BASE + pd.Timedelta(milliseconds=round(r[0] * 1000)) for r in _L2_DEPTH]
        )
        .astype("datetime64[ns]")
        .dt.tz_localize("UTC")
    )
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": np.array([r[2] for r in _L2_DEPTH], dtype=np.int64),
            "volume": np.array([r[3] for r in _L2_DEPTH], dtype=np.int64),
            "direction": pd.Categorical(
                [r[1] for r in _L2_DEPTH],
                categories=["bid", "ask"],
                ordered=True,
            ),
        }
    )


def toy_l2_trades() -> pd.DataFrame:
    """Return the toy L2 session's canonical **trades** DataFrame.

    Four prints consistent with :func:`toy_l2_depth`, in the canonical trades
    layout.  A price-level feed carries no order identity, so
    ``maker_event_id`` / ``taker_event_id`` / ``maker`` / ``taker`` (and the
    ``*_og`` columns) are ``<NA>``.  ``direction`` is the true taker side,
    equal to what Lee–Ready recovers from :func:`toy_l2_depth`'s prevailing
    mid — null it to exercise
    :func:`~ob_analytics.trade_sign.classify_trade_sign`.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp``, ``price``, ``volume``, ``direction``
        (categorical ``buy``/``sell``), and the ``<NA>`` maker/taker
        attribution columns — a
        :func:`~ob_analytics.schemas.validate_trades_df` frame.
    """
    ts = (
        pd.Series(
            [_BASE + pd.Timedelta(milliseconds=round(t[0] * 1000)) for t in _L2_TRADES]
        )
        .astype("datetime64[ns]")
        .dt.tz_localize("UTC")
    )
    n = len(_L2_TRADES)
    na = pd.array([pd.NA] * n, dtype="object")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": np.array([t[1] for t in _L2_TRADES], dtype=np.int64),
            "volume": np.array([t[2] for t in _L2_TRADES], dtype=np.int64),
            "direction": pd.Categorical(
                [t[3] for t in _L2_TRADES], categories=["buy", "sell"], ordered=True
            ),
            "maker_event_id": na,
            "taker_event_id": na.copy(),
            "maker": na.copy(),
            "taker": na.copy(),
            "maker_og": na.copy(),
            "taker_og": na.copy(),
        }
    )
