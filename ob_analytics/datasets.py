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

Timestamps are tz-naive at millisecond resolution starting from an
arbitrary Monday morning; ``exchange_timestamp`` equals ``timestamp``
(as in LOBSTER sessions, where only exchange time exists).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["toy_events", "toy_trades"]

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
    event_id = np.array([e[0] for e in _EVENTS], dtype=np.int64)
    seconds = [e[1] for e in _EVENTS]
    actors = [e[2] for e in _EVENTS]
    ts = pd.Series(
        [_BASE + pd.Timedelta(milliseconds=round(s * 1000)) for s in seconds]
    ).astype("datetime64[ms]")

    return pd.DataFrame(
        {
            "original_number": event_id.copy(),
            "id": np.array([_ACTOR_IDS[a] for a in actors], dtype=np.int64),
            "timestamp": ts,
            "exchange_timestamp": ts.copy(),
            "price": np.array([e[5] for e in _EVENTS], dtype=np.float64),
            "volume": np.array([e[6] for e in _EVENTS], dtype=np.float64),
            "action": pd.Categorical(
                [e[3] for e in _EVENTS],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "direction": pd.Categorical(
                [e[4] for e in _EVENTS],
                categories=["bid", "ask"],
                ordered=True,
            ),
            "event_id": event_id,
            "fill": np.array([e[7] for e in _EVENTS], dtype=np.float64),
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
    events = toy_events()
    eid_to_oid = dict(zip(events["event_id"], events["id"]))
    eid_to_og = dict(zip(events["event_id"], events["original_number"]))
    oid_to_actor = {v: k for k, v in _ACTOR_IDS.items()}

    maker_eid = [t[4] for t in _TRADES]
    taker_eid = [t[5] for t in _TRADES]
    maker = [eid_to_oid[e] for e in maker_eid]
    taker = [eid_to_oid[e] for e in taker_eid]

    return pd.DataFrame(
        {
            "timestamp": pd.Series(
                [_BASE + pd.Timedelta(milliseconds=round(t[0] * 1000)) for t in _TRADES]
            ).astype("datetime64[ms]"),
            "price": np.array([t[1] for t in _TRADES], dtype=np.float64),
            "volume": np.array([t[2] for t in _TRADES], dtype=np.float64),
            "direction": pd.Categorical(
                [t[3] for t in _TRADES], categories=["buy", "sell"], ordered=True
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
