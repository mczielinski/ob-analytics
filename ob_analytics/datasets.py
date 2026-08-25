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

Timestamps are tz-aware UTC nanoseconds (the schema's canonical time model,
issue #154) starting from an arbitrary Monday morning; ``exchange_timestamp``
equals ``timestamp`` (as in LOBSTER sessions, where only exchange time exists).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["toy_events", "toy_l2_depth", "toy_l2_trades", "toy_trades"]

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
    ts = (
        pd.Series([_BASE + pd.Timedelta(milliseconds=round(s * 1000)) for s in seconds])
        .astype("datetime64[ns]")
        .dt.tz_localize("UTC")
    )

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
            )
            .astype("datetime64[ns]")
            .dt.tz_localize("UTC"),
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
    [99.0, 100.0]
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
            "price": np.array([r[2] for r in _L2_DEPTH], dtype=np.float64),
            "volume": np.array([r[3] for r in _L2_DEPTH], dtype=np.float64),
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
            "price": np.array([t[1] for t in _L2_TRADES], dtype=np.float64),
            "volume": np.array([t[2] for t in _L2_TRADES], dtype=np.float64),
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
