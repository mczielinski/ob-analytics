"""Point-in-time order-book reconstruction.

Replays an event stream up to an instant and reports which orders are resting
on the book there, best price first, with the cumulative liquidity and the
distance from the touch each side implies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ob_analytics.engine._events import Action, Direction, OrderEvents

# ── Stable multi-key sorting ──────────────────────────────────────────
#
# The book's row order is part of its output, so the sorts have to be stable
# and reproducible: equal keys keep the order the stream gave them.  NumPy has
# no descending stable sort, so one is composed here, and multi-key orders are
# built by sorting on each key from least to most significant.


def _argsort_desc(values: np.ndarray) -> np.ndarray:
    """Stable descending argsort: ties keep their original relative order.

    ``argsort(kind="stable")`` only sorts ascending, and reversing its result
    also reverses the ties.  Sorting the reversed array instead, then mapping
    the positions back, puts the ties the right way round again.
    """
    n = len(values)
    return n - 1 - np.argsort(values[::-1], kind="stable")[::-1]


def argsort_keys(keys: list[tuple[np.ndarray, bool]]) -> np.ndarray:
    """Stable lexicographic argsort over ``(values, ascending)`` keys.

    *keys* are given **most significant first**.  Each key is applied as its own
    stable sort, least significant first, which composes into the lexicographic
    order — the same result pandas' multi-column ``sort_values(kind="stable")``
    produces.
    """
    order = np.arange(len(keys[0][0]))
    for values, ascending in reversed(keys):
        ranked = values[order]
        within = (
            np.argsort(ranked, kind="stable") if ascending else _argsort_desc(ranked)
        )
        order = order[within]
    return order


# ── Crossed-book eviction ─────────────────────────────────────────────


def crossed_prefix_counts(
    bid_prices: np.ndarray,
    bid_ts: np.ndarray,
    ask_prices: np.ndarray,
    ask_ts: np.ndarray,
) -> tuple[int, int]:
    """How many best-end bids / asks to evict to uncross two book sides.

    *bid_prices* descend from the best bid, *ask_prices* ascend from the best
    ask, each paired with its order timestamp.  Walks the touch: while the top
    bid is priced at or above the top ask (crossed, or locked when equal), evict
    the older of the two touching orders — the static-snapshot analogue of
    :class:`~ob_analytics.depth.DepthMetricsEngine` trusting the fresher quote.
    The evicted orders are exactly the contiguous best-end prefixes, so the two
    returned counts describe the eviction completely.
    """
    bi = 0
    ai = 0
    n_bid = bid_prices.size
    n_ask = ask_prices.size
    while bi < n_bid and ai < n_ask and bid_prices[bi] >= ask_prices[ai]:
        if bid_ts[bi] <= ask_ts[ai]:
            bi += 1
        else:
            ai += 1
    return bi, ai


# ── Reconstructed book ────────────────────────────────────────────────


@dataclass(frozen=True)
class BookSide:
    """One side of a reconstructed book, **best price first**.

    The identity of each resting order — its id, both clocks, price, and
    outstanding size — is not copied here: :attr:`row` points at the event that
    left the order in this state, so a caller reads any column it wants
    straight off its own event table. Only the two derived quantities travel.

    Attributes
    ----------
    row : numpy.ndarray
        For each resting order, the index in the :class:`OrderEvents` arrays of
        its latest event at the snapshot instant (``int64``).
    liquidity : numpy.ndarray
        Cumulative outstanding volume from the touch down to and including this
        order (``float64``).
    bps : numpy.ndarray
        Distance from this side's own touch, in basis points (``float64``).
        Zero at the touch, rising away from it on both sides.
    """

    row: np.ndarray
    liquidity: np.ndarray
    bps: np.ndarray

    def __len__(self) -> int:
        return len(self.row)


@dataclass(frozen=True)
class BookState:
    """The book at one instant: both sides, best price first.

    The instant itself is not repeated here — the caller supplied it as
    :func:`book_state`'s ``at`` and still holds it.

    Attributes
    ----------
    bids : BookSide
        Resting bids, highest price first.
    asks : BookSide
        Resting asks, lowest price first.
    """

    bids: BookSide
    asks: BookSide


def _empty_side() -> BookSide:
    return BookSide(
        row=np.empty(0, dtype=np.int64),
        liquidity=np.empty(0, dtype=np.float64),
        bps=np.empty(0, dtype=np.float64),
    )


def _resting_rows(events: OrderEvents, at: int) -> np.ndarray:
    """Rows holding the state of every order resting on the book at *at*.

    An order rests iff it was submitted (it has a ``created`` row in the window)
    and its latest event at or before *at* is neither a delete nor one that left
    it exhausted.  The outstanding-size check is what removes fully executed
    orders on venues that never emit a delete for them (LOBSTER), which would
    otherwise linger as phantoms crossing the book.

    Returns the rows in ascending stream order, one per resting order.
    """
    window = np.flatnonzero(events.timestamp <= at)
    if window.size == 0:
        return window

    ids = events.order_id[window]
    # Last row per id: group the window by id with a stable sort, keep each
    # run's final entry, then restore stream order.
    by_id = np.argsort(ids, kind="stable")
    grouped = ids[by_id]
    run_end = np.empty(len(grouped), dtype=bool)
    run_end[-1] = True
    np.not_equal(grouped[1:], grouped[:-1], out=run_end[:-1])
    latest = np.sort(window[by_id[run_end]])

    submitted = np.isin(
        events.order_id[latest], ids[events.action[window] == Action.CREATED]
    )
    return latest[
        submitted
        & (events.action[latest] != Action.DELETED)
        & (events.volume[latest] > 0)
    ]


def _uncross_rows(events: OrderEvents, resting: np.ndarray) -> np.ndarray:
    """Drop crossed resting orders so the snapshot satisfies ``best_bid < best_ask``.

    Static-snapshot mirror of the depth engine's crossed-level eviction: at the
    crossed or locked touch, keep the fresher quote and evict the older opposing
    order, repeating until the book is uncrossed.  Recency uses the receive
    clock, the same one the depth engine processes in.  Market rows never rest
    on the book, so they take no part in the crossing test and are always kept.
    """
    book = resting[~events.market_mask()[resting]]
    bids = book[events.direction[book] == Direction.BID]
    asks = book[events.direction[book] == Direction.ASK]
    bids = bids[
        argsort_keys([(events.price[bids], False), (events.timestamp[bids], True)])
    ]
    asks = asks[
        argsort_keys([(events.price[asks], True), (events.timestamp[asks], True)])
    ]

    n_bid, n_ask = crossed_prefix_counts(
        events.price[bids],
        events.timestamp[bids],
        events.price[asks],
        events.timestamp[asks],
    )
    if n_bid == 0 and n_ask == 0:
        return resting
    evicted = np.concatenate([bids[:n_bid], asks[:n_ask]])
    return resting[~np.isin(resting, evicted)]


def _side(events: OrderEvents, resting: np.ndarray, direction: Direction) -> BookSide:
    """One side of the book, best price first, with liquidity and bps.

    Orders are ranked by price — the touch first — then by id, which is a total
    order because at most one row per order id can rest at a time.
    """
    rows = resting[
        (events.direction[resting] == direction) & ~events.market_mask()[resting]
    ]
    if rows.size == 0:
        return _empty_side()

    rows = rows[
        argsort_keys(
            [
                (events.price[rows], direction == Direction.ASK),
                (events.order_id[rows], True),
            ]
        )
    ]
    prices = events.price[rows]
    touch = prices[0]
    away = prices - touch if direction == Direction.ASK else touch - prices
    return BookSide(
        row=rows,
        liquidity=np.cumsum(events.volume[rows]),
        bps=(away / touch) * 10000,
    )


def book_state(events: OrderEvents, *, at: int, uncross: bool = False) -> BookState:
    """Reconstruct the order book at one instant.

    Parameters
    ----------
    events : OrderEvents
        The event stream, in canonical order.
    at : int
        The instant to evaluate the book at, in nanoseconds since the epoch
        (UTC).  Events after it are ignored.
    uncross : bool
        When ``True``, evict crossed resting orders so the snapshot satisfies
        ``best_bid < best_ask`` — a *display* convenience mirroring the depth
        engine's crossed-level eviction.  The default is ``False``: the
        reconstruction stays **faithful** to the feed, so a diff feed's
        genuinely crossed resting orders are replayed as they arrived rather
        than silently uncrossed.  It has no effect on a matched-book feed, which
        is never crossed.

    Returns
    -------
    BookState
        Both sides, best price first, each carrying the row that put every
        resting order in its current state.
    """
    resting = _resting_rows(events, at)
    if uncross:
        resting = _uncross_rows(events, resting)
    return BookState(
        bids=_side(events, resting, Direction.BID),
        asks=_side(events, resting, Direction.ASK),
    )
