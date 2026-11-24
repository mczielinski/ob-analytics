"""FIFO queue-position reconstruction for visible limit orders.

A single time-ordered pass over the event stream rebuilds, per
``(direction, price)`` level, the price-time-priority queue of resting orders.
For each order event it reports the order's rank in its level (1 = front), the
volume ahead of it, the queue length, and its age.

Visible-only: orders with no public identity never join the visible queue and
are excluded (see :meth:`~ob_analytics.engine.OrderEvents.visible`), so the
reconstructed touch volume matches the *visible* book, not the full book.
Market orders never rest, so they are left out too, as the book leaves them out.

Each order is queued at its resting price
(:attr:`~ob_analytics.engine.OrderEvents.resting_price`), the price the book
places it at.  A row that moves the order to a new resting price takes it out
of its old level and puts it at the back of the new one.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from ob_analytics.engine._events import NAT_NS, Action, Direction, OrderEvents
from ob_analytics.engine._queue_replay import elapsed_seconds, replay


class _LiveLevels:
    """The non-empty price levels of one side, with the best one at hand.

    The prices sit in a heap, so finding the best level does not look at every
    live level.  A level that empties leaves its heap entry behind; the entry is
    dropped when it reaches the top, and the heap is rebuilt from the live
    levels when such stale entries outnumber them.
    """

    __slots__ = ("_heap", "_live", "_sign")

    def __init__(self, direction: int) -> None:
        # ``heapq`` keeps the smallest entry on top, so bid prices are stored
        # negated to put the highest bid there.
        self._sign = -1 if direction == Direction.BID else 1
        self._live: set = set()
        self._heap: list = []

    def add(self, price) -> None:
        """Mark *price* as holding at least one order."""
        if price in self._live:
            return
        self._live.add(price)
        heap = self._heap
        if len(heap) > 2 * len(self._live) + 64:
            heap[:] = [self._sign * live for live in self._live]
            heapq.heapify(heap)
        else:
            heapq.heappush(heap, self._sign * price)

    def discard(self, price) -> None:
        """Mark *price* as empty."""
        self._live.discard(price)

    def best(self):
        """The best live price — highest bid or lowest ask — or ``None``."""
        heap = self._heap
        while heap:
            price = self._sign * heap[0]
            if price in self._live:
                return price
            heapq.heappop(heap)
        return None


@dataclass(frozen=True)
class QueuePositions:
    """One row per order event, reporting that order's place in its level.

    A move to a new price gives two rows for its event: the order leaving its
    old level, then joining its new one.

    The event's own columns — time, order id, direction — are not copied:
    :attr:`row` points back at the event in the :class:`OrderEvents` arrays.

    Attributes
    ----------
    row : numpy.ndarray
        Index in the :class:`OrderEvents` arrays of the event this row reports
        on (``int64``).
    price : numpy.ndarray
        The price level the order is queued at: its resting price, which can
        differ from the price on the event's row.
    action : numpy.ndarray
        What the event did to the queue, as
        :class:`~ob_analytics.engine.Action` codes: joined the back
        (``created``), kept its place at a new size (``changed``), or left
        (``deleted``).  A reduction to zero is reported as a ``deleted``,
        whatever the venue called it.  A move to a new price is reported as a
        ``deleted`` at the old level and a ``created`` at the back of the new
        one.  The ``created`` row keeps the order's age from its placement.
    rank : numpy.ndarray
        1-based position from the front of the level (``int64``).  While an
        order stays at one price it is monotone non-increasing: newcomers join
        the back.
    queue_len : numpy.ndarray
        Number of orders resting at the level (``int64``).
    ahead_volume : numpy.ndarray
        Outstanding size of the orders ahead of this one, in the size dtype the
        events carried — ``int64`` lots for a canonical stream.
    remaining : numpy.ndarray
        This order's own outstanding size after the event, in that same dtype.
    age_s : numpy.ndarray
        Seconds since the order was placed (``float64``).
    """

    row: np.ndarray
    price: np.ndarray
    action: np.ndarray
    rank: np.ndarray
    queue_len: np.ndarray
    ahead_volume: np.ndarray
    remaining: np.ndarray
    age_s: np.ndarray

    def __len__(self) -> int:
        return len(self.row)


@dataclass(frozen=True)
class QueueAgeGrid:
    """Touch-queue composition over time: the age of the order at each rank.

    Attributes
    ----------
    ages : numpy.ndarray
        A ``(max_rank, n_samples)`` float array: ``ages[r, t]`` is the age in
        seconds of the order at rank ``r + 1`` (front = row 0) at sample ``t``,
        or ``NaN`` where the queue is shorter than ``r + 1``.
    max_rank : int
        The deepest the touch queue got over the sampled window — the number of
        rows in :attr:`ages`.
    """

    ages: np.ndarray
    max_rank: int


def _check_replayable(events: OrderEvents, visible: np.ndarray) -> None:
    """Raise ``ValueError`` unless every visible row can be queued.

    The compiled loop indexes its arrays by side and price and does not check
    the bounds, so a bad code must be caught before it runs.  Each visible row
    needs a side that is a :class:`Direction`, a time that is not
    :data:`NAT_NS`, and a resting price that is not NaN.
    """
    directions = events.direction[visible]
    if np.any((directions != Direction.BID) & (directions != Direction.ASK)):
        raise ValueError("queue_positions needs every direction to be BID or ASK")
    if np.any(events.timestamp[visible] == NAT_NS):
        raise ValueError("queue_positions needs a timestamp on every visible row")
    resting = events.rests_at()[visible]
    if resting.dtype.kind == "f" and np.isnan(resting).any():
        raise ValueError("queue_positions needs a resting price on every visible row")


def queue_positions(events: OrderEvents, *, touch_only: bool = True) -> QueuePositions:
    """Reconstruct the FIFO queue position of each visible limit order over time.

    Price-time priority: a ``created`` event appends to the back of its level; a
    size reduction (partial fill or partial cancel) keeps the order's place; a
    ``deleted`` — or a reduction to zero — removes it.  A move to a new resting
    price takes the order to the back of its new level.  Market orders never
    join a queue.

    Parameters
    ----------
    events : OrderEvents
        The event stream, in canonical order.  Order matters here more than
        anywhere else in the engine: it *is* the queue's priority.
    touch_only : bool
        Keep only the events where the order rests at the best bid or ask at
        that instant — the input to the touch-queue faces.  ``False`` keeps
        every visible level.

    Returns
    -------
    QueuePositions
        One row per surviving order event, two for a move.
    """
    visible = events.visible()
    resting = events.rests_at()
    _check_replayable(events, visible)
    # The compiled loop takes small integer codes, not ids and prices: each
    # order id becomes its index among the distinct ids, and each price its
    # index among the distinct prices, which keeps the price order.
    ids, orders = np.unique(events.order_id[visible], return_inverse=True)
    prices, price_index = np.unique(resting[visible], return_inverse=True)
    exact = bool(np.issubdtype(events.volume.dtype, np.integer))
    volumes = events.volume[visible].astype(
        np.int64 if exact else np.float64, copy=False
    )
    row, level, action, rank, queue_len, ahead, remaining, age = replay(
        orders.astype(np.int64, copy=False),
        events.timestamp[visible].astype(np.int64, copy=False),
        price_index.astype(np.int64, copy=False),
        volumes,
        events.direction[visible],
        events.action[visible],
        ids.size,
        prices.size,
        exact,
        touch_only,
    )
    return QueuePositions(
        row=visible[row].astype(np.int64, copy=False),
        price=prices[level % max(prices.size, 1)],
        action=action,
        rank=rank,
        queue_len=queue_len,
        # Sizes keep the dtype they arrived in, so a canonical integer-lot
        # stream gives exact queue totals: ``ahead_volume`` is a running sum
        # down a level, which is precisely where a float drifts.
        ahead_volume=ahead.astype(events.volume.dtype),
        remaining=remaining.astype(events.volume.dtype),
        age_s=age,
    )


def queue_age_grid(
    events: OrderEvents, *, side: Direction, at: np.ndarray
) -> QueueAgeGrid:
    """Snapshot one side's touch queue at each of the instants *at*.

    Replays the side's events and, at every sample instant, records the age of
    each order resting at the best price by FIFO rank.  Orders are queued as
    :func:`queue_positions` queues them: at their resting price, with market
    orders left out.

    Parameters
    ----------
    events : OrderEvents
        The event stream, in canonical order.
    side : Direction
        Which touch to compose.
    at : numpy.ndarray
        Sample instants in int64 nanoseconds, ascending.  The caller chooses the
        window and the spacing; the engine only replays to them.

    Returns
    -------
    QueueAgeGrid
        The age-by-rank grid and its depth.
    """
    if side not in (Direction.BID, Direction.ASK):
        raise ValueError(f"side must be a Direction, got {side!r}")

    visible = events.visible(side=side)
    if visible.size == 0 or at.size == 0:
        return QueueAgeGrid(ages=np.empty((0, 0)), max_rank=0)

    order_ids = events.order_id[visible].tolist()
    times = events.timestamp[visible].tolist()
    prices = events.rests_at()[visible].tolist()
    volumes = events.volume[visible].tolist()
    actions = events.action[visible].tolist()

    queues: dict[object, dict[int, float]] = {}  # price -> {order id: remaining}
    # Order id -> the price it is queued at, for the orders in a queue now.
    order_level: dict[int, object] = {}
    placed_at: dict[int, int] = {}
    live = _LiveLevels(side)

    def leave(oid: int, level: object) -> None:
        del order_level[oid]
        del queues[level][oid]
        if not queues[level]:
            live.discard(level)

    def join(oid: int, level: object, volume: float) -> None:
        order_level[oid] = level
        queues.setdefault(level, {})[oid] = float(volume)
        live.add(level)

    snapshots: list[list[float]] = []
    pending = 0
    n = len(order_ids)
    for sample in at.tolist():
        # Apply every event at or before this sample instant.
        while pending < n and times[pending] <= sample:
            oid = order_ids[pending]
            when = times[pending]
            price = prices[pending]
            level = order_level.get(oid)
            if actions[pending] == Action.CREATED:
                if level is not None:
                    # Placed again while still queued: it rests once.
                    leave(oid, level)
                placed_at[oid] = when
                join(oid, price, volumes[pending])
            elif level is not None:
                if actions[pending] == Action.DELETED or volumes[pending] <= 0:
                    leave(oid, level)
                elif price != level:
                    # A move to a new price: onto the back of the new level.
                    leave(oid, level)
                    join(oid, price, volumes[pending])
                else:
                    queues[level][oid] = float(volumes[pending])
            pending += 1

        touch = live.best()
        if touch is None:
            snapshots.append([])
            continue
        snapshots.append(
            [elapsed_seconds(sample - placed_at[oid]) for oid in queues[touch]]
        )

    max_rank = max((len(column) for column in snapshots), default=0)
    ages = np.full((max_rank, len(at)), np.nan, dtype=float)
    for t, column in enumerate(snapshots):
        if column:
            ages[: len(column), t] = column
    return QueueAgeGrid(ages=ages, max_rank=max_rank)
