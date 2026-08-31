"""FIFO queue-position reconstruction for visible limit orders.

A single time-ordered pass over the event stream rebuilds, per
``(direction, price)`` level, the price-time-priority queue of resting orders.
For each order event it reports the order's rank in its level (1 = front), the
volume ahead of it, the queue length, and its age.

Visible-only: orders with no public identity never join the visible queue and
are excluded (see :meth:`~ob_analytics.engine.OrderEvents.visible`), so the
reconstructed touch volume matches the *visible* book, not the full book.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ob_analytics.engine._events import Action, Direction, OrderEvents


def _elapsed_seconds(delta_ns: int) -> float:
    """Seconds between two nanosecond instants.

    Reproduces ``pandas.Timedelta.total_seconds()`` bit for bit, down to its
    two quirks: the result is truncated to whole microseconds, and the whole
    seconds are added to the fraction *after* the division rather than before,
    which moves the last bit on some values.  These ages were measured that way
    before the engine moved to integer nanoseconds and the golden-output gates
    pin the resulting numbers, so the arithmetic is copied rather than
    corrected; giving the ages their full nanosecond precision is a deliberate
    change that has to re-record those baselines.
    """
    seconds, remainder = divmod(delta_ns, 1_000_000_000)
    return seconds + (remainder // 1000) / 1e6


@dataclass(frozen=True)
class QueuePositions:
    """One row per order event, reporting that order's place in its level.

    The event's own columns — time, order id, direction, price — are not
    copied: :attr:`row` points back at the event in the :class:`OrderEvents`
    arrays.

    Attributes
    ----------
    row : numpy.ndarray
        Index in the :class:`OrderEvents` arrays of the event this row reports
        on (``int64``).
    action : numpy.ndarray
        What the event did to the queue, as
        :class:`~ob_analytics.engine.Action` codes: joined the back
        (``created``), kept its place at a smaller size (``changed``), or left
        (``deleted``).  A reduction to zero is reported as a ``deleted``,
        whatever the venue called it.
    rank : numpy.ndarray
        1-based position from the front of the level (``int64``).  Within one
        order's life it is monotone non-increasing: newcomers join the back.
    queue_len : numpy.ndarray
        Number of orders resting at the level (``int64``).
    ahead_volume : numpy.ndarray
        Outstanding size of the orders ahead of this one (``float64``).
    remaining : numpy.ndarray
        This order's own outstanding size after the event (``float64``).
    age_s : numpy.ndarray
        Seconds since the order was placed (``float64``).
    """

    row: np.ndarray
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


def queue_positions(events: OrderEvents, *, touch_only: bool = True) -> QueuePositions:
    """Reconstruct the FIFO queue position of each visible limit order over time.

    Price-time priority: a ``created`` event appends to the back of its level; a
    size reduction (partial fill or partial cancel) keeps the order's place; a
    ``deleted`` — or a reduction to zero — removes it.

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
        One row per surviving order event.
    """
    visible = events.visible()
    order_ids = events.order_id[visible].tolist()
    times = events.timestamp[visible].tolist()
    prices = events.price[visible].tolist()
    volumes = events.volume[visible].tolist()
    directions = events.direction[visible].tolist()
    actions = events.action[visible].tolist()

    # Per level: insertion-ordered {order id: remaining}.  Dicts preserve
    # insertion order, which *is* price-time priority here.
    queues: dict[tuple[int, object], dict[int, float]] = {}
    order_level: dict[int, tuple[int, object]] = {}
    placed_at: dict[int, int] = {}
    # Live (non-empty) price levels per side, for the running touch.
    live: tuple[set, set] = (set(), set())

    def touch(direction: int):
        levels = live[direction]
        if not levels:
            return None
        return max(levels) if direction == Direction.BID else min(levels)

    rows: list[int] = []
    out_action: list[int] = []
    out_rank: list[int] = []
    out_len: list[int] = []
    out_ahead: list[float] = []
    out_remaining: list[float] = []
    out_age: list[float] = []

    def emit(row, when, oid, level, queue, action) -> None:
        direction, price = level
        if touch_only and price != touch(direction):
            return
        ahead = 0.0
        rank = 0
        for other_id, remaining in queue.items():
            rank += 1
            if other_id == oid:
                break
            ahead += remaining
        rows.append(row)
        out_action.append(action)
        out_rank.append(rank)
        out_len.append(len(queue))
        out_ahead.append(ahead)
        out_remaining.append(queue[oid])
        out_age.append(
            _elapsed_seconds(when - placed_at[oid]) if oid in placed_at else 0.0
        )

    for i, oid in enumerate(order_ids):
        row = visible[i]
        when = times[i]
        action = actions[i]
        volume = volumes[i]

        if action == Action.CREATED:
            level = (directions[i], prices[i])
            order_level[oid] = level
            placed_at[oid] = when
            queue = queues.setdefault(level, {})
            queue[oid] = float(volume)
            live[level[0]].add(level[1])
            emit(row, when, oid, level, queue, Action.CREATED)
            continue

        level = order_level.get(oid)
        if level is None:
            # Creation never seen (pre-existing / windowed-in): cannot place it.
            continue
        queue = queues.get(level, {})
        if oid not in queue:
            continue

        if action == Action.DELETED or volume <= 0:
            emit(
                row, when, oid, level, queue, Action.DELETED
            )  # last place before removal
            del queue[oid]
            if not queue:
                live[level[0]].discard(level[1])
            continue

        queue[oid] = float(volume)  # a size reduction keeps the queue place
        emit(row, when, oid, level, queue, Action.CHANGED)

    return QueuePositions(
        row=np.array(rows, dtype=np.int64),
        action=np.array(out_action, dtype=np.int8),
        rank=np.array(out_rank, dtype=np.int64),
        queue_len=np.array(out_len, dtype=np.int64),
        ahead_volume=np.array(out_ahead, dtype=np.float64),
        remaining=np.array(out_remaining, dtype=np.float64),
        age_s=np.array(out_age, dtype=np.float64),
    )


def queue_age_grid(events: OrderEvents, *, side: int, at: np.ndarray) -> QueueAgeGrid:
    """Snapshot one side's touch queue at each of the instants *at*.

    Replays the side's events and, at every sample instant, records the age of
    each order resting at the best price by FIFO rank.

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
    prices = events.price[visible].tolist()
    volumes = events.volume[visible].tolist()
    actions = events.action[visible].tolist()

    queues: dict[object, dict[int, float]] = {}  # price -> {order id: remaining}
    placed_at: dict[int, int] = {}
    live: set = set()

    def best():
        if not live:
            return None
        return max(live) if side == Direction.BID else min(live)

    snapshots: list[list[float]] = []
    pending = 0
    n = len(order_ids)
    for sample in at.tolist():
        # Apply every event at or before this sample instant.
        while pending < n and times[pending] <= sample:
            oid = order_ids[pending]
            when = times[pending]
            price = prices[pending]
            if actions[pending] == Action.CREATED:
                placed_at[oid] = when
                queues.setdefault(price, {})[oid] = float(volumes[pending])
                live.add(price)
            elif price in queues and oid in queues[price]:
                if actions[pending] == Action.DELETED or volumes[pending] <= 0:
                    del queues[price][oid]
                    if not queues[price]:
                        live.discard(price)
                else:
                    queues[price][oid] = float(volumes[pending])
            pending += 1

        touch = best()
        if touch is None:
            snapshots.append([])
            continue
        snapshots.append(
            [_elapsed_seconds(sample - placed_at[oid]) for oid in queues[touch]]
        )

    max_rank = max((len(column) for column in snapshots), default=0)
    ages = np.full((max_rank, len(at)), np.nan, dtype=float)
    for t, column in enumerate(snapshots):
        if column:
            ages[: len(column), t] = column
    return QueueAgeGrid(ages=ages, max_rank=max_rank)
