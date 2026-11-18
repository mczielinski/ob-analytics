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


_TREE_FROM = 32
"""Queue length at which a level starts keeping its Fenwick trees.

Below it, walking the queue from the front is cheaper than keeping the trees up
to date: most levels hold only a few orders at a time.
"""


class _ArrivalQueue:
    """The orders resting at one price level, in arrival order.

    Reports each order's rank and the size queued ahead of it.  A short queue
    is walked from the front.  A long one keeps two Fenwick trees (binary
    indexed trees) instead, so these come from prefix sums in time logarithmic
    in the queue length.  The trees are built when the queue reaches
    :data:`_TREE_FROM` orders and dropped when it falls below half that.

    In the trees, each order holds a slot, numbered in the order the orders
    joined.  One tree counts the orders still resting in each run of slots, the
    other sums their remaining size.  A slot is not reused when its order
    leaves.  When the slots run out, the resting orders are numbered again from
    1 in arrival order and the trees are rebuilt with room for as many again,
    so each join costs constant time on average.

    The trees need exact sums: with float sizes, taking an order out would
    leave rounding error behind in the totals.  So they are kept only for
    integer sizes (``exact=True``).  A queue of float sizes is always walked,
    adding the sizes in queue order, so no error builds up as orders come and
    go.
    """

    __slots__ = (
        "_cap",
        "_count",
        "_exact",
        "_remaining",
        "_slot",
        "_used",
        "_volume",
    )

    def __init__(self, *, exact: bool) -> None:
        self._exact = exact
        self._remaining: dict = {}  # order id -> remaining size, arrival order
        # The trees, present only while the queue is long; see _build_trees.
        self._slot: dict | None = None  # order id -> its 1-based slot
        self._count: list = []  # Fenwick tree of resting orders, 1-based
        self._volume: list = []  # Fenwick tree of remaining size, 1-based
        self._used = 0  # slots handed out since the trees were built
        self._cap = 0  # slots the trees have room for

    def __len__(self) -> int:
        return len(self._remaining)

    def append(self, oid, size) -> None:
        """Put order *oid*, not already resting here, at the back."""
        self._remaining[oid] = size
        slots = self._slot
        if slots is None:
            if self._exact and len(self._remaining) >= _TREE_FROM:
                self._build_trees()
        elif self._used == self._cap:
            self._build_trees()
        else:
            self._used += 1
            slots[oid] = self._used
            self._add(self._used, 1, size)

    def resize(self, oid, size) -> None:
        """Give resting order *oid* a new remaining size; its place is kept."""
        if self._slot is not None:
            self._add(self._slot[oid], 0, size - self._remaining[oid])
        self._remaining[oid] = size

    def remove(self, oid) -> None:
        """Take resting order *oid* out of the queue."""
        size = self._remaining.pop(oid)
        slots = self._slot
        if slots is None:
            return
        if len(self._remaining) < _TREE_FROM // 2:
            self._drop_trees()
        else:
            self._add(slots.pop(oid), -1, -size)

    def position(self, oid) -> tuple[int, object, object]:
        """Resting order *oid*'s place in the queue.

        Returns its 1-based rank, the size queued ahead of it, and its own
        remaining size.
        """
        rank = 1
        ahead = 0
        if self._slot is None:
            for other_id, size in self._remaining.items():
                if other_id == oid:
                    return rank, ahead, size
                rank += 1
                ahead += size
            raise KeyError(oid)
        count, volume = self._count, self._volume
        i = self._slot[oid] - 1
        while i:
            rank += count[i]
            ahead += volume[i]
            i &= i - 1
        return rank, ahead, self._remaining[oid]

    def _add(self, i: int, count_delta: int, size_delta) -> None:
        count, volume, cap = self._count, self._volume, self._cap
        while i <= cap:
            count[i] += count_delta
            volume[i] += size_delta
            i += i & -i

    def _build_trees(self) -> None:
        """Number the resting orders 1, 2, ... and make room for as many again."""
        cap = 2 * len(self._remaining)
        count = [0] * (cap + 1)
        volume = [0] * (cap + 1)
        slots = {}
        for slot, (oid, size) in enumerate(self._remaining.items(), start=1):
            slots[oid] = slot
            count[slot] = 1
            volume[slot] = size
        # Build both trees in one linear pass: each node passes its total on to
        # the next node that covers it.
        for i in range(1, cap + 1):
            parent = i + (i & -i)
            if parent <= cap:
                count[parent] += count[i]
                volume[parent] += volume[i]
        self._slot = slots
        self._count, self._volume = count, volume
        self._cap = cap
        self._used = len(self._remaining)

    def _drop_trees(self) -> None:
        self._slot = None
        self._count, self._volume = [], []
        self._used = self._cap = 0


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
    order_ids = events.order_id[visible].tolist()
    times = events.timestamp[visible].tolist()
    prices = events.rests_at()[visible].tolist()
    volumes = events.volume[visible].tolist()
    directions = events.direction[visible].tolist()
    actions = events.action[visible].tolist()

    # Per level: the resting orders in arrival order, which *is* price-time
    # priority here.
    queues: dict[tuple[int, object], _ArrivalQueue] = {}
    exact = np.issubdtype(events.volume.dtype, np.integer)
    order_level: dict[int, tuple[int, object]] = {}
    placed_at: dict[int, int] = {}
    # Live (non-empty) price levels per side, for the running touch.
    live = (_LiveLevels(Direction.BID), _LiveLevels(Direction.ASK))

    rows: list[int] = []
    out_price: list[object] = []
    out_action: list[int] = []
    out_rank: list[int] = []
    out_len: list[int] = []
    out_ahead: list[object] = []
    out_remaining: list[object] = []
    out_age: list[float] = []

    def emit(row, when, oid, level, queue, action) -> None:
        direction, price = level
        if touch_only and price != live[direction].best():
            return
        rank, ahead, remaining = queue.position(oid)
        rows.append(row)
        out_price.append(price)
        out_action.append(action)
        out_rank.append(rank)
        out_len.append(len(queue))
        out_ahead.append(ahead)
        out_remaining.append(remaining)
        out_age.append(
            _elapsed_seconds(when - placed_at[oid]) if oid in placed_at else 0.0
        )

    def leave(oid: int, level: tuple[int, object]) -> None:
        del order_level[oid]
        queue = queues[level]
        queue.remove(oid)
        if not queue:
            live[level[0]].discard(level[1])

    def join(oid: int, level: tuple[int, object], volume) -> _ArrivalQueue:
        order_level[oid] = level
        queue = queues.get(level)
        if queue is None:
            queue = queues[level] = _ArrivalQueue(exact=exact)
        queue.append(oid, volume)
        live[level[0]].add(level[1])
        return queue

    for i, oid in enumerate(order_ids):
        row = visible[i]
        when = times[i]
        action = actions[i]
        volume = volumes[i]
        level = order_level.get(oid)

        if action == Action.CREATED:
            if level is not None:
                # Placed again while still queued: it leaves its old place, as
                # a cancel does, and rests once, at its latest price.
                emit(row, when, oid, level, queues[level], Action.DELETED)
                leave(oid, level)
            placed_at[oid] = when
            new_level = (directions[i], prices[i])
            emit(row, when, oid, new_level, join(oid, new_level, volume), action)
            continue

        if level is None:
            # Creation never seen (pre-existing / windowed-in), or already
            # gone: cannot place it.
            continue
        queue = queues[level]

        if action == Action.DELETED or volume <= 0:
            emit(row, when, oid, level, queue, Action.DELETED)  # last place
            leave(oid, level)
            continue

        new_level = (level[0], prices[i])
        if new_level != level:
            # A move to a new price: out of the old level, as a cancel leaves
            # it, then onto the back of the new one.
            emit(row, when, oid, level, queue, Action.DELETED)
            leave(oid, level)
            queue = join(oid, new_level, volume)
            emit(row, when, oid, new_level, queue, Action.CREATED)
            continue

        queue.resize(oid, volume)  # a size change keeps the queue place
        emit(row, when, oid, level, queue, Action.CHANGED)

    return QueuePositions(
        row=np.array(rows, dtype=np.int64),
        price=np.asarray(out_price, dtype=events.rests_at().dtype),
        action=np.array(out_action, dtype=np.int8),
        rank=np.array(out_rank, dtype=np.int64),
        queue_len=np.array(out_len, dtype=np.int64),
        # Sizes keep the dtype they arrived in, so a canonical integer-lot
        # stream (issue #226) gives exact queue totals: ``ahead_volume`` is a
        # running sum down a level, which is precisely where a float drifts.
        ahead_volume=np.asarray(out_ahead, dtype=events.volume.dtype),
        remaining=np.asarray(out_remaining, dtype=events.volume.dtype),
        age_s=np.array(out_age, dtype=np.float64),
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
            [_elapsed_seconds(sample - placed_at[oid]) for oid in queues[touch]]
        )

    max_rank = max((len(column) for column in snapshots), default=0)
    ages = np.full((max_rank, len(at)), np.nan, dtype=float)
    for t, column in enumerate(snapshots):
        if column:
            ages[: len(column), t] = column
    return QueueAgeGrid(ages=ages, max_rank=max_rank)
