"""The compiled loop behind :func:`~ob_analytics.engine.queue_positions`.

:func:`replay` rebuilds the FIFO queue at every price level, one event at a
time, and writes a row for each event it reports on.  numba compiles it to
machine code on first use.  In Python, each event cost a handful of dictionary,
heap and list calls, and almost all of the time was the cost of the calls.

Inputs are numbers only
-----------------------
The caller turns the event columns into small integer codes first, so the loop
needs no dictionaries:

* each order id becomes an index from ``0`` to the number of orders, less one;
* each price becomes its index in the sorted array of the distinct prices, so
  a level is a side and a price index: ``side * n_prices + price_index``.  Bid
  levels come first, and on each side a larger level number is a higher price.

The queue state
---------------
Each level keeps its resting orders in arrival order as a doubly linked list
of order indices, with its first and last order, its length, and, for integer
sizes, the total size resting at it.  An order is in at most one level at a
time, so the links, the remaining size and the level are arrays over the
orders.  Each side tracks its best non-empty level.  When that level empties,
the loop looks at the next levels away from the touch until it finds one that
is not empty.

Rank and size ahead
-------------------
For float sizes, the size ahead of an order is summed from the front of the
queue in queue order, as the Python code summed it, so the values are the
same.  For integer sizes the sum is exact in any order, so the loop walks from
both ends at once and stops at whichever end reaches the order first.  Most
events are about the order at the back (a new order) or near the front (a
fill), so both are found within a few steps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit

_CREATED = 0
_CHANGED = 1
_DELETED = 2
"""The :class:`~ob_analytics.engine.Action` codes, as plain numbers.

Copied here because compiled code cannot read an ``IntEnum``.
"""


def _compiled(func: Callable[..., Any]) -> Any:
    """Compile *func* with numba, caching the machine code on disk if it can.

    numba keeps its cache next to this file, or in a cache folder in the
    user's home when this folder is read-only.  When neither can be written,
    as in some containers, caching raises at import.  The function is then
    compiled again in each process instead.

    The depth replay has the same helper.  The engine imports nothing from the
    layers above it, so it keeps its own copy.
    """
    try:
        return njit(cache=True)(func)
    except RuntimeError:
        return njit(func)


@_compiled
def elapsed_seconds(delta_ns):
    """Seconds between two nanosecond instants.

    Reproduces ``pandas.Timedelta.total_seconds()`` bit for bit, down to its
    two quirks: the result is truncated to whole microseconds, and the whole
    seconds are added to the fraction *after* the division rather than before,
    which moves the last bit on some values.  The golden-output tests pin these
    numbers, so the arithmetic is copied rather than corrected.  Integer
    ``//`` and ``%`` round towards minus infinity here, as Python's ``divmod``
    does.
    """
    seconds = delta_ns // 1_000_000_000
    remainder = delta_ns % 1_000_000_000
    return seconds + (remainder // 1000) / 1e6


@_compiled
def _position(
    order, level, head, tail, length, total, nxt, prv, remaining, exact, zero
):
    """The 1-based rank of *order* in *level* and the size queued ahead of it.

    *zero* is zero in the size dtype.
    """
    if not exact:
        rank = 1
        ahead = zero
        k = head[level]
        while k != order:
            ahead += remaining[k]
            rank += 1
            k = nxt[k]
        return rank, ahead
    rank = 1
    ahead = zero
    behind_count = 0
    behind = zero
    front = head[level]
    back = tail[level]
    while True:
        if front == order:
            return rank, ahead
        if back == order:
            return (
                length[level] - behind_count,
                total[level] - behind - remaining[order],
            )
        ahead += remaining[front]
        rank += 1
        front = nxt[front]
        behind += remaining[back]
        behind_count += 1
        back = prv[back]


@_compiled
def _grown(values, size):
    """A copy of *values* with room for at least *size* entries."""
    out = np.empty(max(2 * values.shape[0], size, 16), dtype=values.dtype)
    out[: values.shape[0]] = values
    return out


@_compiled
def replay(
    orders,
    times,
    price_index,
    volumes,
    directions,
    actions,
    n_orders,
    n_prices,
    exact,
    touch_only,
):
    """Rebuild every level's FIFO queue and report each order's place in it.

    Parameters
    ----------
    orders : numpy.ndarray
        Each event's order index, from ``0`` to *n_orders* less one.
    times : numpy.ndarray
        Each event's time, in int64 nanoseconds.
    price_index : numpy.ndarray
        Each event's resting price, as an index from ``0`` to *n_prices* less
        one into the sorted distinct prices.
    volumes : numpy.ndarray
        Each event's remaining size: ``int64`` when *exact*, else ``float64``.
    directions, actions : numpy.ndarray
        Each event's side and action codes.
    exact : bool
        Whether the sizes are integers, so that sums are exact in any order.
    touch_only : bool
        Report only events at the best level of their side at that instant.

    Returns
    -------
    tuple of numpy.ndarray
        For each reported row: the event index, the level, the action code,
        the rank, the queue length, the size ahead, and the remaining size and
        age in seconds of the order.
    """
    n = orders.shape[0]
    n_levels = 2 * n_prices
    zero = volumes[:0].sum()

    level_of = np.full(n_orders, -1, dtype=np.int64)
    remaining = np.zeros(n_orders, dtype=volumes.dtype)
    placed = np.zeros(n_orders, dtype=np.int64)
    nxt = np.full(n_orders, -1, dtype=np.int64)
    prv = np.full(n_orders, -1, dtype=np.int64)

    head = np.full(n_levels, -1, dtype=np.int64)
    tail = np.full(n_levels, -1, dtype=np.int64)
    length = np.zeros(n_levels, dtype=np.int64)
    total = np.zeros(n_levels, dtype=volumes.dtype)
    # The best non-empty level of each side: the highest bid level, the lowest
    # ask level.  -1 and n_levels mean the side is empty.
    best_bid = -1
    best_ask = n_levels

    cap = max(n, 16)
    out_row = np.empty(cap, dtype=np.int64)
    out_level = np.empty(cap, dtype=np.int64)
    out_action = np.empty(cap, dtype=np.int8)
    out_rank = np.empty(cap, dtype=np.int64)
    out_len = np.empty(cap, dtype=np.int64)
    out_ahead = np.empty(cap, dtype=volumes.dtype)
    out_remaining = np.empty(cap, dtype=volumes.dtype)
    out_age = np.empty(cap, dtype=np.float64)
    count = 0

    for i in range(n):
        order = orders[i]
        action = actions[i]
        volume = volumes[i]
        level = level_of[order]

        # What this event does, as up to two steps: take the order out of
        # *leave_level*, and put it at the back of *join_level*.  -1 skips a
        # step.  *update* is a size change that keeps the order's place.
        leave_level = -1
        join_level = -1
        update = False
        if action == _CREATED:
            leave_level = level
            join_level = directions[i] * n_prices + price_index[i]
        elif level < 0:
            # Creation never seen (before the window), or already gone.
            continue
        elif action == _DELETED or volume <= zero:
            leave_level = level
        else:
            moved = (level // n_prices) * n_prices + price_index[i]
            if moved != level:
                leave_level = level
                join_level = moved
            else:
                update = True

        for step in range(3):
            if step == 0:
                if leave_level < 0:
                    continue
                target = leave_level
                code = _DELETED
            elif step == 1:
                if not update:
                    continue
                target = level
                code = _CHANGED
                if exact:
                    total[target] += volume - remaining[order]
                remaining[order] = volume
            else:
                if join_level < 0:
                    continue
                target = join_level
                code = _CREATED
                if action == _CREATED:
                    # Placed now.  A row reporting it leaving an earlier place
                    # still counts its age from that earlier placement.
                    placed[order] = times[i]
                # Onto the back of the new level.
                level_of[order] = target
                remaining[order] = volume
                if exact:
                    total[target] += volume
                last = tail[target]
                prv[order] = last
                nxt[order] = -1
                if last < 0:
                    head[target] = order
                else:
                    nxt[last] = order
                tail[target] = order
                length[target] += 1
                if target < n_prices:
                    best_bid = max(best_bid, target)
                else:
                    best_ask = min(best_ask, target)

            # Report the order's place: after it joins, before it leaves.
            at_touch = target == best_bid or target == best_ask
            if at_touch or not touch_only:
                if count == cap:
                    cap = 2 * cap
                    out_row = _grown(out_row, cap)
                    out_level = _grown(out_level, cap)
                    out_action = _grown(out_action, cap)
                    out_rank = _grown(out_rank, cap)
                    out_len = _grown(out_len, cap)
                    out_ahead = _grown(out_ahead, cap)
                    out_remaining = _grown(out_remaining, cap)
                    out_age = _grown(out_age, cap)
                rank, ahead = _position(
                    order,
                    target,
                    head,
                    tail,
                    length,
                    total,
                    nxt,
                    prv,
                    remaining,
                    exact,
                    zero,
                )
                out_row[count] = i
                out_level[count] = target
                out_action[count] = code
                out_rank[count] = rank
                out_len[count] = length[target]
                out_ahead[count] = ahead
                out_remaining[count] = remaining[order]
                out_age[count] = elapsed_seconds(times[i] - placed[order])
                count += 1

            if step == 0:
                # Out of its level, as a cancel leaves it.
                level_of[order] = -1
                if exact:
                    total[target] -= remaining[order]
                before = prv[order]
                after = nxt[order]
                if before < 0:
                    head[target] = after
                else:
                    nxt[before] = after
                if after < 0:
                    tail[target] = before
                else:
                    prv[after] = before
                length[target] -= 1
                if length[target] == 0:
                    if target == best_bid:
                        while best_bid >= 0 and length[best_bid] == 0:
                            best_bid -= 1
                    elif target == best_ask:
                        while best_ask < n_levels and length[best_ask] == 0:
                            best_ask += 1

    return (
        out_row[:count],
        out_level[:count],
        out_action[:count],
        out_rank[:count],
        out_len[:count],
        out_ahead[:count],
        out_remaining[:count],
        out_age[:count],
    )
