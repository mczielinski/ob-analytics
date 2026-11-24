"""Compiled loops for the L2 depth replay.

:class:`~ob_analytics.depth.PriceLevelBook` and
:class:`~ob_analytics.depth.DepthSummaryEngine` keep their state in plain numpy
arrays and call the functions here, which numba compiles to machine code on
first use.  The replay applies one depth row at a time, and each row did about
a dozen small numpy calls when it ran in Python.  Almost all of that time was
the cost of the calls, not the work in them.  Here each row is a few loops over
the arrays, with no Python in between.

The book state
--------------
Each side is three arrays of the same length: the prices, the volumes, and the
update count at which each level was last set.  Each side keeps its best price
last: the bids ascending by price, the asks descending.  Most rows change a
level near the best price, so adding or removing one moves only the few levels
between it and the best.  Only the first ``n`` entries of a side are levels;
the rest is spare room, so a new level does not allocate.  ``meta`` holds the
counts, in the order of the ``META_*`` indices below.

:func:`apply_row` does not grow a side.  Its caller makes sure the side a row
may add a level to has room for one more before it applies the row.
:func:`replay` does that itself, and returns the new arrays.

Outputs are the same as the numpy code this replaced, value for value.  The
float arithmetic is done in the same order: the bin sums add levels from the
best price outwards, as ``numpy.cumsum`` did, and the window edges round half
to even, as Python's ``round`` does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit

META_BIDS = 0
"""Index in ``meta`` of the number of bid levels."""
META_ASKS = 1
"""Index in ``meta`` of the number of ask levels."""
META_CROSSED_BIDS = 2
"""Index in ``meta`` of the number of bids left out of the quoted book."""
META_CROSSED_ASKS = 3
"""Index in ``meta`` of the number of asks left out of the quoted book."""
META_CLOCK = 4
"""Index in ``meta`` of the number of rows applied so far."""
META_SIZE = 5
"""Length of ``meta``."""


def _compiled(func: Callable[..., Any]) -> Any:
    """Compile *func* with numba, caching the machine code on disk if it can.

    numba keeps its cache next to this file, or in a cache folder in the
    user's home when this folder is read-only.  When neither can be written,
    as in some containers, caching raises at import.  The function is then
    compiled again in each process instead.
    """
    try:
        return njit(cache=True)(func)
    except RuntimeError:
        return njit(func)


@_compiled
def _find(prices, n, price, order):
    """The first index in ``prices[:n]`` not before *price* in the side's order.

    *order* is ``1`` for a side kept ascending and ``-1`` for one kept
    descending.
    """
    lo = 0
    hi = n
    key = order * price
    while lo < hi:
        mid = (lo + hi) >> 1
        if order * prices[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


@_compiled
def _set_level(prices, volumes, stamps, n, price, volume, clock, order):
    """Set, add or remove one level of a side; return the side's new length.

    The arrays must have room for ``n + 1`` levels.  *order* is ``1`` for the
    bids, kept ascending, and ``-1`` for the asks, kept descending.
    """
    i = _find(prices, n, price, order)
    found = i < n and prices[i] == price
    if volume > 0:
        if found:
            volumes[i] = volume
            stamps[i] = clock
            return n
        for j in range(n, i, -1):
            prices[j] = prices[j - 1]
            volumes[j] = volumes[j - 1]
            stamps[j] = stamps[j - 1]
        prices[i] = price
        volumes[i] = volume
        stamps[i] = clock
        return n + 1
    if found:
        for j in range(i, n - 1):
            prices[j] = prices[j + 1]
            volumes[j] = volumes[j + 1]
            stamps[j] = stamps[j + 1]
        return n - 1
    return n


@_compiled
def apply_row(
    bid_prices,
    bid_volumes,
    bid_stamps,
    ask_prices,
    ask_volumes,
    ask_stamps,
    meta,
    price,
    volume,
    side,
):
    """Apply one depth row; return whether the other side's quoted book changed.

    *side* is ``0`` for a bid and ``1`` for an ask.  A positive *volume* sets
    the size at *price*, and zero removes the level.  Afterwards the crossed
    counts in *meta* say how many levels at each side's best end the quoted
    book leaves out: while the best bid is at or above the best ask, the level
    set less recently is left out.  This is the rule of
    ``engine.crossed_prefix_counts``, applied to price levels.
    """
    n_bid = meta[META_BIDS]
    n_ask = meta[META_ASKS]
    # The walk below visits levels from each best end down to the quoted touch
    # only, so a row strictly behind its side's quoted touch cannot change
    # which levels are crossed.  Most rows are behind it.
    if side == 1:
        k = n_ask - 1 - meta[META_CROSSED_ASKS]
        behind = k >= 0 and price > ask_prices[k]
    else:
        k = n_bid - 1 - meta[META_CROSSED_BIDS]
        behind = k >= 0 and price < bid_prices[k]

    meta[META_CLOCK] += 1
    clock = meta[META_CLOCK]
    if side == 1:
        n_ask = _set_level(
            ask_prices, ask_volumes, ask_stamps, n_ask, price, volume, clock, -1
        )
        meta[META_ASKS] = n_ask
    else:
        n_bid = _set_level(
            bid_prices, bid_volumes, bid_stamps, n_bid, price, volume, clock, 1
        )
        meta[META_BIDS] = n_bid
    if behind:
        return False

    if side == 1:
        before = meta[META_CROSSED_BIDS]
    else:
        before = meta[META_CROSSED_ASKS]
    crossed_bids = 0
    crossed_asks = 0
    # Crossed or locked: walk the touch from both best ends, leaving out the
    # level set less recently each time.
    while crossed_bids < n_bid and crossed_asks < n_ask:
        b = n_bid - 1 - crossed_bids
        a = n_ask - 1 - crossed_asks
        if bid_prices[b] < ask_prices[a]:
            break
        if bid_stamps[b] <= ask_stamps[a]:
            crossed_bids += 1
        else:
            crossed_asks += 1
    meta[META_CROSSED_BIDS] = crossed_bids
    meta[META_CROSSED_ASKS] = crossed_asks
    if side == 1:
        return before != crossed_bids
    return before != crossed_asks


@_compiled
def fill_breaks(range_len, bins, out):
    """Write the last offset in each bin of a window of *range_len* ticks.

    The bins share out offsets ``0`` to ``range_len - 2`` in whole ticks; the
    last offset of the window is left out.  The boundaries never go
    backwards, so no bin is negative.  When the window has fewer ticks than
    bins, some boundaries repeat, and a bin that ends on the same boundary as
    the bin before it stays at zero.  The best price (offset ``0``) is always
    in the first bin, even when the window is a single tick.
    """
    # Capping every boundary at the last offset keeps them in order.  With at
    # least as many ticks as bins, only the last boundary changes, by one.
    cap = max(range_len - 2, 0)
    for b in range(bins):
        out[b] = min(((b + 1) * range_len + bins - 1) // bins - 1, cap)


@_compiled
def bin_walk(
    prices,
    volumes,
    start,
    step,
    direction,
    best,
    range_len,
    breaks,
    out,
    offset,
):
    """Sum level volumes into the bins that end at *breaks*.

    Visits ``prices[start]``, ``prices[start + step]`` and so on, while the
    index stays in the array, which must be in order of distance from *best*.
    A level's offset is its distance from *best* in ticks:
    ``direction * (price - best)``, so *direction* is ``1`` for asks and
    ``-1`` for bids.  Bin ``b`` gets the volume of the levels whose offsets
    are above ``breaks[b - 1]`` and at most ``breaks[b]``, written to
    ``out[offset + b]``.  Only offsets below *range_len* count, so a window
    with no ticks leaves every bin at zero.

    The result is the same, value for value, as differencing a running total
    over the dense window sampled at *breaks*: the running total adds levels
    in the same order, and an empty tick adds ``0.0``, which changes nothing.
    A negative break counts from the window's end, as a numpy index does.
    """
    total = 0.0
    previous = 0.0
    j = start
    n = prices.shape[0]
    for b in range(breaks.shape[0]):
        limit = breaks[b]
        if limit < 0:
            limit += range_len
        while 0 <= j < n:
            distance = direction * (prices[j] - best)
            if distance > limit or distance >= range_len:
                break
            total += volumes[j]
            j += step
        out[offset + b] = total if b == 0 else total - previous
        previous = total


@_compiled
def write_side(
    bid_prices,
    bid_volumes,
    ask_prices,
    ask_volumes,
    meta,
    side,
    bps,
    bins,
    breaks,
    out,
):
    """Write one side's columns of a depth summary row into *out*.

    The columns are the best price, the volume at it, and the volume in each
    of *bins* bins of *bps* basis points from the best price outwards, over
    the quoted levels only.  The bids come first in the row, then the asks.
    *breaks* is room for *bins* integers.
    """
    if side == 1:
        offset = 2 + bins
        quoted = meta[META_ASKS] - meta[META_CROSSED_ASKS]
        prices = ask_prices[:quoted]
        volumes = ask_volumes[:quoted]
    else:
        offset = 0
        quoted = meta[META_BIDS] - meta[META_CROSSED_BIDS]
        prices = bid_prices[:quoted]
        volumes = bid_volumes[:quoted]
    if quoted <= 0:
        for c in range(offset, offset + 2 + bins):
            out[c] = 0.0
        return

    best = prices[quoted - 1]
    out[offset] = best
    out[offset + 1] = volumes[quoted - 1]
    # The window covered by the bins, as the depth summary has always drawn
    # it: from the best ask up to end_value inclusive, or from the best bid
    # down to end_value inclusive.  np.rint rounds half to even, as Python's
    # round does.
    width = bps * bins * 0.0001
    if side == 1:
        end_value = np.int64(np.rint((1 + width) * best)) + 1
        range_len = end_value - best + 1
        direction = 1
    else:
        end_value = np.int64(np.rint((1 - width) * best))
        range_len = best - end_value + 1
        direction = -1
    fill_breaks(range_len, bins, breaks)
    bin_walk(
        prices,
        volumes,
        quoted - 1,
        -1,
        direction,
        best,
        range_len,
        breaks,
        out,
        offset + 2,
    )


@_compiled
def grown(values):
    """A copy of *values* with twice the room, or 16 entries if that is more."""
    out = np.empty(max(2 * values.shape[0], 16), dtype=values.dtype)
    out[: values.shape[0]] = values
    return out


@_compiled
def replay(
    bid_prices,
    bid_volumes,
    bid_stamps,
    ask_prices,
    ask_volumes,
    ask_stamps,
    meta,
    prices,
    volumes,
    sides,
    bps,
    bins,
    result,
):
    """Apply every depth row in order and write the summary row after each.

    Row ``i`` of *result* is the book after row ``i`` of *prices*, *volumes*
    and *sides*.  Each row writes the side it changes, and the other side too
    when the row changed which of its levels are quoted; the rest of the row
    is carried over from the row before.  The first row starts from the book
    the arrays already hold.

    Returns the six side arrays, which are new arrays when a side ran out of
    room.
    """
    breaks = np.empty(bins, dtype=np.int64)
    row0 = result[0]
    write_side(
        bid_prices,
        bid_volumes,
        ask_prices,
        ask_volumes,
        meta,
        0,
        bps,
        bins,
        breaks,
        row0,
    )
    write_side(
        bid_prices,
        bid_volumes,
        ask_prices,
        ask_volumes,
        meta,
        1,
        bps,
        bins,
        breaks,
        row0,
    )
    for i in range(prices.shape[0]):
        row = result[i]
        if i > 0:
            row[:] = result[i - 1]
        side = sides[i]
        if side == 1:
            if meta[META_ASKS] == ask_prices.shape[0]:
                ask_prices = grown(ask_prices)
                ask_volumes = grown(ask_volumes)
                ask_stamps = grown(ask_stamps)
        elif meta[META_BIDS] == bid_prices.shape[0]:
            bid_prices = grown(bid_prices)
            bid_volumes = grown(bid_volumes)
            bid_stamps = grown(bid_stamps)
        if apply_row(
            bid_prices,
            bid_volumes,
            bid_stamps,
            ask_prices,
            ask_volumes,
            ask_stamps,
            meta,
            prices[i],
            volumes[i],
            side,
        ):
            write_side(
                bid_prices,
                bid_volumes,
                ask_prices,
                ask_volumes,
                meta,
                1 - side,
                bps,
                bins,
                breaks,
                row,
            )
        write_side(
            bid_prices,
            bid_volumes,
            ask_prices,
            ask_volumes,
            meta,
            side,
            bps,
            bins,
            breaks,
            row,
        )
    return bid_prices, bid_volumes, bid_stamps, ask_prices, ask_volumes, ask_stamps
