"""Frame adapters for the engine's FIFO queue reconstruction (WS-4.1).

The reconstruction itself is :mod:`ob_analytics.engine`: a single time-ordered
pass over the event stream rebuilds, per ``(direction, price)`` level, the
price-time-priority queue of resting orders, and reports each order's **rank**
in its level (1 = front), the **volume ahead** of it, the **queue length**, and
its **age** — the inputs to the ``queue_position`` and ``liquidity_at_touch`` L3
faces.

What lives here is the pandas face of that engine: the canonical same-instant
sort (issue #154), the display-level filter, the sampling window, and the frames
the plotting layer consumes.

Visible-only caveat: hidden orders (LOBSTER ``id == 0`` / type-5 executions)
never join the visible queue and are excluded, so reconstructed touch volume
matches the *visible* book, not the full book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ob_analytics import _engine_frames, engine
from ob_analytics.schemas import time_order_keys

QUEUE_COLUMNS: tuple[str, ...] = _engine_frames.QUEUE_COLUMNS

_ENGINE_COLUMNS = [
    "event_id",
    "id",
    "timestamp",
    "price",
    "volume",
    "direction",
    "action",
]


def _in_canonical_order(events: pd.DataFrame) -> pd.DataFrame:
    """The columns the queue engine reads, in the canonical same-instant order.

    A timestamp alone cannot order events that share an instant, so the frame is
    sorted by the total order :func:`~ob_analytics.schemas.time_order_keys`
    defines (issue #154) before the engine replays it — the replay order *is*
    the queue's priority, so this is what makes the reconstruction reproducible
    run to run and engine to engine.
    """
    ev = events[_ENGINE_COLUMNS]
    return ev.sort_values(time_order_keys(ev), kind="stable")


def queue_positions(
    events: pd.DataFrame,
    *,
    levels: str = "touch",
) -> pd.DataFrame:
    """Reconstruct FIFO queue position for each visible limit order over time.

    Parameters
    ----------
    events : pandas.DataFrame
        Canonical events (``event_id``, ``id``, ``timestamp``, ``price``,
        ``volume`` = outstanding-after-event, ``direction``, ``action``).
    levels : {"touch", "all"}
        ``"touch"`` (default) keeps only rows where the order rests at the best
        bid/ask at that instant — the input to the touch-queue faces. ``"all"``
        keeps every visible level.

    Returns
    -------
    pandas.DataFrame
        One row per order event with columns :data:`QUEUE_COLUMNS`.  ``rank``
        is 1-based from the front; ``ahead_volume`` sums the remaining size of
        the orders ahead; ``age_s`` is seconds since the order's creation.

    Notes
    -----
    Price-time priority: ``created`` appends to the back of its level; a size
    reduction (partial fill or partial cancel) keeps the order's place; a
    ``deleted`` (or a reduction to zero) removes it.  Hidden orders (``id == 0``)
    are skipped.  Within an order's lifetime ``rank`` is monotone non-increasing
    (FIFO: newcomers join the back), which the tests assert.
    """
    if levels not in ("touch", "all"):
        raise ValueError(f"levels must be 'touch' or 'all', got {levels!r}")

    ev = _in_canonical_order(events)
    positions = engine.queue_positions(
        _engine_frames.to_order_events(ev), touch_only=levels == "touch"
    )
    return _engine_frames.queue_frame(ev, positions)


def queue_age_grid(
    events: pd.DataFrame,
    *,
    side: str = "bid",
    n_time: int = 200,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Touch-queue composition over time: the age of the order at each rank.

    Replays one side's events and snapshots its touch (best) level at *n_time*
    evenly spaced instants, recording each resting order's age by FIFO rank.
    The grid feeds the ``liquidity_at_touch`` L3 composition strip (a
    ``pcolormesh``/``Heatmap`` of age over time x rank).

    Parameters
    ----------
    events : pandas.DataFrame
        Canonical events (``event_id``/``id``/``timestamp``/``price``/
        ``volume``/``direction``/``action``).
    side : {"bid", "ask"}
        Which touch to compose (default ``"bid"`` -- the front HFT queue-position
        research lives in).
    n_time : int
        Number of time columns.

    Returns
    -------
    (ages, times, max_rank)
        ``ages`` is a ``(max_rank, n_time)`` float array: ``ages[r, t]`` is the
        age in **seconds** of the order at rank ``r + 1`` (front = row 0) at
        sample ``t``, or ``NaN`` where the queue is shorter than ``r + 1``.
        ``times`` is the length-``n_time`` array of sample timestamps.
        Visible-only (hidden orders absent).
    """
    if side not in ("bid", "ask"):
        raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")

    touch = engine.Direction[side.upper()]
    ev = _in_canonical_order(events)
    arrays = _engine_frames.to_order_events(ev)
    # The window spans exactly the rows the engine will replay, so ask the
    # engine which those are rather than restating its visibility rule here.
    replayed = arrays.visible(side=touch)
    if replayed.size == 0 or n_time < 1:
        return np.empty((0, 0)), np.array([], dtype="datetime64[ns]"), 0

    # The sampling window is a display choice, so it is set here and handed to
    # the engine, which only replays to the instants it is given.
    samples = pd.date_range(
        ev["timestamp"].iloc[replayed[0]],
        ev["timestamp"].iloc[replayed[-1]],
        periods=n_time,
    )
    grid = engine.queue_age_grid(
        arrays, side=touch, at=_engine_frames.nanoseconds(samples)
    )
    return grid.ages, samples.to_numpy(), grid.max_rank
