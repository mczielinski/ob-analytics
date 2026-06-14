"""FIFO queue-position reconstruction for visible limit orders (WS-4.1).

A single time-ordered pass over the canonical events table rebuilds, per
``(direction, price)`` level, the price-time-priority queue of resting orders.
For each order event it emits the order's **rank** in its level (1 = front),
the **volume ahead** of it, the **queue length**, and its **age** — the inputs
to the ``queue_position`` and ``liquidity_at_touch`` L3 faces.

Visible-only caveat: hidden orders (LOBSTER ``id == 0`` / type-5 executions)
never join the visible queue and are excluded, so reconstructed touch volume
matches the *visible* book, not the full book.
"""

from __future__ import annotations

import pandas as pd

QUEUE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "id",
    "direction",
    "price",
    "action",
    "rank",
    "queue_len",
    "ahead_volume",
    "remaining",
    "age_s",
)


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

    cols = ["event_id", "id", "timestamp", "price", "volume", "direction", "action"]
    ev = events.loc[events["id"] != 0, cols]
    # Deterministic price-time order: timestamp, then arrival (event_id).
    ev = ev.sort_values(["timestamp", "event_id"], kind="stable")

    # Per level: insertion-ordered {id: remaining}.  Dicts preserve insertion
    # order, which *is* price-time priority here.
    queues: dict[tuple[str, float], dict[int, float]] = {}
    order_level: dict[int, tuple[str, float]] = {}
    created_ts: dict[int, pd.Timestamp] = {}
    # Live (non-empty) price levels per side, for the running touch.
    live: dict[str, set[float]] = {"bid": set(), "ask": set()}

    def _touch(direction: str) -> float | None:
        prices = live[direction]
        if not prices:
            return None
        return max(prices) if direction == "bid" else min(prices)

    rows: list[tuple] = []
    only_touch = levels == "touch"

    def emit(ts, oid, level, q, action) -> None:
        direction, price = level
        if only_touch and price != _touch(direction):
            return
        ahead = 0.0
        rank = 0
        for other_id, rem in q.items():
            rank += 1
            if other_id == oid:
                break
            ahead += rem
        age_s = (ts - created_ts[oid]).total_seconds() if oid in created_ts else 0.0
        rows.append(
            (ts, oid, direction, price, action, rank, len(q), ahead, q[oid], age_s)
        )

    for event_id, oid, ts, price, volume, direction, action in ev.itertuples(
        index=False, name=None
    ):
        if action == "created":
            level = (direction, price)
            order_level[oid] = level
            created_ts[oid] = ts
            q = queues.setdefault(level, {})
            q[oid] = float(volume)
            live[direction].add(price)
            emit(ts, oid, level, q, "created")
            continue

        level = order_level.get(oid)
        if level is None:
            # Creation never seen (pre-existing / windowed-in): cannot place it.
            continue
        q = queues.get(level, {})
        if oid not in q:
            continue

        if action == "deleted" or volume <= 0:
            emit(ts, oid, level, q, "deleted")  # last position before removal
            del q[oid]
            if not q:
                live[level[0]].discard(level[1])
            continue

        q[oid] = float(volume)  # size reduction keeps queue place
        emit(ts, oid, level, q, "changed")

    return pd.DataFrame(rows, columns=QUEUE_COLUMNS)
