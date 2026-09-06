"""Per-order lifecycles: placement to outcome.

Collapses an event stream into one row per order — when it was placed, how much
of it traded, when it left the book, and how it ended.  One derivation, shared
by the order-book faces and every consumer that asks what became of an order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ob_analytics.engine._events import NAT_NS, Action, Code, OrderEvents

# ── Outcome code vocabulary ───────────────────────────────────────────


class Outcome(Code):
    """How an order ended."""

    RESTING = 0
    FILLED = 1
    PARTIAL = 2
    CANCELLED = 3


OUTCOMES: tuple[str, ...] = Outcome.labels()
"""Outcome labels, indexed by :class:`Outcome` code."""

DEFAULT_FILL_TOLERANCE: float = 1e-9
"""Default tolerance on "fully executed" (:func:`order_lifecycles`).

Venue volumes are 8-decimal floats, and the fills summed over one order can
drift by a float epsilon, so an order counts as filled when its executed total
reaches its placed size to within this much."""


@dataclass(frozen=True)
class OrderLifecycles:
    """One row per order, in the order the orders were placed.

    As with :class:`~ob_analytics.engine.BookSide`, the placement columns are
    not copied: :attr:`created_row` points at the order's ``created`` event, so
    a caller reads the placement price, size, direction, and any label it
    attached (the classifier ``type``, the placement aggressiveness) off its own
    event table at that row.

    Orders with no ``created`` row are absent — a pre-existing opening book and
    hidden executions have no placement to anchor a lifecycle to.

    Attributes
    ----------
    order_id : numpy.ndarray
        The order's identifier.
    created_row : numpy.ndarray
        Index in the :class:`OrderEvents` arrays of the order's first
        ``created`` event (``int64``).
    filled_vol : numpy.ndarray
        Total quantity executed over the order's life, in integer lots
        (``int64``).
    end_ts : numpy.ndarray
        Termination time in int64 nanoseconds, or :data:`~ob_analytics.engine.
        NAT_NS` while the order is still resting.
    outcome : numpy.ndarray
        :class:`Outcome` codes.  Flashed orders are the cancelled subset the classifier labelled
        ``flashed-limit``.
    """

    order_id: np.ndarray
    created_row: np.ndarray
    filled_vol: np.ndarray
    end_ts: np.ndarray
    outcome: np.ndarray

    def __len__(self) -> int:
        return len(self.order_id)


def _grouped_sum(slot: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Sum *values* into *n* groups, exactly.

    Sizes are integer lots, so ordinary integer accumulation is
    exact and this is one ``np.bincount``.

    It used to be Kahan summation over a per-position loop, because *values*
    were floats in the base asset and a plain accumulation drifted in the last
    bits once an order collected many small fills against a large running
    total — which moved ``filled_vol`` and, at the margin, the outcome derived
    from it.  Integers do not drift, so the compensation and the loop are both
    gone.
    """
    total = np.bincount(slot, weights=values, minlength=n)
    # ``bincount`` returns float64 whenever weights are given, so put the exact
    # integer back; every summand is an integer lot count, so nothing is lost.
    return total.astype(np.int64)


def _slots(order_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Row in *order_ids* for each entry of *values*, or ``-1`` when absent."""
    if order_ids.size == 0:
        return np.full(len(values), -1, dtype=np.int64)
    ranked = np.argsort(order_ids, kind="stable")
    sorted_ids = order_ids[ranked]
    found = np.clip(np.searchsorted(sorted_ids, values), 0, len(sorted_ids) - 1)
    return np.where(sorted_ids[found] == values, ranked[found], -1)


def order_lifecycles(
    events: OrderEvents, *, fill_tolerance: float = DEFAULT_FILL_TOLERANCE
) -> OrderLifecycles:
    """Collapse *events* into one row per order.

    Termination follows the schema's volume contract: an order ends when a
    ``deleted`` row arrives **or** its outstanding size reaches zero.  The
    second is how a fully executed order ends on a venue that emits no delete
    for it (LOBSTER); the ``created`` row itself is excluded from the test so a
    zero-size placement does not terminate itself.

    Parameters
    ----------
    events : OrderEvents
        The event stream, in canonical order, carrying ``fill``.
    fill_tolerance : float
        How far short of its placed size an order's executed total may fall and
        still count as fully filled.  Defaults to
        :data:`DEFAULT_FILL_TOLERANCE`; raise it for a venue whose quantities
        are coarser than 8 decimal places.

    Returns
    -------
    OrderLifecycles
        One row per submitted order, ordered by first placement.
    """
    fill = events.require_fill("order_lifecycles")

    # Identity: the first `created` row of each order, in placement order.
    created_rows = np.flatnonzero(events.action == Action.CREATED)
    created_ids = events.order_id[created_rows]
    unique_ids, first_seen = np.unique(created_ids, return_index=True)
    placement_order = np.argsort(first_seen, kind="stable")
    order_id = unique_ids[placement_order]
    created_row = created_rows[first_seen[placement_order]]
    n = len(order_id)

    slot = _slots(order_id, events.order_id)
    known = slot >= 0

    # A missing fill contributes nothing, matching the aggregation this
    # replaced; the schema says the column is 0 when nothing traded.
    counted = known & ~np.isnan(fill)
    filled_vol = _grouped_sum(slot[counted], fill[counted], n)

    # Termination: an explicit delete, or an outstanding size exhausted after
    # placement.  The earliest of the two ends the order.
    terminal = np.flatnonzero(
        known
        & (
            (events.action == Action.DELETED)
            | ((events.action != Action.CREATED) & (events.volume <= 0))
        )
    )
    end_ts = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(end_ts, slot[terminal], events.timestamp[terminal])
    terminated = end_ts != np.iinfo(np.int64).max
    end_ts[~terminated] = NAT_NS

    placed_vol = events.volume[created_row]
    # Both sides are integer lot counts (issue #226), so the tolerance no
    # longer does anything: any two distinct lot counts differ by at least 1.
    # It stays because it is a documented parameter, and it costs nothing.
    fully_executed = filled_vol >= placed_vol - fill_tolerance
    outcome = np.full(n, Outcome.RESTING, dtype=np.int8)
    outcome[terminated & fully_executed & (placed_vol > 0)] = Outcome.FILLED
    outcome[terminated & ~fully_executed & (filled_vol > 0)] = Outcome.PARTIAL
    outcome[terminated & (filled_vol <= 0)] = Outcome.CANCELLED

    return OrderLifecycles(
        order_id=order_id,
        created_row=created_row,
        filled_vol=filled_vol,
        end_ts=end_ts,
        outcome=outcome,
    )
