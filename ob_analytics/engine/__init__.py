"""The order-book engine: order events in, book states and lifecycles out.

This is the stateful core of ob-analytics — the part that replays a stream of
order events and works out what the book looked like, who was queued behind
whom, and what became of each order.  It is deliberately the *only* part of the
library with that knowledge, and deliberately the smallest surface the rest of
the library can be built on.

The boundary
------------
Everything crosses this interface as **numpy arrays**: the shared event schema
(issue #112) in column form (:class:`OrderEvents`) on the way in, and small
result records on the way out.  The engine imports no pandas, knows nothing
about analytics, plots, sources, or files, and never reads a column it was not
handed.  That is what lets the inside be replaced — numba first, maybe Rust
later (issue #138) — or fed one event at a time (issue #139) without touching
anything above it.

Row indices instead of copies
-----------------------------
Output records carry a ``row`` (or ``created_row``) index into the input arrays
rather than copying the event's own columns back out.  A caller reads the
``exchange_timestamp``, the classifier ``type``, the venue and symbol, or
anything else it tracks straight off its own table at that row.  So adding a
column to the schema never widens this interface, and the engine never has to
learn a vocabulary — order types, venue names — that belongs to the layer above.

The interface
-------------
* :func:`book_state` — the book at one instant, both sides.
* :func:`order_lifecycles` — one row per order: placement to outcome.
* :func:`queue_positions` — each order's FIFO place, event by event.
* :func:`queue_age_grid` — the touch queue's composition over time.
* :func:`crossed_prefix_counts` — how much of a crossed touch to evict.

Time, price, and codes
----------------------
Timestamps are **int64 nanoseconds since the epoch, UTC** (issue #154): a zone
is a presentation detail, so it is stripped on the way in and re-attached on the
way out.  Prices are **integer ticks** (issue #155); the engine only compares
and subtracts them, so a float column from a pre-tick frame still works.

Categorical columns arrive as :class:`Direction`, :class:`Action`, and
:class:`Outcome` codes.  Each is an ``IntEnum`` that derives the schema's string
from its own member name, so the integer the engine compares and the label a
frame carries cannot drift apart.

Using it from pandas
--------------------
:mod:`ob_analytics._engine_frames` is the adapter that converts a canonical
events DataFrame into :class:`OrderEvents` and turns these records back into
frames.  It is the one place in the library where pandas and the engine meet;
:func:`ob_analytics.analytics.order_book`,
:func:`ob_analytics.analytics.order_lifecycles`, and the functions in
:mod:`ob_analytics.queue` are its callers.
"""

from __future__ import annotations

from ob_analytics.engine._book import (
    BookSide,
    BookState,
    book_state,
    crossed_prefix_counts,
)
from ob_analytics.engine._events import (
    ACTIONS,
    DIRECTIONS,
    HIDDEN_ORDER_ID,
    NAT_NS,
    Action,
    Code,
    Direction,
    OrderEvents,
)
from ob_analytics.engine._lifecycles import (
    DEFAULT_FILL_TOLERANCE,
    OUTCOMES,
    OrderLifecycles,
    Outcome,
    order_lifecycles,
)
from ob_analytics.engine._queue import (
    QueueAgeGrid,
    QueuePositions,
    queue_age_grid,
    queue_positions,
)

__all__ = [
    "ACTIONS",
    "DEFAULT_FILL_TOLERANCE",
    "DIRECTIONS",
    "HIDDEN_ORDER_ID",
    "NAT_NS",
    "OUTCOMES",
    "Action",
    "BookSide",
    "BookState",
    "Code",
    "Direction",
    "OrderEvents",
    "OrderLifecycles",
    "Outcome",
    "QueueAgeGrid",
    "QueuePositions",
    "book_state",
    "crossed_prefix_counts",
    "order_lifecycles",
    "queue_age_grid",
    "queue_positions",
]
