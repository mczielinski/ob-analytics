"""The engine's input: the canonical event schema as numpy columns.

:class:`OrderEvents` is the array form of the shared event schema (issue #112)
documented in :mod:`ob_analytics.schemas`.  It carries only the columns the
rebuild actually reads; everything else a caller wants on an output row —
``exchange_timestamp``, ``event_id``, the classifier ``type``, instrument
identity — travels back as a **row index** into the same arrays, so the engine
never has to know those columns exist.

Categorical columns arrive as small integer codes rather than strings: the
replay loops compare them millions of times, and integers keep the engine free
of any string vocabulary the analytics layer may change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Code(IntEnum):
    """An integer code that also knows the schema's string for itself.

    The engine stores and compares the integer; a caller converting to or from
    a DataFrame needs the string.  Deriving one from the other here means the
    two can never drift apart, which a hand-maintained parallel list of labels
    would eventually do.
    """

    @property
    def label(self) -> str:
        """The string this code stands for in the shared schema."""
        return self.name.lower()

    @classmethod
    def labels(cls) -> tuple[str, ...]:
        """Every label of this vocabulary, indexed by its code."""
        return tuple(member.label for member in cls)


class Direction(Code):
    """Which side of the book an order sits on."""

    BID = 0
    ASK = 1


class Action(Code):
    """What an event did to the order it names."""

    CREATED = 0
    CHANGED = 1
    DELETED = 2


DIRECTIONS: tuple[str, ...] = Direction.labels()
"""Direction labels, indexed by :class:`Direction` code."""

ACTIONS: tuple[str, ...] = Action.labels()
"""Action labels, indexed by :class:`Action` code."""

HIDDEN_ORDER_ID: int = 0
"""Order id reserved for orders with no public identity (LOBSTER hidden
executions).  They never join the visible queue, so the queue reconstructions
skip them — see :meth:`OrderEvents.visible`."""

NAT_NS: int = np.iinfo(np.int64).min
"""Sentinel for "no time" in an int64-nanosecond column.

It is the integer NumPy stores for ``datetime64`` ``NaT``, so a caller turns a
nanosecond column into a datetime column — sentinels and all — with a plain
``values.view("datetime64[ns]")``."""


@dataclass(frozen=True)
class OrderEvents:
    """One order-event stream as parallel numpy columns.

    Every array has the same length: one entry per event, in stream order.

    Ordering contract
    -----------------
    Rows must arrive in the venue's canonical event order — the total order
    :func:`ob_analytics.schemas.time_order_keys` defines (``timestamp``, then
    the tie-breaks the frame carries).  At minimum, the rows belonging to one
    ``order_id`` must be chronological: the reconstructions read the last row
    per order as that order's current state, and replay the stream front to
    back.  Sorting is the caller's job, so the engine never has to guess which
    tie-break columns a venue publishes.

    Attributes
    ----------
    order_id : numpy.ndarray
        The venue's per-order identifier (``int64``).  :data:`HIDDEN_ORDER_ID`
        marks an order with no public identity.
    timestamp : numpy.ndarray
        Receive-clock time as **int64 nanoseconds since the epoch, UTC**
        (issue #154).  The zone is dropped on the way in and re-attached on the
        way out, so the engine compares plain integers.
    price : numpy.ndarray
        Price as a whole number of ticks (``int64``; issue #155).  A float
        column from a pre-tick frame also works — the engine only ever compares
        and subtracts prices.
    volume : numpy.ndarray
        The order's outstanding size **after** the event (``float64``), per the
        schema's volume contract.
    direction : numpy.ndarray
        :class:`Direction` codes.
    action : numpy.ndarray
        :class:`Action` codes.
    fill : numpy.ndarray or None
        Quantity executed at this event (``float64``), 0 when nothing traded.
        Required by :func:`~ob_analytics.engine.order_lifecycles`; ignored by
        the other reconstructions.
    is_market : numpy.ndarray or None
        ``True`` where the classifier labelled the order *market* — an order
        that crosses rather than rests.  Market rows never join the book, so
        :func:`~ob_analytics.engine.book_state` excludes them.  ``None`` means
        "nothing is a market order".

    Raises
    ------
    ValueError
        If the columns are not all one-dimensional and the same length.
    """

    order_id: np.ndarray
    timestamp: np.ndarray
    price: np.ndarray
    volume: np.ndarray
    direction: np.ndarray
    action: np.ndarray
    fill: np.ndarray | None = None
    is_market: np.ndarray | None = None

    def __post_init__(self) -> None:
        lengths = {}
        for name in (
            "order_id",
            "timestamp",
            "price",
            "volume",
            "direction",
            "action",
            "fill",
            "is_market",
        ):
            column = getattr(self, name)
            if column is None:
                continue
            if column.ndim != 1:
                raise ValueError(
                    f"OrderEvents.{name} must be one-dimensional, got "
                    f"{column.ndim} dimensions"
                )
            lengths[name] = len(column)
        if len(set(lengths.values())) > 1:
            raise ValueError(f"OrderEvents columns have unequal lengths: {lengths}")

    def __len__(self) -> int:
        return len(self.order_id)

    def visible(self, *, side: Direction | None = None) -> np.ndarray:
        """Rows holding visible orders, optionally on one side only.

        Orders sharing :data:`HIDDEN_ORDER_ID` have no public identity, so they
        never join the visible queue.  Both queue reconstructions select rows
        through here, and so does any caller that needs to size a window over
        the same rows the engine will replay — the rule lives in one place.
        """
        keep = self.order_id != HIDDEN_ORDER_ID
        if side is not None:
            keep = keep & (self.direction == side)
        return np.flatnonzero(keep)

    def require_fill(self, who: str) -> np.ndarray:
        """Return :attr:`fill`, or raise when the caller omitted it."""
        if self.fill is None:
            raise ValueError(f"{who}: OrderEvents.fill is required but was not given")
        return self.fill

    def market_mask(self) -> np.ndarray:
        """Return :attr:`is_market`, or an all-``False`` mask when it is absent."""
        if self.is_market is None:
            return np.zeros(len(self), dtype=bool)
        return self.is_market
