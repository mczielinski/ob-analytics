"""The pandas side of the order-book engine's boundary.

:mod:`ob_analytics.engine` speaks numpy arrays and integer nanoseconds; the rest
of the library speaks pandas DataFrames with the column contract in
:mod:`ob_analytics.schemas`.  This module is the one place the two meet.  It
converts a canonical events frame into the engine's :class:`~ob_analytics.
engine.OrderEvents` input, and turns the engine's result records back into the
frames the analytics and plotting layers expect.

Keeping the conversion here — rather than inside the engine — is what lets the
engine stay pandas-free, so it can be replaced with a faster implementation
without any of this moving.  Nothing here is public API; the callers are
:func:`ob_analytics.analytics.order_book`,
:func:`ob_analytics.analytics.order_lifecycles`, and :mod:`ob_analytics.queue`.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ob_analytics.engine import (
    ACTIONS,
    DIRECTIONS,
    OUTCOMES,
    Action,
    BookSide,
    OrderEvents,
    OrderLifecycles,
    QueuePositions,
)

TimestampLike = datetime | np.datetime64 | str | pd.Timestamp
"""What :func:`instant_ns` accepts as a snapshot instant."""


# ── Time ──────────────────────────────────────────────────────────────
#
# The engine works in int64 nanoseconds since the epoch, UTC (issue #154): a
# time zone is a presentation detail, so it is stripped on the way in and
# re-attached on the way out from whichever column the caller is mirroring.


def nanoseconds(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return a datetime column as int64 nanoseconds since the epoch, UTC."""
    return pd.DatetimeIndex(times).as_unit("ns").astype("int64").to_numpy()


def instant_ns(value: TimestampLike, *, like: pd.Series) -> int:
    """Return one snapshot instant as int64 nanoseconds, on *like*'s clock.

    Raises
    ------
    TypeError
        If *value* and *like* disagree about being time-zone aware — the same
        refusal pandas makes when comparing a naive instant with an aware
        column, raised here because the engine compares plain integers and would
        otherwise silently mix the two clocks.
    """
    stamp = pd.Timestamp(value)
    column_tz = getattr(like.dtype, "tz", None)
    if (stamp.tz is None) != (column_tz is None):
        raise TypeError(
            "Cannot compare tz-naive and tz-aware timestamps: the snapshot "
            f"instant is {'naive' if stamp.tz is None else str(stamp.tz)} and "
            f"the event timestamps are "
            f"{'naive' if column_tz is None else str(column_tz)}."
        )
    return int(pd.DatetimeIndex([stamp]).as_unit("ns").astype("int64")[0])


def timestamps(values: np.ndarray, *, like: pd.Series) -> pd.DatetimeIndex:
    """Return int64-nanosecond *values* as datetimes on *like*'s clock.

    ``NaT`` sentinels survive the round trip: NumPy stores ``NaT`` as the
    smallest int64, which is what the engine writes for "no time".
    """
    index = pd.DatetimeIndex(np.asarray(values, dtype=np.int64).view("datetime64[ns]"))
    if getattr(like.dtype, "tz", None) is not None:
        index = index.tz_localize("UTC")
    # `astype` carries the column's own time zone and resolution across, so the
    # rebuilt column is indistinguishable from one the loader produced.
    return pd.DatetimeIndex(index.astype(like.dtype))


# ── Events in ─────────────────────────────────────────────────────────


def _codes(values: pd.Series, labels: tuple[str, ...]) -> np.ndarray:
    """Map a label column onto its engine codes; ``-1`` for anything unknown.

    ``-1`` is deliberate rather than an error: an unrecognised direction or
    action matches no branch in the engine, which is exactly how the string
    comparisons it replaced behaved.
    """
    return pd.Categorical(values, categories=labels).codes


def _sizes(values: pd.Series) -> np.ndarray:
    """Return a size column as the engine's array, keeping it exact.

    Canonical sizes are integer lots, and the engine sums them —
    ``liquidity`` down a book side, ``ahead_volume`` along a queue — so passing
    the integers through keeps those totals exact.  An older frame still
    holding float sizes in the base asset is passed through unchanged rather
    than rounded onto a lot grid this function does not know.
    """
    array = values.to_numpy()
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64)
    return array.astype(np.float64)


def resting_price(events: pd.DataFrame) -> pd.Series:
    """Return the price each row's order rests at once the row is applied.

    This is the library's one rule for where an order rests.  The order's
    first row, and any ``created`` row, sets the price.  After that, only a
    ``changed`` row with no fill moves it: that is a modify to a new price,
    such as Databento's.  Every other row keeps the price the order already
    had, whatever price the row carries.  A venue can report a fill at the
    price it traded at, which need not be the order's own (Bitstamp does this
    for an order that crosses and then rests), and can report a ``deleted``
    row at another price.

    An order is an id on one side of the book.  The rows must be in the
    canonical event order.

    Parameters
    ----------
    events : pandas.DataFrame
        Events with ``id``, ``direction``, ``action``, ``price`` and ``fill``.

    Returns
    -------
    pandas.Series
        One price per row, on *events*' index and in the dtype of its
        ``price`` column.
    """
    price = events["price"].to_numpy()
    order_id = events["id"].to_numpy()
    if not np.issubdtype(order_id.dtype, np.number):
        # Ids that numpy cannot sort quickly, such as UUID strings.
        order_id = pd.factorize(events["id"])[0]
    is_bid = (events["direction"] == "bid").to_numpy()
    # Group each order's rows together, keeping their order within the order.
    by_order = np.lexsort((is_bid, order_id))
    grouped_id = order_id[by_order]
    grouped_bid = is_bid[by_order]
    first = np.ones(len(by_order), dtype=bool)
    first[1:] = (grouped_id[1:] != grouped_id[:-1]) | (
        grouped_bid[1:] != grouped_bid[:-1]
    )
    action = events["action"]
    sets = (action == "created") | ((action == "changed") & (events["fill"] == 0))
    sets_price = first | sets.to_numpy()[by_order]
    # Each row reads the price of the latest row at or before it that set one.
    # An order's first row always sets one, so this never reads across orders.
    setter = np.maximum.accumulate(np.where(sets_price, np.arange(len(by_order)), 0))
    rests_at = np.empty_like(price)
    rests_at[by_order] = price[by_order][setter]
    return pd.Series(rests_at, index=events.index, name="price").astype(
        events["price"].dtype
    )


def to_order_events(
    events: pd.DataFrame, *, fill: bool = False, market: bool = False
) -> OrderEvents:
    """Convert a canonical events frame into the engine's array input.

    Only the columns the engine reads are converted.  Everything else stays in
    the frame and is read back through the row indices the engine returns.
    The resting price of each row is worked out here, by
    :func:`resting_price`, so every reconstruction places an order at the
    same price.

    Parameters
    ----------
    events : pandas.DataFrame
        Events satisfying the :mod:`ob_analytics.schemas` contract, already in
        the caller's canonical order.  The ``fill`` column is always read, to
        work out the resting price.
    fill : bool
        Also pass the ``fill`` column itself to the engine (needed for order
        lifecycles).  The column is read either way, for the resting price.
    market : bool
        Also derive the market-order mask from the classifier ``type`` column
        (needed to keep crossing orders off the reconstructed book and out of
        the queues).
    """
    return OrderEvents(
        order_id=events["id"].to_numpy(),
        timestamp=nanoseconds(events["timestamp"]),
        price=events["price"].to_numpy(),
        resting_price=resting_price(events).to_numpy(),
        # Sizes cross the boundary as the integer lots the schema stores
        # (issue #226), so the cumulative sums the engine builds from them —
        # ``liquidity`` on a book side, ``ahead_volume`` in the queue — stay
        # exact.  A pre-#226 frame still carrying float sizes is passed through
        # as it is rather than truncated to whole lots.
        volume=_sizes(events["volume"]),
        direction=_codes(events["direction"], DIRECTIONS),
        action=_codes(events["action"], ACTIONS),
        fill=_sizes(events["fill"]) if fill else None,
        is_market=(events["type"] == "market").to_numpy() if market else None,
    )


# ── Results out ───────────────────────────────────────────────────────

BOOK_SIDE_ORDER_COLUMNS: tuple[str, ...] = (
    "id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "volume",
)
"""Per-order columns of a book side: the order's latest event, with its resting price."""

BOOK_SIDE_COLUMNS: tuple[str, ...] = (*BOOK_SIDE_ORDER_COLUMNS, "liquidity", "bps")
"""Columns of a reconstructed order-book side, in order."""


def book_side_frame(events: pd.DataFrame, side: BookSide) -> pd.DataFrame:
    """Build one reconstructed book side as a frame, best price first.

    The per-order columns are taken straight off *events* at the rows the engine
    reported, so every dtype — both clocks, integer tick prices, the id column —
    survives untouched.  ``price`` is the price the order rests at, which the
    engine reports, not the price on the order's latest row.
    """
    frame = events.iloc[side.row][list(BOOK_SIDE_ORDER_COLUMNS)].copy()
    frame["price"] = pd.array(side.price).astype(events["price"].dtype)
    frame["liquidity"] = side.liquidity
    frame["bps"] = side.bps
    return frame


LIFECYCLE_PLACEMENT_COLUMNS: dict[str, str] = {
    "timestamp": "placed_ts",
    "volume": "placed_vol",
    "price": "price",
    "direction": "direction",
}
"""Columns copied from an order's ``created`` row, and their lifecycle names."""

LIFECYCLE_OPTIONAL_COLUMNS: tuple[str, ...] = ("type", "aggressiveness_bps")
"""Labels carried onto the lifecycle row when the events frame has them."""


def lifecycles_frame(events: pd.DataFrame, lifecycles: OrderLifecycles) -> pd.DataFrame:
    """Build the lifecycle table: one row per order, placement to outcome.

    Placement values are taken **per column**, as the first non-null value among
    the order's ``created`` rows.  That is only distinguishable from reading the
    engine's single ``created_row`` when one order id carries more than one
    ``created`` event, which a venue that recycles ids can produce; taking each
    column separately is what the frame contract has always done, so it is kept.
    """
    carried = [c for c in LIFECYCLE_OPTIONAL_COLUMNS if c in events.columns]
    columns = [*LIFECYCLE_PLACEMENT_COLUMNS, *carried]
    created = events.loc[events["action"] == ACTIONS[Action.CREATED]]
    placement = (
        created.groupby("id", sort=False)[columns].first().reindex(lifecycles.order_id)
    )
    frame = placement.rename(columns=LIFECYCLE_PLACEMENT_COLUMNS).reset_index(drop=True)
    frame.insert(0, "id", lifecycles.order_id)
    frame["filled_vol"] = lifecycles.filled_vol
    frame["end_ts"] = timestamps(lifecycles.end_ts, like=events["timestamp"])
    frame["outcome"] = np.asarray(OUTCOMES)[lifecycles.outcome]
    return frame


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
"""Columns of a queue-position table, in order."""


def queue_frame(events: pd.DataFrame, positions: QueuePositions) -> pd.DataFrame:
    """Build the queue-position table: one row per surviving order event.

    A move to a new price has two rows, leaving and joining.

    ``price`` is the price level the order is queued at, which the engine
    reports, not the price on the event's own row.
    """
    frame = events.iloc[positions.row][
        ["timestamp", "id", "direction", "price"]
    ].reset_index(drop=True)
    frame["price"] = pd.array(positions.price).astype(events["price"].dtype)
    frame["direction"] = frame["direction"].astype(str)
    frame["action"] = np.asarray(ACTIONS)[positions.action]
    frame["rank"] = positions.rank
    frame["queue_len"] = positions.queue_len
    frame["ahead_volume"] = positions.ahead_volume
    frame["remaining"] = positions.remaining
    frame["age_s"] = positions.age_s
    return frame
