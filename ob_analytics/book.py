"""The frame faces of the order-book engine: the book at an instant and the
per-order lifecycles.

The reconstructions themselves live in :mod:`ob_analytics.engine`, which holds
no pandas.  :func:`order_book` and :func:`order_lifecycles` are its frame
adapters: they validate the column contract, hand the engine arrays, and dress
the result back up as DataFrames.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

import pandas as pd

from ob_analytics import _engine_frames, engine
from ob_analytics._utils import require_per_order_events
from ob_analytics.schemas import validate_columns

__all__ = [
    "BookSnapshot",
    "order_book",
    "order_lifecycles",
]

# ---------------------------------------------------------------------------
# Order lifecycles
# ---------------------------------------------------------------------------


def order_lifecycles(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse events into one row per order: placement → outcome.

    The canonical lifecycle table (one derivation, shared by the L3 faces
    and the order-book reconstruction).  A thin frame wrapper over
    :func:`ob_analytics.engine.order_lifecycles`, which relies on the
    schemas.py volume contract: ``volume`` is the outstanding size after each
    event and ``fill`` the executed delta, so an order is *terminated* when a
    ``deleted`` row arrives **or its outstanding size reaches zero** — the
    latter is how fully-executed LOBSTER orders end, which never emit a
    ``deleted`` event.

    Parameters
    ----------
    events : pandas.DataFrame
        Events satisfying the schemas.py contract.  Orders without a
        ``created`` row (pre-existing book, hidden executions) are
        excluded — their placement is unknown.

    Returns
    -------
    pandas.DataFrame
        One row per order id:

        * ``id``, ``direction``, ``price`` — placement identity.
        * ``type`` — classifier label (when the column is present).
        * ``placed_ts``, ``placed_vol`` — from the ``created`` row.
        * ``filled_vol`` — total executed quantity (Σ ``fill``).
        * ``end_ts`` — termination time (``NaT`` while still resting;
          callers clip to their window end for display).
        * ``outcome`` — ``filled`` / ``partial`` / ``cancelled`` /
          ``resting``.  Flashed orders are the ``cancelled`` subset whose
          ``type`` is ``flashed-limit``.
        * ``aggressiveness_bps`` — placement distance (when present).
    """
    validate_columns(
        events,
        {"id", "timestamp", "price", "volume", "direction", "action", "fill"},
        "order_lifecycles",
    )
    require_per_order_events(events, "order_lifecycles")

    lifecycles = engine.order_lifecycles(
        _engine_frames.to_order_events(events, fill=True)
    )
    return _engine_frames.lifecycles_frame(events, lifecycles)


# ---------------------------------------------------------------------------
# Order book reconstruction
# ---------------------------------------------------------------------------


class BookSnapshot(TypedDict):
    """The order book at one instant, as :func:`order_book` returns it.

    A plain dictionary at run time; the type names its three fixed keys so
    that callers indexing it get a DataFrame rather than a wide union.

    Attributes
    ----------
    timestamp : datetime.datetime or pandas.Timestamp
        The instant the book was evaluated at.
    bids : pandas.DataFrame
        Active bid orders, best first.
    asks : pandas.DataFrame
        Active ask orders, best last.
    """

    timestamp: datetime | pd.Timestamp
    bids: pd.DataFrame
    asks: pd.DataFrame


def order_book(
    events: pd.DataFrame,
    at: datetime | None = None,
    max_levels: int | None = None,
    bps_range: int = 0,
    uncross: bool = False,
) -> BookSnapshot:
    """Reconstruct the order book at a specific point in time.

    The reconstruction itself is :func:`ob_analytics.engine.order_book`; this
    function is its frame adapter, and owns the display window (*max_levels*,
    *bps_range*) the engine has no opinion about.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing order events.
    at : datetime.datetime or pandas.Timestamp, optional
        The point in time at which to evaluate the order book.
        If None, uses the latest event timestamp in the data.
    max_levels : int, optional
        The maximum number of price levels to include for bids and asks.
    bps_range : int, optional
        Basis points range to filter the bids and asks. Default is 0.
    uncross : bool, optional
        When ``True``, evict crossed resting orders so the snapshot satisfies
        ``best_bid < best_ask`` — a *display* convenience, by the rule the
        depth summary uncrosses its touch with. At the crossed touch, the order
        whose latest event is older goes first, and orders at one price are
        walked in that order. This is the library's only way to uncross a
        per-order book: to draw an uncrossed ``book_snapshot`` or
        ``depth_chart``, pass the book it returns. The default is ``False``:
        the reconstruction stays **faithful** to the feed, so a diff feed's
        genuinely crossed resting orders (see
        :class:`~ob_analytics.protocols.FeedType`) are replayed as-is rather
        than silently uncrossed. Has no effect on a matched-book feed, which is
        never crossed.

    Returns
    -------
    BookSnapshot
        A dictionary containing:
        - 'timestamp': The evaluation timestamp.
        - 'asks': DataFrame of active ask orders.
        - 'bids': DataFrame of active bid orders.

        Each order's ``price`` is the price it rests at: the price of its
        latest ``created`` row, or of a later ``changed`` row with no fill.  A row that
        reports a fill does not move the order, so ``price`` can differ from
        the price on the order's latest row.  The depth table and the queue
        tables place orders at the same price.
    """
    validate_columns(
        events,
        {
            "action",
            "timestamp",
            "id",
            "direction",
            "type",
            "price",
            "volume",
            "fill",
            "exchange_timestamp",
        },
        "order_book",
    )
    require_per_order_events(events, "order_book")

    if at is None:
        at = events["timestamp"].max()

    pct_range = bps_range * 0.0001

    book = engine.order_book(
        _engine_frames.to_order_events(events, market=True),
        at=_engine_frames.instant_ns(at, like=events["timestamp"]),
        uncross=uncross,
    )

    # The engine hands both sides back best-first; asks are presented the other
    # way round, so the touch sits at the end of the frame.
    asks = _engine_frames.book_side_frame(events, book.asks)
    asks = asks.iloc[::-1].reset_index(drop=True)
    bids = _engine_frames.book_side_frame(events, book.bids)

    if pct_range > 0:
        if not asks.empty:
            max_ask_price = asks.iloc[-1]["price"] * (1 + pct_range)
            asks = asks[asks["price"] <= max_ask_price]
        if not bids.empty:
            min_bid_price = bids.iloc[0]["price"] * (1 - pct_range)
            bids = bids[bids["price"] >= min_bid_price]

    if max_levels is not None:
        asks = asks.tail(max_levels).reset_index(drop=True)
        bids = bids.head(max_levels).reset_index(drop=True)

    return {"timestamp": at, "asks": asks, "bids": bids}
