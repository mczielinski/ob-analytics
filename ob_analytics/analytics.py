"""Format-agnostic post-processing analytics.

Contains functions that operate on the outputs of any pipeline run,
regardless of the originating data format (Bitstamp, LOBSTER, etc.):
aggressiveness, trade impacts, order type classification, sequence-gap and
data-quality reporting, and the frame-level face of the order-book engine.

The reconstructions themselves — the book at an instant, the per-order
lifecycles — live in :mod:`ob_analytics.engine`, which holds no pandas.
:func:`order_book` and :func:`order_lifecycles` here are its frame adapters:
they validate the column contract, hand the engine arrays, and dress the result
back up as DataFrames.
"""

from __future__ import annotations

import heapq
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics import _engine_frames, engine
from ob_analytics._utils import (
    epoch_ns,
    price_scale,
    require_per_order_events,
    size_scale,
    ticks_to_price,
    validate_non_empty,
)
from ob_analytics.depth import (
    BookLevels,
    _book_levels,
    _price_level_volume,
    price_level_volume,
)
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import Clocks, FeedType, SequenceKind, TradeAttribution
from ob_analytics.schemas import (
    INGEST_SEQ_COLUMN,
    MAYBE_BAD_BOOK_ATTR,
    ORDER_TYPE_DTYPE,
    ORIGIN_COLUMN,
    SEQUENCE_COLUMN,
    SNAPSHOT_ORIGIN,
    time_order_keys,
    unsized_order_ids,
    validate_columns,
)
from ob_analytics.trade_sign import quote_before


def _event_diff_bps(
    orders: pd.DataFrame,
    standing: np.ndarray,
    depth_summary: pd.DataFrame,
    direction: int,
) -> pd.DataFrame:
    """Per-order aggressiveness in BPS vs the best price standing before it.

    *standing* is :func:`~ob_analytics.trade_sign.quote_before` for *orders*.
    *direction* is ``1`` for bids, ``-1`` for asks. Helper for
    :func:`order_aggressiveness`.
    """
    side = "bid" if direction == 1 else "ask"
    best_price_col = f"best_{side}_price"
    has_quote = (orders["direction"] == side).to_numpy() & (standing >= 0)
    merged = orders[has_quote]
    best = (
        depth_summary[best_price_col].iloc[standing[has_quote]].set_axis(merged.index)
    )

    diff_price = direction * (merged["price"] - best)
    # A distance from the touch has no meaning when the touch is not a
    # tradeable price, and dividing by it yields a signed infinity that then
    # travels through every downstream mean.  The bundled Bitstamp snapshot
    # carries orders priced at zero (``audit`` reports them as
    # ``nonpositive_price``), and the depth summary can report one of them as
    # the best when it is the more recently updated of two crossed levels, so
    # this is reachable on real data.
    best = best.where(best > 0)
    diff_bps = 10000 * diff_price / best
    return pd.DataFrame({"event_id": merged["event_id"], "diff_bps": diff_bps})


def order_aggressiveness(
    events: pd.DataFrame, depth_summary: pd.DataFrame
) -> pd.DataFrame:
    """Calculate order aggressiveness with respect to the best bid or ask in BPS.

    Each new limit order is measured against the best price on its own side of
    the book as it stood just before the order arrived: the last
    ``depth_summary`` row strictly earlier in the canonical event order
    (:func:`~ob_analytics.schemas.time_order_keys` -- ``timestamp``, then
    ``event_id`` between rows at the same instant), found by
    :func:`~ob_analytics.trade_sign.quote_before`.  A positive value means the
    order improved on that price, a negative one sat behind it.

    Parameters
    ----------
    events : pandas.DataFrame
        The events DataFrame (must contain ``direction``, ``action``, ``type``,
        ``timestamp``, ``event_id``, ``price`` columns).
    depth_summary : pandas.DataFrame
        The order book summary statistics DataFrame (must contain
        ``timestamp`` and the ``best_bid_price`` / ``best_ask_price`` columns;
        an ``event_id`` column orders rows that share a timestamp).

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with an added ``aggressiveness_bps`` column.
    """
    validate_columns(
        events,
        {"direction", "action", "type", "timestamp", "event_id", "price"},
        "order_aggressiveness(events)",
    )
    validate_columns(
        depth_summary,
        {"timestamp"},
        "order_aggressiveness(depth_summary)",
    )
    require_per_order_events(events, "order_aggressiveness")

    orders = events[
        (events["action"] != "changed")
        & events["type"].isin(["flashed-limit", "resting-limit"])
    ]
    missing = ~orders["timestamp"].isin(depth_summary["timestamp"])
    if missing.any():
        logger.debug(
            "order_aggressiveness: {}/{} order timestamps not in "
            "depth_summary (each is read against the last earlier row)",
            missing.sum(),
            len(orders),
        )
    standing = quote_before(orders, depth_summary)
    bid_diff = _event_diff_bps(orders, standing, depth_summary, 1)
    ask_diff = _event_diff_bps(orders, standing, depth_summary, -1)
    # Work on a copy: the caller's frame must not grow columns as a side
    # effect (the merges below already produce new frames).
    events = events.copy()
    events["aggressiveness_bps"] = np.nan

    if not bid_diff.empty:
        events = pd.merge(events, bid_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(
            events["diff_bps"]
        )
        events = events.drop(columns=["diff_bps"])

    if not ask_diff.empty:
        events = pd.merge(events, ask_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(
            events["diff_bps"]
        )
        events = events.drop(columns=["diff_bps"])

    return events


def trade_impacts(trades: pd.DataFrame) -> pd.DataFrame:
    """Generate a DataFrame containing order book impact summaries.

    Aggregates trade records by taker order ID to summarise how each
    aggressive order swept through the book (price range, number of fills,
    total volume, VWAP, duration).

    Parameters
    ----------
    trades : pandas.DataFrame
        The trades DataFrame (must contain ``taker``, ``price``, ``volume``,
        ``timestamp``, ``direction`` columns).

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarising market order impacts with columns:
        ``id``, ``min_price``, ``max_price``, ``vwap``, ``hits``, ``vol``,
        ``start_time``, ``end_time``, ``dir``.  The price-valued columns
        (``min_price``, ``max_price``, ``vwap``) are in the quote currency
        and ``vol`` in the instrument's own units, whether the trades hold
        integer ticks and lots (with ``tick_size`` and ``lot_size`` on the
        frame's ``attrs``, as on a pipeline result) or display prices and
        sizes.

    Raises
    ------
    ConfigError
        If a required column is missing, or the prices or sizes are integers
        and the frame carries no ``tick_size`` or ``lot_size``.
        Also if *trades* is empty.
    """
    validate_columns(
        trades,
        {"taker", "price", "volume", "timestamp", "direction"},
        "trade_impacts",
    )
    validate_non_empty(trades, "trade_impacts")

    price = trades["price"].to_numpy(dtype=np.float64) / price_scale(
        trades, "trade_impacts"
    )
    volume = trades["volume"].to_numpy(dtype=np.float64) / size_scale(
        trades, "trade_impacts"
    )
    trades_pv = trades.assign(_price=price, _volume=volume, _pv=price * volume)
    impacts = (
        trades_pv.groupby("taker")
        .agg(
            id=("taker", "last"),
            min_price=("_price", "min"),
            max_price=("_price", "max"),
            hits=("taker", "size"),
            vol=("_volume", "sum"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            dir=("direction", "last"),
            pv_sum=("_pv", "sum"),
        )
        .reset_index(drop=True)
    )
    impacts["vwap"] = impacts["pv_sum"] / impacts["vol"]
    cols = [
        "id",
        "min_price",
        "max_price",
        "vwap",
        "hits",
        "vol",
        "start_time",
        "end_time",
        "dir",
    ]
    return impacts[cols]


# ---------------------------------------------------------------------------
# Order type classification
# ---------------------------------------------------------------------------


def set_order_types(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Determine limit order types.

    Classifies each order as one of: *market*, *resting-limit*,
    *flashed-limit*, or *market-limit*, based on how the order interacts
    with the book over its lifetime.

    *market* and *market-limit* mark takers, so they need the trades' taker
    ids.  A feed that shows resting orders only (see
    :class:`~ob_analytics.protocols.TradeAttribution`) has none, and no order
    is labelled either; that is the feed, not a session without market orders.
    LOBSTER fills its taker ids with a guess, so its labels are only as good
    as the guess.

    Parameters
    ----------
    events : pandas.DataFrame
        The limit order events DataFrame.
    trades : pandas.DataFrame
        The executions DataFrame.

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with an updated 'type' column indicating order types.
    """
    validate_columns(
        events,
        {"id", "price", "action", "event_id", "direction"},
        "set_order_types(events)",
    )
    validate_columns(
        trades,
        {"maker_event_id", "taker_event_id"},
        "set_order_types(trades)",
    )
    require_per_order_events(events, "set_order_types")

    facts = _OrderFacts.of(events, _named_event_ids(trades))
    return _with_order_types(events, _order_types(facts, trades))


def _named_event_ids(trades: pd.DataFrame) -> np.ndarray:
    """Return every ``event_id`` a trade names, as maker or as taker."""
    named = pd.concat([trades["maker_event_id"], trades["taker_event_id"]])
    return named.dropna().unique()


@dataclass(frozen=True)
class _OrderFacts:
    """What the order-type rule reads of the events: a few values per order.

    :func:`set_order_types` reads them off a whole run at once.  A windowed run
    reads them one window at a time and joins the parts with :meth:`combine`,
    so it never holds the whole run's events.  The rule itself is
    :func:`_order_types`.

    Attributes
    ----------
    ids : pandas.Index
        Every order id seen, once each.
    created, deleted : pandas.DataFrame
        The ``id`` and ``volume`` of each ``created`` and each ``deleted`` row.
    changed : pandas.Index
        The ids with a ``changed`` row, once each.
    named : pandas.DataFrame
        The ``event_id`` and ``id`` of each event a trade names.
    """

    ids: pd.Index
    created: pd.DataFrame
    deleted: pd.DataFrame
    changed: pd.Index
    named: pd.DataFrame

    @classmethod
    def of(cls, events: pd.DataFrame, named_event_ids: Any) -> _OrderFacts:
        """Read the facts off *events*.

        *named_event_ids* are the ``event_id`` values the run's trades name,
        as maker or taker; an event not among them is never read as one.
        """
        action = events["action"]
        return cls(
            ids=pd.Index(events["id"]).unique(),
            created=events.loc[action == "created", ["id", "volume"]],
            deleted=events.loc[action == "deleted", ["id", "volume"]],
            changed=pd.Index(events.loc[action == "changed", "id"]).unique(),
            named=events.loc[
                events["event_id"].isin(named_event_ids), ["event_id", "id"]
            ],
        )

    @classmethod
    def combine(cls, parts: Iterable[_OrderFacts]) -> _OrderFacts:
        """Join the facts of consecutive parts of one run."""
        parts = list(parts)
        if not parts:
            raise ValueError("_OrderFacts.combine: no parts to combine")
        return cls(
            ids=parts[0].ids.append([p.ids for p in parts[1:]]).unique(),
            created=pd.concat([p.created for p in parts], ignore_index=True),
            deleted=pd.concat([p.deleted for p in parts], ignore_index=True),
            changed=parts[0].changed.append([p.changed for p in parts[1:]]).unique(),
            named=pd.concat([p.named for p in parts], ignore_index=True),
        )


def _order_types(facts: _OrderFacts, trades: pd.DataFrame) -> pd.Series:
    """Return each order's type, on an index of order ids.

    The rule :func:`set_order_types` applies; see it for what each type means.
    An order whose type the rule cannot tell is ``"unknown"``.
    """
    created, deleted = facts.created, facts.deleted
    created_ids, deleted_ids = created["id"], deleted["id"]
    changed = facts.changed

    # Created and deleted with no change in between: a flashed order, when it
    # left with the size it came with.
    created_deleted_ids = created_ids[
        created_ids.isin(deleted_ids) & ~created_ids.isin(changed)
    ]
    cd_created = created[created_ids.isin(created_deleted_ids)]
    cd_deleted = deleted[deleted_ids.isin(created_deleted_ids)]
    cd_merged = cd_created.merge(cd_deleted, on="id", suffixes=("_created", "_deleted"))
    flashed_ids = cd_merged.loc[
        cd_merged["volume_created"] == cd_merged["volume_deleted"], "id"
    ]
    forever_ids = created_ids[
        ~created_ids.isin(changed) & ~created_ids.isin(deleted_ids)
    ]

    named = facts.named
    maker_ids = pd.Index(
        named.loc[named["event_id"].isin(trades["maker_event_id"].dropna()), "id"]
    )
    taker_ids = pd.Index(
        named.loc[named["event_id"].isin(trades["taker_event_id"].dropna()), "id"]
    )

    ids = facts.ids
    is_maker, is_taker = ids.isin(maker_ids), ids.isin(taker_ids)
    # Taker labels win over resting ones, and a resting label over a flash.
    # An order first seen mid-stream (no created row) is from the book that
    # was already there, or a hidden execution: structurally unclassifiable,
    # so it gets a class of its own instead of "unknown".
    kind = np.select(
        [
            is_taker & ~is_maker,
            is_taker & is_maker,
            ids.isin(forever_ids) | (is_maker & ~is_taker),
            ids.isin(flashed_ids),
            ~ids.isin(created_ids),
        ],
        ["market", "market-limit", "resting-limit", "flashed-limit", "pre-existing"],
        default="unknown",
    )
    unidentified = int((kind == "unknown").sum())
    if unidentified:
        warnings.warn(
            f"Could not classify {unidentified} orders; their type is 'unknown'",
            # The caller of set_order_types.
            stacklevel=3,
        )
    return pd.Series(kind, index=ids, name="type")


def _with_order_types(events: pd.DataFrame, types: pd.Series) -> pd.DataFrame:
    """Return a copy of *events* with each row's order type from *types*.

    *types* is what :func:`_order_types` returns.  The copy keeps the caller's
    frame free of the new column.
    """
    events = events.copy()
    events["type"] = pd.Categorical(
        events["id"].map(types).to_numpy(), dtype=ORDER_TYPE_DTYPE
    )
    return events


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


class OrderBookSnapshot(TypedDict):
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
    tp: datetime | None = None,
    max_levels: int | None = None,
    bps_range: int = 0,
    uncross: bool = False,
) -> OrderBookSnapshot:
    """Reconstruct the order book at a specific point in time.

    The reconstruction itself is :func:`ob_analytics.engine.book_state`; this
    function is its frame adapter, and owns the display window (*max_levels*,
    *bps_range*) the engine has no opinion about.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing order events.
    tp : datetime.datetime or pandas.Timestamp, optional
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
    OrderBookSnapshot
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

    if tp is None:
        tp = events["timestamp"].max()

    pct_range = bps_range * 0.0001

    book = engine.book_state(
        _engine_frames.to_order_events(events, market=True),
        at=_engine_frames.instant_ns(tp, like=events["timestamp"]),
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

    return {"timestamp": tp, "asks": asks, "bids": bids}


# ---------------------------------------------------------------------------
# Venue sequence gap detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceGapReport:
    """Result of scanning a frame's venue ``sequence`` column for breaks.

    Built by :func:`detect_sequence_gaps`.  A feed numbers each message on a
    channel with a sequence that should rise by exactly one; a skip is a
    dropped message and a step that does not rise is a reordered (or repeated)
    one.  Rows are read in ingest order (``ingest_seq`` when present, else the
    frame's row order) and grouped per instrument / venue when those columns
    exist, so interleaved channels are scored independently.

    Attributes
    ----------
    n_sequenced : int
        Rows carrying a non-null venue ``sequence``.  ``0`` means the frame has
        nothing to check (the report is then trivially clean).
    n_updates : int
        Distinct consecutive sequence values seen — one per venue message (a
        book update that emits several rows shares one sequence, counted once).
    n_missing : int
        Total count of skipped sequence numbers: the dropped-message count.
    n_out_of_order : int
        Steps where the sequence did not advance (a repeat or a decrease) — a
        reordered or duplicated message.
    max_gap : int
        Largest single run of consecutive missing numbers (``0`` when none).
    first_break_seq : int or None
        The last in-order sequence value before the first break, a hint for
        where to resync; ``None`` when the frame is clean.
    """

    n_sequenced: int
    n_updates: int
    n_missing: int
    n_out_of_order: int
    max_gap: int
    first_break_seq: int | None

    @property
    def has_sequence(self) -> bool:
        """Whether any row carried a venue sequence to check."""
        return self.n_sequenced > 0

    @property
    def clean(self) -> bool:
        """True when no missing and no out-of-order sequence values were found."""
        return self.n_missing == 0 and self.n_out_of_order == 0

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain, JSON-serialisable dict."""
        return {
            "n_sequenced": self.n_sequenced,
            "n_updates": self.n_updates,
            "n_missing": self.n_missing,
            "n_out_of_order": self.n_out_of_order,
            "max_gap": self.max_gap,
            "first_break_seq": self.first_break_seq,
        }


_SEQUENCE_GROUP_COLUMNS: tuple[str, ...] = ("venue", "symbol")


def detect_sequence_gaps(
    frame: pd.DataFrame,
    *,
    sequence_col: str = SEQUENCE_COLUMN,
    order_col: str = INGEST_SEQ_COLUMN,
    group_cols: Sequence[str] = _SEQUENCE_GROUP_COLUMNS,
    kind: SequenceKind = SequenceKind.CONTIGUOUS,
) -> SequenceGapReport:
    """Report missing or out-of-order venue sequence numbers in *frame*.

    Parameters
    ----------
    frame : pandas.DataFrame
        Events or depth rows.  A missing *sequence_col* (a source with no venue
        sequence) yields an empty, clean report — the column is optional.
    sequence_col : str
        Column holding the venue's per-event sequence (nullable integer).
    order_col : str
        Column defining ingest order; when absent, the frame's current row
        order is used instead.
    group_cols : sequence of str
        Columns that identify one channel (instrument / venue).  Only those
        present in *frame* are used; with none present the whole frame is one
        channel.  Sequences from different channels are not comparable, so each
        group is scored on its own.
    kind : SequenceKind
        What the sequence promises.  With
        :attr:`~ob_analytics.protocols.SequenceKind.MONOTONIC` a skipped number
        is normal, so nothing is counted as missing and only steps that do not
        rise are reported.

    Returns
    -------
    SequenceGapReport

    Notes
    -----
    Consecutive equal sequence values are collapsed before comparison, so a
    single venue update that emits several rows (e.g. every changed level of one
    book diff) counts once.  A non-consecutive repeat still shows as
    out-of-order.
    """
    if sequence_col not in frame.columns or frame.empty:
        return SequenceGapReport(0, 0, 0, 0, 0, None)

    groups = [c for c in group_cols if c in frame.columns]
    work = frame
    if order_col in frame.columns:
        work = frame.sort_values(order_col, kind="stable")

    if groups:
        parts: Iterable[pd.DataFrame] = (
            part for _, part in work.groupby(groups, sort=False, dropna=False)
        )
    else:
        parts = (work,)

    n_sequenced = n_updates = n_missing = n_out_of_order = max_gap = 0
    first_break: int | None = None

    for part in parts:
        seq = part[sequence_col]
        seq = seq[seq.notna()]
        if seq.empty:
            continue
        vals = seq.to_numpy(dtype="int64")
        n_sequenced += int(vals.size)

        # Collapse consecutive repeats: one venue message can emit several rows
        # (every changed level of a book diff shares that update's sequence).
        keep = np.ones(vals.size, dtype=bool)
        keep[1:] = vals[1:] != vals[:-1]
        uniq = vals[keep]
        n_updates += int(uniq.size)
        if uniq.size < 2:
            continue

        diffs = np.diff(uniq)
        broken = diffs <= 0
        n_out_of_order += int(np.count_nonzero(broken))
        # A skip is a dropped message only when every message adds one.
        if kind is SequenceKind.CONTIGUOUS:
            gap_steps = diffs[diffs > 1] - 1
            if gap_steps.size:
                n_missing += int(gap_steps.sum())
                max_gap = max(max_gap, int(gap_steps.max()))
            broken |= diffs > 1

        if first_break is None:
            where = np.nonzero(broken)[0]
            if where.size:
                first_break = int(uniq[where[0]])

    return SequenceGapReport(
        n_sequenced, n_updates, n_missing, n_out_of_order, max_gap, first_break
    )


# ---------------------------------------------------------------------------
# Per-run data-quality summary
# ---------------------------------------------------------------------------


def _faithful_best_series(depth: pd.DataFrame) -> pd.DataFrame:
    """Faithful best bid / best ask over time from a price-level-volume frame.

    Unlike ``depth_summary`` — which the depth engine already *uncrosses* —
    this reads the raw resting-book depth from
    :func:`~ob_analytics.depth.price_level_volume` and reports the touch
    without uncrossing it, so a diff feed's crossed intervals (``best_bid >
    best_ask``) survive to be measured.  One chronological pass; each side
    keeps a lazy-deletion heap, so the running best is amortised ``O(log L)``.

    Parameters
    ----------
    depth : pandas.DataFrame
        Columns ``timestamp``, ``price``, ``volume``, ``direction`` — the
        cumulative resting volume per price level (``0`` = level empty), as
        emitted by :func:`~ob_analytics.depth.price_level_volume`.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp``, ``best_bid``, ``best_ask`` (``NaN`` where the
        side is empty), one row per input row in timestamp order.
    """
    validate_columns(
        depth, {"timestamp", "price", "volume", "direction"}, "_faithful_best_series"
    )
    ordered = depth.sort_values("timestamp", kind="stable")
    ts = ordered["timestamp"].to_numpy()
    # Read the columns as Python lists: the loop takes one element at a time,
    # which is faster from a list than from a numpy array.
    prices = ordered["price"].to_numpy(dtype=np.float64).tolist()
    vols = ordered["volume"].to_numpy(dtype=np.float64).tolist()
    is_bid = (ordered["direction"].to_numpy() == "bid").tolist()

    n = len(ordered)
    best_bid = np.full(n, np.nan)
    best_ask = np.full(n, np.nan)

    bid_vol: dict[float, float] = {}
    ask_vol: dict[float, float] = {}
    bid_heap: list[float] = []  # max-heap by price, stored negated
    ask_heap: list[float] = []  # min-heap by price

    for i, (p, v, bid) in enumerate(zip(prices, vols, is_bid, strict=True)):
        if bid:
            bid_vol[p] = v
            if v > 0:
                heapq.heappush(bid_heap, -p)
            while bid_heap and bid_vol.get(-bid_heap[0], 0.0) <= 0:
                heapq.heappop(bid_heap)
        else:
            ask_vol[p] = v
            if v > 0:
                heapq.heappush(ask_heap, p)
            while ask_heap and ask_vol.get(ask_heap[0], 0.0) <= 0:
                heapq.heappop(ask_heap)
        if bid_heap:
            best_bid[i] = -bid_heap[0]
        if ask_heap:
            best_ask[i] = ask_heap[0]

    return pd.DataFrame({"timestamp": ts, "best_bid": best_bid, "best_ask": best_ask})


def _crossed_time_fraction(best: pd.DataFrame) -> tuple[float, int]:
    """Time-weighted crossed fraction (0–1) and crossed-episode count.

    A state is *crossed* when both sides exist and ``best_bid > best_ask``.
    The book holds each row's state over the interval to the next row, so the
    fraction is duration-weighted (matching the "book was crossed for ~90 s"
    reading) rather than row-weighted.  Episodes count contiguous crossed
    intervals (rising edges).
    """
    # Raw UTC nanoseconds, where ``to_numpy()`` on the tz-aware clock would
    # give an object array of Timestamps (issue #154).
    ts = epoch_ns(best["timestamp"])
    bid = best["best_bid"].to_numpy()
    ask = best["best_ask"].to_numpy()
    crossed = (~np.isnan(bid)) & (~np.isnan(ask)) & (bid > ask)
    if crossed.size == 0:
        return 0.0, 0

    episodes = int(crossed[0]) + int(np.sum(crossed[1:] & ~crossed[:-1]))
    if crossed.size < 2:
        return (1.0 if crossed[0] else 0.0), episodes

    dt = np.diff(ts).astype(np.float64)  # interval lengths in ns
    total = float(dt.sum())
    if total <= 0:  # all events share a timestamp -> fall back to row share
        return float(crossed.mean()), episodes
    crossed_time = float(dt[crossed[:-1]].sum())
    return crossed_time / total, episodes


def _book_crossing(
    depth: pd.DataFrame,
) -> tuple[pd.DataFrame | None, float, int]:
    """How long the book in a price-level-volume frame is crossed.

    Returns the faithful best bid / best ask series (``None`` for an empty
    frame), the percentage (0–100) of the time the book is crossed, and the
    number of crossed intervals.
    """
    if depth.empty:
        return None, 0.0, 0
    best = _faithful_best_series(depth)
    share, episodes = _crossed_time_fraction(best)
    return best, 100.0 * share, episodes


# ---------------------------------------------------------------------------
# Stale resting orders
# ---------------------------------------------------------------------------

# How long a venue may take to report an order after a trade shows it is gone.
# A diff feed lags: a filled or cancelled order keeps resting until its own
# report arrives.  On the bundled Bitstamp sample that report comes a median
# 23 ms and a 95th percentile 234 ms after the trade, and the two orders it never
# arrives for outlive the trade by more than 25 minutes.  One second sits well
# clear of both, so the threshold does not need tuning per feed.
STALE_GRACE: pd.Timedelta = pd.Timedelta(seconds=1)


@dataclass(frozen=True)
class StaleOrder:
    """A resting order a trade printed through, which the venue did not report.

    A matching engine cannot print a trade at a price worse than a resting order
    on the other side of it.  So a trade above a resting ask, or below a resting
    bid, shows that the order had already left the book.  When the venue then
    does not report the order again within a grace period, the rebuilt book goes
    on holding an order that is not there.

    Attributes
    ----------
    id : int or str
        The order id, as the feed writes it: an integer on most venues, a UUID
        string on Independent Reserve.
    direction : str
        ``"bid"`` or ``"ask"``.
    price : float
        The price the order rests at, in the quote currency
        (``ticks * tick_size``).
    disproved_at : pandas.Timestamp
        The time of the first trade that printed through it.
    stale_seconds : float
        Seconds from that trade to the order's next event row, or to the end of
        the capture when it has none.
    touch_seconds : float
        Seconds of that span in which the order's price was the best price on
        its side of the faithful book: how long it held the touch after the
        trade showed it was gone.
    """

    id: int | str
    direction: str
    price: float
    disproved_at: pd.Timestamp
    stale_seconds: float
    touch_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return the order as a plain, JSON-serialisable dict."""
        return {
            "id": self.id,
            "direction": self.direction,
            "price": self.price,
            "disproved_at": self.disproved_at.isoformat(),
            "stale_seconds": self.stale_seconds,
            "touch_seconds": self.touch_seconds,
        }


def _capture_end_ns(events: pd.DataFrame, trades: pd.DataFrame) -> int:
    """The last instant either frame covers, in UTC nanoseconds."""
    ends = [
        int(epoch_ns(frame["timestamp"]).max())
        for frame in (events, trades)
        if not frame.empty
    ]
    return max(ends)


def _range_extreme(
    values: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, highest: bool
) -> np.ndarray:
    """Max (or min) of ``values[lo[i]:hi[i]]`` for every ``i``; needs ``hi > lo``.

    A sparse table answers every range in constant time, so the cost does not
    grow with how many trades an order rests across.
    """
    reduce = np.maximum if highest else np.minimum
    length = hi - lo
    level = np.floor(np.log2(length)).astype(np.int64)
    table = [values]
    width = 1
    while 2 * width <= int(length.max()):
        prev = table[-1]
        table.append(reduce(prev[:-width], prev[width:]))
        width *= 2
    out = np.empty(lo.size, dtype=values.dtype)
    for k in np.unique(level):
        rows = level == k
        span = 1 << int(k)
        out[rows] = reduce(table[k][lo[rows]], table[k][hi[rows] - span])
    return out


def _stale_spans(
    events: pd.DataFrame, trades: pd.DataFrame, grace: pd.Timedelta, end_ns: int
) -> pd.DataFrame:
    """Each order's earliest resting span that a trade printed through.

    A span runs from a ``created`` or ``changed`` row that leaves size on the
    book to the order's next row, or to *end_ns*.  It is stale when a trade
    prints through its price more than *grace* before the span ends.

    Returns
    -------
    pandas.DataFrame
        Columns ``id``, ``is_ask``, ``price`` (ticks), ``disproved_ns`` and
        ``end_ns``; one row per stale order.
    """
    empty = pd.DataFrame(
        {
            "id": pd.Series(dtype=np.int64),
            "is_ask": pd.Series(dtype=bool),
            "price": pd.Series(dtype=np.float64),
            "disproved_ns": pd.Series(dtype=np.int64),
            "end_ns": pd.Series(dtype=np.int64),
        }
    )
    rows = events
    if "type" in rows.columns:
        # Market orders never rest; the price-level rebuild skips them too.
        rows = rows[rows["type"] != "market"]
    if rows.empty or trades.empty:
        return empty
    rows = rows.sort_values(["id", *time_order_keys(rows)], kind="stable")

    ids = rows["id"].to_numpy()
    ts = epoch_ns(rows["timestamp"])
    ends = np.full(ts.size, end_ns, dtype=np.int64)
    same_order = ids[1:] == ids[:-1]
    ends[:-1] = np.where(same_order, ts[1:], end_ns)

    # A non-positive price is not a tradeable level, so no trade can print
    # through it.
    resting = (
        (rows["action"] != "deleted").to_numpy()
        & (rows["volume"] > 0).to_numpy()
        & (rows["price"] > 0).to_numpy()
    )
    ids, ts, ends = ids[resting], ts[resting], ends[resting]
    prices = rows["price"].to_numpy(dtype=np.float64)[resting]
    is_ask = (rows["direction"] == "ask").to_numpy()[resting]

    ordered = trades.sort_values("timestamp", kind="stable")
    t_ns = epoch_ns(ordered["timestamp"])
    t_px = ordered["price"].to_numpy(dtype=np.float64)

    # Only trades strictly after the row count: a trade stamped in the same
    # instant as the row cannot be put before or after it.  And only trades
    # more than *grace* before the span ends, so an order the venue reports
    # promptly is never flagged.
    lo = np.searchsorted(t_ns, ts, side="right")
    hi = np.searchsorted(t_ns, ends - int(grace.value), side="left")
    has_trades = hi > lo
    if not has_trades.any():
        return empty
    lo, hi = lo[has_trades], hi[has_trades]
    ids, ends = ids[has_trades], ends[has_trades]
    prices, is_ask = prices[has_trades], is_ask[has_trades]

    highest = _range_extreme(t_px, lo, hi, highest=True)
    lowest = _range_extreme(t_px, lo, hi, highest=False)
    through = np.where(is_ask, highest > prices, lowest < prices)
    if not through.any():
        return empty

    disproved = []
    for i in np.nonzero(through)[0]:
        window = t_px[lo[i] : hi[i]]
        hit = window > prices[i] if is_ask[i] else window < prices[i]
        disproved.append(t_ns[lo[i] + int(np.argmax(hit))])

    spans = pd.DataFrame(
        {
            "id": ids[through],
            "is_ask": is_ask[through],
            "price": prices[through],
            "disproved_ns": np.asarray(disproved, dtype=np.int64),
            "end_ns": ends[through],
        }
    )
    first = spans.groupby("id", sort=False)["disproved_ns"].idxmin()
    return spans.loc[first].reset_index(drop=True)


def _stale_orders(
    spans: pd.DataFrame, best: pd.DataFrame, end_ns: int, tick_size: float
) -> tuple[StaleOrder, ...]:
    """Measure how long each stale span held the touch, worst first."""
    bt = epoch_ns(best["timestamp"])
    held_until = np.append(bt[1:], end_ns)
    touch = {True: best["best_ask"].to_numpy(), False: best["best_bid"].to_numpy()}
    found = []
    for oid, is_ask, price, disproved, end in zip(
        spans["id"].to_numpy(),
        spans["is_ask"].to_numpy(),
        spans["price"].to_numpy(),
        spans["disproved_ns"].to_numpy(),
        spans["end_ns"].to_numpy(),
        strict=True,
    ):
        overlap = np.minimum(held_until, end) - np.maximum(bt, disproved)
        at_touch = (touch[bool(is_ask)] == price) & (overlap > 0)
        found.append(
            StaleOrder(
                # A NumPy scalar becomes the plain Python value, so the
                # report serialises to JSON.
                id=oid.item() if isinstance(oid, np.generic) else oid,
                direction="ask" if is_ask else "bid",
                price=float(ticks_to_price(price, tick_size)),
                disproved_at=pd.Timestamp(int(disproved), tz="UTC"),
                stale_seconds=int(end - disproved) / 1e9,
                touch_seconds=float(overlap[at_touch].sum()) / 1e9,
            )
        )
    found.sort(key=lambda o: (o.touch_seconds, o.stale_seconds), reverse=True)
    return tuple(found)


def detect_stale_orders(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    grace: pd.Timedelta = STALE_GRACE,
    depth: pd.DataFrame | None = None,
    tick_size: float = 1.0,
) -> tuple[StaleOrder, ...]:
    """Find resting orders a trade printed through that the venue left in place.

    A trade above a resting ask, or below a resting bid, shows that the order
    had already left the book: a matching engine fills the better price first.
    The venue normally reports the order within milliseconds.  An order it does
    not report again within *grace* stays in the rebuilt book although it is no
    longer at the venue, and distorts the spread, the depth and the queue from
    then on.

    This only reports.  It does not change what :func:`order_book` returns: the
    test needs to know what the feed said *after* the trade, which a live
    capture cannot know in time.

    The tables can be the pipeline's own (prices in integer ticks, sizes in
    integer lots; pass ``tick_size``) or the tables
    :func:`~ob_analytics.visualization.display_result` converts for display
    (prices in the quote currency, sizes in the base asset; leave
    ``tick_size`` at ``1.0``).  Both give the same orders and the same times.

    Parameters
    ----------
    events : pandas.DataFrame
        Order events with ``id``, ``timestamp``, ``price``, ``volume``,
        ``direction`` and ``action``.  Rows typed ``market`` (see
        :func:`set_order_types`) are skipped when a ``type`` column is present.
    trades : pandas.DataFrame
        Trades with ``timestamp`` and ``price``, in the same price units as
        *events*.
    grace : pandas.Timedelta, optional
        How long after the trade the venue may take to report the order.
        Defaults to :data:`STALE_GRACE` (one second).
    depth : pandas.DataFrame, optional
        The faithful price-level-volume frame, used to measure how long each
        stale order held the touch.  Computed from *events* with
        :func:`~ob_analytics.depth.price_level_volume` when ``None`` and an
        order is found.
    tick_size : float, optional
        Quote-currency size of one price tick, so the reported prices read in
        the quote currency.  Leave at ``1.0`` to report them in the units of
        ``events["price"]``: ticks for the pipeline's tables, the quote
        currency for display tables.

    Returns
    -------
    tuple of StaleOrder
        Worst first: the longest time at the touch, then the longest time
        stale.  Empty when the venue reported every order in time.
    """
    validate_columns(
        events,
        {"id", "timestamp", "price", "volume", "direction", "action"},
        "detect_stale_orders(events)",
    )
    validate_columns(trades, {"timestamp", "price"}, "detect_stale_orders(trades)")
    if events.empty or trades.empty:
        return ()
    end_ns = _capture_end_ns(events, trades)
    spans = _stale_spans(events, trades, grace, end_ns)
    if spans.empty:
        return ()
    if depth is None:
        depth = price_level_volume(events)
    return _stale_orders(spans, _faithful_best_series(depth), end_ns, tick_size)


class Severity(str, Enum):
    """How much a failed data-quality check matters.

    A check carries its severity so the policy — what fails a run — lives with
    the measurement rather than in each caller.  The enum mixes in ``str``
    (``Severity.ERROR == "error"``), which keeps CLI and JSON output plain.

    Attributes
    ----------
    ERROR
        The data contradicts something that must hold (a duplicate
        ``event_id``, a dropped venue message, a negative volume).  Any failing
        error check fails the run.
    WARNING
        A signal worth reading before trusting the feed, but one a sound
        capture can legitimately show (orders resting before the capture
        began, zero-priced levels, messages reordered in transit).  Fails the
        run only under ``ob-analytics audit --strict``.
    INFO
        Reported for context; never fails a run.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class QualityCheck:
    """One named data-quality check and how it read on this run.

    Attributes
    ----------
    name : str
        Stable identifier, matching the summary field it reads
        (e.g. ``"duplicate_event_ids"``).
    passed : bool
        Whether the data satisfied the check.  ``False`` for a skipped check,
        which did not pass.
    severity : Severity
        What a failure means (see :class:`Severity`).
    detail : str
        One line saying what was found and how to read it.
    skipped : bool
        Whether the check did not run, because the data cannot answer it.
        A skipped check is :attr:`Severity.INFO` and never fails a run;
        *detail* says why it did not run.
    """

    name: str
    passed: bool
    severity: Severity
    detail: str
    skipped: bool = False

    @classmethod
    def skip(cls, name: str, why: str) -> QualityCheck:
        """A check that did not run, and *why*."""
        return cls(name, False, Severity.INFO, why, skipped=True)

    def to_dict(self) -> dict[str, Any]:
        """Return the check as a plain, JSON-serialisable dict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "skipped": self.skipped,
            "severity": str(self.severity.value),
            "detail": self.detail,
        }


# A matched book that is crossed for less than this share of session time is
# treated as uncrossed: floating-point ties at the touch, not a defect.
CROSSED_TOLERANCE_PCT: float = 0.05

# How much a crossed book matters, by feed type.  A matched book and a
# price-level book must not cross; a diff feed can, faithfully.  An undeclared
# feed type is scored between the two.
_CROSSED_SEVERITY: dict[FeedType, Severity] = {
    FeedType.MATCHED_BOOK: Severity.ERROR,
    FeedType.PRICE_LEVELS: Severity.ERROR,
    FeedType.DIFF_FEED: Severity.INFO,
}

# For the feed types that must not cross: their name in the crossing note, and
# the most likely cause of a crossing.
_MUST_NOT_CROSS: dict[FeedType, tuple[str, str]] = {
    FeedType.MATCHED_BOOK: ("a matched book", "check reconstruction/data"),
    FeedType.PRICE_LEVELS: (
        "price levels",
        "most often the capture kept a level the venue removed",
    ),
}

# Above this share, unresolved maker/taker attribution stops being incidental
# (trades against orders that were resting before the capture began) and starts
# suggesting the trades and events do not describe the same session.
UNMATCHED_TRADES_WARN_PCT: float = 5.0

# The events columns the price-level rebuild reads.  ``data_quality_summary``
# counts overdrawn levels only when the events carry all of them.
_LEVEL_COLUMNS = frozenset(
    {"event_id", "id", "timestamp", "exchange_timestamp", "price", "volume"}
    | {"direction", "action", "fill", "type"}
)


@dataclass(frozen=True)
class BookDifference:
    """The first place the book rebuilt from the events differs from the venue's.

    Attributes
    ----------
    message_row : int
        The ``original_number`` of the event after which the two books
        differ: the row of the venue's file that states the venue's book.
    direction : str
        ``"bid"`` or ``"ask"``.
    price_level : int
        The price level, counted from the touch (1 is the best price).
    price, venue_price : float or None
        The price of that level in the rebuilt book and in the venue's, in
        the quote currency (``ticks * tick_size``); ``None`` where that book
        has no level there.
    volume, venue_volume : int
        The size at that level in the rebuilt book and in the venue's, in
        integer lots; ``0`` where that book has no level there.
    """

    message_row: int
    direction: str
    price_level: int
    price: float | None
    volume: int
    venue_price: float | None
    venue_volume: int

    def to_dict(self) -> dict[str, Any]:
        """Return the difference as a plain, JSON-serialisable dict."""
        return {
            "message_row": self.message_row,
            "direction": self.direction,
            "price_level": self.price_level,
            "price": self.price,
            "volume": self.volume,
            "venue_price": self.venue_price,
            "venue_volume": self.venue_volume,
        }

    def describe(self) -> str:
        """The difference in words, for the report."""

        def level(price: float | None, volume: int) -> str:
            if price is None:
                return "empty"
            return f"{volume:,} lot(s) at {_format_price(price)}"

        return (
            f"after message row {self.message_row}: {self.direction} price "
            f"level {self.price_level} is {level(self.price, self.volume)}, the venue's is "
            f"{level(self.venue_price, self.venue_volume)}"
        )


@dataclass(frozen=True)
class DataQualitySummary:
    """Per-run data-quality metrics for a reconstructed session.

    Built by :func:`data_quality_summary`.  All percentages are 0–100 floats.

    The fields are the measurements; :attr:`checks` turns them into pass/fail
    verdicts with a :class:`Severity` each, and :attr:`ok` is the one-line
    answer to "is this feed trustworthy?" that ``ob-analytics audit`` exits on.

    Attributes
    ----------
    feed_type : FeedType
        The source's declared crossing invariant (see
        :class:`~ob_analytics.protocols.FeedType`); sets expectations for
        ``crossed_pct``.
    n_events, n_orders, n_trades : int
        Row / distinct-order / trade counts.
    crossed_pct : float
        Percentage of session *time* the faithful book is crossed
        (``best_bid > best_ask``).  Expected ``~0`` for a matched book or a
        price-level book; a genuine, faithfully-replayed property of a diff
        feed.
    crossed_episodes : int
        Number of distinct crossed intervals.
    per_order_crossed_pct : float
        Percentage of session time the book rebuilt from the per-order
        events is crossed.  It differs from ``crossed_pct`` only when the
        depth table comes from somewhere else: a LOBSTER run reads its depth
        from the orderbook file, and its message file holds only the events
        inside the requested price levels, so an order cancelled outside
        them stays in the per-order book.  The per-order views (order book
        snapshots, queue positions, order lifecycles) read that book.  ``0``
        on a price-level (L2) run, which has no per-order events.
    per_order_crossed_episodes : int
        Number of distinct intervals the per-order book is crossed.
    venue_book_price_levels : int
        How many price levels per side the venue's own book states (the
        ``venue_book`` passed to :func:`data_quality_summary`; for LOBSTER,
        the orderbook file).  ``0`` when there is no venue book, or no book
        rebuilt from per-order events to compare with it: the
        ``venue_book`` check is then skipped.
    venue_book_rows : int
        Book states compared: one per event, each after its own message.
    venue_book_differing_rows : int
        Of those, the states where the book rebuilt from the per-order events
        differs from the venue's in any price or size of the best
        ``venue_book_price_levels`` levels of either side.
    venue_book_first_difference : BookDifference or None
        Where the first of those states differs, or ``None`` when none does.
    unmatched_trades_pct : float
        Percentage of trades with an order the feed can name left unresolved:
        a missing ``maker_event_id`` or ``taker_event_id`` when the feed names
        both, a missing ``maker_event_id`` when it names the maker only, and
        ``0`` when it names neither (see ``trade_attribution``).
    duplicate_event_ids : int
        Count of ``event_id`` values occurring more than once (``event_id``
        should be globally unique — any non-zero value is suspect).
    duplicate_created_ids : int
        Count of order ids with more than one ``created`` event.
    pre_existing_orders : int
        Distinct orders resting before the capture window (classifier label
        ``pre-existing`` — structurally unclassifiable, not failures).
    events_with_sequence : int
        Rows carrying a non-null venue ``sequence`` (see
        :func:`detect_sequence_gaps`).  ``0`` when the source has no sequence or
        ``track_sequence`` was off at load — the sequence metrics below are then
        trivially zero.
    sequence_gaps : int
        Dropped-message count: skipped venue sequence numbers.  Always ``0``
        when ``sequence_kind`` is ``MONOTONIC``, where a skip is normal.
    sequence_out_of_order : int
        Reordered or duplicated messages: sequence steps that did not advance,
        leaving out ``sequence_restarts``.
    sequence_restarts : int
        Steps back where the source started again from a new opening book
        and the venue's count started again with it.  The source reports them
        (``sequence_restarts`` in ``meta.json``); they are not faults of the
        sequence, and the resync is reported by the capture's own checks.
    sequence_kind : SequenceKind
        What the venue sequence promises, and so whether ``sequence_gaps`` was
        checked (see :class:`~ob_analytics.protocols.SequenceKind`).
    orphan_orders : int
        Distinct order ids with a ``changed`` or ``deleted`` event but no
        ``created`` one.  Every order resting before the capture began is an
        orphan, so a capture that starts mid-stream reports a small, stable
        count; this is also the signal a live capture's stream drifting from
        its opening snapshot shows up as.
    orphan_events : int
        Rows belonging to those orphan orders.
    nonpositive_price_rows : int
        Rows priced at or below zero.  Legal in the schema (prices are signed
        integer ticks) but not a tradeable level.
    negative_volume_rows : int
        Rows with a negative ``volume`` (or negative ``fill``): impossible size.
    negative_level_rows : int
        Depth rows where the events would take a price level below zero: they
        took more size off the level than they put on it.  The depth table
        holds such a level at zero, so it reads less than the orders the
        events show resting there.  Counted from the events.  Always ``0`` on
        a price-level (L2) run, whose depth rows ``negative_volume_rows``
        already counts.
    exchange_time_after_receive : int
        Rows whose venue clock (``exchange_timestamp``) is later than the local
        receive clock (``timestamp``) — an event received before it happened.
        Always ``0`` when ``clocks`` is not ``BOTH``.  Rows from a capture's
        opening book (``origin`` ``snapshot``) are not counted: the capture
        may not have measured their clocks.
    exchange_time_reordered : int
        Steps where the venue clock goes backwards while the receive clock
        moves forward: messages that reached the capture out of order.
        Always ``0`` when ``clocks`` is not ``BOTH``; the opening book's rows
        are left out, as for ``exchange_time_after_receive``.
    clocks : Clocks
        Which clocks the data carries (see
        :class:`~ob_analytics.protocols.Clocks`).  The two clock checks run
        only when it is ``BOTH``; with one clock there is nothing to compare,
        and the report says so.
    stale_orders : tuple of StaleOrder
        Resting orders a trade printed through that the venue did not report
        again within :data:`STALE_GRACE`, worst first (see
        :func:`detect_stale_orders`).  Each one stays in the rebuilt book after
        it has gone, so a diff feed's crossing is then partly this defect
        rather than the feed's normal lag.
    trade_attribution : TradeAttribution
        Which orders of a trade the feed can name (see
        :class:`~ob_analytics.protocols.TradeAttribution`); sets which sides
        ``unmatched_trades_pct`` and ``trades_matched_by_order`` count.
    trades_matched_by_order : int
        Trades whose taker is an order created with size 0.  Bitstamp creates
        an instant order bought by quote amount that way and never reports
        what it executed, so a trade with no row of its size is matched to
        the order by its id (see
        :class:`~ob_analytics.bitstamp.BitstampTradeReader`).  The order and
        its type are right; its rows show less fill than its trades.  Always
        ``0`` when the feed does not name takers.
    maybe_bad_book_records : int or None
        Records the venue flagged as coming after a gap in its feed that it
        could not repair, so the book may be wrong from the first of them on
        (Databento's ``F_MAYBE_BAD_BOOK``).  Read from the events ``attrs``
        (:data:`~ob_analytics.schemas.MAYBE_BAD_BOOK_ATTR`).  ``None`` when
        the events carry no count, as from a source that does not flag such
        records, and the check does not run.
    """

    feed_type: FeedType
    n_events: int
    n_orders: int
    n_trades: int
    crossed_pct: float
    crossed_episodes: int
    unmatched_trades_pct: float
    duplicate_event_ids: int
    duplicate_created_ids: int
    pre_existing_orders: int
    events_with_sequence: int = 0
    sequence_gaps: int = 0
    sequence_out_of_order: int = 0
    sequence_restarts: int = 0
    sequence_kind: SequenceKind = SequenceKind.CONTIGUOUS
    orphan_orders: int = 0
    orphan_events: int = 0
    nonpositive_price_rows: int = 0
    negative_volume_rows: int = 0
    negative_level_rows: int = 0
    exchange_time_after_receive: int = 0
    exchange_time_reordered: int = 0
    clocks: Clocks = Clocks.BOTH
    stale_orders: tuple[StaleOrder, ...] = ()
    trade_attribution: TradeAttribution = TradeAttribution.BOTH
    trades_matched_by_order: int = 0
    per_order_crossed_pct: float = 0.0
    per_order_crossed_episodes: int = 0
    venue_book_price_levels: int = 0
    venue_book_rows: int = 0
    venue_book_differing_rows: int = 0
    venue_book_first_difference: BookDifference | None = None
    maybe_bad_book_records: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a plain, JSON-serialisable dict."""
        return {
            "feed_type": str(self.feed_type.value),
            "n_events": self.n_events,
            "n_orders": self.n_orders,
            "n_trades": self.n_trades,
            "crossed_pct": self.crossed_pct,
            "crossed_episodes": self.crossed_episodes,
            "per_order_crossed_pct": self.per_order_crossed_pct,
            "per_order_crossed_episodes": self.per_order_crossed_episodes,
            "venue_book_price_levels": self.venue_book_price_levels,
            "venue_book_rows": self.venue_book_rows,
            "venue_book_differing_rows": self.venue_book_differing_rows,
            "venue_book_differing_pct": self.venue_book_differing_pct,
            "venue_book_first_difference": (
                self.venue_book_first_difference.to_dict()
                if self.venue_book_first_difference is not None
                else None
            ),
            "unmatched_trades_pct": self.unmatched_trades_pct,
            "duplicate_event_ids": self.duplicate_event_ids,
            "duplicate_created_ids": self.duplicate_created_ids,
            "pre_existing_orders": self.pre_existing_orders,
            "events_with_sequence": self.events_with_sequence,
            "sequence_gaps": self.sequence_gaps,
            "sequence_out_of_order": self.sequence_out_of_order,
            "sequence_restarts": self.sequence_restarts,
            "sequence_kind": str(self.sequence_kind.value),
            "orphan_orders": self.orphan_orders,
            "orphan_events": self.orphan_events,
            "nonpositive_price_rows": self.nonpositive_price_rows,
            "negative_volume_rows": self.negative_volume_rows,
            "negative_level_rows": self.negative_level_rows,
            "exchange_time_after_receive": self.exchange_time_after_receive,
            "exchange_time_reordered": self.exchange_time_reordered,
            "clocks": str(self.clocks.value),
            "stale_orders": [o.to_dict() for o in self.stale_orders],
            "trade_attribution": str(self.trade_attribution.value),
            "trades_matched_by_order": self.trades_matched_by_order,
            "maybe_bad_book_records": self.maybe_bad_book_records,
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
        }

    def _crossed_note(self) -> str:
        """One-line reading of ``crossed_pct`` given the feed type."""
        if self.feed_type in _MUST_NOT_CROSS:
            name, cause = _MUST_NOT_CROSS[self.feed_type]
            if self.crossed_pct <= CROSSED_TOLERANCE_PCT:
                return f"as expected for {name}"
            return f"UNEXPECTED for {name} — {cause}"
        if self.feed_type == FeedType.DIFF_FEED:
            if self.stale_orders:
                return (
                    f"diff feed, but {len(self.stale_orders)} stale resting "
                    "order(s) stay in the book — see stale resting orders"
                )
            return "expected for a diff feed — faithful replay, not a bug"
        return "feed type undeclared"

    def _per_order_crosses_more(self) -> bool:
        """Whether the per-order book is crossed for longer than the depth table."""
        extra = self.per_order_crossed_pct - self.crossed_pct
        return extra > CROSSED_TOLERANCE_PCT

    def _per_order_crossed_note(self) -> str:
        """One-line reading of ``per_order_crossed_pct``."""
        if self.n_events == 0:
            return "no per-order events"
        if self.per_order_crossed_pct <= CROSSED_TOLERANCE_PCT:
            return "not crossed"
        if not self._per_order_crosses_more():
            return "no more than the depth table — see the crossed resting book"
        return (
            "orders the events never close stay in it, so the per-order views "
            "are wrong; for LOBSTER, the message file has too few price "
            "levels: request more levels"
        )

    @property
    def venue_book_differing_pct(self) -> float:
        """Percentage of the compared book states that differ from the venue's."""
        if not self.venue_book_rows:
            return 0.0
        return 100.0 * self.venue_book_differing_rows / self.venue_book_rows

    def _venue_book(self) -> str:
        """The ``venue_book`` comparison in words, or why it did not run."""
        if not self.venue_book_price_levels and not self.n_events:
            return "not checked: no per-order events to rebuild a book from"
        if not self.venue_book_price_levels:
            return (
                "not checked: no venue book to compare with; only a LOBSTER "
                "run from its files has one"
            )
        levels = f"[best {self.venue_book_price_levels} price level(s) per side]"
        first = self.venue_book_first_difference
        if first is None:
            return (
                "rebuilt book matches the venue's book in all "
                f"{self.venue_book_rows:,} row(s) {levels}"
            )
        return (
            "rebuilt book differs from the venue's book in "
            f"{self.venue_book_differing_rows:,} of {self.venue_book_rows:,} "
            f"row(s) ({self.venue_book_differing_pct:.2f}%) {levels}; first "
            f"{first.describe()}"
        )

    def _venue_book_check(self) -> QualityCheck:
        """The ``venue_book`` check, or a skipped one when there is no venue book."""
        if not self.venue_book_price_levels:
            return QualityCheck.skip("venue_book", self._venue_book())
        detail = self._venue_book()
        if self.venue_book_first_difference is not None:
            detail += (
                ". The per-order views read the rebuilt book. For LOBSTER, the "
                "usual cause is orders the message file does not cover: those "
                "resting before it starts, and those outside its price levels"
            )
        return QualityCheck(
            "venue_book",
            self.venue_book_first_difference is None,
            Severity.WARNING,
            detail,
        )

    def _unmatched_sides(self) -> str:
        """The order(s) a trade must resolve, in words, for the check's text."""
        if self.trade_attribution == TradeAttribution.BOTH:
            return "maker or taker"
        return "maker"

    def _unmatched_note(self) -> str:
        """Which orders of a trade ``unmatched_trades_pct`` looked for."""
        if self.trade_attribution == TradeAttribution.MAKER_ONLY:
            return "maker only: this feed does not show takers"
        if self.trade_attribution == TradeAttribution.NONE:
            return "not checked: this feed names no orders"
        return "maker and taker"

    def _matched_by_order(self) -> str:
        """The ``trades_matched_by_order`` count in words, or why it is not."""
        if self.trade_attribution == TradeAttribution.MAKER_ONLY:
            return "not checked: this feed does not show takers"
        if self.trade_attribution == TradeAttribution.NONE:
            return "not checked: this feed names no orders"
        return f"{self.trades_matched_by_order} trade(s) [taker created with size 0]"

    def _matched_by_order_detail(self) -> str:
        """The ``trades_matched_by_order`` check's text."""
        if not self.trades_matched_by_order:
            return self._matched_by_order()
        return (
            f"{self._matched_by_order()}: these orders do not report what "
            "they executed, so a trade with no row of its size is matched to "
            "the order by its id"
        )

    def _maybe_bad_book(self) -> str:
        """The ``maybe_bad_book_records`` count in words, or why it is not."""
        if self.maybe_bad_book_records is None:
            return (
                "not checked: the events carry no count of records flagged "
                "after a gap (only Databento flags them)"
            )
        return f"{self.maybe_bad_book_records} record(s) flagged after a gap"

    def _maybe_bad_book_check(self) -> QualityCheck:
        """The flagged-records check, or a skipped one when nothing was counted."""
        if self.maybe_bad_book_records is None:
            return QualityCheck.skip("maybe_bad_book", self._maybe_bad_book())
        return QualityCheck(
            "maybe_bad_book",
            self.maybe_bad_book_records == 0,
            Severity.WARNING,
            f"{self.maybe_bad_book_records} record(s) carry the venue's flag "
            "for a gap in its feed that it could not repair, so the book may "
            "be wrong from the first of them on",
        )

    def _clocks_note(self) -> str:
        """Why the clock checks did not run, or ``""`` when they did."""
        if self.clocks is Clocks.RECEIVE_ONLY:
            return (
                "not checked: the data has no venue time, so exchange_timestamp "
                "copies the receive time"
            )
        if self.clocks is Clocks.VENUE_ONLY:
            return (
                "not checked: the data has no receive time, so timestamp copies "
                "the venue time"
            )
        return ""

    def _worst_stale(self) -> str:
        """The worst stale order: its id, side, price and time at the touch."""
        worst = self.stale_orders[0]
        return (
            f"{worst.direction} {worst.id} at {_format_price(worst.price)} held "
            f"the {worst.direction} touch for "
            f"{_format_duration(worst.touch_seconds)} after a trade printed "
            "through it"
        )

    @property
    def checks(self) -> tuple[QualityCheck, ...]:
        """Every check this run was scored against, errors first.

        The crossing check reads its severity off :attr:`feed_type`: a crossed
        resting book is a defect in a matched book or a price-level book and a
        faithful property of a diff feed, so the same number means opposite
        things and only the declared feed type can tell them apart.
        """
        crossed = self.crossed_pct > CROSSED_TOLERANCE_PCT
        crossed_severity = _CROSSED_SEVERITY.get(self.feed_type, Severity.WARNING)

        checks = [
            QualityCheck(
                "duplicate_event_ids",
                self.duplicate_event_ids == 0,
                Severity.ERROR,
                f"{self.duplicate_event_ids} event_id value(s) occur more than "
                "once; event_id is the unique key of an event",
            ),
            QualityCheck(
                "duplicate_created_ids",
                self.duplicate_created_ids == 0,
                Severity.ERROR,
                f"{self.duplicate_created_ids} order id(s) have more than one "
                "created event; an order is created once",
            ),
            QualityCheck(
                "sequence_gaps",
                self.sequence_gaps == 0,
                Severity.ERROR,
                f"{self.sequence_gaps} venue sequence number(s) missing: "
                "dropped messages, so the book is rebuilt from an incomplete feed",
            ),
            QualityCheck(
                "sequence_out_of_order",
                self.sequence_out_of_order == 0,
                Severity.ERROR,
                f"{self.sequence_out_of_order} venue sequence step(s) did not "
                "advance: repeated or reordered messages",
            ),
            QualityCheck(
                "negative_volume",
                self.negative_volume_rows == 0,
                Severity.ERROR,
                f"{self.negative_volume_rows} row(s) carry a negative volume "
                "or fill: an impossible size",
            ),
            QualityCheck(
                "crossed_book",
                not crossed,
                crossed_severity,
                f"the resting book is crossed for {self.crossed_pct:.2f}% of "
                f"session time ({self.crossed_episodes} episode(s)) "
                f"[{self._crossed_note()}]",
            ),
            QualityCheck(
                "per_order_book_crossed",
                not self._per_order_crosses_more(),
                Severity.WARNING,
                f"the book rebuilt from the per-order events is crossed for "
                f"{self.per_order_crossed_pct:.2f}% of session time "
                f"({self.per_order_crossed_episodes} episode(s)) "
                f"[{self._per_order_crossed_note()}]",
            ),
            self._venue_book_check(),
            QualityCheck(
                "unmatched_trades",
                self.unmatched_trades_pct <= UNMATCHED_TRADES_WARN_PCT,
                Severity.WARNING,
                f"{self.unmatched_trades_pct:.2f}% of trades have no resolvable "
                f"{self._unmatched_sides()} order "
                f"[{self._unmatched_note()}]; above "
                f"{UNMATCHED_TRADES_WARN_PCT:.0f}% the trades and events may not "
                "describe the same session",
            ),
            QualityCheck(
                "orphan_orders",
                self.orphan_orders == 0,
                Severity.WARNING,
                f"{self.orphan_orders} order(s) are changed or deleted with no "
                f"created event ({self.orphan_events} row(s)); expected for the "
                "book already resting when the capture began, but a rise "
                "mid-session means the stream lost messages",
            ),
            QualityCheck(
                "stale_orders",
                not self.stale_orders,
                Severity.WARNING,
                f"{len(self.stale_orders)} resting order(s) a trade printed "
                "through and the venue did not report again within "
                f"{STALE_GRACE.total_seconds():g} s; each stays in the rebuilt "
                "book after it has gone"
                + (f". Worst: {self._worst_stale()}" if self.stale_orders else ""),
            ),
            QualityCheck(
                "negative_level",
                self.negative_level_rows == 0,
                Severity.WARNING,
                f"{self.negative_level_rows} depth row(s) would take a price "
                "level below zero: the events take more size off the level than "
                "they put on it, so the level, held at zero, reads short",
            ),
            QualityCheck(
                "nonpositive_price",
                self.nonpositive_price_rows == 0,
                Severity.WARNING,
                f"{self.nonpositive_price_rows} row(s) are priced at or below "
                "zero: not a tradeable level",
            ),
            QualityCheck(
                "pre_existing_orders",
                True,
                Severity.INFO,
                f"{self.pre_existing_orders} order(s) were resting before the "
                "capture window: structurally unclassifiable, not failures",
            ),
            QualityCheck(
                "trades_matched_by_order",
                True,
                Severity.INFO,
                self._matched_by_order_detail(),
            ),
            QualityCheck(
                "venue_sequence",
                True,
                Severity.INFO,
                f"{self.events_with_sequence} row(s) carry a venue sequence"
                if self.events_with_sequence
                else "no venue sequence in this feed, so gaps cannot be detected",
            ),
            *self._clock_checks(),
            self._maybe_bad_book_check(),
        ]
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        return tuple(sorted(checks, key=lambda c: order[c.severity]))

    def _clock_checks(self) -> list[QualityCheck]:
        """The two clock checks, or one note saying why they did not run."""
        skipped = self._clocks_note()
        if skipped:
            return [QualityCheck.skip("clocks", skipped)]
        return [
            QualityCheck(
                "exchange_time_after_receive",
                self.exchange_time_after_receive == 0,
                Severity.ERROR,
                f"{self.exchange_time_after_receive} row(s) have a venue "
                "timestamp later than the receive timestamp: an event received "
                "before it happened (clock skew, or the two clocks swapped)",
            ),
            QualityCheck(
                "exchange_time_reordered",
                self.exchange_time_reordered == 0,
                Severity.WARNING,
                f"{self.exchange_time_reordered} message(s) arrived out of venue "
                "order (the venue clock goes back while the receive clock moves "
                "forward)",
            ),
        ]

    @property
    def errors(self) -> tuple[QualityCheck, ...]:
        """Failed checks whose severity is :attr:`Severity.ERROR`."""
        return tuple(
            c for c in self.checks if not c.passed and c.severity == Severity.ERROR
        )

    @property
    def warnings(self) -> tuple[QualityCheck, ...]:
        """Failed checks whose severity is :attr:`Severity.WARNING`."""
        return tuple(
            c for c in self.checks if not c.passed and c.severity == Severity.WARNING
        )

    @property
    def ok(self) -> bool:
        """True when no error-severity check failed (warnings may still stand)."""
        return not self.errors

    def render(self) -> str:
        """Return a fixed-width, human-readable report block."""
        missing = (
            f"{self.sequence_gaps} missing"
            if self.sequence_kind is SequenceKind.CONTIGUOUS
            else "gaps not checked (sequence only rises)"
        )
        clock_order = self._clocks_note() or (
            f"{self.exchange_time_after_receive} venue-after-receive / "
            f"{self.exchange_time_reordered} reordered"
        )
        lines = [
            "Data quality summary",
            f"  feed type             : {self.feed_type.value}",
            f"  events / orders       : {self.n_events:,} / {self.n_orders:,}",
            f"  trades                : {self.n_trades:,}",
            (
                f"  crossed resting book  : {self.crossed_pct:.2f}% of session "
                f"({self.crossed_episodes} episode(s)) [{self._crossed_note()}]"
            ),
            (
                f"  per-order book crossed: {self.per_order_crossed_pct:.2f}% of "
                f"session ({self.per_order_crossed_episodes} episode(s)) "
                f"[{self._per_order_crossed_note()}]"
            ),
            f"  venue book            : {self._venue_book()}",
            (
                f"  stale resting orders  : {len(self.stale_orders)}"
                + (f" (worst: {self._worst_stale()})" if self.stale_orders else "")
            ),
            (
                f"  unmatched trades      : {self.unmatched_trades_pct:.2f}% "
                f"[{self._unmatched_note()}]"
            ),
            f"  matched by order      : {self._matched_by_order()}",
            f"  duplicate event ids   : {self.duplicate_event_ids}",
            f"  duplicate created ids : {self.duplicate_created_ids}",
            f"  pre-existing orders   : {self.pre_existing_orders}",
            (
                f"  orphan orders         : {self.orphan_orders} "
                f"({self.orphan_events} event(s), no created row)"
            ),
            (
                f"  impossible values     : {self.nonpositive_price_rows} "
                f"non-positive price(s) / {self.negative_volume_rows} "
                f"negative volume(s) / {self.negative_level_rows} level row(s) "
                "below zero"
            ),
            f"  clock order           : {clock_order}",
            f"  records after a gap   : {self._maybe_bad_book()}",
            (
                f"  venue sequence        : {missing} / "
                f"{self.sequence_out_of_order} out-of-order "
                + (
                    f"/ {self.sequence_restarts} restart(s) at a resync "
                    if self.sequence_restarts
                    else ""
                )
                + f"({self.events_with_sequence} row(s) numbered)"
            ),
        ]

        # INFO checks never fail a run and their numbers are already in the
        # block above, so the verdict lists only what a reader must act on.
        failed = self.errors + self.warnings
        if failed:
            lines.append(
                f"Checks: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
            )
            lines += [
                f"  {c.severity.value.upper():<7} {c.name}: {c.detail}" for c in failed
            ]
        else:
            lines.append("Checks: all passed")
        return "\n".join(lines)


def _format_price(price: float) -> str:
    """A price with thousands separators and only the decimals it needs."""
    decimals = next((d for d in range(9) if round(price, d) == price), 8)
    return f"{price:,.{decimals}f}"


def _format_duration(seconds: float) -> str:
    """A duration in the largest unit that keeps it readable."""
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _clock_order_counts(frame: pd.DataFrame) -> tuple[int, int]:
    """Count the two clock-order defects in *frame*: ``(after_receive, reordered)``.

    The schema carries two clocks: ``exchange_timestamp``, stamped by the
    venue, and ``timestamp``, stamped on receipt.  Two things follow from that,
    and neither depends on how the frame happens to be sorted:

    * an event cannot be received before the venue stamped it, so
      ``exchange_timestamp > timestamp`` is impossible;
    * read in canonical time order (:func:`~ob_analytics.schemas.time_order_keys`),
      the venue clock should not go backwards while the receive clock moves
      forward — where it does, those messages reached the capture out of order.

    Rows from a capture's opening book (``origin`` ``snapshot``) are left
    out of both counts: the capture may not have measured their clocks (an
    opening book sent without a venue time copies the receive time).  Rows
    sharing a receive instant are skipped for the second count: their relative
    order is set by the tie-break key, not by arrival, so a venue-clock step
    across them says nothing.  A frame without both columns scores zero on
    both counts.
    """
    if "exchange_timestamp" not in frame.columns or "timestamp" not in frame.columns:
        return 0, 0
    if frame.empty:
        return 0, 0

    both = frame[["timestamp", "exchange_timestamp"]].notna().all(axis=1)
    if ORIGIN_COLUMN in frame.columns:
        # Only rows marked as the opening book; a blank origin is checked.
        opening = frame[ORIGIN_COLUMN].eq(SNAPSHOT_ORIGIN).fillna(False)
        both &= ~opening.astype(bool)
    after_receive = int(
        (frame.loc[both, "exchange_timestamp"] > frame.loc[both, "timestamp"]).sum()
    )

    ordered = frame.loc[both].sort_values(time_order_keys(frame), kind="stable")
    if len(ordered) < 2:
        return after_receive, 0
    receive = np.diff(epoch_ns(ordered["timestamp"]))
    venue = np.diff(epoch_ns(ordered["exchange_timestamp"]))
    reordered = int(np.count_nonzero((venue < 0) & (receive > 0)))
    return after_receive, reordered


def _trades_matched_by_order(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    trade_attribution: TradeAttribution,
) -> int:
    """Count the trades whose taker is an order created with size 0.

    Such an order (Bitstamp's instant order bought by quote amount) never
    reports what it executed, so its trades are matched by its order id.
    Only a feed that names takers is read: a maker always reports its fills.
    """
    if trade_attribution != TradeAttribution.BOTH:
        return 0
    instant = unsized_order_ids(events)
    if instant.empty:
        return 0
    rows = events.loc[events["id"].isin(instant), "event_id"]
    taker = pd.to_numeric(trades["taker_event_id"], errors="coerce")
    return int(taker.isin(rows).sum())


def _compare_venue_book(
    events: pd.DataFrame,
    rebuilt: pd.DataFrame,
    venue_book: BookLevels,
    tick_size: float,
) -> tuple[int, BookDifference | None]:
    """Compare the book rebuilt from *events* with the venue's, row by row.

    *rebuilt* is the depth table of the per-order events.  Each event is
    matched with the venue's row of its own message, named by
    ``original_number``, not with the row at its position: a venue file can
    state the book after messages the events leave out.  Both books are
    compared on every price and size of the best price levels the venue
    states; the rebuilt book keeps every price level its orders rest at,
    crossed or not.

    Returns the number of events after which the books differ, and the
    first difference.
    """
    validate_columns(
        events, {"original_number"}, "data_quality_summary(venue_book=...)"
    )
    if not pd.api.types.is_integer_dtype(rebuilt["price"]):
        raise ConfigError(
            "data_quality_summary: a venue book is compared in integer ticks; "
            "pass the pipeline's own tables, not the ones display_result "
            "converts."
        )
    ordered = events.sort_values("original_number", kind="stable")
    rows = ordered["original_number"].to_numpy(dtype=np.int64) - 1
    n_venue, num_levels = venue_book.ask_prices.shape
    outside = (rows < 0) | (rows >= n_venue)
    if outside.any():
        raise ConfigError(
            f"data_quality_summary: the venue book has {n_venue} rows, but an "
            f"event names message row {int(rows[outside][0]) + 1}.  Pass the "
            "venue book of the same message file."
        )
    mine = _book_levels(ordered, rebuilt, num_levels, quoted=False)
    venue = BookLevels(*(side[rows] for side in venue_book))

    # Per side, which levels of each row differ in price or size.
    sides = {
        "ask": (mine.ask_prices != venue.ask_prices)
        | (mine.ask_volumes != venue.ask_volumes),
        "bid": (mine.bid_prices != venue.bid_prices)
        | (mine.bid_volumes != venue.bid_volumes),
    }
    differs = sides["ask"].any(axis=1) | sides["bid"].any(axis=1)
    n_differing = int(differs.sum())
    if not n_differing:
        return 0, None

    # The first differing row, at its level nearest the touch (asks first
    # on a tie).
    r = int(np.argmax(differs))
    level, direction = min(
        (int(np.argmax(diff[r])), name) for name, diff in sides.items() if diff[r].any()
    )

    def side_level(book: BookLevels) -> tuple[float | None, int]:
        price = getattr(book, f"{direction}_prices")[r, level]
        volume = int(getattr(book, f"{direction}_volumes")[r, level])
        if not volume:
            return None, 0
        return float(ticks_to_price(price, tick_size)), volume

    price, volume = side_level(mine)
    venue_price, venue_volume = side_level(venue)
    first = BookDifference(
        message_row=int(rows[r]) + 1,
        direction=direction,
        price_level=level + 1,
        price=price,
        volume=volume,
        venue_price=venue_price,
        venue_volume=venue_volume,
    )
    return n_differing, first


def data_quality_summary(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    feed_type: FeedType = FeedType.UNKNOWN,
    depth: pd.DataFrame | None = None,
    tick_size: float = 1.0,
    sequence_kind: SequenceKind = SequenceKind.CONTIGUOUS,
    sequence_restarts: int = 0,
    trade_attribution: TradeAttribution = TradeAttribution.BOTH,
    clocks: Clocks = Clocks.BOTH,
    venue_book: BookLevels | None = None,
) -> DataQualitySummary:
    """Summarise the data quality of one reconstructed session.

    Surfaces the health signals that matter before trusting a feed — most
    importantly how crossed the resting book is, which distinguishes a matched
    book from a diff feed (see :class:`~ob_analytics.protocols.FeedType`).

    A source whose feed flags records that come after a gap it could not
    repair (Databento) puts their count on ``events.attrs`` under
    :data:`~ob_analytics.schemas.MAYBE_BAD_BOOK_ATTR`; the summary reports
    it as ``maybe_bad_book_records``.

    For a pipeline run, call
    :meth:`~ob_analytics.pipeline.PipelineResult.audit` instead: it passes
    the run's tables, tick size and declarations
    (:class:`~ob_analytics.protocols.FeedDeclarations`) for you.  Pass the
    declarations here yourself only for tables that are not a run's.

    Like :func:`detect_stale_orders`, this takes either the pipeline's own
    tables or the tables :func:`~ob_analytics.visualization.display_result`
    converts for display, and measures the same crossing and the same stale
    orders on both.

    Parameters
    ----------
    events : pandas.DataFrame
        Classified events (must carry the canonical columns **and** the
        ``type`` column from :func:`set_order_types`).  For a price-level
        (L2) run this is the empty, schema-valid frame from
        :func:`~ob_analytics._utils.empty_events`: the per-order metrics then
        report zero and crossing is read from *depth* (pass
        ``PipelineResult.depth``).
    trades : pandas.DataFrame
        The trades frame, with ``maker_event_id`` / ``taker_event_id``.
    feed_type : FeedType, optional
        The source's declared feed type, recorded on the summary and used to
        interpret ``crossed_pct``.
    depth : pandas.DataFrame, optional
        A faithful price-level-volume frame (e.g. ``PipelineResult.depth``).
        When ``None`` it is computed from *events* via
        :func:`~ob_analytics.depth.price_level_volume`.  **Do not** pass
        ``depth_summary`` — that is already uncrossed and would report ~0%.
    tick_size : float, optional
        Quote-currency size of one price tick (``PipelineResult.config.tick_size``),
        so the prices in ``stale_orders`` read in the quote currency.  Leave at
        ``1.0`` to report them in the units of ``events["price"]``: ticks for
        the pipeline's tables, the quote currency for display tables.
    sequence_kind : SequenceKind, optional
        What the venue ``sequence`` promises, passed to
        :func:`detect_sequence_gaps`.  A capture records it in ``meta.json``
        (read it with :func:`~ob_analytics.capture_record.read_record`).
    sequence_restarts : int, optional
        How many of the sequence's steps back the source counted as the count
        starting again at a new opening book.  That many are left out of
        ``sequence_out_of_order``.  A capture records it in ``meta.json``
        (read it with :func:`~ob_analytics.capture_record.read_record`).
    trade_attribution : TradeAttribution, optional
        Which orders of a trade the feed can name, so the unmatched-trades
        check looks only for those.  Read it off the source with
        :func:`~ob_analytics.protocols.trade_attribution_of`; a capture records
        it in ``meta.json``.
    clocks : Clocks, optional
        Which clocks the data carries, so the clock checks run only when there
        are two to compare.  With two, the checks still leave out rows from a
        capture's opening book (``origin`` ``snapshot``).  Read it off the source with
        :func:`~ob_analytics.protocols.clocks_of`; a live capture records it in
        ``meta.json`` (read it with
        :func:`~ob_analytics.capture_record.read_record`).  Data with no venue
        time at all (a price-level file without ``exchange_timestamp``) is
        read as :attr:`~ob_analytics.protocols.Clocks.RECEIVE_ONLY` whatever
        is declared; an empty frame keeps the declaration.
    venue_book : BookLevels, optional
        The venue's own book after every message, when the source supplies
        one: for LOBSTER, the orderbook file read by
        :func:`~ob_analytics.lobster.lobster_venue_book`.  Row ``i`` is the
        book after the message whose ``original_number`` is ``i + 1``, so
        *events* must carry ``original_number`` and be the pipeline's own
        tables (integer ticks and lots).  The book rebuilt from *events* is
        then compared with it on every price and size of the levels it
        states.  Without it, the ``venue_book`` check is skipped.

    Returns
    -------
    DataQualitySummary

    Raises
    ------
    ConfigError
        If a *venue_book* is passed with events that lack
        ``original_number``, are display tables, or name a message row the
        venue book does not have.
    """
    validate_columns(
        events,
        {"event_id", "id", "action", "direction", "price", "type"},
        "data_quality_summary(events)",
    )
    validate_columns(
        trades,
        {"maker_event_id", "taker_event_id"},
        "data_quality_summary(trades)",
    )

    # A price-level (L2) run has no per-order events: ``events`` is empty (but
    # schema-valid, see :func:`~ob_analytics._utils.empty_events`).  The
    # per-order metrics then report zero and the crossed book is read from the
    # supplied depth frame (the price-level book) rather than reconstructed
    # from events.
    l2 = events.empty
    # The depth rows where the events overdraw a price level.  The depth table
    # holds such a level at zero, so they are counted from the events.  An L2
    # run's depth rows are counted by ``negative_volume_rows`` below.
    negative_level_rows = 0
    # The book rebuilt from the per-order events, when the depth table is not
    # that book: a LOBSTER run reads its depth from the orderbook file.
    # ``rebuilt_depth`` is the per-order book whichever table holds it.
    per_order_depth = None
    rebuilt_depth = None
    if depth is None and l2:
        depth = events
    elif depth is None:
        depth, overdrawn = _price_level_volume(events)
        negative_level_rows = len(overdrawn)
        rebuilt_depth = depth
    elif not l2 and _LEVEL_COLUMNS <= set(events.columns):
        per_order_depth, overdrawn = _price_level_volume(events)
        negative_level_rows = len(overdrawn)
        rebuilt_depth = per_order_depth

    # The venue's own book against the book rebuilt from the per-order
    # events.  An L2 run has no such book to compare.
    venue_book_price_levels = venue_book_rows = venue_book_differing_rows = 0
    venue_book_first_difference = None
    if venue_book is not None and not l2:
        if rebuilt_depth is None:
            missing = sorted(_LEVEL_COLUMNS - set(events.columns))
            raise ConfigError(
                "data_quality_summary: a venue book is compared with the book "
                f"rebuilt from the events, which needs their columns {missing}."
            )
        venue_book_differing_rows, venue_book_first_difference = _compare_venue_book(
            events, rebuilt_depth, venue_book, tick_size
        )
        venue_book_price_levels = int(venue_book.ask_prices.shape[1])
        venue_book_rows = len(events)

    best, crossed_pct, crossed_episodes = _book_crossing(depth)
    if l2:
        per_order_crossed_pct, per_order_crossed_episodes = 0.0, 0
    elif per_order_depth is None:
        per_order_crossed_pct, per_order_crossed_episodes = (
            crossed_pct,
            crossed_episodes,
        )
    else:
        _, per_order_crossed_pct, per_order_crossed_episodes = _book_crossing(
            per_order_depth
        )

    # Stale resting orders need per-order events and priced, timed trades.  An
    # L2 run has neither, and a trades frame without prices has nothing to
    # print through a resting order.
    stale_orders: tuple[StaleOrder, ...] = ()
    if (
        best is not None
        and not l2
        and not trades.empty
        and {"timestamp", "price"} <= set(trades.columns)
    ):
        end_ns = _capture_end_ns(events, trades)
        spans = _stale_spans(events, trades, STALE_GRACE, end_ns)
        if not spans.empty:
            stale_orders = _stale_orders(spans, best, end_ns, tick_size)

    # Look only for the orders the feed can name: a feed that shows resting
    # orders only never shows a taker, so a missing taker there is the feed,
    # not a match failure.  L2 trades carry no attribution at all.
    n_trades = len(trades)
    if n_trades and not l2 and trade_attribution != TradeAttribution.NONE:
        unmatched = trades["maker_event_id"].isna()
        if trade_attribution == TradeAttribution.BOTH:
            unmatched = unmatched | trades["taker_event_id"].isna()
        unmatched_pct = 100.0 * float(unmatched.sum()) / n_trades
        matched_by_order = _trades_matched_by_order(events, trades, trade_attribution)
    else:
        unmatched_pct = 0.0
        matched_by_order = 0

    event_id_counts = events["event_id"].value_counts()
    duplicate_event_ids = int((event_id_counts > 1).sum())

    created_ids = events.loc[events["action"] == "created", "id"]
    created_counts = created_ids.value_counts()
    duplicate_created_ids = int((created_counts > 1).sum())

    pre_existing_orders = int(
        events.loc[events["type"] == "pre-existing", "id"].nunique()
    )

    # Venue sequence gaps: read from events on the L3 path, or from the
    # price-level depth on the L2 path (where sequence, when present, rides on
    # depth rather than the empty events frame).  Absent columns score zero.
    gaps = detect_sequence_gaps(depth if l2 else events, kind=sequence_kind)
    # Steps back the source accounts for: never more than the data shows.
    restarts = min(max(sequence_restarts, 0), gaps.n_out_of_order)

    # Orders changed or deleted with no created row.  On the L2 path there are
    # no per-order events, so there is nothing to orphan.
    if l2:
        orphan_orders = orphan_events = 0
    else:
        created_ids = set(events.loc[events["action"] == "created", "id"])
        after = events.loc[events["action"] != "created", "id"]
        orphaned = after[~after.isin(created_ids)]
        orphan_events = int(orphaned.size)
        orphan_orders = int(orphaned.nunique())

    # Impossible values, read from whichever frame carries the price levels.
    levels = depth if l2 else events
    nonpositive_price_rows = int((levels["price"] <= 0).sum())
    negative_volume_rows = int((levels["volume"] < 0).sum())
    if not l2 and "fill" in events.columns:
        negative_volume_rows += int((events["fill"] < 0).sum())
    # Rows with no venue time have one clock, whatever the source declares.
    # An empty frame has no rows to say so, and keeps the declaration.
    clocks = Clocks(clocks)
    no_venue_time = "exchange_timestamp" not in levels.columns or bool(
        levels["exchange_timestamp"].isna().all()
    )
    if clocks is Clocks.BOTH and not levels.empty and no_venue_time:
        clocks = Clocks.RECEIVE_ONLY
    if clocks is Clocks.BOTH:
        after_receive, reordered = _clock_order_counts(levels)
    else:
        after_receive, reordered = 0, 0
    # The source puts the count on the events when its feed can flag records
    # after a gap; without the key, nothing was checked.
    maybe_bad_book = events.attrs.get(MAYBE_BAD_BOOK_ATTR)

    return DataQualitySummary(
        feed_type=feed_type,
        n_events=len(events),
        n_orders=int(events["id"].nunique()),
        n_trades=n_trades,
        crossed_pct=crossed_pct,
        crossed_episodes=crossed_episodes,
        per_order_crossed_pct=per_order_crossed_pct,
        per_order_crossed_episodes=per_order_crossed_episodes,
        unmatched_trades_pct=unmatched_pct,
        duplicate_event_ids=duplicate_event_ids,
        duplicate_created_ids=duplicate_created_ids,
        pre_existing_orders=pre_existing_orders,
        events_with_sequence=gaps.n_sequenced,
        sequence_gaps=gaps.n_missing,
        sequence_out_of_order=gaps.n_out_of_order - restarts,
        sequence_restarts=restarts,
        sequence_kind=sequence_kind,
        orphan_orders=orphan_orders,
        orphan_events=orphan_events,
        nonpositive_price_rows=nonpositive_price_rows,
        negative_volume_rows=negative_volume_rows,
        negative_level_rows=negative_level_rows,
        exchange_time_after_receive=after_receive,
        exchange_time_reordered=reordered,
        clocks=clocks,
        stale_orders=stale_orders,
        trade_attribution=trade_attribution,
        trades_matched_by_order=matched_by_order,
        venue_book_price_levels=venue_book_price_levels,
        venue_book_rows=venue_book_rows,
        venue_book_differing_rows=venue_book_differing_rows,
        venue_book_first_difference=venue_book_first_difference,
        maybe_bad_book_records=(
            None if maybe_bad_book is None else int(maybe_bad_book)
        ),
    )
