"""Format-agnostic post-processing analytics.

Contains functions that operate on the outputs of any pipeline run,
regardless of the originating data format (Bitstamp, LOBSTER, etc.):
order aggressiveness, trade impacts, and order type classification.

The book at an instant and the per-order lifecycles are in
:mod:`ob_analytics.book`; the data-quality checks are in
:mod:`ob_analytics.quality`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import (
    price_scale,
    require_per_order_events,
    size_scale,
    validate_non_empty,
)
from ob_analytics.schemas import ORDER_TYPE_DTYPE, validate_columns
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
