"""Format-agnostic post-processing analytics.

Contains functions that operate on the outputs of any pipeline run,
regardless of the originating data format (Bitstamp, LOBSTER, etc.):
aggressiveness, trade impacts, order type classification, and
point-in-time order book reconstruction.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.exceptions import InvalidDataError


def order_aggressiveness(
    events: pd.DataFrame, depth_summary: pd.DataFrame
) -> pd.DataFrame:
    """Calculate order aggressiveness with respect to the best bid or ask in BPS.

    Parameters
    ----------
    events : pandas.DataFrame
        The events DataFrame (must contain ``direction``, ``action``, ``type``,
        ``timestamp``, ``event_id``, ``price`` columns).
    depth_summary : pandas.DataFrame
        The order book summary statistics DataFrame (must contain ``timestamp``
        and ``event_id`` columns).

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

    def event_diff_bps(events: pd.DataFrame, direction: int) -> pd.DataFrame:
        side = "bid" if direction == 1 else "ask"
        orders = events[
            (events["direction"] == side)
            & (events["action"] != "changed")
            & events["type"].isin(["flashed-limit", "resting-limit"])
        ].sort_values(by="timestamp", kind="stable")

        missing = ~orders["timestamp"].isin(depth_summary["timestamp"])
        if missing.any():
            logger.debug(
                "order_aggressiveness: {}/{} {} order timestamps not in "
                "depth_summary (merge_asof will handle gracefully)",
                missing.sum(),
                len(orders),
                side,
            )

        best_price_col = f"best_{side}_price"

        depth_summary_sorted = depth_summary.sort_values("event_id")
        orders = orders.sort_values("event_id")

        merged = pd.merge_asof(
            orders,
            depth_summary_sorted[["event_id", best_price_col]],
            on="event_id",
            direction="backward",
            allow_exact_matches=False,
        )

        merged = merged.dropna(subset=[best_price_col]).copy()
        best = merged[best_price_col]

        diff_price = direction * (merged["price"] - best)
        diff_bps = 10000 * diff_price / best
        return pd.DataFrame({"event_id": merged["event_id"], "diff_bps": diff_bps})

    bid_diff = event_diff_bps(events, 1)
    ask_diff = event_diff_bps(events, -1)
    events["aggressiveness_bps"] = np.nan

    if not bid_diff.empty:
        events = pd.merge(events, bid_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(
            events["diff_bps"]
        )
        events.drop(columns=["diff_bps"], inplace=True)

    if not ask_diff.empty:
        events = pd.merge(events, ask_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(
            events["diff_bps"]
        )
        events.drop(columns=["diff_bps"], inplace=True)

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
        ``start_time``, ``end_time``, ``dir``.
    """
    validate_columns(
        trades,
        {"taker", "price", "volume", "timestamp", "direction"},
        "trade_impacts",
    )
    validate_non_empty(trades, "trade_impacts")

    trades_pv = trades.assign(_pv=trades["price"] * trades["volume"])
    impacts = (
        trades_pv.groupby("taker")
        .agg(
            id=("taker", "last"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            hits=("taker", "size"),
            vol=("volume", "sum"),
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
    *flashed-limit*, *pacman*, or *market-limit*, based on how the order
    interacts with the book over its lifetime.

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
    validate_non_empty(events, "set_order_types")

    events["type"] = pd.Categorical(
        np.repeat("unknown", len(events)),
        categories=[
            "unknown",
            "flashed-limit",
            "resting-limit",
            "market-limit",
            "pacman",
            "market",
        ],
        ordered=True,
    )

    price_ranges = events.groupby("id")["price"].agg(["min", "max"])
    pacman_ids = set(price_ranges[price_ranges["min"] != price_ranges["max"]].index)
    events.loc[events["id"].isin(pacman_ids), "type"] = "pacman"

    created = events[events["action"] == "created"].sort_values(by="id", kind="stable")
    deleted = events[events["action"] == "deleted"].sort_values(by="id", kind="stable")
    changed = events[events["action"] == "changed"]

    created_deleted_ids = created[
        created["id"].isin(deleted["id"]) & ~created["id"].isin(changed["id"])
    ]["id"]

    cd_created = created[created["id"].isin(created_deleted_ids)][["id", "volume"]]
    cd_deleted = deleted[deleted["id"].isin(created_deleted_ids)][["id", "volume"]]
    cd_merged = cd_created.merge(cd_deleted, on="id", suffixes=("_created", "_deleted"))
    flashed_ids = set(
        cd_merged.loc[cd_merged["volume_created"] == cd_merged["volume_deleted"], "id"]
    )
    forever_ids = set(
        created[
            ~created["id"].isin(changed["id"]) & ~created["id"].isin(deleted["id"])
        ]["id"]
    )

    maker_event_ids_set = set(trades["maker_event_id"].dropna())
    taker_event_ids_set = set(trades["taker_event_id"].dropna())

    maker_ids = set(events[events["event_id"].isin(maker_event_ids_set)]["id"])
    taker_ids = set(events[events["event_id"].isin(taker_event_ids_set)]["id"])

    pure_maker_ids = maker_ids - taker_ids - pacman_ids
    ml_ids = (taker_ids & maker_ids) - pacman_ids
    mo_ids = taker_ids - maker_ids - pacman_ids

    events.loc[events["id"].isin(flashed_ids), "type"] = "flashed-limit"
    events.loc[events["id"].isin(forever_ids | pure_maker_ids), "type"] = (
        "resting-limit"
    )
    events.loc[events["id"].isin(ml_ids), "type"] = "market-limit"
    events.loc[events["id"].isin(mo_ids), "type"] = "market"

    unidentified = (events["type"] == "unknown").sum()
    if unidentified > 0:
        logger.warning("Could not identify {} orders", unidentified)

    return events


# ---------------------------------------------------------------------------
# Order book reconstruction
# ---------------------------------------------------------------------------


def order_book(
    events: pd.DataFrame,
    tp: datetime | None = None,
    max_levels: int | None = None,
    bps_range: int = 0,
    min_bid: float = 0,
    max_ask: float = np.inf,
) -> dict[str, datetime | pd.Timestamp | pd.DataFrame]:
    """Reconstruct the order book at a specific point in time.

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
    min_bid : float, optional
        Minimum bid price. Default is 0.
    max_ask : float, optional
        Maximum ask price. Default is infinity.

    Returns
    -------
    dict[str, datetime.datetime or pandas.DataFrame]
        A dictionary containing:
        - 'timestamp': The evaluation timestamp.
        - 'asks': DataFrame of active ask orders.
        - 'bids': DataFrame of active bid orders.
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
            "exchange_timestamp",
        },
        "order_book",
    )
    validate_non_empty(events, "order_book")

    if tp is None:
        tp = events["timestamp"].max()

    pct_range = bps_range * 0.0001

    def active_bids(active_orders: pd.DataFrame) -> pd.DataFrame:
        bids = active_orders[
            (active_orders["direction"] == "bid") & (active_orders["type"] != "market")
        ]
        bids = bids.sort_values(
            by=["price", "id"], ascending=[False, True], kind="stable"
        )
        first_price = bids.iloc[0]["price"] if not bids.empty else np.nan
        bids["bps"] = (
            ((first_price - bids["price"]) / first_price) * 10000
            if not bids.empty
            else np.nan
        )
        bids["liquidity"] = bids["volume"].cumsum()
        return bids

    def active_asks(active_orders: pd.DataFrame) -> pd.DataFrame:
        asks = active_orders[
            (active_orders["direction"] == "ask") & (active_orders["type"] != "market")
        ]
        asks = asks.sort_values(
            by=["price", "id"], ascending=[True, True], kind="stable"
        )
        first_price = asks.iloc[0]["price"] if not asks.empty else np.nan
        asks["bps"] = (
            ((asks["price"] - first_price) / first_price) * 10000
            if not asks.empty
            else np.nan
        )
        asks["liquidity"] = asks["volume"].cumsum()
        return asks

    created_before = events[
        (events["action"] == "created") & (events["timestamp"] <= tp)
    ]["id"]

    deleted_before = events[
        (events["action"] == "deleted") & (events["timestamp"] <= tp)
    ]["id"]

    active_order_ids = set(created_before) - set(deleted_before)
    active_orders = events[events["id"].isin(active_order_ids)]
    active_orders = active_orders[active_orders["timestamp"] <= tp]

    changed_orders_mask = active_orders["action"] == "changed"
    changed_before = active_orders[changed_orders_mask]
    changed_before = changed_before.sort_values(
        by="timestamp", ascending=False, kind="stable"
    )
    changed_before = changed_before.drop_duplicates(subset="id", keep="first")

    active_orders = active_orders[~changed_orders_mask]
    active_orders = active_orders[~active_orders["id"].isin(changed_before["id"])]
    active_orders = pd.concat([active_orders, changed_before], ignore_index=True)

    if not all(active_orders["timestamp"] <= tp):
        raise InvalidDataError(
            f"Some active orders have timestamps after the requested time {tp}."
        )
    if active_orders["id"].duplicated().any():
        raise InvalidDataError(
            "Duplicate order IDs found in active orders. "
            "This indicates a data integrity issue."
        )

    asks = active_asks(active_orders)
    asks = asks[
        ["id", "timestamp", "exchange_timestamp", "price", "volume", "liquidity", "bps"]
    ]
    asks = asks.iloc[::-1].reset_index(drop=True)

    bids = active_bids(active_orders)
    bids = bids[
        ["id", "timestamp", "exchange_timestamp", "price", "volume", "liquidity", "bps"]
    ]

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
