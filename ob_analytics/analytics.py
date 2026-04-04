"""Format-agnostic post-processing analytics.

Contains functions that operate on the outputs of any pipeline run,
regardless of the originating data format (Bitstamp, LOBSTER, etc.).

This module is the home for analytics that sit above the format layer:
aggressiveness, trade impacts, and future additions such as fill rates,
slippage, or market impact metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty


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
