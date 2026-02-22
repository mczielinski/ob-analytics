"""Order type classification for limit order book events.

Classifies each order as one of: *market*, *resting-limit*,
*flashed-limit*, *pacman*, or *market-limit*, based on how the order
interacts with the book over its lifetime.
"""

import numpy as np
import pandas as pd

from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty


def set_order_types(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Determine limit order types.

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

    # Pacman orders
    price_ranges = events.groupby("id")["price"].agg(["min", "max"])
    pacman_ids = set(price_ranges[price_ranges["min"] != price_ranges["max"]].index)
    events.loc[events["id"].isin(pacman_ids), "type"] = "pacman"

    # Flashed and resting limit orders
    created = events[events["action"] == "created"].sort_values(by="id", kind="stable")
    deleted = events[events["action"] == "deleted"].sort_values(by="id", kind="stable")
    changed = events[events["action"] == "changed"]

    created_deleted_ids = created[
        created["id"].isin(deleted["id"]) & ~created["id"].isin(changed["id"])
    ]["id"]

    # Merge on id to safely compare created vs deleted volumes
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

    # Pure maker: was a maker, never a taker, never a pacman
    pure_maker_ids = maker_ids - taker_ids - pacman_ids

    # Market limit: was both a maker and a taker, never a pacman
    ml_ids = (taker_ids & maker_ids) - pacman_ids

    # Pure market: was a taker, never a maker, never a pacman
    mo_ids = taker_ids - maker_ids - pacman_ids

    events.loc[events["id"].isin(flashed_ids), "type"] = "flashed-limit"
    events.loc[
        events["id"].isin(forever_ids | pure_maker_ids), "type"
    ] = "resting-limit"

    # Market limit orders
    events.loc[events["id"].isin(ml_ids), "type"] = "market-limit"

    # Market orders
    events.loc[events["id"].isin(mo_ids), "type"] = "market"

    unidentified = (events["type"] == "unknown").sum()
    if unidentified > 0:
        logger.warning("Could not identify {} orders", unidentified)

    return events
