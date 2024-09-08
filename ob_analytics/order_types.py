import pandas as pd
import numpy as np


def set_order_types(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Determine limit order types.

    Args:
      events: The limit order events DataFrame.
      trades: The executions DataFrame.

    Returns:
      The events DataFrame with an updated 'type' column indicating order types.
    """

    def is_pacman(events: pd.DataFrame) -> pd.Series:
        return events.groupby("id")["price"].transform(lambda x: x.diff().any())

    events["type"] = "unknown"
    events["type"] = pd.Categorical(
        events["type"],
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
    pacman_ids = events[is_pacman(events)]["id"]
    events.loc[events["id"].isin(pacman_ids), "type"] = "pacman"

    # Flashed and resting limit orders
    created = events[events["action"] == "created"].sort_values(by="id")
    deleted = events[events["action"] == "deleted"].sort_values(by="id")
    changed = events[events["action"] == "changed"]

    created_deleted_ids = created[
        created["id"].isin(deleted["id"]) & ~created["id"].isin(changed["id"])
    ]["id"].reset_index(drop=True)
    # Get volumes from the 'deleted' DataFrame with matched IDs and reset the index
    deleted_volumes = deleted.loc[
        deleted["id"].isin(created_deleted_ids), "volume"
    ].reset_index(drop=True)

    # Get volumes from the 'created' DataFrame with matched IDs and reset the index
    created_volumes = created.loc[
        created["id"].isin(created_deleted_ids), "volume"
    ].reset_index(drop=True)

    # Compare volumes
    volume_matched = deleted_volumes == created_volumes
    flashed_ids = created_deleted_ids[volume_matched]
    forever_ids = created[
        ~created["id"].isin(changed["id"]) & ~created["id"].isin(deleted["id"])
    ]["id"].reset_index(drop=True)

    maker_ids = events[events["event.id"].isin(trades["maker.event.id"])]["id"].unique()
    taker_ids = events[events["event.id"].isin(trades["taker.event.id"])]["id"].unique()
    maker_ids = maker_ids[~np.isin(maker_ids, taker_ids)]
    maker_ids = maker_ids[~np.isin(maker_ids, pacman_ids)]

    events.loc[events["id"].isin(flashed_ids), "type"] = "flashed-limit"
    events.loc[
        events["id"].isin(forever_ids) | events["id"].isin(maker_ids), "type"
    ] = "resting-limit"

    # Market limit orders
    ml_ids = taker_ids[
        np.isin(
            taker_ids,
            events[events["event.id"].isin(trades["maker.event.id"])]["id"].unique(),
        )
    ]
    ml_ids = ml_ids[~np.isin(ml_ids, pacman_ids)]
    events.loc[events["id"].isin(ml_ids), "type"] = "market-limit"

    # Market orders
    mo_ids = taker_ids[
        ~np.isin(
            taker_ids,
            events[events["event.id"].isin(trades["maker.event.id"])]["id"].unique(),
        )
    ]
    mo_ids = mo_ids[~np.isin(mo_ids, pacman_ids)]
    events.loc[events["id"].isin(mo_ids), "type"] = "market"

    unidentified = (events["type"] == "unknown").sum()
    if unidentified > 0:
        print(f"Could not identify {unidentified} orders")

    return events
