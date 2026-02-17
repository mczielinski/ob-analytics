import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_event_data(
    file: str, price_digits: int = 2, volume_digits: int = 8
) -> pd.DataFrame:
    """
    Read raw limit order event data from a CSV file.

    Parameters
    ----------
    file : str
        The path to the CSV file containing limit order events.
    price_digits : int, optional
        The number of decimal places for the 'price' column. Default is 2.
    volume_digits : int, optional
        The number of decimal places for the 'volume' column. Default is 8.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the raw limit order events data.
    """

    def remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate delete events, matching R's removeDuplicates logic.

        R's logic:
        1. Get all 'deleted' events
        2. Sort by (id, volume)
        3. Find ids that have multiple delete events
        4. Keep the first occurrence per id, remove subsequent ones
        5. Remove those event.ids from the full events DataFrame
        """
        deletes = events[events["action"] == "deleted"].sort_values(
            by=["id", "volume"], kind="stable"
        )
        # Find ids with multiple delete events
        dup_ids = deletes.loc[deletes["id"].duplicated(), "id"]
        duplicate_deletes = deletes[deletes["id"].isin(dup_ids)]
        # Get event.ids of the 2nd+ occurrence for each id (keep first)
        duplicate_event_ids = duplicate_deletes.loc[
            duplicate_deletes["id"].duplicated(), "event.id"
        ]

        rem_dup = len(duplicate_event_ids)
        if rem_dup > 0:
            removed_ids = events.loc[
                events["event.id"].isin(duplicate_event_ids), "id"
            ]
            logger.warning(
                "Removed %d duplicate order cancellations: %s",
                rem_dup,
                " ".join(removed_ids.astype(str)),
            )

        return events[~events["event.id"].isin(duplicate_event_ids)]

    events = pd.read_csv(file)
    events = events[events["volume"] >= 0]
    events = events.reset_index().rename(columns={"index": "original_number"})
    events.original_number = events.original_number + 1
    events["volume"] = events["volume"].round(volume_digits)
    events["price"] = events["price"].round(price_digits)

    events["timestamp"] = pd.to_datetime(events["timestamp"] / 1000, unit="s")
    events["exchange.timestamp"] = pd.to_datetime(
        events["exchange.timestamp"] / 1000, unit="s"
    )
    events["action"] = pd.Categorical(
        events["action"], categories=["created", "changed", "deleted"], ordered=True
    )
    events["direction"] = pd.Categorical(
        events["direction"], categories=["bid", "ask"], ordered=True
    )

    # Sort by id ASC, volume DESC, action ASC, timestamp ASC
    # (matches R: order(id, -volume, action, timestamp))
    events = events.sort_values(
        by=["id", "volume", "action", "timestamp"],
        ascending=[True, False, True, True],
        kind="stable",
    )

    # Assign event.id BEFORE removing duplicates (matches R)
    # This means event.id will have gaps after duplicate removal
    events["event.id"] = np.arange(1, len(events) + 1)

    # Remove duplicate delete events (after event.id assignment, matching R)
    events = remove_duplicates(events)

    # Calculate fill deltas (volume change between consecutive events for same order)
    # Using vectorDiff approach: c(0, diff(v)) per group
    fill_deltas = events.groupby("id")["volume"].diff().fillna(0)

    # For pacman orders: zero out fill when price changes
    price_deltas = events.groupby("id")["price"].diff().fillna(0)
    fill_deltas = fill_deltas.where(price_deltas == 0, 0)

    events["fill"] = fill_deltas.abs().round(volume_digits)

    # Fix timestamps: re-sort timestamps within each order id group
    # to match the logical lifecycle ordering (id, -volume, action, timestamp).
    # R does: ts.ordered <- unlist(tapply(events$timestamp, events$id, sort))
    ts_sorted = (
        events.groupby("id")["timestamp"]
        .transform(lambda x: np.sort(x.values, kind="stable"))
    )
    events["timestamp"] = ts_sorted

    return events


def order_aggressiveness(
    events: pd.DataFrame, depth_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate order aggressiveness with respect to the best bid or ask in BPS.

    Parameters
    ----------
    events : pandas.DataFrame
        The events DataFrame.
    depth_summary : pandas.DataFrame
        The order book summary statistics DataFrame.

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with an added 'aggressiveness.bps' column.
    """

    def event_diff_bps(events: pd.DataFrame, direction: int) -> pd.DataFrame:
        """
        Calculate the price difference in basis points for orders in a given direction.

        Parameters
        ----------
        events : pandas.DataFrame
            The events DataFrame.
        direction : int
            The direction of the orders: 1 for bids, -1 for asks.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with 'event.id' and 'diff.bps' columns.
        """
        side = "bid" if direction == 1 else "ask"
        orders = events[
            (events["direction"] == side)
            & (events["action"] != "changed")
            & events["type"].isin(["flashed-limit", "resting-limit"])
        ].sort_values(by="timestamp", kind="stable")

        assert all(orders["timestamp"].isin(depth_summary["timestamp"])), (
            "Not all timestamps in orders are present in depth_summary"
        )

        best_price_col = f"best.{side}.price"

        # Replicate R's `match` behavior with a left merge
        # Drop duplicates from depth_summary to ensure a 1-to-1 merge like R's match()
        unique_depth_summary = depth_summary.drop_duplicates(subset=["timestamp"])
        merged = pd.merge(
            orders,
            unique_depth_summary[["timestamp", best_price_col]],
            on="timestamp",
            how="left",
        )

        # Replicate R's `head(best, -1)` by shifting
        best = merged[best_price_col].shift(1)

        # Drop the first row which now has a NaN `best` price
        merged = merged.iloc[1:].copy()
        best = best.iloc[1:]

        diff_price = direction * (merged["price"] - best)
        diff_bps = 10000 * diff_price / best
        return pd.DataFrame({"event.id": merged["event.id"], "diff.bps": diff_bps})

    bid_diff = event_diff_bps(events, 1)
    ask_diff = event_diff_bps(events, -1)
    events["aggressiveness.bps"] = np.nan

    # Use merge to update aggressiveness.bps for bids
    if not bid_diff.empty:
        events = pd.merge(events, bid_diff, on="event.id", how="left")
        events["aggressiveness.bps"] = events["aggressiveness.bps"].fillna(events["diff.bps"])
        events.drop(columns=["diff.bps"], inplace=True)

    # Use merge to update aggressiveness.bps for asks
    if not ask_diff.empty:
        events = pd.merge(events, ask_diff, on="event.id", how="left")
        events["aggressiveness.bps"] = events["aggressiveness.bps"].fillna(events["diff.bps"])
        events.drop(columns=["diff.bps"], inplace=True)

    return events
