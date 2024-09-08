import pandas as pd
import numpy as np
from ob_analytics.auxiliary import interval_sum_breaks


def price_level_volume(events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cumulative volume for each price level over time.

    Args:
      events: A pandas DataFrame containing limit order events.

    Returns:
      A pandas DataFrame with the cumulative volume for each price level.
    """

    def directional_price_level_volume(dir_events):
        cols = [
            "event.id",
            "id",
            "timestamp",
            "exchange.timestamp",
            "price",
            "volume",
            "direction",
            "action",
        ]

        # Added Volume
        added_volume = dir_events[
            (
                (dir_events["action"] == "created")
                | ((dir_events["action"] == "changed") & (dir_events["fill"] == 0))
            )
            & (dir_events["type"] != "pacman")
            & (dir_events["type"] != "market")
        ][cols]

        # Cancelled Volume
        cancelled_volume = dir_events[
            (dir_events["action"] == "deleted")
            & (dir_events["volume"] > 0)
            & (dir_events["type"] != "pacman")
            & (dir_events["type"] != "market")
        ][cols]
        cancelled_volume["volume"] = -cancelled_volume["volume"]
        cancelled_volume = cancelled_volume[
            cancelled_volume["id"].isin(added_volume["id"])
        ]

        # Filled Volume
        filled_volume = dir_events[
            (dir_events["fill"] > 0)
            & (dir_events["type"] != "pacman")
            & (dir_events["type"] != "market")
        ][
            [
                "event.id",
                "id",
                "timestamp",
                "exchange.timestamp",
                "price",
                "fill",
                "direction",
                "action",
            ]
        ]
        filled_volume["fill"] = -filled_volume["fill"]
        filled_volume = filled_volume[filled_volume["id"].isin(added_volume["id"])]
        filled_volume.columns = cols

        # Combine Volumes
        volume_deltas = pd.concat([added_volume, cancelled_volume, filled_volume])
        volume_deltas = volume_deltas.sort_values(by=["price", "timestamp"])

        # Calculate Cumulative Volume
        volume_deltas["volume"] = volume_deltas.groupby("price")["volume"].cumsum()
        volume_deltas["volume"] = volume_deltas["volume"].clip(lower=0)

        return volume_deltas[["timestamp", "price", "volume", "direction"]]

    bids = events[events["direction"] == "bid"]
    depth_bid = directional_price_level_volume(bids)
    asks = events[events["direction"] == "ask"]
    depth_ask = directional_price_level_volume(asks)
    depth_data = pd.concat([depth_bid, depth_ask])
    return depth_data.sort_values(by="timestamp")


def filter_depth(
    d: pd.DataFrame, from_timestamp: pd.Timestamp, to_timestamp: pd.Timestamp
) -> pd.DataFrame:
    # 1. Get all active price levels before start of range
    pre = d[d["timestamp"] <= from_timestamp]
    pre = pre.sort_values(by=["price", "timestamp"])

    # Last update for each price level <= from. This becomes the starting point for all updates within the range.
    pre = pre.drop_duplicates(subset="price", keep="last")
    pre = pre[pre["volume"] > 0]

    # Clamp range (reset timestamp to from if price level active before start of range)
    if not pre.empty:
        pre["timestamp"] = pre["timestamp"].apply(lambda r: max(from_timestamp, r))

    # 2. Add all volume changes within the range
    mid = d[(d["timestamp"] > from_timestamp) & (d["timestamp"] < to_timestamp)]
    range_combined = pd.concat([pre, mid])

    # 3. At the end of the range, set all price level volume to 0
    open_ends = range_combined.drop_duplicates(subset="price", keep="last")
    open_ends = open_ends[open_ends["volume"] > 0].copy()
    open_ends["timestamp"] = to_timestamp
    open_ends["volume"] = 0

    # Combine pre, mid, and open_ends, ensure it is in order
    range_combined = pd.concat([range_combined, open_ends])
    range_combined = range_combined.sort_values(by=["price", "timestamp"])

    return range_combined


def depth_metrics(depth, bps=25, bins=20):
    def pct_names(name):
        return [f"{name}{i}bps" for i in range(bps, bps * bins + 1, bps)]

    ordered_depth = depth.sort_values(by="timestamp")
    ordered_depth["price"] = (100 * ordered_depth["price"]).round().astype(int)
    depth_matrix = np.column_stack(
        (
            ordered_depth["price"],
            ordered_depth["volume"],
            np.where(ordered_depth["direction"] == "bid", 0, 1),
        )
    )

    metrics = pd.DataFrame(
        0,
        index=range(len(ordered_depth)),
        columns=["best.bid.price", "best.bid.vol"]
        + pct_names("bid.vol")
        + ["best.ask.price", "best.ask.vol"]
        + pct_names("ask.vol"),
    )

    # the volume state for all price level depths. (updated in loop)
    asks_state = np.zeros(1000000, dtype=int)
    asks_state[999999] = 1  # trick (so there is an initial best ask)
    bids_state = np.zeros(1000000, dtype=int)
    bids_state[0] = 1  # trick
    # initial best bid/ask
    best_ask = ordered_depth[ordered_depth["direction"] == "ask"]["price"].max()
    best_bid = ordered_depth[ordered_depth["direction"] == "bid"]["price"].min()
    best_ask_vol = 0
    best_bid_vol = 0

    for i in range(len(ordered_depth)):
        depth_row = depth_matrix[i, :]
        price = int(depth_row[0])
        volume = depth_row[1]
        side = depth_row[2]

        # ask
        if side > 0:  # if side is 0, bid, if side is 1, ask (this is doing asks)
            # If the current price is higher than the best bid, it's a valid ask
            if price > best_bid:
                asks_state[price] = volume  # Update the volume at this price level
                # If there's volume at this price level
                if volume > 0:
                    # If the price is lower than the current best ask, update best ask and volume
                    if price < best_ask:
                        best_ask = price
                        best_ask_vol = volume
                    # If the price is the same as the current best ask, update only the volume
                    elif price == best_ask:
                        best_ask_vol = volume
                # If there's no volume at this price level
                else:
                    # If this price was the best ask, find the new best ask
                    if price == best_ask:
                        best_ask = np.where(asks_state > 0)[0][0]
                        best_ask_vol = asks_state[best_ask]

                # Calculate the price range and volume range for the ask side
                end_value = round((1 + bps * bins * 0.0001) * best_ask) + 1
                price_range = np.arange(best_ask, end_value, 1)

                volume_range = asks_state[price_range]

                # Calculate breaks for binning the volume data
                breaks = (
                    np.ceil(np.cumsum(np.repeat(len(price_range) / bins, bins))).astype(
                        int
                    )
                    - 1
                )
                breaks[-1] = breaks[-1] - 1

                # Update the metrics DataFrame with ask-side data
                metrics.iloc[i, bins + 2] = best_ask
                metrics.iloc[i, bins + 3] = best_ask_vol

                metrics.iloc[i, (bins + 4) : (2 * (2 + bins))] = interval_sum_breaks(
                    volume_range, breaks
                )

                # Copy the last bid data (no need to re-calculate it)
                if i > 0:
                    metrics.iloc[i, : (2 + bins)] = metrics.iloc[i - 1, : (2 + bins)]
            # If the price is not higher than the best bid, it's not a valid ask, so no changes are made
            else:
                # Copy the last data (no change)
                if i > 0:
                    metrics.iloc[i, :] = metrics.iloc[i - 1, :]

        # --- BID SIDE LOGIC ---
        # This section follows a similar logic as the ask side, but for bids
        else:  # bid
            # If the current price is lower than the best ask, it's a valid bid
            if price < best_ask:
                bids_state[price] = volume  # Update the volume at this price level
                # If there's volume at this price level
                if volume > 0:
                    # If the price is higher than the current best bid, update best bid and volume
                    if price > best_bid:
                        best_bid = price
                    # If the price is the same as the current best bid, update only the volume
                    elif price == best_bid:
                        best_bid_vol = volume
                # If there's no volume at this price level
                else:
                    # If this price was the best bid, find the new best bid
                    if price == best_bid:
                        best_bid = np.where(bids_state > 0)[0][-1]
                        best_bid_vol = bids_state[best_bid]

                # Calculate the price range and volume range for the bid side

                # Calculate the end value for the price_range
                end_value = round((1 - bps * bins * 0.0001) * best_bid)

                # Create the price_range array using np.arange, ensuring it includes the end value by adding 1 to the end value
                price_range = np.arange(best_bid, end_value - 1, -1)
                volume_range = bids_state[price_range]

                # Calculate breaks for binning the volume data
                breaks = (
                    np.ceil(np.cumsum(np.repeat(len(price_range) / bins, bins))).astype(
                        int
                    )
                    - 1
                )
                breaks[-1] = breaks[-1] - 1

                # Update the metrics DataFrame with bid-side data
                metrics.iloc[i, 0] = best_bid
                metrics.iloc[i, 1] = best_bid_vol

                metrics.iloc[i, 2 : (2 + bins)] = interval_sum_breaks(
                    volume_range, breaks
                )

                # Copy the last ask data (no need to re-calculate it)
                if i > 0:
                    metrics.iloc[i, (bins + 2) : (2 * (2 + bins))] = metrics.iloc[
                        i - 1, (bins + 2) : (2 * (2 + bins))
                    ]
            # If the price is not lower than the best ask, it's not a valid bid, so no changes are made
            else:
                # Copy the last data (no change)
                if i > 0:
                    metrics.iloc[i, :] = metrics.iloc[i - 1, :]

    # back into $
    # res = pd.concat([ordered_depth['timestamp'], metrics], axis=1)
    res = pd.concat([ordered_depth.reset_index()["timestamp"], metrics], axis=1)
    keys = ["best.bid.price", "best.ask.price"]
    res[keys] = round(0.01 * res[keys], 2)

    return res


def get_spread(depth_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the bid/ask spread from the depth summary.

    Args:
      depth_summary: A pandas DataFrame containing depth summary statistics.

    Returns:
      A pandas DataFrame with the bid/ask spread data.
    """
    spread = depth_summary[
        [
            "timestamp",
            "best_bid_price",
            "best_bid_vol",
            "best_ask_price",
            "best_ask_vol",
        ]
    ]
    changes = (
        spread[
            ["best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"]
        ].diff()
        != 0
    ).any(axis=1)
    return spread[changes]
