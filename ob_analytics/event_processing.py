import numpy as np
import pandas as pd


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
        Remove duplicate events from the DataFrame.

        Parameters
        ----------
        events : pandas.DataFrame
            DataFrame containing limit order events.

        Returns
        -------
        pandas.DataFrame
            DataFrame with duplicate events removed.
        """
        dups = events[
            events.duplicated(subset=["id", "price", "volume", "action"])
            & (events["action"] != "changed")
        ].index
        if len(dups) > 0:
            print(
                f"Removed {len(dups)} duplicate events: {', '.join(events.loc[dups, 'id'].astype(str))}"
            )
            events = events.drop(dups)
        return events

    events = pd.read_csv(file)
    events = events[events["volume"] >= 0]
    events = events.reset_index().rename(columns={"index": "original_number"})
    events.original_number = events.original_number + 1
    events["volume"] = events["volume"].round(volume_digits)
    events["price"] = events["price"].round(price_digits)
    events = remove_duplicates(events)
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
    events = events.sort_values(by=["id", "action", "timestamp"])
    events["event.id"] = np.arange(1, len(events) + 1)
    events["fill"] = (
        events.groupby("id")["volume"].diff().abs().fillna(0).round(volume_digits)
    )
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
        orders = events[
            (events["direction"] == ("bid" if direction == 1 else "ask"))
            & (events["action"] != "changed")
            & events["type"].isin(["flashed-limit", "resting-limit"])
        ].sort_values(by="timestamp")

        # Equivalent to R's stopifnot
        assert all(
            orders["timestamp"].isin(depth_summary["timestamp"])
        ), "Not all timestamps in orders are present in depth_summary"

        best = depth_summary.loc[
            depth_summary["timestamp"].isin(orders["timestamp"]),
            "best.bid.price" if direction == 1 else "best.ask.price",
        ]
        diff_price = direction * (orders["price"][1:] - best[:-1])
        diff_bps = 10000 * diff_price / best[:-1]
        return pd.DataFrame({"event.id": orders["event.id"][1:], "diff.bps": diff_bps})

    bid_diff = event_diff_bps(events, 1)
    ask_diff = event_diff_bps(events, -1)
    events["aggressiveness.bps"] = np.nan

    # Merge bid_diff with events based on 'event.id'
    events = events.merge(
        bid_diff[["event.id", "diff.bps"]],
        on="event.id",
        how="left",
        suffixes=("", "_bid"),
    )
    events["aggressiveness.bps"].fillna(events["diff.bps"], inplace=True)
    events.drop(columns=["diff.bps"], inplace=True)

    # Merge ask_diff with events based on 'event.id'
    events = events.merge(
        ask_diff[["event.id", "diff.bps"]],
        on="event.id",
        how="left",
        suffixes=("", "_ask"),
    )
    events["aggressiveness.bps"].fillna(events["diff.bps"], inplace=True)
    events.drop(columns=["diff.bps"], inplace=True)

    return events
