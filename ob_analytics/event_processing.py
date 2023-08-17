import pandas as pd
import numpy as np
from ob_analytics.auxiliary import vector_diff

def load_event_data(file: str, price_digits: int = 2, volume_digits: int = 8) -> pd.DataFrame:
    """
    Load event data, clean and preprocess it.

    Parameters
    ----------
    file : str
        Path to the CSV file containing the event data.
    price_digits : int, optional
        Number of decimal places to round the price to. Default is 2.
    volume_digits : int, optional
        Number of decimal places to round the volume to. Default is 8.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed event data.
    """

    def remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the events dataframe.

        Parameters
        ----------
        events : pd.DataFrame
            Events dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe after removing duplicates.
        """
        mask = events.duplicated(subset=["id", "price", "volume", "action"]) & (events["action"] != "changed")
        dups = events[mask]
        if not dups.empty:
            ids = " ".join(map(str, dups["id"].unique()))
            print(f"Warning: removed {len(dups)} duplicate events: {ids}")
            events = events[~mask]
        return events

    # Load data
    events = pd.read_csv(file)

    # Remove negative volumes
    negative_vol = events[events["volume"] < 0]
    if not negative_vol.empty:
        print(f"Warning: removed {len(negative_vol)} negative volume events")
        events = events[events["volume"] >= 0]

    # Round volume and price
    events["volume"] = events["volume"].round(volume_digits)
    events["price"] = events["price"].round(price_digits)

    # Remove duplicates
    events = remove_duplicates(events)

    # Convert timestamps
    events["timestamp"] = pd.to_datetime(events["timestamp"] / 1000, unit="s", origin="unix")
    events["exchange.timestamp"] = pd.to_datetime(events["exchange.timestamp"] / 1000, unit="s", origin="unix")

    # Factorize columns
    events["action"] = pd.Categorical(events["action"], categories=["created", "changed", "deleted"], ordered=True)
    events["direction"] = pd.Categorical(events["direction"], categories=["bid", "ask"], ordered=True)

    # Sort data
    events = events.sort_values(by=["id", "action", "timestamp"])

    # Add event ID
    events["event.id"]= range(1, len(events) + 1)
    # Compute fill deltas
    events_grouped = events.groupby("id")["volume"].apply(vector_diff).explode().reset_index()
    events = pd.merge(events, events_grouped, on="id", how="left")
    events["fill"] = events["volume_y"].astype(float).abs().round(volume_digits)
    events.drop(columns=["volume_y"], inplace=True)

    return events

def order_aggressiveness(events: pd.DataFrame, depth_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate order aggressiveness with respect to best bid or ask in BPS.

    Parameters
    ----------
    events : pd.DataFrame
        The events dataframe.
    depth_summary : pd.DataFrame
        Order book summary statistics.

    Returns
    -------
    pd.DataFrame
        The events dataframe containing a new aggressiveness.bps column.
    """
    def event_diff_bps(events_subset: pd.DataFrame, direction: int) -> pd.DataFrame:
        # Filter based on direction and order type
        if direction == 1:
            condition = "bid"
        else:
            condition = "ask"

        orders = events_subset[
            (events_subset["direction"] == condition) & 
            (events_subset["action"] != "changed") & 
            (events_subset["type"].isin(["flashed-limit", "resting-limit"]))
        ]
        orders = orders.sort_values(by="timestamp")
        
        # Ensuring all timestamps are present in depth_summary
        assert all(orders["timestamp"].isin(depth_summary["timestamp"]))
        
        best_col = "best.bid.price" if direction == 1 else "best.ask.price"
        best_prices = depth_summary.loc[depth_summary["timestamp"].isin(orders["timestamp"]), best_col].values
        
        diff_price = direction * (orders["price"].values - best_prices[:-1])
        diff_bps = 10000 * diff_price / best_prices[:-1]
        
        return pd.DataFrame({"event.id": orders["event.id"].values[1:], "diff.bps": diff_bps})

    # Calculate aggressiveness for bids and asks
    bid_diff = event_diff_bps(events, 1)
    ask_diff = event_diff_bps(events, -1)
    
    # Add aggressiveness values to the main events dataframe
    events["aggressiveness.bps"] = np.nan
    events.loc[events["event.id"].isin(bid_diff["event.id"]), "aggressiveness.bps"] = bid_diff["diff.bps"].values
    events.loc[events["event.id"].isin(ask_diff["event.id"]), "aggressiveness.bps"] = ask_diff["diff.bps"].values
    
    return events
