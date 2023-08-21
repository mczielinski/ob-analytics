import pandas as pd
import numpy as np
from ob_analytics.auxiliary import vector_diff

def load_event_data(file: str, price_digits: int = 2, volume_digits: int = 8) -> pd.DataFrame:
    """
    Load and preprocess event data from the provided file.

    Parameters
    ----------
    file : str
        Path to the file containing event data.
    price_digits : int, optional
        Number of decimal digits to round the price column, by default 2.
    volume_digits : int, optional
        Number of decimal digits to round the volume column, by default 8.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataframe containing event data.
    """

    def remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate events from the data.
    
        Parameters
        ----------
        events : pd.DataFrame
            The dataframe containing event data.
    
        Returns
        -------
        pd.DataFrame
            The dataframe after removing duplicate events.
        """
        mask = (events.duplicated(subset=["id", "price", "volume", "action"])) & (events["action"] != "changed")
        dups = events[mask].index
        if len(dups) > 0:
            ids = " ".join(map(str, events.loc[dups, "id"].unique()))
            events = events.drop(dups)
            print(f"Warning: removed {len(dups)} duplicate events: {ids}")
        return events
    
    # Load the data
    events = pd.read_csv(file)
    
    # Remove rows with negative volume
    negative_vol = events[events['volume'] < 0].index
    if len(negative_vol) > 0:
        events = events.drop(negative_vol)
        print(f"Warning: removed {len(negative_vol)} negative volume events")
    
    # Round volume and price columns
    events['volume'] = events['volume'].round(volume_digits)
    events['price'] = events['price'].round(price_digits)
    
    # Remove duplicates
    events = remove_duplicates(events)
    
    # Convert timestamps
    events['timestamp'] = pd.to_datetime(events['timestamp'] / 1000, unit='s', origin="1970-01-01").dt.floor('S')
    events['exchange.timestamp'] = pd.to_datetime(events['exchange.timestamp'] / 1000, unit='s', origin="1970-01-01").dt.floor('S')
    
    # Factorize columns
    events['action'] = pd.Categorical(events['action'], categories=["created", "changed", "deleted"], ordered=True)
    events['direction'] = pd.Categorical(events['direction'], categories=["bid", "ask"], ordered=True)
    
    # Order data
    events = events.sort_values(by=["id", "action", "timestamp"])
    
    # Add event.id column
    events.insert(0, 'event.id', range(1, len(events) + 1))
    
    # Compute fill column
    fill_deltas = events.groupby('id')['volume'].transform(vector_diff)
    events['fill'] = fill_deltas.abs().round(volume_digits)
    
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
