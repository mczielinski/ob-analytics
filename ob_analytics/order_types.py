import pandas as pd
import numpy as np

def is_pacman(events: pd.DataFrame) -> pd.Series:
    """Determine if the events are of type 'pacman'.
    
    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing the events.
    
    Returns
    -------
    pd.Series
        Boolean series indicating if each event id corresponds to a 'pacman' order.
    """
    def check_diff(prices: pd.Series) -> bool:
        return any(np.diff(prices) != 0)
    
    return events.groupby("id")["price"].apply(check_diff)

def set_order_types(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Determine limit order types.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing the events.
    trades : pd.DataFrame
        DataFrame containing the trades.

    Returns
    -------
    pd.DataFrame
        DataFrame with the inferred order types.
    """
    # Initialize order type to 'unknown'
    events["type"] = "unknown"
    
    # Identify 'pacman' orders
    pacman_ids = events[events["id"].isin(is_pacman(events).index[is_pacman(events).values])]["event.id"].values
    events.loc[events["event.id"].isin(pacman_ids), "type"] = "pacman"
    
    # Identify 'flashed-limit' orders
    flashed_limit_orders = events[
        (events["action"] == "created") & 
        events["event.id"].isin(events.loc[events["action"] == "deleted", "event.id"])
    ]
    events.loc[events["event.id"].isin(flashed_limit_orders["event.id"]), "type"] = "flashed-limit"
    
    # Identify 'resting-limit' orders
    resting_limit_orders = events[
        (events["action"] == "created") & 
        events["event.id"].isin(events.loc[events["action"] == "hit", "event.id"])
    ]
    events.loc[events["event.id"].isin(resting_limit_orders["event.id"]), "type"] = "resting-limit"
    

    # Identify 'market-limit' orders
    ml_ids = set(trades["taker.event.id"]) & set(events.loc[events["type"] == "flashed-limit", "event.id"])
    ml_ids = ml_ids - set(events.loc[events["type"] == "pacman", "event.id"])
    events.loc[events["event.id"].isin(ml_ids), "type"] = "market-limit"
    
    # Identify 'market' orders
    mo_ids = set(trades["taker.event.id"]) - set(trades["maker.event.id"])
    mo_ids = mo_ids - set(events.loc[events["type"] == "pacman", "event.id"])
    events.loc[events["event.id"].isin(mo_ids), "type"] = "market"
    
    return events

