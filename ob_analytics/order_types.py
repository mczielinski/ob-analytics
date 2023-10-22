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
    pacman_ids = set(events[events["id"].isin(is_pacman(events).index[is_pacman(events).values])]["id"].values)
    events.loc[events["id"].isin(pacman_ids), "type"] = "pacman"
    
    # Identify orders
    created = events[events["action"] == "created"].sort_values("id")
    created_ids = set(created["id"])
    deleted_ids = set(events.loc[events["action"] == "deleted", "id"])
    changed_ids = set(events.loc[events["action"] == "changed", "id"])
    
    # Identify orders that don't have a creation event
    original_ids = set(events["id"]) - created_ids
    
    # Identify 'flashed-limit' orders
    created_deleted_ids = created_ids - changed_ids & deleted_ids
    volumes_created = {id_: created.loc[created["id"] == id_, "volume"].values[0] for id_ in created_deleted_ids}
    volumes_deleted = {id_: events.loc[events["id"] == id_, "volume"].values[0] for id_ in created_deleted_ids & deleted_ids}
    volume_matched = {id_ for id_ in created_deleted_ids if volumes_created[id_] == volumes_deleted[id_]}
    flashed_ids = created_deleted_ids & volume_matched

    if flashed_ids:
        events.loc[events["id"].isin(flashed_ids), "type"] = "flashed-limit"
        
    # Identify 'resting-limit' orders
    forever_ids = created_ids - changed_ids - deleted_ids | (original_ids - changed_ids - deleted_ids)
    if forever_ids:
        events.loc[events["id"].isin(forever_ids), "type"] = "resting-limit"
    
    # Identify 'market-limit' orders
    maker_ids = set(pd.unique(events.loc[events["event.id"].isin(trades["maker.event.id"]), "id"]))
    taker_ids = set(pd.unique(events.loc[events["event.id"].isin(trades["taker.event.id"]), "id"]))
    
    ml_ids = taker_ids.intersection(maker_ids) - pacman_ids
    if ml_ids:
        events.loc[events["id"].isin(ml_ids), "type"] = "market-limit"

    # Identify 'market' orders
    mo_ids = taker_ids - maker_ids - pacman_ids
    if mo_ids:
        events.loc[events["id"].isin(mo_ids), "type"] = "market"
    
    return events