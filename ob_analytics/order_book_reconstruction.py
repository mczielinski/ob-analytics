import pandas as pd
import numpy as np

def is_sorted(array: np.ndarray, ascending: bool = True) -> bool:
    """Check if an array is sorted."""
    return np.all(array[:-1] <= array[1:]) if ascending else np.all(array[:-1] >= array[1:])

def sanity_check_active_orders(active_orders: pd.DataFrame, tp: pd.Timestamp):
    """Perform sanity checks on the active orders."""
    if not is_sorted(active_orders['timestamp'].to_numpy()):
        raise ValueError("Active orders dataframe is not sorted by timestamp.")
    if not is_sorted(active_orders['id'].to_numpy()):
        raise ValueError("Active orders dataframe is not sorted by id.")
    if active_orders['timestamp'].iloc[-1] > tp:
        raise ValueError("Last order timestamp exceeds the specified timestamp.")

def get_active_orders(events: pd.DataFrame, tp: pd.Timestamp) -> pd.DataFrame:
    """Retrieve active orders up to a specified timestamp."""
    active_orders = events[(events['action'] == 'added') & (events['timestamp'] <= tp)]
    sanity_check_active_orders(active_orders, tp)
    return active_orders

def active_asks(active_orders: pd.DataFrame) -> pd.DataFrame:
    """Get active ask orders from the active orders dataframe."""
    asks = active_orders[active_orders["direction"] == "ask"]
    asks = asks.sort_values(by=["price", "id"], ascending=[True, True])
    if not asks.empty:
        first_price = asks["price"].iloc[0]
        asks["bps"] = ((asks["price"] - first_price) / first_price) * 10000
        asks["liquidity"] = asks["volume"].cumsum()
    return asks

def active_bids(active_orders: pd.DataFrame) -> pd.DataFrame:
    """Get active bid orders from the active orders dataframe."""
    bids = active_orders[active_orders["direction"] == "bid"]
    bids = bids.sort_values(by=["price", "id"], ascending=[False, True])
    if not bids.empty:
        first_price = bids["price"].iloc[0]
        bids["bps"] = ((first_price - bids["price"]) / first_price) * 10000
        bids["liquidity"] = bids["volume"].cumsum()
    return bids

def order_book(events: pd.DataFrame, tp: pd.Timestamp = pd.Timestamp.now(tz="UTC"),
               max_levels: int = None, bps_range: float = 0, 
               min_bid: float = 0, max_ask: float = float('inf')) -> dict:
    """Construct the order book."""
    pct_range = bps_range * 0.0001
    active_orders = get_active_orders(events, tp)
    sanity_check_active_orders(active_orders, tp)
    asks = active_asks(active_orders)[["id", "timestamp", "exchange.timestamp", "price", "volume", "liquidity", "bps"]]
    asks = asks.iloc[::-1]
    bids = active_bids(active_orders)[["id", "timestamp", "exchange.timestamp", "price", "volume", "liquidity", "bps"]]
    if pct_range > 0 and not asks.empty:
        max_ask_adjusted = asks["price"].iloc[-1] * (1 + pct_range)
        asks = asks[asks["price"] <= max_ask_adjusted]
    if pct_range > 0 and not bids.empty:
        min_bid_adjusted = bids["price"].iloc[0] * (1 - pct_range)
        bids = bids[bids["price"] >= min_bid_adjusted]
    if max_levels is not None:
        asks = asks.tail(max_levels)
        bids = bids.head(max_levels)
    asks.reset_index(drop=True, inplace=True)
    bids.reset_index(drop=True, inplace=True)
    return {
        "timestamp": tp,
        "asks": asks,
        "bids": bids
    }
