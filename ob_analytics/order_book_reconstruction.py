import pandas as pd
import numpy as np


def order_book(
    events: pd.DataFrame,
    tp: pd.Timestamp = pd.Timestamp.now(tz="UTC"),
    max_levels: int = None,
    bps_range: float = 0,
    min_bid: float = 0,
    max_ask: float = np.inf,
) -> dict:
    """
    Reconstruct a limit order book for a specific point in time.

    Args:
      events: A pandas DataFrame of limit order events.
      tp: The time point at which to reconstruct the order book.
      max_levels: The maximum number of price levels to return.
      bps_range: The maximum depth to return in basis points from the best bid/ask.
      min_bid: The minimum bid price to include.
      max_ask: The maximum ask price to include.

    Returns:
      A dictionary containing the reconstructed order book with 'timestamp', 'asks', and 'bids' DataFrames.
    """
    pct_range = bps_range * 0.0001

    def active_bids(active_orders: pd.DataFrame) -> pd.DataFrame:
        bids = active_orders[
            (active_orders["direction"] == "bid") & (active_orders["type"] != "market")
        ]
        bids = bids.sort_values(by=["price", "id"], ascending=[False, True])
        first_price = bids["price"].iloc[0]
        bids["bps"] = ((first_price - bids["price"]) / first_price) * 10000
        bids["liquidity"] = bids["volume"].cumsum()
        return bids

    def active_asks(active_orders: pd.DataFrame) -> pd.DataFrame:
        asks = active_orders[
            (active_orders["direction"] == "ask") & (active_orders["type"] != "market")
        ]
        asks = asks.sort_values(by=["price", "id"], ascending=[True, True])
        first_price = asks["price"].iloc[0]
        asks["bps"] = ((asks["price"] - first_price) / first_price) * 10000
        asks["liquidity"] = asks["volume"].cumsum()
        return asks

    # Active orders processing
    created_before = events[
        (events["action"] == "created") & (events["timestamp"] <= tp)
    ]["id"]
    deleted_before = events[
        (events["action"] == "deleted") & (events["timestamp"] <= tp)
    ]["id"]
    active_order_ids = set(created_before) - set(deleted_before)
    active_orders = events[events["id"].isin(active_order_ids)]
    active_orders = active_orders[active_orders["timestamp"] <= tp]

    # Handle changed orders
    changed_orders = active_orders[active_orders["action"] == "changed"]
    changed_orders = (
        changed_orders.sort_values(by="timestamp", ascending=False)
        .groupby("id")
        .head(1)
    )
    active_orders = pd.concat(
        [active_orders[active_orders["action"] != "changed"], changed_orders]
    )
    assert not active_orders["id"].duplicated().any()

    asks = active_asks(active_orders)[
        ["id", "timestamp", "exchange.timestamp", "price", "volume", "liquidity", "bps"]
    ]
    asks = asks.iloc[::-1]  # Reverse asks for ascending price order
    bids = active_bids(active_orders)[
        ["id", "timestamp", "exchange.timestamp", "price", "volume", "liquidity", "bps"]
    ]

    if pct_range > 0:
        max_ask = asks["price"].iloc[-1] * (1 + pct_range)
        asks = asks[asks["price"] <= max_ask]
        min_bid = bids["price"].iloc[0] * (1 - pct_range)
        bids = bids[bids["price"] >= min_bid]

    if max_levels is not None:
        asks = asks.iloc[-max_levels:]
        bids = bids.iloc[:max_levels]

    return {"timestamp": tp, "asks": asks, "bids": bids}
