from datetime import datetime, timezone

import numpy as np
import pandas as pd


def order_book(
    events, tp=None, max_levels=None, bps_range=0, min_bid=0, max_ask=np.inf
):
    if tp is None:
        tp = datetime.now(timezone.utc)

    pct_range = bps_range * 0.0001

    def active_bids(active_orders):
        bids = active_orders[
            (active_orders["direction"] == "bid") & (active_orders["type"] != "market")
        ]
        # Order by price descending, then by id ascending (FIFO)
        bids = bids.sort_values(by=["price", "id"], ascending=[False, True])
        first_price = bids.iloc[0]["price"] if not bids.empty else np.nan
        bids["bps"] = (
            ((first_price - bids["price"]) / first_price) * 10000
            if not bids.empty
            else np.nan
        )
        bids["liquidity"] = bids["volume"].cumsum()
        return bids

    def active_asks(active_orders):
        asks = active_orders[
            (active_orders["direction"] == "ask") & (active_orders["type"] != "market")
        ]
        # Order by price ascending, then by id ascending (FIFO)
        asks = asks.sort_values(by=["price", "id"], ascending=[True, True])
        first_price = asks.iloc[0]["price"] if not asks.empty else np.nan
        asks["bps"] = (
            ((asks["price"] - first_price) / first_price) * 10000
            if not asks.empty
            else np.nan
        )
        asks["liquidity"] = asks["volume"].cumsum()
        return asks

    # Determine active orders
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
    changed_orders_mask = active_orders["action"] == "changed"
    changed_before = active_orders[changed_orders_mask]
    changed_before = changed_before.sort_values(by="timestamp", ascending=False)
    changed_before = changed_before.drop_duplicates(subset="id", keep="first")

    # Remove changed orders and their initial creations
    active_orders = active_orders[~changed_orders_mask]
    active_orders = active_orders[~active_orders["id"].isin(changed_before["id"])]
    active_orders = pd.concat([active_orders, changed_before], ignore_index=True)

    # Sanity checks
    assert all(active_orders["timestamp"] <= tp), "Timestamps exceed current time."
    assert not active_orders["id"].duplicated().any(), "Duplicate order IDs found."

    asks = active_asks(active_orders)
    asks = asks[
        ["id", "timestamp", "exchange.timestamp", "price", "volume", "liquidity", "bps"]
    ]
    # Reverse the asks (ascending price)
    asks = asks.iloc[::-1].reset_index(drop=True)

    bids = active_bids(active_orders)
    bids = bids[
        ["id", "timestamp", "exchange.timestamp", "price", "volume", "liquidity", "bps"]
    ]

    # Apply percentage range filter
    if pct_range > 0:
        if not asks.empty:
            max_ask_price = asks.iloc[-1]["price"] * (1 + pct_range)
            asks = asks[asks["price"] <= max_ask_price]
        if not bids.empty:
            min_bid_price = bids.iloc[0]["price"] * (1 - pct_range)
            bids = bids[bids["price"] >= min_bid_price]

    # Limit the number of levels
    if max_levels is not None:
        asks = asks.tail(max_levels).reset_index(drop=True)
        bids = bids.head(max_levels).reset_index(drop=True)

    return {"timestamp": tp, "asks": asks, "bids": bids}
