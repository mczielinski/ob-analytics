from datetime import datetime, timezone

import numpy as np
import pandas as pd


def order_book(
    events: pd.DataFrame,
    tp: datetime | None = None,
    max_levels: int | None = None,
    bps_range: int = 0,
    min_bid: float = 0,
    max_ask: float = np.inf,
) -> dict[str, datetime | pd.DataFrame]:
    """
    Reconstruct the order book at a specific point in time.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing order events.
    tp : datetime.datetime, optional
        The point in time at which to evaluate the order book.
        If None, uses the current UTC time.
    max_levels : int, optional
        The maximum number of price levels to include for bids and asks.
    bps_range : int, optional
        Basis points range to filter the bids and asks. Default is 0.
    min_bid : float, optional
        Minimum bid price. Default is 0.
    max_ask : float, optional
        Maximum ask price. Default is infinity.

    Returns
    -------
    dict[str, datetime.datetime or pandas.DataFrame]
        A dictionary containing:
        - 'timestamp': The evaluation timestamp.
        - 'asks': DataFrame of active ask orders.
        - 'bids': DataFrame of active bid orders.
    """
    if tp is None:
        tp = datetime.now(timezone.utc)

    pct_range = bps_range * 0.0001

    def active_bids(active_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Extract active bid orders and calculate BPS and cumulative liquidity.

        Parameters
        ----------
        active_orders : pandas.DataFrame
            DataFrame of active orders.

        Returns
        -------
        pandas.DataFrame
            DataFrame of active bids with BPS and liquidity calculations.
        """
        bids = active_orders[
            (active_orders["direction"] == "bid") & (active_orders["type"] != "market")
        ]
        # Order by price descending, then by id ascending (FIFO)
        bids = bids.sort_values(by=["price", "id"], ascending=[False, True], kind="stable")
        first_price = bids.iloc[0]["price"] if not bids.empty else np.nan
        bids["bps"] = (
            ((first_price - bids["price"]) / first_price) * 10000
            if not bids.empty
            else np.nan
        )
        bids["liquidity"] = bids["volume"].cumsum()
        return bids

    def active_asks(active_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Extract active ask orders and calculate BPS and cumulative liquidity.

        Parameters
        ----------
        active_orders : pandas.DataFrame
            DataFrame of active orders.

        Returns
        -------
        pandas.DataFrame
            DataFrame of active asks with BPS and liquidity calculations.
        """
        asks = active_orders[
            (active_orders["direction"] == "ask") & (active_orders["type"] != "market")
        ]
        # Order by price ascending, then by id ascending (FIFO)
        asks = asks.sort_values(by=["price", "id"], ascending=[True, True], kind="stable")
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
    changed_before = changed_before.sort_values(by="timestamp", ascending=False, kind="stable")
    changed_before = changed_before.drop_duplicates(subset="id", keep="first")

    # Remove changed orders and their initial creations
    active_orders = active_orders[~changed_orders_mask]
    active_orders = active_orders[~active_orders["id"].isin(changed_before["id"])]
    active_orders = pd.concat([active_orders, changed_before], ignore_index=True)

    if not all(active_orders["timestamp"] <= tp):
        raise ValueError(
            f"Some active orders have timestamps after the requested time {tp}."
        )
    if active_orders["id"].duplicated().any():
        raise RuntimeError(
            "Duplicate order IDs found in active orders. "
            "This indicates a data integrity issue."
        )

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
