"""Tests for order_book_reconstruction.py — edge cases in order_book()."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.order_book_reconstruction import order_book


def _events(*rows):
    """Build events from (event_id, id, action, direction, type, timestamp, price, volume)."""
    df = pd.DataFrame(
        rows,
        columns=["event_id", "id", "action", "direction", "type", "timestamp", "price", "volume"],
    ).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"]),
        exchange_timestamp=lambda d: pd.to_datetime(d["timestamp"]),
    )
    df["action"] = pd.Categorical(df["action"], categories=["created", "changed", "deleted"], ordered=True)
    df["direction"] = pd.Categorical(df["direction"], categories=["bid", "ask"], ordered=True)
    return df


class TestOrderBook:

    def test_default_tp_uses_latest_timestamp(self):
        """When tp=None, order_book uses events['timestamp'].max()."""
        events = _events(
            (1, 10, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 100.0, 5.0),
            (2, 20, "created", "ask", "resting-limit", "2015-01-01 00:00:02", 110.0, 3.0),
        )
        result = order_book(events, tp=None)
        assert result["timestamp"] == pd.Timestamp("2015-01-01 00:00:02")

    def test_max_levels_limits_output(self):
        """max_levels caps the number of price levels returned."""
        events = _events(
            (1, 10, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 100.0, 5.0),
            (2, 20, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 99.0, 3.0),
            (3, 30, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 98.0, 2.0),
            (4, 40, "created", "ask", "resting-limit", "2015-01-01 00:00:01", 110.0, 1.0),
        )
        result = order_book(events, max_levels=2)
        assert len(result["bids"]) <= 2

    def test_deleted_orders_not_in_book(self):
        """Orders that were deleted before tp should not appear."""
        events = _events(
            (1, 10, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 100.0, 5.0),
            (2, 10, "deleted", "bid", "resting-limit", "2015-01-01 00:00:02", 100.0, 5.0),
            (3, 20, "created", "ask", "resting-limit", "2015-01-01 00:00:01", 110.0, 3.0),
        )
        result = order_book(events, tp=pd.Timestamp("2015-01-01 00:00:03"))
        assert 10 not in result["bids"]["id"].values

    def test_changed_order_uses_latest_state(self):
        """A changed order should use its most recent volume/price."""
        events = _events(
            (1, 10, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 100.0, 5.0),
            (2, 10, "changed", "bid", "resting-limit", "2015-01-01 00:00:02", 100.0, 3.0),
            (3, 20, "created", "ask", "resting-limit", "2015-01-01 00:00:01", 110.0, 1.0),
        )
        result = order_book(events, tp=pd.Timestamp("2015-01-01 00:00:03"))
        bid_row = result["bids"][result["bids"]["id"] == 10]
        assert len(bid_row) == 1
        assert bid_row.iloc[0]["volume"] == 3.0  # changed volume, not original

    def test_bps_range_filter(self):
        """bps_range filters out orders too far from the best price."""
        events = _events(
            (1, 10, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 100.0, 5.0),
            (2, 20, "created", "bid", "resting-limit", "2015-01-01 00:00:01", 50.0, 5.0),  # 50% away
            (3, 30, "created", "ask", "resting-limit", "2015-01-01 00:00:01", 110.0, 3.0),
        )
        result = order_book(events, bps_range=100)  # 1% range
        # Bid at 50 is ~50% away from best bid 100, should be filtered
        assert 20 not in result["bids"]["id"].values
        # Bid at 100 should survive
        assert 10 in result["bids"]["id"].values
