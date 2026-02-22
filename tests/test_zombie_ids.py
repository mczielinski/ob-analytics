"""Tests for get_zombie_ids — vectorised O(N log M) zombie detection."""

import pandas as pd
import pytest

from ob_analytics.data import get_zombie_ids


def _events(*rows):
    """Helper: build an events DataFrame from (id, action, direction, timestamp, price) tuples."""
    return pd.DataFrame(rows, columns=["id", "action", "direction", "timestamp", "price"]).assign(
        timestamp=lambda df: pd.to_datetime(df["timestamp"])
    )


def _trades(*rows):
    """Helper: build a trades DataFrame from (direction, timestamp, price) tuples."""
    return pd.DataFrame(rows, columns=["direction", "timestamp", "price"]).assign(
        timestamp=lambda df: pd.to_datetime(df["timestamp"])
    )


class TestGetZombieIds:
    """Unit tests for get_zombie_ids boundary conditions."""

    def test_no_zombies_when_all_cancelled(self):
        """Every order has a matching 'deleted' action → no zombies."""
        events = _events(
            (1, "created", "bid", "2015-01-01 00:00:00", 100),
            (1, "deleted", "bid", "2015-01-01 00:00:01", 100),
            (2, "created", "ask", "2015-01-01 00:00:00", 110),
            (2, "deleted", "ask", "2015-01-01 00:00:01", 110),
        )
        trades = _trades(("sell", "2015-01-01 00:00:02", 105))
        assert get_zombie_ids(events, trades) == []

    def test_bid_zombie_detected(self):
        """Uncancelled bid with price > lowest future sell → zombie."""
        events = _events(
            (1, "created", "bid", "2015-01-01 00:00:00", 100),
            # no deletion for id=1
        )
        trades = _trades(("sell", "2015-01-01 00:00:01", 90))  # sell at 90 < bid 100
        result = get_zombie_ids(events, trades)
        assert 1 in result

    def test_ask_zombie_detected(self):
        """Uncancelled ask with price < highest future buy → zombie."""
        events = _events(
            (1, "created", "ask", "2015-01-01 00:00:00", 100),
        )
        trades = _trades(("buy", "2015-01-01 00:00:01", 110))  # buy at 110 > ask 100
        result = get_zombie_ids(events, trades)
        assert 1 in result

    def test_no_zombie_when_trade_does_not_cross(self):
        """Uncancelled bid but no future trade crosses its price → not a zombie."""
        events = _events(
            (1, "created", "bid", "2015-01-01 00:00:00", 100),
        )
        trades = _trades(("sell", "2015-01-01 00:00:01", 110))  # sell at 110 > bid 100
        result = get_zombie_ids(events, trades)
        assert 1 not in result

    def test_empty_trades(self):
        """Empty trades DataFrame → no zombies, no crash."""
        events = _events(
            (1, "created", "bid", "2015-01-01 00:00:00", 100),
        )
        trades = _trades()  # empty
        assert get_zombie_ids(events, trades) == []

    def test_duplicate_timestamps_in_trades(self):
        """Multiple trades at the exact same ms shouldn't break accumulate."""
        events = _events(
            (1, "created", "bid", "2015-01-01 00:00:00", 100),
        )
        trades = _trades(
            ("sell", "2015-01-01 00:00:01", 95),
            ("sell", "2015-01-01 00:00:01", 90),  # same timestamp, lower price
        )
        result = get_zombie_ids(events, trades)
        assert 1 in result  # bid at 100 > min future sell of 90
