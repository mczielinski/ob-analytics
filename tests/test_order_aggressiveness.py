"""Tests for order_aggressiveness — event_id merge_asof boundary conditions."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.event_processing import order_aggressiveness
from ob_analytics.exceptions import InvalidDataError


def _make_events_and_depth(event_rows, depth_rows):
    """Build synthetic events and depth_summary DataFrames.

    event_rows: list of (event_id, direction, action, type, timestamp, price)
    depth_rows: list of (event_id, timestamp, best_bid_price, best_ask_price)
    """
    events = pd.DataFrame(
        event_rows,
        columns=["event_id", "direction", "action", "type", "timestamp", "price"],
    ).assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))

    depth = pd.DataFrame(
        depth_rows,
        columns=["event_id", "timestamp", "best_bid_price", "best_ask_price"],
    ).assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))

    return events, depth


class TestOrderAggressiveness:
    """Unit tests for the event_id-based order_aggressiveness function."""

    def test_bid_more_aggressive_than_best(self):
        """A bid priced above the best bid → positive aggressiveness."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                (2, "bid", "created", "resting-limit", "2015-01-01 00:00:02", 105),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                (2, "2015-01-01 00:00:02", 100, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        row2 = result[result["event_id"] == 2].iloc[0]
        assert row2["aggressiveness_bps"] > 0  # 105 vs best_bid 100

    def test_ask_more_aggressive_than_best(self):
        """An ask priced below the best ask → positive aggressiveness."""
        events, depth = _make_events_and_depth(
            [
                (1, "ask", "created", "resting-limit", "2015-01-01 00:00:01", 110),
                (2, "ask", "created", "resting-limit", "2015-01-01 00:00:02", 105),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                (2, "2015-01-01 00:00:02", 100, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        row2 = result[result["event_id"] == 2].iloc[0]
        assert row2["aggressiveness_bps"] > 0  # 105 vs best_ask 110

    def test_passive_bid(self):
        """A bid priced below the best bid → negative aggressiveness."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "flashed-limit", "2015-01-01 00:00:01", 100),
                (2, "bid", "created", "flashed-limit", "2015-01-01 00:00:02", 95),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                (2, "2015-01-01 00:00:02", 100, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        row2 = result[result["event_id"] == 2].iloc[0]
        assert row2["aggressiveness_bps"] < 0  # 95 vs best_bid 100

    def test_burst_timestamp_no_future_peeking(self):
        """Two bids at identical timestamps are both evaluated against PRIOR depth."""
        events, depth = _make_events_and_depth(
            [
                # Seed depth event
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                # Two bids in the same millisecond burst
                (2, "bid", "created", "flashed-limit", "2015-01-01 00:00:02", 105),
                (3, "bid", "created", "flashed-limit", "2015-01-01 00:00:02", 104),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                (2, "2015-01-01 00:00:02", 105, 110),  # depth AFTER event 2
                (3, "2015-01-01 00:00:02", 105, 110),  # depth AFTER event 3
            ],
        )
        result = order_aggressiveness(events, depth)
        agg2 = result[result["event_id"] == 2]["aggressiveness_bps"].iloc[0]
        agg3 = result[result["event_id"] == 3]["aggressiveness_bps"].iloc[0]

        # Event 2 looks back to depth at event_id=1 (best_bid=100)
        # aggressiveness = 10000 * (105 - 100) / 100 = 500
        assert abs(agg2 - 500) < 0.01

        # Event 3 looks back to depth at event_id=2 (best_bid=105)
        # aggressiveness = 10000 * (104 - 105) / 105 ≈ -95.24
        assert agg3 < 0  # passive relative to post-event-2 book

    def test_first_event_gets_nan(self):
        """The first event has no prior depth → NaN aggressiveness."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        assert result["aggressiveness_bps"].isna().all()

    def test_changed_action_excluded(self):
        """Orders with action='changed' are not scored."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                (2, "bid", "changed", "resting-limit", "2015-01-01 00:00:02", 95),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                (2, "2015-01-01 00:00:02", 100, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        row2 = result[result["event_id"] == 2].iloc[0]
        assert pd.isna(row2["aggressiveness_bps"])

    def test_missing_timestamps_raises(self):
        """Raise InvalidDataError when order timestamps aren't in depth_summary."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                (2, "bid", "created", "resting-limit", "2015-01-01 00:00:05", 105),
            ],
            [
                # Only depth at 00:00:01 — missing 00:00:05
                (1, "2015-01-01 00:00:01", 100, 110),
            ],
        )
        with pytest.raises(InvalidDataError, match="Not all order timestamps"):
            order_aggressiveness(events, depth)
