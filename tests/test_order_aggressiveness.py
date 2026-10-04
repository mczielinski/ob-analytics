"""Tests for order_aggressiveness — which book stands before each order."""

import pandas as pd

from ob_analytics.analytics import order_aggressiveness


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
    """Unit tests for the standing-quote lookup in order_aggressiveness."""

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

    def test_missing_timestamps_handled_gracefully(self):
        """Missing depth_summary timestamps don't crash -- the last earlier row is used."""
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
        result = order_aggressiveness(events, depth)
        assert "aggressiveness_bps" in result.columns

    def test_event_ids_out_of_time_order(self):
        """The standing quote is found by time, not by event_id.

        Bitstamp numbers its events after sorting by order id, so the event
        with the next-lower id can be much later in time.  The order must be
        read against the book at its own time, not at that event's time.
        """
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                (3, "bid", "created", "resting-limit", "2015-01-01 00:00:02", 101),
                # Next-lower event id than 3's would-be neighbour, but later.
                (2, "bid", "deleted", "resting-limit", "2015-01-01 00:30:00", 100),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                (3, "2015-01-01 00:00:02", 101, 110),
                (2, "2015-01-01 00:30:00", 50, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        agg3 = result[result["event_id"] == 3]["aggressiveness_bps"].iloc[0]
        # Against best bid 100 at 00:00:01, not 50 from the 00:30:00 row.
        assert abs(agg3 - 100) < 1e-9

    def test_same_instant_ties_broken_by_event_id(self):
        """At one instant, rows with a lower event_id stand before the order."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                (5, "bid", "created", "resting-limit", "2015-01-01 00:00:02", 104),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                # Same instant as event 5; rows are not in event_id order.
                (6, "2015-01-01 00:00:02", 103, 110),
                (4, "2015-01-01 00:00:02", 102, 110),
                (5, "2015-01-01 00:00:02", 104, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        agg5 = result[result["event_id"] == 5]["aggressiveness_bps"].iloc[0]
        # Event 4 (best bid 102) is the last row before event 5.
        assert abs(agg5 - 10000 * 2 / 102) < 1e-9

    def test_depth_summary_without_event_id(self):
        """Without event_id, only rows at an earlier timestamp stand before."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                (2, "bid", "created", "resting-limit", "2015-01-01 00:00:02", 105),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
                (2, "2015-01-01 00:00:02", 105, 110),
            ],
        )
        result = order_aggressiveness(events, depth.drop(columns="event_id"))
        agg = result.set_index("event_id")["aggressiveness_bps"]
        assert pd.isna(agg[1])
        assert abs(agg[2] - 500) < 1e-9

    def test_missing_order_timestamp_has_no_standing_book(self):
        """An order with no timestamp is not read against the last book."""
        events, depth = _make_events_and_depth(
            [
                (1, "bid", "created", "resting-limit", "2015-01-01 00:00:01", 100),
                (2, "bid", "created", "resting-limit", None, 105),
            ],
            [
                (1, "2015-01-01 00:00:01", 100, 110),
            ],
        )
        result = order_aggressiveness(events, depth)
        assert pd.isna(result.set_index("event_id")["aggressiveness_bps"][2])
