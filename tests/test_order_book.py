"""Tests for order_book_reconstruction.py — edge cases in order_book()."""

import pandas as pd

from ob_analytics.analytics import order_book


def _events(*rows):
    """Build events from (event_id, id, action, direction, type, timestamp, price, volume)."""
    df = pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "id",
            "action",
            "direction",
            "type",
            "timestamp",
            "price",
            "volume",
        ],
    ).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"]),
        exchange_timestamp=lambda d: pd.to_datetime(d["timestamp"]),
    )
    df["action"] = pd.Categorical(
        df["action"], categories=["created", "changed", "deleted"], ordered=True
    )
    df["direction"] = pd.Categorical(
        df["direction"], categories=["bid", "ask"], ordered=True
    )
    return df


class TestOrderBook:
    def test_default_tp_uses_latest_timestamp(self):
        """When tp=None, order_book uses events['timestamp'].max()."""
        events = _events(
            (
                1,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                100.0,
                5.0,
            ),
            (
                2,
                20,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:02",
                110.0,
                3.0,
            ),
        )
        result = order_book(events, tp=None)
        assert result["timestamp"] == pd.Timestamp("2015-01-01 00:00:02")

    def test_max_levels_limits_output(self):
        """max_levels caps the number of price levels returned."""
        events = _events(
            (
                1,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                100.0,
                5.0,
            ),
            (
                2,
                20,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                99.0,
                3.0,
            ),
            (
                3,
                30,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                98.0,
                2.0,
            ),
            (
                4,
                40,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:01",
                110.0,
                1.0,
            ),
        )
        result = order_book(events, max_levels=2)
        assert len(result["bids"]) <= 2

    def test_deleted_orders_not_in_book(self):
        """Orders that were deleted before tp should not appear."""
        events = _events(
            (
                1,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                100.0,
                5.0,
            ),
            (
                2,
                10,
                "deleted",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:02",
                100.0,
                5.0,
            ),
            (
                3,
                20,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:01",
                110.0,
                3.0,
            ),
        )
        result = order_book(events, tp=pd.Timestamp("2015-01-01 00:00:03"))
        assert 10 not in result["bids"]["id"].values

    def test_changed_order_uses_latest_state(self):
        """A changed order should use its most recent volume/price."""
        events = _events(
            (
                1,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                100.0,
                5.0,
            ),
            (
                2,
                10,
                "changed",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:02",
                100.0,
                3.0,
            ),
            (
                3,
                20,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:01",
                110.0,
                1.0,
            ),
        )
        result = order_book(events, tp=pd.Timestamp("2015-01-01 00:00:03"))
        bid_row = result["bids"][result["bids"]["id"] == 10]
        assert len(bid_row) == 1
        assert bid_row.iloc[0]["volume"] == 3.0  # changed volume, not original

    def test_bps_range_filter(self):
        """bps_range filters out orders too far from the best price."""
        events = _events(
            (
                1,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                100.0,
                5.0,
            ),
            (
                2,
                20,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                50.0,
                5.0,
            ),  # 50% away
            (
                3,
                30,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:01",
                110.0,
                3.0,
            ),
        )
        result = order_book(events, bps_range=100)  # 1% range
        # Bid at 50 is ~50% away from best bid 100, should be filtered
        assert 20 not in result["bids"]["id"].values
        # Bid at 100 should survive
        assert 10 in result["bids"]["id"].values


class TestCanonicalActiveSet:
    """Active set under the schemas.py contract: deleted OR exhausted ends an
    order; orders without a created row never enter the book."""

    def test_fully_executed_order_without_delete_is_excluded(self):
        # The LOBSTER phantom regression: full execution emits no delete;
        # the exhausted order must still leave the book.
        events = _events(
            (
                1,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                100.0,
                5.0,
            ),
            (
                2,
                10,
                "changed",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:02",
                100.0,
                0.0,
            ),
            (
                3,
                20,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:03",
                110.0,
                3.0,
            ),
        )
        result = order_book(events)
        assert 10 not in result["bids"]["id"].values
        assert 20 in result["asks"]["id"].values

    def test_partially_executed_order_keeps_outstanding_size(self):
        events = _events(
            (
                1,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:01",
                100.0,
                5.0,
            ),
            (
                2,
                10,
                "changed",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:02",
                100.0,
                2.0,
            ),
        )
        result = order_book(events)
        row = result["bids"].set_index("id").loc[10]
        assert row["volume"] == 2.0

    def test_pre_existing_order_without_created_is_excluded(self):
        events = _events(
            (
                1,
                9,
                "changed",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:01",
                105.0,
                7.0,
            ),
            (
                2,
                10,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:02",
                100.0,
                5.0,
            ),
        )
        result = order_book(events)
        assert 9 not in result["asks"]["id"].values
        assert 10 in result["bids"]["id"].values

    def test_book_is_not_crossed_after_exhaustions(self):
        # An exhausted best ask must not pin the touch below newer bids.
        events = _events(
            (
                1,
                1,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:01",
                101.0,
                5.0,
            ),
            (
                2,
                1,
                "changed",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:02",
                101.0,
                0.0,
            ),
            (
                3,
                2,
                "created",
                "ask",
                "resting-limit",
                "2015-01-01 00:00:03",
                103.0,
                5.0,
            ),
            (
                4,
                3,
                "created",
                "bid",
                "resting-limit",
                "2015-01-01 00:00:04",
                102.0,
                5.0,
            ),
        )
        result = order_book(events)
        best_bid = result["bids"].iloc[0]["price"]
        best_ask = result["asks"].iloc[-1]["price"]
        assert best_bid < best_ask
