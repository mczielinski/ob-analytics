"""Tests for filter_depth — time-range filtering with boundary handling."""

import pandas as pd

from ob_analytics.depth import filter_depth


def _depth(*rows):
    """Build a depth DataFrame from (timestamp, price, volume) tuples."""
    return pd.DataFrame(rows, columns=["timestamp", "price", "volume"]).assign(
        timestamp=lambda df: pd.to_datetime(df["timestamp"])
    )


class TestFilterDepth:
    """Unit tests for filter_depth boundary conditions."""

    def test_pre_range_timestamps_clipped(self):
        """Timestamps before from_timestamp are clipped upward to from_timestamp."""
        d = _depth(
            ("2015-01-01 00:00:00", 100, 5.0),  # before range
            ("2015-01-01 00:00:30", 100, 3.0),   # in range
        )
        from_ts = pd.Timestamp("2015-01-01 00:00:10")
        to_ts = pd.Timestamp("2015-01-01 00:01:00")
        result = filter_depth(d, from_ts, to_ts)

        # The pre-range row at price=100 should take the last state (vol=5)
        # but its timestamp should be clipped to from_ts
        pre_rows = result[result["timestamp"] == from_ts]
        assert len(pre_rows) >= 1
        assert (pre_rows["price"] == 100).any()

    def test_zero_volume_excluded_from_pre_range(self):
        """Price levels with volume=0 before the range are dead and excluded."""
        d = _depth(
            ("2015-01-01 00:00:00", 100, 5.0),
            ("2015-01-01 00:00:05", 100, 0.0),   # vol went to 0 before range
            ("2015-01-01 00:00:30", 200, 3.0),    # in range
        )
        from_ts = pd.Timestamp("2015-01-01 00:00:10")
        to_ts = pd.Timestamp("2015-01-01 00:01:00")
        result = filter_depth(d, from_ts, to_ts)

        # Price 100 had vol=0 as its last pre-range state → excluded from pre
        pre_rows = result[(result["timestamp"] == from_ts) & (result["price"] == 100)]
        assert len(pre_rows) == 0

    def test_open_end_rows_added(self):
        """Surviving levels get a volume=0 closing row at to_timestamp."""
        d = _depth(
            ("2015-01-01 00:00:30", 100, 5.0),  # in range, still alive
        )
        from_ts = pd.Timestamp("2015-01-01 00:00:00")
        to_ts = pd.Timestamp("2015-01-01 00:01:00")
        result = filter_depth(d, from_ts, to_ts)

        close_rows = result[(result["timestamp"] == to_ts) & (result["price"] == 100)]
        assert len(close_rows) == 1
        assert close_rows.iloc[0]["volume"] == 0

    def test_empty_pre_range(self):
        """All data within range — no pre-range clipping needed."""
        d = _depth(
            ("2015-01-01 00:00:20", 100, 5.0),
            ("2015-01-01 00:00:30", 200, 3.0),
        )
        from_ts = pd.Timestamp("2015-01-01 00:00:10")
        to_ts = pd.Timestamp("2015-01-01 00:01:00")
        result = filter_depth(d, from_ts, to_ts)

        # Should contain both original rows plus closing rows
        assert len(result) >= 2
        # No rows at from_ts since nothing was pre-range
        assert (result["timestamp"] >= from_ts).all()
