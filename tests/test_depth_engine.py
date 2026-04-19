"""Tests for DepthMetricsEngine — crossed-book guards, deletions, event_id passthrough."""

import numpy as np
import pandas as pd

from ob_analytics.depth import DepthMetricsEngine


def _depth(*rows):
    """Build a depth DataFrame from (timestamp, price, volume, direction) tuples."""
    return pd.DataFrame(
        rows, columns=["timestamp", "price", "volume", "direction"]
    ).assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))


class TestCrossedBookGuards:
    """The strict < / > guards (not <= / >=) ensure touching orders are processed."""

    def test_ask_at_best_bid_is_processed(self):
        """An ask at exactly the best bid should NOT be silently dropped."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        # Set up: bid at 100
        engine.update(100, 5.0, 0, out)
        assert engine._best_bid == 100

        # Ask at exactly 100 — with strict < guard, this IS processed
        engine.update(100, 3.0, 1, out)
        assert 100 in engine._ask_levels

    def test_bid_at_best_ask_is_processed(self):
        """A bid at exactly the best ask should NOT be silently dropped."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        # Set up: ask at 110
        engine.update(110, 5.0, 1, out)
        assert engine._best_ask == 110

        # Bid at exactly 110 — with strict > guard, this IS processed
        engine.update(110, 3.0, 0, out)
        assert 110 in engine._bid_levels

    def test_ask_below_best_bid_is_dropped(self):
        """An ask strictly below the best bid is a crossed book → dropped."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update(100, 5.0, 0, out)  # bid at 100
        engine.update(99, 3.0, 1, out)  # ask at 99 < bid 100

        assert 99 not in engine._ask_levels

    def test_bid_above_best_ask_is_dropped(self):
        """A bid strictly above the best ask is a crossed book → dropped."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update(110, 5.0, 1, out)  # ask at 110
        engine.update(111, 3.0, 0, out)  # bid at 111 > ask 110

        assert 111 not in engine._bid_levels


class TestBestPriceRecalculation:
    """When the best level is deleted, the next-best must promote."""

    def test_ask_deletion_promotes_next_best(self):
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update(100, 5.0, 1, out)  # ask at 100
        engine.update(105, 3.0, 1, out)  # ask at 105
        assert engine._best_ask == 100

        engine.update(100, 0.0, 1, out)  # delete best ask
        assert engine._best_ask == 105
        assert engine._best_ask_vol == 3.0

    def test_bid_deletion_promotes_next_best(self):
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update(100, 5.0, 0, out)  # bid at 100
        engine.update(95, 3.0, 0, out)  # bid at 95
        assert engine._best_bid == 100

        engine.update(100, 0.0, 0, out)  # delete best bid
        assert engine._best_bid == 95
        assert engine._best_bid_vol == 3.0

    def test_delete_only_ask_clears_best(self):
        """Deleting the sole ask level sets best_ask to None."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update(100, 5.0, 1, out)
        engine.update(100, 0.0, 1, out)

        assert engine._best_ask is None
        assert engine._best_ask_vol == 0.0


class TestEventIdPassthrough:
    """If depth has event_id, depth_summary should preserve it."""

    def test_event_id_in_output(self):
        depth = pd.DataFrame(
            {
                "event_id": [1, 2, 3],
                "timestamp": pd.to_datetime(
                    [
                        "2015-01-01 00:00:01",
                        "2015-01-01 00:00:02",
                        "2015-01-01 00:00:03",
                    ]
                ),
                "price": [100.0, 110.0, 100.0],
                "volume": [5.0, 3.0, 0.0],
                "direction": ["bid", "ask", "bid"],
            }
        )
        engine = DepthMetricsEngine()
        result = engine.compute(depth)
        assert "event_id" in result.columns
        assert list(result["event_id"]) == [1, 2, 3]

    def test_no_event_id_still_works(self):
        """Backward compat: depth without event_id still works."""
        depth = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2015-01-01 00:00:01", "2015-01-01 00:00:02"]
                ),
                "price": [100.0, 110.0],
                "volume": [5.0, 3.0],
                "direction": ["bid", "ask"],
            }
        )
        engine = DepthMetricsEngine()
        result = engine.compute(depth)
        assert "event_id" not in result.columns
        assert "timestamp" in result.columns


class TestBpsBinVolumes:
    """Lock in the BPS-bin volume aggregation used by the depth summary."""

    def test_ask_volumes_bucketed_by_bps(self):
        """Two ask levels at +25bps and +50bps should land in their bins."""
        # Use a coarse grid: bps=100, bins=5 → 100bps-wide buckets.
        from ob_analytics.config import PipelineConfig

        cfg = PipelineConfig(depth_bps=100, depth_bins=5, price_decimals=2)
        engine = DepthMetricsEngine(cfg)
        out = np.zeros(engine._row_len)

        # best ask at 10000 (=$100.00 in cents)
        engine.update(10000, 4.0, 1, out)
        # additional ask 100bps higher = 10100
        engine.update(10100, 7.0, 1, out)

        # Column layout: best_bid_price, best_bid_vol, bid_vol... (5),
        #                best_ask_price, best_ask_vol, ask_vol... (5)
        ask_offset = 2 + 5
        # best_ask volume is in best_ask_vol slot (offset+1), bin volumes
        # follow.  The first bin (0-100bps) covers the best ask itself.
        assert out[ask_offset] == 10000
        assert out[ask_offset + 1] == 4.0
        # The first 100bps window contains both the best (4.0) and the
        # +100bps level (7.0)
        bins = out[ask_offset + 2 : ask_offset + 2 + 5]
        assert bins.sum() == 11.0
        # The +100bps level falls in the first bin in this layout
        assert bins[0] > 0

    def test_bid_volumes_bucketed_by_bps(self):
        from ob_analytics.config import PipelineConfig

        cfg = PipelineConfig(depth_bps=100, depth_bins=5, price_decimals=2)
        engine = DepthMetricsEngine(cfg)
        out = np.zeros(engine._row_len)

        engine.update(10000, 4.0, 0, out)  # best bid
        engine.update(9900, 7.0, 0, out)  # 100bps below
        engine.update(9000, 1.0, 0, out)  # 1000bps below — outside 5*100bps window

        assert out[0] == 10000
        assert out[1] == 4.0
        bins = out[2 : 2 + 5]
        # 4.0 (best) + 7.0 (+100bps) inside window; 1.0 outside
        assert bins.sum() == 11.0
