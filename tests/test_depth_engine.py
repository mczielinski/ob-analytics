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
    """Touching (equal-price) orders are kept; strictly-crossed stale levels are evicted.

    A resting bid and ask can coexist only when ``bid_price < ask_price``.  When a
    fresh quote strictly crosses the opposing best, the *stale* opposing levels are
    evicted (the fresh quote is trusted) rather than the fresh quote dropped.  This
    keeps an orphaned best level (e.g. one whose delete event is missing from the
    feed) from freezing the touch forever.  Equal-price ``touching`` levels are
    still permitted (locks are tolerated, matching the original design).
    """

    def test_ask_at_best_bid_is_processed(self):
        """An ask at exactly the best bid should NOT be silently dropped."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        # Set up: bid at 100
        engine.update_side(100, 5.0, 0, out)
        assert engine._best_bid == 100

        # Ask at exactly 100 — with strict < guard, this IS processed
        engine.update_side(100, 3.0, 1, out)
        assert 100 in engine._ask_levels

    def test_bid_at_best_ask_is_processed(self):
        """A bid at exactly the best ask should NOT be silently dropped."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        # Set up: ask at 110
        engine.update_side(110, 5.0, 1, out)
        assert engine._best_ask == 110

        # Bid at exactly 110 — with strict > guard, this IS processed
        engine.update_side(110, 3.0, 0, out)
        assert 110 in engine._bid_levels

    def test_ask_below_best_bid_evicts_crossed_bid(self):
        """An ask strictly below the best bid is the fresh truth: evict the stale bid."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update_side(100, 5.0, 0, out)  # bid at 100
        engine.update_side(99, 3.0, 1, out)  # ask at 99 < bid 100 (strict cross)

        assert 99 in engine._ask_levels  # fresh ask kept
        assert 100 not in engine._bid_levels  # stale crossed bid evicted
        assert engine._best_ask == 99
        assert engine._best_bid is None

    def test_bid_above_best_ask_evicts_crossed_ask(self):
        """A bid strictly above the best ask is the fresh truth: evict the stale ask."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update_side(110, 5.0, 1, out)  # ask at 110
        engine.update_side(111, 3.0, 0, out)  # bid at 111 > ask 110 (strict cross)

        assert 111 in engine._bid_levels  # fresh bid kept
        assert 110 not in engine._ask_levels  # stale crossed ask evicted
        assert engine._best_bid == 111
        assert engine._best_ask is None

    def test_crossing_bid_unfreezes_orphaned_best_ask(self):
        """Regression: an orphaned ask (missing delete) must not freeze best_ask.

        Reproduces the depth-tracker freeze: two ask orders rest at a price
        whose delete events never arrive, pinning best_ask.  Once bids climb
        strictly above that stale level, it must be evicted so best_ask tracks
        the genuine liquidity above it.
        """
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update_side(100, 0.17, 1, out)  # orphaned ask at 100 (never deleted)
        engine.update_side(105, 1.0, 1, out)  # genuine ask above
        assert engine._best_ask == 100

        engine.update_side(101, 1.0, 0, out)  # bid climbs above the stale ask

        assert 100 not in engine._ask_levels  # stale level evicted
        assert engine._best_ask == 105  # best_ask tracks up, no longer frozen
        assert engine._best_bid == 101

    def test_compute_writes_opposing_side_after_eviction(self):
        """End-to-end: depth_summary best_ask must track, and best_bid be written.

        Eviction touches the *opposing* side, so ``compute`` must refresh and
        write the opposing metrics columns rather than carry the stale prior row.
        """
        engine = DepthMetricsEngine()
        depth = _depth(
            ("2026-01-01T00:00:00", 100.0, 0.17, "ask"),  # orphan ask, never deleted
            ("2026-01-01T00:00:01", 105.0, 1.0, "ask"),  # genuine ask above
            ("2026-01-01T00:00:02", 101.0, 1.0, "bid"),  # bid crosses the orphan
        )

        out = engine.compute(depth)
        last = out.iloc[-1]

        assert last["best_ask_price"] == 105.0  # tracked, not frozen at 100
        assert last["best_bid_price"] == 101.0  # opposing side written
        assert last["best_bid_vol"] == 1.0


class TestBestPriceRecalculation:
    """When the best level is deleted, the next-best must promote."""

    def test_ask_deletion_promotes_next_best(self):
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update_side(100, 5.0, 1, out)  # ask at 100
        engine.update_side(105, 3.0, 1, out)  # ask at 105
        assert engine._best_ask == 100

        engine.update_side(100, 0.0, 1, out)  # delete best ask
        assert engine._best_ask == 105
        assert engine._best_ask_vol == 3.0

    def test_bid_deletion_promotes_next_best(self):
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update_side(100, 5.0, 0, out)  # bid at 100
        engine.update_side(95, 3.0, 0, out)  # bid at 95
        assert engine._best_bid == 100

        engine.update_side(100, 0.0, 0, out)  # delete best bid
        assert engine._best_bid == 95
        assert engine._best_bid_vol == 3.0

    def test_delete_only_ask_clears_best(self):
        """Deleting the sole ask level sets best_ask to None."""
        engine = DepthMetricsEngine()
        out = np.zeros(engine._row_len)

        engine.update_side(100, 5.0, 1, out)
        engine.update_side(100, 0.0, 1, out)

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
        engine.update_side(10000, 4.0, 1, out)
        # additional ask 100bps higher = 10100
        engine.update_side(10100, 7.0, 1, out)

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

        engine.update_side(10000, 4.0, 0, out)  # best bid
        engine.update_side(9900, 7.0, 0, out)  # 100bps below
        engine.update_side(9000, 1.0, 0, out)  # 1000bps below — outside 5*100bps window

        assert out[0] == 10000
        assert out[1] == 4.0
        bins = out[2 : 2 + 5]
        # 4.0 (best) + 7.0 (+100bps) inside window; 1.0 outside
        assert bins.sum() == 11.0


def test_interval_sums_sparse_matches_dense():
    """_interval_sums_sparse must byte-match the dense cumsum reference.

    This is the correctness proof for the depth-engine performance fix: summing
    only the active levels into the bins must reproduce, bit-for-bit, the legacy
    ``np.cumsum(dense)[breaks]`` differencing over the full zero-padded window.
    """
    from ob_analytics.depth import _cached_breaks, _interval_sums_sparse

    rng = np.random.default_rng(20260602)
    for _ in range(3000):
        bins = int(rng.integers(1, 7))
        range_len = int(rng.choice([1, 2, 3, 5, 10, 100, 1000, 50000]))
        side = int(rng.integers(0, 2))
        best = int(rng.integers(1000, 6_000_000))
        n_active = int(rng.integers(0, 20))
        raw_idx = rng.integers(-3, max(range_len, 1) + 3, size=n_active)
        levels: dict[int, float] = {}
        for idx in raw_idx:
            idx = int(idx)
            price = best + idx if side == 1 else best - idx
            if price <= 0 or price in levels:
                continue
            levels[price] = float(rng.random() * 1000)

        breaks = _cached_breaks(range_len, bins)
        dense = np.zeros(range_len, dtype=np.float64)
        for p, v in levels.items():
            idx = (p - best) if side == 1 else (best - p)
            if 0 <= idx < range_len:
                dense[idx] = v
        # Legacy reference: cumulative sum sampled at the bin breaks, then
        # differenced into per-bin sums (the original interval_sum_breaks).
        cumulative = np.cumsum(dense)[breaks]
        ref = np.concatenate((np.array([cumulative[0]]), np.diff(cumulative)))
        got = _interval_sums_sparse(levels, best, side, range_len, breaks)
        assert np.array_equal(got, ref), (
            f"mismatch: side={side} range_len={range_len} bins={bins} "
            f"levels={levels} got={got} ref={ref}"
        )
