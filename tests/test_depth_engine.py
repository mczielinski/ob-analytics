"""Tests for DepthMetricsEngine — crossed-book guards, deletions, event_id passthrough.

Also the price-level book it is built on (:class:`PriceLevelBook`) and the
replay of that book at many instants (:func:`price_level_snapshots`).
"""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.depth import (
    DepthMetricsEngine,
    PriceLevelBook,
    depth_metrics,
    price_level_snapshots,
)


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


class TestSameInstantOrder:
    """Rows that share an instant are replayed in the canonical event order.

    The order is ``schemas.time_order_keys``: the timestamp, then the
    ``event_id`` between rows at one instant.  Each depth_summary row is then
    the book after the event it names, as the per-order rebuild sees it.
    """

    def test_rows_at_one_instant_replay_by_event_id(self):
        # Listed bid first, but the ask (event 1) came first.  In event order
        # the later bid crosses the ask and evicts it; replayed as listed, the
        # ask would evict the bid instead.
        depth = pd.DataFrame(
            {
                "event_id": [2, 1],
                "timestamp": pd.to_datetime(["2015-01-01 00:00:01"] * 2),
                "price": [100, 99],
                "volume": [5, 3],
                "direction": ["bid", "ask"],
            }
        )
        result = DepthMetricsEngine().compute(depth)
        assert list(result["event_id"]) == [1, 2]
        last = result.iloc[-1]
        assert (last["best_bid_price"], last["best_ask_price"]) == (100, 0)

    def test_every_row_matches_the_per_order_rebuild(self):
        """On a feed that is never crossed, the two rebuilds agree after every event."""
        import dataclasses

        from ob_analytics import engine
        from ob_analytics._engine_frames import to_order_events
        from ob_analytics.pipeline import Pipeline
        from ob_analytics.schemas import time_order_keys
        from ob_analytics.synth import (
            SynthConfig,
            SyntheticLoader,
            SyntheticTradeSource,
            generate_session,
        )

        session = generate_session(SynthConfig(seed=3, duration=60.0))
        result = Pipeline(
            loader=SyntheticLoader(session),
            trade_source=SyntheticTradeSource(session),
        ).run(source=None)
        events = result.events.sort_values(
            time_order_keys(result.events), kind="stable"
        ).reset_index(drop=True)
        stream = to_order_events(events, market=True)
        position = pd.Series(np.arange(len(events)), index=events["event_id"])

        def best_after(event_id: int) -> tuple[int, int]:
            # The per-order book after every event up to and including this one.
            end = int(position[event_id]) + 1
            upto = dataclasses.replace(
                stream,
                **{
                    f.name: getattr(stream, f.name)[:end]
                    for f in dataclasses.fields(stream)
                    if getattr(stream, f.name) is not None
                },
            )
            book = engine.book_state(upto, at=int(stream.timestamp[end - 1]))
            bid = stream.price[book.bids.row[0]] if book.bids.row.size else 0
            ask = stream.price[book.asks.row[0]] if book.asks.row.size else 0
            return int(bid), int(ask)

        # An event can write more than one depth row; its last row is the book
        # once the whole event is applied.
        per_event = result.depth_summary.groupby("event_id", sort=False).tail(1)
        assert len(per_event) > 500
        disagree = [
            event_id
            for event_id, bid, ask in zip(
                per_event["event_id"].tolist(),
                per_event["best_bid_price"].tolist(),
                per_event["best_ask_price"].tolist(),
            )
            if (bid, ask) != best_after(event_id)
        ]
        assert disagree == []


class TestPriceLevelBook:
    """The book at L2 that DepthMetricsEngine and the book replay share (#117)."""

    def test_sets_replaces_and_removes_levels(self):
        book = PriceLevelBook()
        book.update(100, 5.0, 0)
        book.update(99, 2.0, 0)
        book.update(102, 3.0, 1)
        book.update(100, 4.0, 0)  # a new size for an existing level
        assert book.bid_prices.tolist() == [99, 100]
        assert book.bid_volumes.tolist() == [2.0, 4.0]
        book.update(100, 0, 0)  # zero removes the level
        assert book.bid_prices.tolist() == [99]
        assert book.ask_prices.tolist() == [102]

    def test_fresh_quote_evicts_the_levels_it_crosses(self):
        book = PriceLevelBook()
        book.update(100, 1.0, 0)
        book.update(103, 1.0, 0)
        # A new ask at 102 crosses the stale bid at 103 only.
        assert book.update(102, 1.0, 1) is True
        assert book.bid_prices.tolist() == [100]
        assert book.ask_prices.tolist() == [102]

    def test_locked_book_is_kept(self):
        book = PriceLevelBook()
        book.update(100, 1.0, 0)
        assert book.update(100, 1.0, 1) is False
        assert book.bid_prices.tolist() == [100]
        assert book.ask_prices.tolist() == [100]

    def test_float_prices_are_not_rounded(self):
        book = PriceLevelBook(price_dtype=np.float64)
        book.update(100.25, 1.0, 0)
        book.update(100.75, 1.0, 1)
        assert book.bid_prices.tolist() == [100.25]
        assert book.ask_prices.tolist() == [100.75]


def _replay_depth():
    """Six depth rows; the 00:00:04 ask at 101 crosses the stale bid at 102."""
    return _depth(
        ("2026-01-01 00:00:01", 100, 5, "bid"),
        ("2026-01-01 00:00:01", 103, 4, "ask"),
        ("2026-01-01 00:00:02", 102, 2, "bid"),
        ("2026-01-01 00:00:03", 104, 1, "ask"),
        ("2026-01-01 00:00:04", 101, 3, "ask"),
        ("2026-01-01 00:00:05", 100, 0, "bid"),
    ).assign(timestamp=lambda df: df["timestamp"].dt.tz_localize("UTC"))


class TestPriceLevelSnapshots:
    """The L2 book at many instants, in one pass over the depth table (#117)."""

    def test_touch_matches_depth_summary_at_every_row(self):
        depth = _replay_depth()
        summary = depth_metrics(depth)
        times = depth["timestamp"].drop_duplicates()
        for t, snap in zip(times, price_level_snapshots(depth, times), strict=True):
            row = summary[summary["timestamp"] <= t].iloc[-1]
            bids, asks = snap["bids"], snap["asks"]
            assert (bids["price"].iloc[0] if len(bids) else 0) == row["best_bid_price"]
            assert (bids["volume"].iloc[0] if len(bids) else 0) == row["best_bid_vol"]
            assert (asks["price"].iloc[-1] if len(asks) else 0) == row["best_ask_price"]
            assert (asks["volume"].iloc[-1] if len(asks) else 0) == row["best_ask_vol"]

    def test_rows_at_one_instant_replay_in_event_order(self):
        # Listed bid first, but the ask (event 1) came first, so the bid
        # evicts it -- as in the depth summary.
        t = pd.Timestamp("2026-01-01 00:00:01Z")
        depth = pd.DataFrame(
            {
                "event_id": [2, 1],
                "timestamp": [t, t],
                "price": [100, 99],
                "volume": [5, 3],
                "direction": ["bid", "ask"],
            }
        )
        (snap,) = price_level_snapshots(depth, [t])
        last = depth_metrics(depth).iloc[-1]
        assert snap["bids"]["price"].tolist() == [last["best_bid_price"]] == [100]
        assert snap["asks"].empty and last["best_ask_price"] == 0

    def test_crossed_level_is_evicted(self):
        depth = _replay_depth()
        (snap,) = price_level_snapshots(depth, [pd.Timestamp("2026-01-01 00:00:04Z")])
        assert snap["bids"]["price"].tolist() == [100]
        # order_book's convention: asks best last.
        assert snap["asks"]["price"].tolist() == [104, 103, 101]
        assert snap["asks"]["liquidity"].tolist() == [8, 7, 3]

    def test_keeps_the_order_of_times_and_their_units(self):
        depth = _replay_depth()
        times = [
            pd.Timestamp("2026-01-01 00:00:05Z"),
            pd.Timestamp("2026-01-01 00:00:02Z"),
        ]
        late, early = price_level_snapshots(depth, times)
        assert late["timestamp"] == times[0] and early["timestamp"] == times[1]
        assert early["bids"]["price"].tolist() == [102, 100]
        assert late["bids"]["price"].tolist() == []
        assert early["bids"]["volume"].dtype == depth["volume"].dtype

    def test_max_levels(self):
        depth = _replay_depth()
        (snap,) = price_level_snapshots(
            depth, [pd.Timestamp("2026-01-01 00:00:04Z")], max_levels=2
        )
        assert snap["asks"]["price"].tolist() == [103, 101]

    def test_naive_instant_on_aware_table_is_refused(self):
        with pytest.raises(TypeError, match="tz-naive"):
            price_level_snapshots(
                _replay_depth(), [pd.Timestamp("2026-01-01 00:00:04")]
            )
