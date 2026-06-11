"""Tests for LOBSTER format components.

Focused on behaviors that are easy to break during performance
refactors: ``LobsterTradeReader._find_takers`` (taker identification
heuristic) and ``LobsterWriter._reconstruct_orderbook`` (book replay).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.lobster import (
    LobsterLoader,
    LobsterTradeReader,
    LobsterWriter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events(rows: list[dict]) -> pd.DataFrame:
    """Build an events DataFrame matching the LOBSTER pipeline schema."""
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "original_number" not in df.columns:
        df["original_number"] = df["event_id"]
    return df


# ---------------------------------------------------------------------------
# LobsterTradeReader._find_takers
# ---------------------------------------------------------------------------


class TestFindTakers:
    def test_no_submissions_returns_all_na(self):
        execs = _events(
            [
                {
                    "event_id": 10,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                }
            ]
        )
        all_events = execs.copy()  # only execs; no type-1 rows
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert len(result) == 1
        assert pd.isna(result[0])

    def test_empty_execs_returns_empty_array(self):
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 1,
                }
            ]
        )
        execs = all_events.iloc[0:0]
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert len(result) == 0

    def test_marketable_bid_exec_matched_to_recent_ask_submission(self):
        """A bid execution with a prior ask submission at price <= exec price matches."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 1

    def test_non_marketable_ask_submission_does_not_match_bid_exec(self):
        """ask submission priced ABOVE the bid exec is non-marketable -> NA."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 200.0,  # > exec price 100
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert pd.isna(result[0])

    def test_marketable_ask_exec_matched_to_recent_bid_submission(self):
        """An ask execution with a prior bid submission at price >= exec price matches."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,  # >= exec ask price 100
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 200,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 1

    def test_uses_most_recent_opposite_side_submission(self):
        """When multiple opposite-side submissions exist, the latest is picked."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 2,
                    "id": 2,
                    "timestamp": "2024-01-01 09:30:00.005",  # more recent
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                {
                    "event_id": 3,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.010",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 2  # most recent ask submission

    def test_handles_mixed_directions_and_preserves_order(self):
        """Multiple execs on both sides, each gets its own taker (or NA)."""
        all_events = _events(
            [
                # Ask sub before bid exec[0] (matchable)
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                # Bid sub before ask exec[1] (matchable)
                {
                    "event_id": 2,
                    "id": 2,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 101.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 1,
                },
                # bid exec  -> should match event_id 1
                {
                    "event_id": 3,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.002",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
                # ask exec  -> should match event_id 2
                {
                    "event_id": 4,
                    "id": 200,
                    "timestamp": "2024-01-01 09:30:00.003",
                    "price": 101.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        # execs preserves order: bid exec first, then ask exec
        assert int(result[0]) == 1
        assert int(result[1]) == 2

    def test_sparse_matches_keep_position_alignment(self):
        """An unmatched exec sits between matched ones; result indexed by exec position."""
        all_events = _events(
            [
                {
                    "event_id": 1,
                    "id": 1,
                    "timestamp": "2024-01-01 09:30:00.000",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "ask",
                    "raw_event_type": 1,
                },
                # bid exec[0]: matchable
                {
                    "event_id": 2,
                    "id": 100,
                    "timestamp": "2024-01-01 09:30:00.001",
                    "price": 100.0,
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
                # bid exec[1]: no fresh ask sub at >= this price -> NA
                {
                    "event_id": 3,
                    "id": 101,
                    "timestamp": "2024-01-01 09:30:00.002",
                    "price": 50.0,  # ask sub @100 is NOT marketable for buy@50
                    "volume": 1.0,
                    "direction": "bid",
                    "raw_event_type": 4,
                },
            ]
        )
        execs = all_events[all_events["raw_event_type"] == 4].reset_index(drop=True)
        result = LobsterTradeReader._find_takers(all_events, execs)
        assert int(result[0]) == 1
        assert pd.isna(result[1])


# ---------------------------------------------------------------------------
# LobsterWriter._reconstruct_orderbook
# ---------------------------------------------------------------------------


class TestReconstructOrderbook:
    @pytest.fixture
    def writer(self) -> LobsterWriter:
        return LobsterWriter(trading_date="2024-01-01", price_divisor=10000)

    def _book_events(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "id" not in df.columns:
            df["id"] = np.arange(1, len(df) + 1)
        df["action"] = pd.Categorical(
            df["action"], categories=["created", "changed", "deleted"], ordered=True
        )
        df["direction"] = pd.Categorical(
            df["direction"], categories=["bid", "ask"], ordered=True
        )
        return df

    def test_single_creation_populates_first_level(self, writer):
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 5.0,
                    "raw_event_type": 1,
                }
            ]
        )
        ob = writer._reconstruct_orderbook(events, num_levels=2)
        assert len(ob) == 1
        # Bid level 1 = (price * 10000, size)
        assert ob.iloc[0]["bid_price_1"] == 1000000
        assert ob.iloc[0]["bid_size_1"] == 5.0
        # No asks yet -> dummy ask sentinel and 0 size
        assert ob.iloc[0]["ask_size_1"] == 0

    def test_creates_then_deletes_clears_level(self, writer):
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 3.0,
                    "raw_event_type": 1,
                },
                {
                    "timestamp": "2024-01-01 09:30:01",
                    "action": "deleted",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 3.0,
                    "raw_event_type": 3,
                },
            ]
        )
        ob = writer._reconstruct_orderbook(events, num_levels=1)
        assert len(ob) == 2
        assert ob.iloc[0]["ask_size_1"] == 3.0
        assert ob.iloc[1]["ask_size_1"] == 0

    def test_changed_executions_decrement_volume(self, writer):
        """raw_event_type 2/4/5 'changed' rows reduce the price level."""
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 10.0,
                    "raw_event_type": 1,
                },
                {
                    "timestamp": "2024-01-01 09:30:01",
                    "action": "changed",
                    "direction": "bid",
                    "price": 100.0,
                    "volume": 4.0,
                    "raw_event_type": 4,
                },
            ]
        )
        ob = writer._reconstruct_orderbook(events, num_levels=1)
        assert ob.iloc[0]["bid_size_1"] == 10.0
        assert ob.iloc[1]["bid_size_1"] == 6.0

    def test_changed_with_other_raw_type_does_not_decrement(self, writer):
        """A 'changed' row with raw_event_type not in (2,4,5) is a no-op for the book."""
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 10.0,
                    "raw_event_type": 1,
                },
                {
                    "timestamp": "2024-01-01 09:30:01",
                    "action": "changed",
                    "direction": "ask",
                    "price": 100.0,
                    "volume": 4.0,
                    "raw_event_type": np.nan,  # missing/unknown
                },
            ]
        )
        ob = writer._reconstruct_orderbook(events, num_levels=1)
        assert ob.iloc[0]["ask_size_1"] == 10.0
        assert ob.iloc[1]["ask_size_1"] == 10.0

    def test_multiple_levels_sorted_correctly(self, writer):
        events = self._book_events(
            [
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 101.0,
                    "volume": 1.0,
                    "raw_event_type": 1,
                },
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "ask",
                    "price": 100.0,  # better ask
                    "volume": 2.0,
                    "raw_event_type": 1,
                },
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 99.0,
                    "volume": 3.0,
                    "raw_event_type": 1,
                },
                {
                    "timestamp": "2024-01-01 09:30:00",
                    "action": "created",
                    "direction": "bid",
                    "price": 98.0,
                    "volume": 4.0,
                    "raw_event_type": 1,
                },
            ]
        )
        ob = writer._reconstruct_orderbook(events, num_levels=2)
        last = ob.iloc[-1]
        # Asks ascending
        assert last["ask_price_1"] == 1000000  # 100.0
        assert last["ask_size_1"] == 2.0
        assert last["ask_price_2"] == 1010000  # 101.0
        assert last["ask_size_2"] == 1.0
        # Bids descending
        assert last["bid_price_1"] == 990000  # 99.0
        assert last["bid_size_1"] == 3.0
        assert last["bid_price_2"] == 980000  # 98.0
        assert last["bid_size_2"] == 4.0

    def test_row_per_event(self, writer):
        events = self._book_events(
            [
                {
                    "timestamp": f"2024-01-01 09:30:0{i}",
                    "action": "created",
                    "direction": "bid",
                    "price": 100.0 - i,
                    "volume": 1.0,
                    "raw_event_type": 1,
                }
                for i in range(5)
            ]
        )
        ob = writer._reconstruct_orderbook(events, num_levels=3)
        assert len(ob) == 5


# ---------------------------------------------------------------------------
# original_number convention (cross-format parity)
# ---------------------------------------------------------------------------


class TestOriginalNumberConvention:
    """``original_number`` tracks the 1-based source-file row, not ``event_id``.

    The Bitstamp and LOBSTER loaders historically diverged: Bitstamp carried
    the original CSV row through its ``[id, volume, action, timestamp]`` sort,
    while LOBSTER simply aliased ``original_number = event_id`` and discarded
    the source position of filtered rows (halts / cross trades). They now share
    one convention so trade provenance (``maker_og`` / ``taker_og``) is
    comparable across formats. This pins the LOBSTER side of that contract.
    """

    def test_original_number_tracks_source_rows(self, tmp_path):
        # LOBSTER message schema: time,event_type,id,volume,price,direction.
        # Row 2 (halt, type 7) and row 5 (cross, type 6) are filtered out, so
        # the surviving events keep their 1-based source-file row in
        # original_number while event_id is renumbered contiguously.
        msg = tmp_path / "AAPL_2024-01-01_34200000_57600000_message_1.csv"
        msg.write_text(
            "34200.0,1,1,100,1000000,1\n"  # row 1 -> kept (created bid)
            "34200.1,7,0,0,0,1\n"  # row 2 -> halt (dropped)
            "34200.2,1,2,100,1010000,-1\n"  # row 3 -> kept (created ask)
            "34200.3,4,1,50,1000000,1\n"  # row 4 -> kept (execution)
            "34200.4,6,0,0,0,1\n"  # row 5 -> cross (dropped)
            "34200.5,3,2,100,1010000,-1\n"  # row 6 -> kept (deleted ask)
        )
        events = LobsterLoader(trading_date="2024-01-01").load(msg)

        assert list(events["event_id"]) == [1, 2, 3, 4]
        assert list(events["original_number"]) == [1, 3, 4, 6]
        # The two columns are distinct concepts once any source row is filtered.
        assert not events["original_number"].equals(events["event_id"])
        assert events["original_number"].is_unique


class TestLobsterDepthFromOrderbook:
    """Oracle test: the vectorized per-side diff must match a dict-diff
    reference (the algorithm it replaced) modulo within-event row order,
    which is now deterministic (asks then bids, ascending price)."""

    def test_matches_dict_diff_reference(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.lobster import (
            _DUMMY_ASK_PRICE,
            _DUMMY_BID_PRICE,
            lobster_depth_from_orderbook,
        )

        rng = np.random.default_rng(20260611)
        n, levels, divisor, dec = 60, 3, 10_000, 2

        # Random walk of a tiny book: occasional empty levels (dummy price),
        # occasional zero sizes, occasional unchanged consecutive rows.
        rows = []
        for i in range(n):
            row = []
            base = 5_000_000 + int(rng.integers(-3, 4)) * 100
            for lv in range(levels):
                ap = base + (lv + 1) * 100
                bp = base - (lv + 1) * 100
                av = int(rng.integers(0, 4)) * 100
                bv = int(rng.integers(0, 4)) * 100
                if rng.random() < 0.15:
                    ap, av = _DUMMY_ASK_PRICE, 0
                if rng.random() < 0.15:
                    bp, bv = _DUMMY_BID_PRICE, 0
                row.extend([ap, av, bp, bv])
            rows.append(row)
            if rng.random() < 0.2 and i + 1 < n:
                rows.append(list(row))  # unchanged consecutive row
        ob = pd.DataFrame(rows[:n])
        ob_path = tmp_path / "TEST_orderbook_3.csv"
        ob.to_csv(ob_path, index=False, header=False)

        events = pd.DataFrame(
            {
                "event_id": np.arange(1, n + 1),
                "timestamp": pd.date_range("2012-06-21 09:30", periods=n, freq="s"),
                "raw_event_type": np.ones(n, dtype=int),
            }
        )

        cfg = PipelineConfig(
            price_decimals=dec, price_divisor=divisor, volume_decimals=0
        )
        depth, _summary = lobster_depth_from_orderbook(events, ob_path, cfg)

        # Dict-diff reference (previous implementation, verbatim semantics).
        arr = ob.to_numpy()
        ref_rows, prev = [], {}
        for i in range(n):
            curr: dict[tuple[str, float], float] = {}
            for j in range(levels):
                b = j * 4
                ap, av, bp, bv = arr[i, b], arr[i, b + 1], arr[i, b + 2], arr[i, b + 3]
                if ap != _DUMMY_ASK_PRICE and av > 0:
                    pr = round(ap / divisor, dec)
                    curr[("ask", pr)] = curr.get(("ask", pr), 0) + av
                if bp != _DUMMY_BID_PRICE and bv > 0:
                    pr = round(bp / divisor, dec)
                    curr[("bid", pr)] = curr.get(("bid", pr), 0) + bv
            for key in set(prev) | set(curr):
                pv, cv = prev.get(key, 0.0), curr.get(key, 0.0)
                if pv != cv:
                    ref_rows.append(
                        {
                            "event_id": events["event_id"].iloc[i],
                            "timestamp": events["timestamp"].iloc[i],
                            "price": key[1],
                            "volume": float(cv),
                            "direction": key[0],
                        }
                    )
            prev = curr
        ref = pd.DataFrame(ref_rows)

        def canon(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["direction"] = out["direction"].astype(str)
            return out.sort_values(
                ["event_id", "direction", "price"], kind="stable"
            ).reset_index(drop=True)[
                ["event_id", "timestamp", "price", "volume", "direction"]
            ]

        pd.testing.assert_frame_equal(canon(ref), canon(depth), check_dtype=False)

    def test_within_event_order_is_asks_then_bids_ascending(self, tmp_path):
        from ob_analytics.config import PipelineConfig
        from ob_analytics.lobster import lobster_depth_from_orderbook

        # One row: two asks + two bids -> first event emits everything.
        ob = pd.DataFrame([[5001000, 100, 4999000, 200, 5002000, 300, 4998000, 400]])
        ob_path = tmp_path / "TEST_orderbook_2.csv"
        ob.to_csv(ob_path, index=False, header=False)
        events = pd.DataFrame(
            {
                "event_id": [1],
                "timestamp": pd.to_datetime(["2012-06-21 09:30"]),
                "raw_event_type": [1],
            }
        )
        cfg = PipelineConfig(price_decimals=2, price_divisor=10_000)
        depth, _ = lobster_depth_from_orderbook(events, ob_path, cfg)
        assert list(depth["direction"].astype(str)) == ["ask", "ask", "bid", "bid"]
        assert list(depth["price"]) == [500.1, 500.2, 499.8, 499.9]
