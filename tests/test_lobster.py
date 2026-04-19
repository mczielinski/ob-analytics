"""Tests for LOBSTER format components.

Focused on behaviors that are easy to break during performance
refactors: ``LobsterTradeInferrer._find_takers`` (taker identification
heuristic) and ``LobsterWriter._reconstruct_orderbook`` (book replay).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.lobster import (
    LobsterTradeInferrer,
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
# LobsterTradeInferrer._find_takers
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
        result = LobsterTradeInferrer._find_takers(all_events, execs)
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
