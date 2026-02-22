"""Tests for trades.py — _fix_price_jumps and trade_impacts."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.trades import DefaultTradeInferrer, trade_impacts
from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import MatchingError


def _matched_events(pairs, jump_price=None):
    """Build a minimal matched-events DataFrame.

    pairs: list of (bid_eid, ask_eid, bid_price, ask_price, fill, bid_ts_off, ask_ts_off)
        bid_ts_off / ask_ts_off are seconds offset from base time.
    """
    base = pd.Timestamp("2015-01-01 00:00:00")
    rows = []
    eid_counter = 1
    for bid_eid, ask_eid, bid_price, ask_price, fill, bid_off, ask_off in pairs:
        rows.append({
            "event_id": bid_eid,
            "id": bid_eid * 10,
            "direction": "bid",
            "action": "created",
            "price": bid_price,
            "volume": fill,
            "fill": fill,
            "timestamp": base + pd.Timedelta(seconds=bid_off),
            "exchange_timestamp": base + pd.Timedelta(seconds=bid_off),
            "matching_event": ask_eid,
            "original_number": bid_eid,
        })
        rows.append({
            "event_id": ask_eid,
            "id": ask_eid * 10,
            "direction": "ask",
            "action": "created",
            "price": ask_price,
            "volume": fill,
            "fill": fill,
            "timestamp": base + pd.Timedelta(seconds=ask_off),
            "exchange_timestamp": base + pd.Timedelta(seconds=ask_off),
            "matching_event": bid_eid,
            "original_number": ask_eid,
        })
    df = pd.DataFrame(rows)
    df["direction"] = pd.Categorical(df["direction"], categories=["bid", "ask"], ordered=True)
    df["action"] = pd.Categorical(df["action"], categories=["created", "changed", "deleted"], ordered=True)
    return df


class TestFixPriceJumps:
    """Tests for the maker/taker swap heuristic on consecutive price jumps."""

    def test_no_jumps_no_swap(self):
        """Consecutive prices within threshold → no swap happens."""
        events = _matched_events([
            (1, 2, 100.0, 100.0, 1.0, 0, 1),
            (3, 4, 101.0, 101.0, 1.0, 2, 3),
        ])
        inferrer = DefaultTradeInferrer(PipelineConfig(price_jump_threshold=10.0))
        trades = inferrer.infer_trades(events)
        # No swap needed
        assert len(trades) == 2

    def test_jump_triggers_swap(self):
        """A price jump > threshold swaps maker and taker for that trade."""
        events = _matched_events([
            (1, 2, 100.0, 100.0, 1.0, 0, 1),
            (3, 4, 100.0, 100.0, 1.0, 2, 3),
            (5, 6, 200.0, 200.0, 1.0, 4, 5),  # $100 jump!
        ])
        inferrer = DefaultTradeInferrer(PipelineConfig(price_jump_threshold=10.0))
        trades = inferrer.infer_trades(events)
        assert len(trades) == 3

        # The third trade should have had its maker/taker swapped
        jumped_trade = trades.iloc[2]
        # After swap, maker_event_id and taker_event_id should be reversed
        # from what they'd normally be
        assert jumped_trade["maker_event_id"] != jumped_trade["taker_event_id"]

    def test_jump_at_index_zero_is_skipped(self):
        """If the first trade itself is a 'jump', it gets skipped (no previous trade)."""
        events = _matched_events([
            (1, 2, 500.0, 500.0, 1.0, 0, 1),
            (3, 4, 100.0, 100.0, 1.0, 2, 3),
        ])
        inferrer = DefaultTradeInferrer(PipelineConfig(price_jump_threshold=10.0))
        trades = inferrer.infer_trades(events)
        assert len(trades) == 2

    def test_misaligned_matching_raises(self):
        """If bid event_ids don't align with ask matching_events, raise MatchingError."""
        events = _matched_events([
            (1, 2, 100.0, 100.0, 1.0, 0, 1),
        ])
        # Corrupt the matching_event value
        events.loc[events["direction"] == "ask", "matching_event"] = 999
        inferrer = DefaultTradeInferrer()
        with pytest.raises(MatchingError):
            inferrer.infer_trades(events)


class TestTradeImpacts:
    """Tests for the trade_impacts aggregation function."""

    def test_single_taker_impact(self):
        """A single taker → one impact row with correct VWAP."""
        trades = pd.DataFrame({
            "taker": [10, 10],
            "price": [100.0, 102.0],
            "volume": [2.0, 3.0],
            "timestamp": pd.to_datetime(["2015-01-01 00:00:00", "2015-01-01 00:00:01"]),
            "direction": ["buy", "buy"],
        })
        result = trade_impacts(trades)
        assert len(result) == 1
        row = result.iloc[0]
        # VWAP = (100*2 + 102*3) / (2+3) = 506/5 = 101.2
        assert abs(row["vwap"] - 101.2) < 1e-6
        assert row["hits"] == 2
        assert row["vol"] == 5.0
        assert row["min_price"] == 100.0
        assert row["max_price"] == 102.0

    def test_multiple_takers(self):
        """Multiple takers each get their own impact row."""
        trades = pd.DataFrame({
            "taker": [10, 20],
            "price": [100.0, 200.0],
            "volume": [1.0, 1.0],
            "timestamp": pd.to_datetime(["2015-01-01 00:00:00", "2015-01-01 00:00:01"]),
            "direction": ["buy", "sell"],
        })
        result = trade_impacts(trades)
        assert len(result) == 2

    def test_output_columns(self):
        """trade_impacts returns exactly the expected columns."""
        trades = pd.DataFrame({
            "taker": [10],
            "price": [100.0],
            "volume": [1.0],
            "timestamp": pd.to_datetime(["2015-01-01"]),
            "direction": ["buy"],
        })
        result = trade_impacts(trades)
        expected_cols = {"id", "min_price", "max_price", "vwap", "hits", "vol", "start_time", "end_time", "dir"}
        assert set(result.columns) == expected_cols

    def test_single_trade_vwap_equals_price(self):
        """A taker with one trade → VWAP equals the trade price."""
        trades = pd.DataFrame({
            "taker": [10],
            "price": [42.5],
            "volume": [7.0],
            "timestamp": pd.to_datetime(["2015-01-01"]),
            "direction": ["sell"],
        })
        result = trade_impacts(trades)
        assert abs(result.iloc[0]["vwap"] - 42.5) < 1e-10
