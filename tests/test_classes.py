"""Tests for the protocol-satisfying classes.

Covers BitstampLoader, NeedlemanWunschMatcher, DefaultTradeInferrer,
and DepthMetricsEngine with synthetic data.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ob_analytics.config import PipelineConfig
from ob_analytics.depth import DepthMetricsEngine
from ob_analytics.event_processing import BitstampLoader
from ob_analytics.exceptions import InsufficientDataError, InvalidDataError
from ob_analytics.matching_engine import NeedlemanWunschMatcher
from ob_analytics.protocols import EventLoader, MatchingEngine, TradeInferrer
from ob_analytics.trades import DefaultTradeInferrer


class TestBitstampLoader:
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Write a minimal Bitstamp-format CSV and return its path."""
        csv = tmp_path / "orders.csv"
        ts1 = 1430441820000  # 2015-05-01 01:37:00 UTC in milliseconds
        ts2 = ts1 + 10       # +10ms
        ts3 = ts1 + 100      # +100ms
        rows = [
            f"100,{ts1},{ts1 - 1000},236.50,50000000,created,bid",
            f"200,{ts1},{ts1 - 1000},237.00,30000000,created,ask",
            f"100,{ts2},{ts2 - 1000},236.50,40000000,changed,bid",
            f"200,{ts3},{ts3 - 1000},237.00,20000000,changed,ask",
        ]
        csv.write_text(
            "id,timestamp,exchange.timestamp,price,volume,action,direction\n"
            + "\n".join(rows)
            + "\n"
        )
        return csv

    def test_satisfies_protocol(self):
        assert isinstance(BitstampLoader(), EventLoader)

    def test_loads_csv(self, sample_csv: Path):
        loader = BitstampLoader()
        events = loader.load(sample_csv)
        assert isinstance(events, pd.DataFrame)
        assert len(events) == 4
        for col in ["id", "timestamp", "price", "volume", "action", "direction", "fill", "event.id"]:
            assert col in events.columns

    def test_fill_computed(self, sample_csv: Path):
        events = BitstampLoader().load(sample_csv)
        assert events["fill"].sum() > 0

    def test_rejects_missing_columns(self, tmp_path: Path):
        csv = tmp_path / "bad.csv"
        csv.write_text("id,price\n1,100\n")
        with pytest.raises(InvalidDataError, match="missing required columns"):
            BitstampLoader().load(csv)

    def test_config_precision(self, sample_csv: Path):
        config = PipelineConfig(price_decimals=4, volume_decimals=4)
        events = BitstampLoader(config).load(sample_csv)
        assert events["price"].iloc[0] == 236.5


class TestNeedlemanWunschMatcher:
    def test_satisfies_protocol(self):
        assert isinstance(NeedlemanWunschMatcher(), MatchingEngine)

    def test_match_simple_pairs(self, tiny_events):
        matcher = NeedlemanWunschMatcher()
        result = matcher.match(tiny_events)
        assert "matching.event" in result.columns
        matched_count = result["matching.event"].notna().sum()
        assert matched_count == 4

    def test_rejects_empty_dataframe(self):
        df = pd.DataFrame(
            columns=["direction", "fill", "original_number", "event.id", "timestamp"]
        )
        with pytest.raises(InsufficientDataError):
            NeedlemanWunschMatcher().match(df)

    def test_custom_cutoff_narrow_window(self, tiny_events):
        """A very narrow cutoff should match fewer (or zero) events."""
        config = PipelineConfig(match_cutoff_ms=1)
        matcher = NeedlemanWunschMatcher(config)
        result = matcher.match(tiny_events)
        assert "matching.event" in result.columns
        matched_count = result["matching.event"].notna().sum()
        assert matched_count <= 4


class TestDefaultTradeInferrer:
    def test_satisfies_protocol(self):
        assert isinstance(DefaultTradeInferrer(), TradeInferrer)

    def test_infer_trades_basic(self, matched_events):
        inferrer = DefaultTradeInferrer()
        trades = inferrer.infer_trades(matched_events)
        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 2
        for col in ["timestamp", "price", "volume", "direction", "maker.event.id", "taker.event.id"]:
            assert col in trades.columns

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(InvalidDataError):
            DefaultTradeInferrer().infer_trades(df)


class TestDepthMetricsEngine:
    def test_compute_returns_correct_shape(self, tiny_depth):
        engine = DepthMetricsEngine()
        result = engine.compute(tiny_depth)
        assert len(result) == len(tiny_depth)
        assert "timestamp" in result.columns
        assert "best.bid.price" in result.columns
        assert "best.ask.price" in result.columns

    def test_compute_column_count(self, tiny_depth):
        config = PipelineConfig(depth_bins=5)
        engine = DepthMetricsEngine(config)
        result = engine.compute(tiny_depth)
        expected_cols = 1 + 2 * (2 + 5)  # timestamp + bid(price,vol,5bins) + ask(price,vol,5bins)
        assert len(result.columns) == expected_cols

    def test_best_prices_populated(self, tiny_depth):
        engine = DepthMetricsEngine()
        result = engine.compute(tiny_depth)
        assert result["best.bid.price"].iloc[-1] > 0
        assert result["best.ask.price"].iloc[-1] > 0

    def test_compat_mode_flag(self, tiny_depth):
        engine_compat = DepthMetricsEngine(compat_mode=True)
        engine_correct = DepthMetricsEngine(compat_mode=False)
        r1 = engine_compat.compute(tiny_depth)
        r2 = engine_correct.compute(tiny_depth)
        assert r1.shape == r2.shape

    def test_rejects_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])
        with pytest.raises(InsufficientDataError):
            DepthMetricsEngine().compute(df)

    def test_dynamic_price_range(self):
        """Prices >$9999.99 should not overflow (unlike the old np.zeros(1M) approach)."""
        ts = pd.Timestamp("2025-01-01")
        depth = pd.DataFrame(
            {
                "timestamp": [ts, ts + pd.Timedelta(seconds=1)],
                "price": [50000.00, 50100.00],
                "volume": [100, 200],
                "direction": pd.Categorical(
                    ["bid", "ask"], categories=["bid", "ask"], ordered=True
                ),
            }
        )
        engine = DepthMetricsEngine()
        result = engine.compute(depth)
        assert result["best.bid.price"].iloc[-1] == 50000.00
        assert result["best.ask.price"].iloc[-1] == 50100.00
