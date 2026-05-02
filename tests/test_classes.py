"""Tests for the protocol-satisfying classes.

Covers BitstampLoader and DepthMetricsEngine with synthetic data.
"""

from pathlib import Path

import pandas as pd
import pytest

from ob_analytics.config import PipelineConfig
from ob_analytics.depth import DepthMetricsEngine
from ob_analytics.bitstamp import BitstampLoader
from ob_analytics.exceptions import InsufficientDataError, InvalidDataError
from ob_analytics.protocols import EventLoader


class TestBitstampLoader:
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Write a minimal Bitstamp-format CSV and return its path."""
        csv = tmp_path / "orders.csv"
        ts1 = 1430441820000  # 2015-05-01 01:37:00 UTC in milliseconds
        ts2 = ts1 + 10  # +10ms
        ts3 = ts1 + 100  # +100ms
        rows = [
            f"100,{ts1},{ts1 - 1000},236.50,50000000,created,bid",
            f"200,{ts1},{ts1 - 1000},237.00,30000000,created,ask",
            f"100,{ts2},{ts2 - 1000},236.50,40000000,changed,bid",
            f"200,{ts3},{ts3 - 1000},237.00,20000000,changed,ask",
        ]
        csv.write_text(
            "id,timestamp,exchange_timestamp,price,volume,action,direction\n"
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
        for col in [
            "id",
            "timestamp",
            "price",
            "volume",
            "action",
            "direction",
            "fill",
            "event_id",
        ]:
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


class TestDepthMetricsEngine:
    def test_compute_returns_correct_shape(self, tiny_depth):
        engine = DepthMetricsEngine()
        result = engine.compute(tiny_depth)
        assert len(result) == len(tiny_depth)
        assert "timestamp" in result.columns
        assert "best_bid_price" in result.columns
        assert "best_ask_price" in result.columns

    def test_compute_column_count(self, tiny_depth):
        config = PipelineConfig(depth_bins=5)
        engine = DepthMetricsEngine(config)
        result = engine.compute(tiny_depth)
        expected_cols = 1 + 2 * (
            2 + 5
        )  # timestamp + bid(price,vol,5bins) + ask(price,vol,5bins)
        assert len(result.columns) == expected_cols

    def test_best_prices_populated(self, tiny_depth):
        engine = DepthMetricsEngine()
        result = engine.compute(tiny_depth)
        assert result["best_bid_price"].iloc[-1] > 0
        assert result["best_ask_price"].iloc[-1] > 0

    def test_best_bid_vol_updated_on_new_best(self):
        """best_bid_vol must reflect the new volume when a higher bid arrives."""
        ts = pd.Timestamp("2025-01-01")
        depth = pd.DataFrame(
            {
                "timestamp": [
                    ts,
                    ts + pd.Timedelta(seconds=1),
                    ts + pd.Timedelta(seconds=2),
                ],
                "price": [100.00, 200.00, 101.00],
                "volume": [5, 10, 7],
                "direction": pd.Categorical(
                    ["bid", "ask", "bid"],
                    categories=["bid", "ask"],
                    ordered=True,
                ),
            }
        )
        result = DepthMetricsEngine().compute(depth)
        # After the second bid (101 > 100), best_bid_vol must be 7, not stale 5
        assert result["best_bid_vol"].iloc[-1] == 7

    def test_initialise_best_correct(self):
        """Initial best ask = min(ask prices), best bid = max(bid prices)."""
        ts = pd.Timestamp("2025-01-01")
        depth = pd.DataFrame(
            {
                "timestamp": [ts] * 4,
                "price": [100.00, 102.00, 200.00, 198.00],
                "volume": [10, 20, 30, 40],
                "direction": pd.Categorical(
                    ["bid", "bid", "ask", "ask"],
                    categories=["bid", "ask"],
                    ordered=True,
                ),
            }
        )
        result = DepthMetricsEngine().compute(depth)
        # best bid should be max(100, 102) = 102
        assert result["best_bid_price"].iloc[-1] == 102.00
        # best ask should be min(200, 198) = 198
        assert result["best_ask_price"].iloc[-1] == 198.00

    def test_rejects_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])
        with pytest.raises(InsufficientDataError):
            DepthMetricsEngine().compute(df)

    def test_fractional_volume_preserved(self):
        """Sub-integer volumes (e.g. 0.5 BTC) must not be truncated to 0."""
        ts = pd.Timestamp("2025-01-01")
        depth = pd.DataFrame(
            {
                "timestamp": [ts, ts + pd.Timedelta(seconds=1)],
                "price": [100.00, 200.00],
                "volume": [0.5, 0.75],
                "direction": pd.Categorical(
                    ["bid", "ask"], categories=["bid", "ask"], ordered=True
                ),
            }
        )
        result = DepthMetricsEngine().compute(depth)
        assert result["best_bid_vol"].iloc[-1] == 0.5
        assert result["best_ask_vol"].iloc[-1] == 0.75

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
        assert result["best_bid_price"].iloc[-1] == 50000.00
        assert result["best_ask_price"].iloc[-1] == 50100.00
