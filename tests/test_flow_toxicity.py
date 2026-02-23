"""Tests for flow_toxicity.py — VPIN, Kyle's Lambda, and OFI."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.exceptions import InsufficientDataError, InvalidDataError
from ob_analytics.flow_toxicity import (
    KyleLambdaResult,
    compute_kyle_lambda,
    compute_vpin,
    order_flow_imbalance,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _trades(directions, volumes=None, prices=None, base_sec_offsets=None):
    """Build a minimal trades DataFrame."""
    n = len(directions)
    base = pd.Timestamp("2015-05-01 00:00:00")
    if base_sec_offsets is None:
        base_sec_offsets = list(range(n))
    if volumes is None:
        volumes = [1.0] * n
    if prices is None:
        prices = [100.0] * n
    return pd.DataFrame(
        {
            "timestamp": [
                base + pd.Timedelta(seconds=s) for s in base_sec_offsets
            ],
            "price": prices,
            "volume": volumes,
            "direction": directions,
        }
    )


# ── VPIN ─────────────────────────────────────────────────────────────


class TestComputeVpin:
    def test_uniform_buys_vpin_one(self):
        """All-buy trades → VPIN = 1.0 for every bucket."""
        trades = _trades(["buy"] * 10, volumes=[1.0] * 10)
        result = compute_vpin(trades, bucket_volume=2.0)
        assert len(result) == 5
        assert all(abs(v - 1.0) < 1e-10 for v in result["vpin"])

    def test_balanced_flow_vpin_zero(self):
        """Alternating buy/sell with equal volume → VPIN ≈ 0."""
        directions = ["buy", "sell"] * 5
        trades = _trades(directions, volumes=[2.0] * 10)
        result = compute_vpin(trades, bucket_volume=4.0)
        # Each bucket gets one buy (2.0) and one sell (2.0) → perfectly balanced
        assert all(abs(v) < 1e-10 for v in result["vpin"])

    def test_bucket_boundaries(self):
        """A large trade split across two buckets is handled correctly."""
        # One 3-unit trade, bucket_volume=2.0 → first bucket full (2.0),
        # second gets 1.0 (incomplete, not in output)
        trades = _trades(["buy"], volumes=[3.0])
        result = compute_vpin(trades, bucket_volume=2.0)
        assert len(result) == 1  # only one complete bucket
        assert abs(result.iloc[0]["vpin"] - 1.0) < 1e-10

    def test_output_columns(self):
        """Returns the expected column set."""
        trades = _trades(["buy"] * 4, volumes=[1.0] * 4)
        result = compute_vpin(trades, bucket_volume=2.0)
        expected = {
            "bucket",
            "timestamp_start",
            "timestamp_end",
            "buy_volume",
            "sell_volume",
            "vpin",
            "vpin_avg",
        }
        assert set(result.columns) == expected

    def test_vpin_avg_is_rolling_mean(self):
        """vpin_avg is a rolling mean over n_buckets."""
        trades = _trades(
            ["buy"] * 6 + ["sell"] * 6,
            volumes=[1.0] * 12,
        )
        result = compute_vpin(trades, bucket_volume=2.0, n_buckets=3)
        # First 3 buckets are all-buy (vpin=1), next 3 are all-sell (vpin=1)
        # Rolling mean with window=3 and min_periods=1
        assert len(result) == 6
        assert abs(result.iloc[0]["vpin_avg"] - 1.0) < 1e-10

    def test_empty_trades_raises(self):
        """Empty DataFrame raises InsufficientDataError."""
        empty = pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])
        with pytest.raises(InsufficientDataError):
            compute_vpin(empty, bucket_volume=1.0)

    def test_missing_columns_raises(self):
        """Missing required columns raises InvalidDataError."""
        bad = pd.DataFrame({"timestamp": [1], "price": [100]})
        with pytest.raises(InvalidDataError):
            compute_vpin(bad, bucket_volume=1.0)

    def test_negative_bucket_volume_raises(self):
        """Non-positive bucket_volume raises ValueError."""
        trades = _trades(["buy"])
        with pytest.raises(ValueError, match="positive"):
            compute_vpin(trades, bucket_volume=-1.0)


# ── Kyle's Lambda ────────────────────────────────────────────────────


class TestComputeKyleLambda:
    def test_positive_impact(self):
        """Buys push price up → positive lambda."""
        # Window 1: all buys, price goes 100 → 105
        # Window 2: all sells, price goes 105 → 100
        trades = _trades(
            ["buy", "buy", "sell", "sell"],
            volumes=[10.0, 10.0, 10.0, 10.0],
            prices=[100.0, 105.0, 105.0, 100.0],
            base_sec_offsets=[0, 60, 300, 360],
        )
        result = compute_kyle_lambda(trades, window="5min")
        assert isinstance(result, KyleLambdaResult)
        assert result.lambda_ > 0
        assert result.n_windows == 2

    def test_single_window_returns_nan(self):
        """Only 1 data point → can't run OLS → NaN."""
        trades = _trades(["buy"], volumes=[1.0], prices=[100.0])
        result = compute_kyle_lambda(trades, window="5min")
        assert np.isnan(result.lambda_)
        assert result.n_windows == 1

    def test_result_fields(self):
        """KyleLambdaResult has all expected attributes."""
        trades = _trades(
            ["buy", "sell"],
            volumes=[1.0, 1.0],
            prices=[100.0, 99.0],
            base_sec_offsets=[0, 300],
        )
        result = compute_kyle_lambda(trades, window="5min")
        assert hasattr(result, "lambda_")
        assert hasattr(result, "t_stat")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "n_windows")
        assert hasattr(result, "regression_df")
        assert isinstance(result.regression_df, pd.DataFrame)

    def test_regression_df_columns(self):
        """Regression DataFrame has expected columns."""
        trades = _trades(
            ["buy", "sell"],
            volumes=[1.0, 1.0],
            prices=[100.0, 99.0],
            base_sec_offsets=[0, 300],
        )
        result = compute_kyle_lambda(trades, window="5min")
        assert set(result.regression_df.columns) == {
            "timestamp",
            "delta_price",
            "signed_volume",
        }

    def test_empty_raises(self):
        """Empty DataFrame raises InsufficientDataError."""
        empty = pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])
        with pytest.raises(InsufficientDataError):
            compute_kyle_lambda(empty)

    def test_missing_columns_raises(self):
        """Missing columns raises InvalidDataError."""
        bad = pd.DataFrame({"timestamp": [1]})
        with pytest.raises(InvalidDataError):
            compute_kyle_lambda(bad)


# ── Order Flow Imbalance ─────────────────────────────────────────────


class TestOrderFlowImbalance:
    def test_output_columns(self):
        """Returns expected columns."""
        trades = _trades(["buy", "sell"], volumes=[1.0, 1.0])
        result = order_flow_imbalance(trades, window="1min")
        expected = {"timestamp", "buy_volume", "sell_volume", "net_volume", "ofi"}
        assert set(result.columns) == expected

    def test_all_buys_ofi_one(self):
        """All buys → OFI = 1.0."""
        trades = _trades(["buy"] * 5, volumes=[1.0] * 5)
        result = order_flow_imbalance(trades, window="1min")
        assert all(abs(v - 1.0) < 1e-10 for v in result["ofi"])

    def test_all_sells_ofi_negative_one(self):
        """All sells → OFI = −1.0."""
        trades = _trades(["sell"] * 5, volumes=[1.0] * 5)
        result = order_flow_imbalance(trades, window="1min")
        assert all(abs(v - (-1.0)) < 1e-10 for v in result["ofi"])

    def test_balanced_ofi_zero(self):
        """Equal buy and sell volume in the same window → OFI = 0."""
        trades = _trades(
            ["buy", "sell"],
            volumes=[5.0, 5.0],
            base_sec_offsets=[0, 1],
        )
        result = order_flow_imbalance(trades, window="1min")
        assert all(abs(v) < 1e-10 for v in result["ofi"])

    def test_empty_raises(self):
        """Empty DataFrame raises InsufficientDataError."""
        empty = pd.DataFrame(columns=["timestamp", "volume", "direction"])
        with pytest.raises(InsufficientDataError):
            order_flow_imbalance(empty)

    def test_missing_columns_raises(self):
        """Missing columns raises InvalidDataError."""
        bad = pd.DataFrame({"timestamp": [1]})
        with pytest.raises(InvalidDataError):
            order_flow_imbalance(bad)
