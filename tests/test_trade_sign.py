"""Tests for trade_sign.py — tick rule, Lee–Ready, BVC, and the classifier.

Includes the #107 acceptance harness: on the bundled L3 Bitstamp sample the
per-trade classifiers must agree with the true maker/taker side well above
chance, and VPIN / OFI must run end-to-end once the native ``direction`` is
stripped (the L2 / aggregated-feed scenario).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.exceptions import ConfigError, ObAnalyticsError
from ob_analytics.flow_toxicity import compute_vpin, order_flow_imbalance
from ob_analytics.trade_sign import (
    bulk_volume_classification,
    classify_trade_sign,
    lee_ready,
    tick_rule,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _trades(prices, volumes=None, sec_offsets=None, index=None):
    """Build a minimal trades DataFrame (no native ``direction``)."""
    n = len(prices)
    base = pd.Timestamp("2015-05-01 00:00:00")
    if sec_offsets is None:
        sec_offsets = list(range(n))
    if volumes is None:
        volumes = [1.0] * n
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(seconds=s) for s in sec_offsets],
            "price": np.asarray(prices, dtype=float),
            "volume": np.asarray(volumes, dtype=float),
        },
        index=index,
    )


# ── Tick rule ────────────────────────────────────────────────────────


class TestTickRule:
    def test_up_down_ticks(self):
        # up, up, down, down, up
        signs = tick_rule([10, 11, 12, 11, 10, 12])
        assert signs.tolist() == [1, 1, 1, -1, -1, 1]
        assert signs.dtype == np.int8

    def test_zero_tick_inherits_previous_sign(self):
        # 10→11 up(+1); 11→11 zero → inherit +1; 11→10 down(-1); 10→10 → -1
        assert tick_rule([10, 11, 11, 10, 10]).tolist() == [1, 1, 1, -1, -1]

    def test_leading_zero_run_backfilled(self):
        # First move is a downtick; the flat lead-in inherits it.
        assert tick_rule([5, 5, 4]).tolist() == [-1, -1, -1]

    def test_flat_series_defaults_to_buy(self):
        assert tick_rule([7, 7, 7, 7]).tolist() == [1, 1, 1, 1]

    def test_single_trade(self):
        assert tick_rule([42.0]).tolist() == [1]

    def test_empty(self):
        out = tick_rule([])
        assert out.shape == (0,)
        assert out.dtype == np.int8

    def test_accepts_series(self):
        s = pd.Series([1.0, 2.0, 1.0])
        assert tick_rule(s).tolist() == [1, 1, -1]


# ── Lee–Ready ────────────────────────────────────────────────────────


class TestLeeReady:
    def test_above_below_mid(self):
        prices = np.array([10.0, 9.0])
        mid = np.array([9.5, 9.5])
        assert lee_ready(prices, mid).tolist() == [1, -1]

    def test_at_mid_falls_back_to_tick(self):
        # Second trade sits exactly at the mid → tick rule (uptick 9→10 = buy).
        prices = np.array([9.0, 10.0])
        mid = np.array([9.5, 10.0])
        assert lee_ready(prices, mid).tolist() == [-1, 1]

    def test_nan_mid_falls_back_to_tick(self):
        prices = np.array([10.0, 11.0, 10.0])
        mid = np.array([np.nan, np.nan, np.nan])
        # No quote anywhere → pure tick rule.
        assert lee_ready(prices, mid).tolist() == tick_rule(prices).tolist()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ConfigError, match="same length"):
            lee_ready(np.array([1.0, 2.0]), np.array([1.0]))


# ── classify_trade_sign ──────────────────────────────────────────────


class TestClassifyTradeSign:
    def test_tick_preserves_index_and_order(self):
        # Rows out of chronological order and with a non-default index.
        trades = _trades(
            [11.0, 10.0, 10.5],
            sec_offsets=[2, 0, 1],
            index=[7, 3, 5],
        )
        out = classify_trade_sign(trades, method="tick")
        assert out.name == "direction"
        assert list(out.index) == [7, 3, 5]
        assert list(out.cat.categories) == ["buy", "sell"]
        # Chronological prices 10.0 → 10.5 → 11.0 are all upticks → all buy.
        assert out.astype(str).tolist() == ["buy", "buy", "buy"]

    def test_lee_ready_with_bid_ask_quotes(self):
        trades = _trades([10.0, 10.6], sec_offsets=[1, 2])
        quotes = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2015-05-01 00:00:00")],
                "best_bid_price": [10.2],
                "best_ask_price": [10.4],  # mid 10.3
            }
        )
        out = classify_trade_sign(trades, method="lee_ready", quotes=quotes)
        # 10.0 < 10.3 → sell; 10.6 > 10.3 → buy
        assert out.astype(str).tolist() == ["sell", "buy"]

    def test_lee_ready_accepts_mid_column(self):
        trades = _trades([10.0, 10.6], sec_offsets=[1, 2])
        quotes = pd.DataFrame(
            {"timestamp": [pd.Timestamp("2015-05-01 00:00:00")], "mid": [10.3]}
        )
        out = classify_trade_sign(trades, method="lee_ready", quotes=quotes)
        assert out.astype(str).tolist() == ["sell", "buy"]

    def test_lee_ready_trade_before_first_quote_uses_tick(self):
        # Trade at t=0 precedes the only quote (t=5) → NaN mid → tick fallback.
        trades = _trades([10.0, 11.0], sec_offsets=[0, 1])
        quotes = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2015-05-01 00:00:05")],
                "mid": [10.5],
            }
        )
        out = classify_trade_sign(trades, method="lee_ready", quotes=quotes)
        # Both fall back to tick: 10→11 uptick, backfilled first → buy, buy.
        assert out.astype(str).tolist() == ["buy", "buy"]

    def test_lee_ready_requires_quotes(self):
        with pytest.raises(ConfigError, match="requires quotes"):
            classify_trade_sign(_trades([1.0, 2.0]), method="lee_ready")

    def test_lee_ready_bad_quote_columns(self):
        trades = _trades([10.0])
        quotes = pd.DataFrame({"timestamp": [pd.Timestamp("2015-05-01")], "foo": [1]})
        with pytest.raises(ConfigError, match="mid column"):
            classify_trade_sign(trades, method="lee_ready", quotes=quotes)

    def test_bvc_method_is_rejected_with_guidance(self):
        with pytest.raises(ConfigError, match="bulk_volume_classification"):
            classify_trade_sign(_trades([1.0, 2.0]), method="bvc")

    def test_unknown_method(self):
        with pytest.raises(ConfigError, match="unknown method"):
            classify_trade_sign(_trades([1.0]), method="nope")

    def test_missing_columns(self):
        with pytest.raises(ConfigError, match="missing required columns"):
            classify_trade_sign(pd.DataFrame({"timestamp": [1]}), method="tick")

    def test_empty(self):
        with pytest.raises(ObAnalyticsError):
            classify_trade_sign(_trades([]).iloc[0:0], method="tick")

    def test_method_case_insensitive(self):
        out = classify_trade_sign(_trades([1.0, 2.0]), method="TICK")
        assert out.astype(str).tolist() == ["buy", "buy"]


# ── Bulk volume classification (BVC) ─────────────────────────────────


class TestBulkVolumeClassification:
    def test_buckets_partition_volume(self):
        trades = _trades([100, 101, 102, 101], volumes=[1.0] * 4)
        bvc = bulk_volume_classification(trades, bucket_volume=2.0)
        assert len(bvc) == 2
        # Each bucket's buy + sell volume equals the bucket volume.
        np.testing.assert_allclose(bvc["buy_volume"] + bvc["sell_volume"], 2.0)
        assert set(bvc["buy_fraction"].between(0, 1)) == {True}

    def test_rising_prices_lean_buy(self):
        trades = _trades(list(range(100, 120)), volumes=[1.0] * 20)
        bvc = bulk_volume_classification(trades, bucket_volume=4.0)
        assert (bvc["buy_fraction"] > 0.5).all()

    def test_falling_prices_lean_sell(self):
        trades = _trades(list(range(120, 100, -1)), volumes=[1.0] * 20)
        bvc = bulk_volume_classification(trades, bucket_volume=4.0)
        assert (bvc["buy_fraction"] < 0.5).all()

    def test_flat_prices_are_neutral(self):
        trades = _trades([100.0] * 10, volumes=[1.0] * 10)
        bvc = bulk_volume_classification(trades, bucket_volume=2.0)
        np.testing.assert_allclose(bvc["buy_fraction"], 0.5)

    def test_explicit_sigma(self):
        trades = _trades([100, 101, 102, 103], volumes=[1.0] * 4)
        bvc = bulk_volume_classification(trades, bucket_volume=2.0, sigma=1.0)
        assert len(bvc) == 2
        assert (bvc["buy_fraction"] > 0.5).all()

    def test_incomplete_bucket_returns_empty(self):
        # 1.5 units of volume, bucket needs 2.0 → no completed bucket.
        trades = _trades([100, 101], volumes=[0.5, 1.0])
        bvc = bulk_volume_classification(trades, bucket_volume=2.0)
        assert bvc.empty
        assert "buy_fraction" in bvc.columns

    def test_bad_bucket_volume(self):
        with pytest.raises(ValueError, match="bucket_volume must be positive"):
            bulk_volume_classification(_trades([1.0]), bucket_volume=0.0)

    def test_bad_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            bulk_volume_classification(_trades([1.0]), bucket_volume=1.0, sigma=-1.0)

    def test_missing_columns(self):
        with pytest.raises(ConfigError):
            bulk_volume_classification(
                pd.DataFrame({"timestamp": [1], "price": [1.0]}), bucket_volume=1.0
            )


# ── Fallback wiring in flow_toxicity ─────────────────────────────────


class TestFlowToxicityFallback:
    def _prices(self):
        return [100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 102, 101]

    def test_vpin_without_direction_uses_tick(self):
        trades = _trades(self._prices())
        got = compute_vpin(trades, bucket_volume=2.0)
        # Equivalent to classifying first, then running the native path.
        seeded = trades.copy()
        seeded["direction"] = classify_trade_sign(trades, method="tick")
        expected = compute_vpin(seeded, bucket_volume=2.0)
        pd.testing.assert_frame_equal(got, expected)

    def test_vpin_native_direction_is_backward_compatible(self):
        trades = _trades(self._prices())
        trades["direction"] = ["buy", "sell"] * 6
        # sign_method=None must honor the native direction (not reclassify).
        got = compute_vpin(trades, bucket_volume=2.0)
        # Reclassifying would change the split; assert it did NOT.
        reclassified = compute_vpin(trades, bucket_volume=2.0, sign_method="tick")
        assert not np.allclose(got["vpin"].to_numpy(), reclassified["vpin"].to_numpy())

    def test_vpin_lee_ready_with_quotes(self):
        trades = _trades(self._prices())
        quotes = pd.DataFrame(
            {
                "timestamp": trades["timestamp"],
                "best_bid_price": trades["price"] - 0.5,
                "best_ask_price": trades["price"] + 0.5,
            }
        )
        got = compute_vpin(
            trades, bucket_volume=2.0, sign_method="lee_ready", quotes=quotes
        )
        assert not got.empty
        assert {"vpin", "vpin_avg"}.issubset(got.columns)

    def test_vpin_bvc_path(self):
        trades = _trades(self._prices())
        got = compute_vpin(trades, bucket_volume=2.0, sign_method="bvc")
        assert list(got.columns) == [
            "bucket",
            "timestamp_start",
            "timestamp_end",
            "buy_volume",
            "sell_volume",
            "vpin",
            "vpin_avg",
        ]
        assert (got["vpin"].between(0, 1)).all()

    def test_ofi_without_direction_uses_tick(self):
        trades = _trades(self._prices())
        got = order_flow_imbalance(trades, window="5s")
        seeded = trades.copy()
        seeded["direction"] = classify_trade_sign(trades, method="tick")
        expected = order_flow_imbalance(seeded, window="5s")
        pd.testing.assert_frame_equal(got, expected)

    def test_ofi_bvc_rejected(self):
        trades = _trades(self._prices())
        with pytest.raises(ConfigError, match="only supported by compute_vpin"):
            order_flow_imbalance(trades, sign_method="bvc")

    def test_vpin_missing_price_for_classification(self):
        # No direction and no price → the classifier fallback can't run.
        trades = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2015-05-01 00:00:00"]),
                "volume": [1.0],
            }
        )
        with pytest.raises(ConfigError):
            order_flow_imbalance(trades)


# ── #107 acceptance harness: Bitstamp L3 sample ──────────────────────

# Documented tolerances: the per-trade classifiers must agree with the true
# maker/taker side well above the 0.5 chance level.  Observed on the bundled
# sample (284 trades): tick ≈ 0.83, Lee–Ready ≈ 0.79.  Thresholds sit a
# margin below to stay stable across pandas/numpy point releases.
_TICK_MIN_AGREEMENT = 0.75
_LEE_READY_MIN_AGREEMENT = 0.70


def _agreement(true_side: pd.Series, predicted: pd.Series) -> float:
    return float((true_side.to_numpy() == predicted.astype(str).to_numpy()).mean())


@pytest.fixture(scope="module")
def sample_pipeline_result(bitstamp_sample_dir):
    """Full pipeline run on the bundled Bitstamp sample (loaded once)."""
    from ob_analytics import Pipeline

    return Pipeline().run(bitstamp_sample_dir / "orders.csv.gz")


class TestBitstampValidationHarness:
    def test_classifiers_beat_chance_and_meet_tolerance(self, sample_pipeline_result):
        res = sample_pipeline_result
        trades = res.trades
        true = trades["direction"].astype(str)
        classifiable = true.isin(["buy", "sell"])
        assert classifiable.sum() > 0
        true = true[classifiable]

        tick = classify_trade_sign(trades, method="tick")[classifiable]
        quotes = res.depth_summary[["timestamp", "best_bid_price", "best_ask_price"]]
        lee = classify_trade_sign(trades, method="lee_ready", quotes=quotes)[
            classifiable
        ]

        tick_agreement = _agreement(true, tick)
        lee_agreement = _agreement(true, lee)

        # Both must beat chance and clear the documented tolerance.
        assert tick_agreement > 0.5
        assert lee_agreement > 0.5
        assert tick_agreement >= _TICK_MIN_AGREEMENT
        assert lee_agreement >= _LEE_READY_MIN_AGREEMENT

    def test_vpin_ofi_run_on_direction_stripped_sample(self, sample_pipeline_result):
        # Simulate an L2 / aggregated capture: same trades + quotes, but with
        # the native aggressor side removed.  VPIN and OFI must still run.
        res = sample_pipeline_result
        l2_trades = res.trades.drop(columns=["direction"])
        quotes = res.depth_summary[["timestamp", "best_bid_price", "best_ask_price"]]
        total_volume = float(res.trades["volume"].sum())
        bucket = total_volume / 10.0

        vpin_tick = compute_vpin(l2_trades, bucket_volume=bucket, n_buckets=5)
        assert not vpin_tick.empty
        assert vpin_tick["vpin"].between(0, 1).all()

        vpin_lee = compute_vpin(
            l2_trades,
            bucket_volume=bucket,
            n_buckets=5,
            sign_method="lee_ready",
            quotes=quotes,
        )
        assert not vpin_lee.empty

        vpin_bvc = compute_vpin(
            l2_trades, bucket_volume=bucket, n_buckets=5, sign_method="bvc"
        )
        assert not vpin_bvc.empty

        ofi = order_flow_imbalance(l2_trades, window="1min")
        assert not ofi.empty
        assert ofi["ofi"].between(-1, 1).all()
