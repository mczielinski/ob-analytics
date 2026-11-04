"""Size and price settings are in the instrument's own units, however called.

A pipeline result stores sizes as whole lots and prices as whole ticks, while
a display frame (:func:`~ob_analytics.visualization.display_result`) holds
them as floats in BTC and dollars.  A setting a user types (a VPIN bucket, a
bar threshold, a price correction) means the same amount on either frame.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ob_analytics import (
    Pipeline,
    bars,
    bulk_volume_classification,
    compute_vpin,
    features,
    vpin_bucket_volume,
)
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth import micro_price
from ob_analytics.depth_l2 import DepthCsvWriter
from ob_analytics.exceptions import ConfigError
from ob_analytics.metrics import get_metric
from ob_analytics.visualization.gallery import _metric_payload, display_result

FRAMES = ("events", "trades", "depth", "depth_summary")


@pytest.fixture(scope="module")
def sample(sample_csv_path):
    return Pipeline().run(sample_csv_path)


@pytest.fixture(scope="module")
def shown(sample):
    return display_result(sample)


@pytest.fixture(autouse=True)
def _quiet():
    # VPIN warns when a capture fills fewer than n_buckets buckets.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _lots_without_units() -> pd.DataFrame:
    """Trades with integer sizes and nothing saying what one lot is."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1s", tz="UTC"),
            "price": np.array([100, 101, 102, 101], dtype=np.int64),
            "volume": np.array([10, 20, 30, 40], dtype=np.int64),
            "direction": ["buy", "sell", "buy", "sell"],
        }
    )


# ── A run's frames carry their units ─────────────────────────────────


def test_a_run_puts_its_tick_and_lot_size_on_every_frame(sample) -> None:
    for name in FRAMES:
        frame = getattr(sample, name)
        assert frame.attrs["tick_size"] == 0.01, name
        assert frame.attrs["lot_size"] == 1e-8, name


def test_a_price_level_run_puts_them_on_every_frame_too(tmp_path) -> None:
    DepthCsvWriter().write(
        {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, tmp_path
    )
    result = Pipeline.from_source("depth_csv").run(tmp_path)

    for name in FRAMES:
        frame = getattr(result, name)
        assert frame.attrs["tick_size"] == result.config.tick_size, name
        assert frame.attrs["lot_size"] == result.config.lot_size, name


# ── VPIN: one bucket size, three ways to ask ─────────────────────────


def test_the_three_vpin_calls_use_the_same_half_btc_buckets(sample, shown) -> None:
    from_metric = sample.metric("vpin", bucket_volume=0.5)
    from_function = compute_vpin(sample.trades, bucket_volume=0.5)
    # What result.plot("vpin", bucket_volume=0.5) draws.
    from_plot = _metric_payload(get_metric("vpin"), sample, bucket_volume=0.5)[
        "vpin_df"
    ]

    # Complete buckets: the BTC traded, in whole half-BTC buckets.
    expected = int(shown.trades["volume"].sum() // 0.5)
    for frame in (from_metric, from_function, from_plot):
        assert len(frame) == expected
        assert frame.attrs["bucket_volume"] == 0.5
    pd.testing.assert_frame_equal(from_metric, from_function)
    pd.testing.assert_frame_equal(from_metric, from_plot, check_exact=False)


def test_result_plot_vpin_draws_with_a_size_in_btc(sample) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = sample.plot("vpin", bucket_volume=0.5)
    plt.close("all")
    assert figure is not None


def test_the_readme_vpin_example_runs(sample) -> None:
    # As the README shows it: 0.25 BTC buckets, about sixty on the sample.
    vpin = compute_vpin(sample.trades, bucket_volume=0.25)

    assert vpin.attrs["bucket_volume"] == 0.25
    assert len(vpin) >= 50


def test_the_default_bucket_is_reported_in_btc(sample, shown) -> None:
    picked = vpin_bucket_volume(sample.trades)

    assert picked == pytest.approx(vpin_bucket_volume(shown.trades))
    vpin = compute_vpin(sample.trades)
    assert vpin.attrs["bucket_volume"] == pytest.approx(picked)
    # The default round-trips: passing it back cuts the same buckets.
    again = compute_vpin(sample.trades, bucket_volume=vpin.attrs["bucket_volume"])
    assert len(again) == len(vpin)


def test_bulk_volume_classification_buckets_are_in_btc(sample, shown) -> None:
    raw = bulk_volume_classification(sample.trades, 0.5)
    displayed = bulk_volume_classification(shown.trades, 0.5)

    # close and delta_price are prices, in ticks on one frame and dollars on
    # the other; the bucket cut and the volume split are the same.
    same = ["timestamp_end", "buy_fraction", "buy_volume", "sell_volume"]
    pd.testing.assert_frame_equal(raw[same], displayed[same], check_exact=False)
    np.testing.assert_allclose(raw["buy_volume"] + raw["sell_volume"], 0.5)


def test_a_bvc_sigma_is_in_the_quote_currency(sample, shown) -> None:
    raw = bulk_volume_classification(sample.trades, 0.5, sigma=5.0)
    displayed = bulk_volume_classification(shown.trades, 0.5, sigma=5.0)

    np.testing.assert_allclose(raw["buy_fraction"], displayed["buy_fraction"])


# ── Bars: size, turnover and signed-size thresholds ──────────────────


@pytest.mark.parametrize(
    ("rule", "threshold"),
    [("volume", 0.5), ("dollar", 50_000.0), ("imbalance", 0.5)],
)
def test_a_bar_threshold_cuts_the_same_bars_on_either_frame(
    sample, shown, rule: str, threshold: float
) -> None:
    raw = bars(sample.trades, rule, threshold)
    displayed = bars(shown.trades, rule, threshold)

    assert len(raw) > 1
    np.testing.assert_array_equal(raw["n_trades"], displayed["n_trades"])
    assert raw.attrs["bar_threshold"] == threshold


@pytest.mark.parametrize("rule", ["volume", "dollar", "imbalance"])
def test_a_default_bar_threshold_is_reported_in_instrument_units(
    sample, shown, rule: str
) -> None:
    raw = bars(sample.trades, rule)
    displayed = bars(shown.trades, rule)

    assert raw.attrs["bar_threshold"] == pytest.approx(displayed.attrs["bar_threshold"])


def test_features_pass_the_threshold_through_in_btc(sample, shown) -> None:
    raw = features(sample.trades, None, "volume", 0.5, include=["returns"])

    assert len(raw) == len(bars(shown.trades, "volume", 0.5))
    assert raw.attrs["bar_threshold"] == 0.5


# ── Price settings ───────────────────────────────────────────────────


def test_a_stoikov_correction_is_in_the_quote_currency(sample) -> None:
    summary = sample.depth_summary
    plain = micro_price(summary)
    # One cent on a 0.01 tick is one tick.
    corrected = micro_price(summary, stoikov_adjustment=0.01)

    np.testing.assert_allclose((corrected - plain).dropna(), 1.0)


# ── Integer sizes with no lot size ───────────────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        lambda t: compute_vpin(t, bucket_volume=0.5),
        lambda t: compute_vpin(t),
        lambda t: vpin_bucket_volume(t),
        lambda t: bulk_volume_classification(t, 0.5),
        lambda t: bars(t, "volume", 0.5),
        lambda t: bars(t, "dollar", 50.0),
    ],
    ids=["vpin", "vpin-default", "bucket-volume", "bvc", "volume-bars", "dollar-bars"],
)
def test_integer_sizes_with_no_lot_size_are_refused(call) -> None:
    with pytest.raises(ConfigError, match="lot_size.*display_result"):
        call(_lots_without_units())


def test_a_lot_size_on_attrs_is_enough() -> None:
    trades = _lots_without_units()
    trades.attrs["lot_size"] = 0.1  # sizes of 1, 2, 3 and 4 units

    out = bars(trades, "volume", 3.0)

    assert out.attrs["bar_threshold"] == 3.0
    assert out["n_trades"].tolist() == [2, 1, 1]


def test_integer_prices_with_no_tick_size_are_refused() -> None:
    summary = pd.DataFrame(
        {
            "best_bid_price": np.array([100], dtype=np.int64),
            "best_bid_vol": np.array([1], dtype=np.int64),
            "best_ask_price": np.array([101], dtype=np.int64),
            "best_ask_vol": np.array([1], dtype=np.int64),
        }
    )
    with pytest.raises(ConfigError, match="tick_size.*display_result"):
        micro_price(summary, stoikov_adjustment=0.01)
    # With no setting in price units there is nothing to convert.
    assert micro_price(summary).iloc[0] == 100.5


def test_float_sizes_are_used_as_given() -> None:
    trades = _lots_without_units().astype({"volume": "float64", "price": "float64"})

    out = bars(trades, "volume", 30.0)

    assert out.attrs["bar_threshold"] == 30.0
    assert out["n_trades"].tolist() == [2, 1, 1]
