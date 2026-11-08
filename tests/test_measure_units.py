"""Every measure reports in the instrument's own units, on every path.

``result.metric(name)``, ``result.plot(name)`` and the bare function the
metric wraps must give the same numbers on the same run: prices in the quote
currency and sizes in BTC.  The bare function must also give them on a frame
converted for display, since a measure reads ``tick_size`` and ``lot_size``
from the frame it is handed.  All of it runs on the bundled BTC sample.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ob_analytics import metrics
from ob_analytics.cost import amihud, roll_spread, transaction_costs
from ob_analytics.depth import depth_signals
from ob_analytics.flow_toxicity import (
    compute_kyle_lambda,
    compute_vpin,
    ofi_by_horizon,
    order_flow_imbalance,
)
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.visualization import gallery
from ob_analytics.visualization.gallery import display_result


@pytest.fixture(scope="module")
def result(sample_csv_path) -> PipelineResult:
    return Pipeline().run(sample_csv_path)


@pytest.fixture(scope="module")
def shown(result) -> PipelineResult:
    return display_result(result)


@pytest.fixture(autouse=True)
def _quiet():
    # VPIN warns when the sample fills fewer than n_buckets buckets.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield
    plt.close("all")


def _kyle(run: PipelineResult) -> pd.DataFrame:
    fit = compute_kyle_lambda(run.trades, quotes=run.depth_summary)
    return fit.regression_df.assign(lambda_=fit.lambda_)


def _ofi(run: PipelineResult) -> pd.DataFrame:
    return order_flow_imbalance(run.trades, quotes=run.depth_summary)


def _l1(run: PipelineResult) -> pd.DataFrame:
    # The quote card has no function of its own: the metric is the measure.
    return metrics.get_metric("l1_ticker").compute(run)


#: Built-in metric name → the bare call it wraps, with the metric's defaults.
#: The run's own depth_summary is the quotes, as the metrics pass it.
BARE: dict[str, Callable[[PipelineResult], pd.DataFrame]] = {
    "l1_ticker": _l1,
    "book_signals": lambda run: depth_signals(run.depth_summary),
    "vpin": lambda run: compute_vpin(run.trades, quotes=run.depth_summary),
    "kyle_lambda": _kyle,
    "order_flow_imbalance": _ofi,
    "ofi_horizon": lambda run: ofi_by_horizon(run.trades, quotes=run.depth_summary),
    "transaction_costs": lambda run: transaction_costs(run.trades, run.depth_summary),
    "amihud": lambda run: amihud(run.trades),
    "roll_spread": lambda run: roll_spread(run.trades),
}


def _metric_table(result: PipelineResult, name: str) -> pd.DataFrame:
    frame = result.metric(name)
    if name == "kyle_lambda":
        frame = frame.assign(lambda_=frame.attrs["lambda_"])
    return frame


def _common(frame: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """*frame* cut to the columns of *other*, with plain float numbers."""
    out = frame[list(other.columns)].reset_index(drop=True)
    numeric = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    return out.astype(dict.fromkeys(numeric, float))


def _assert_same_table(got: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        _common(got, expected), _common(expected, expected), check_dtype=False
    )


def _assert_same_payload(got: dict[str, Any], expected: dict[str, Any]) -> None:
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(got[key], value)
        elif isinstance(value, pd.Series):
            pd.testing.assert_series_equal(got[key], value)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got[key], value)
        elif isinstance(value, float) and np.isnan(value):
            assert np.isnan(got[key]), key
        else:
            assert got[key] == value, key


def test_every_built_in_metric_has_a_case() -> None:
    # A new built-in metric must be checked here too.
    assert set(BARE) <= set(metrics.list_metrics())
    shipped = {
        name
        for name in metrics.list_metrics()
        if type(metrics.get_metric(name)).__module__ == "ob_analytics.metrics"
    }
    assert shipped == set(BARE)


#: depth_signals still reports in the units of the frame it is handed, so its
#: table in the instrument's units is the one it gives on a display frame.
FRAME_UNITS = frozenset({"book_signals"})


@pytest.mark.parametrize("name", sorted(BARE))
def test_metric_is_the_bare_function(result, shown, name) -> None:
    run = shown if name in FRAME_UNITS else result
    _assert_same_table(_metric_table(result, name), BARE[name](run))


@pytest.mark.parametrize("name", sorted(BARE.keys() - FRAME_UNITS))
def test_bare_function_reads_a_display_frame_the_same(result, shown, name) -> None:
    _assert_same_table(BARE[name](result), BARE[name](shown))


@pytest.mark.parametrize("name", sorted(BARE))
def test_plot_draws_the_metric_table(result, name, monkeypatch) -> None:
    drawn: dict[str, Any] = {}

    def capture(concept, level, *, backend, **data):
        drawn.update(data)

    def no_display(_result):
        raise AssertionError("result.plot converted the result before a measure")

    monkeypatch.setattr(gallery, "plot", capture)
    monkeypatch.setattr(gallery, "display_result", no_display)
    result.plot(name)

    metric = metrics.get_metric(name)
    _assert_same_payload(drawn, metric.prepare(result.metric(name)))


def test_gallery_card_draws_the_metric_table(result) -> None:
    # The gallery's own cards measure the run's frames too.
    model = gallery.build_gallery_model(result)
    names = [spec.name for spec in model.analytics]
    assert len(names) == len(set(names))
    card = next(spec for spec in model.analytics if spec.name == "amihud")
    expected = metrics.get_metric("amihud").prepare(result.metric("amihud"))
    _assert_same_payload(card.prepare(**card.prep_kwargs), expected)


def test_measures_are_in_the_quote_currency_and_btc(result, shown) -> None:
    # A spot check of the units themselves, not only that the paths agree.
    trade_price = shown.trades["price"]
    assert trade_price.between(100, 1_000_000).all()  # dollars, not ticks

    costs = result.metric("transaction_costs")
    assert costs["price"].to_numpy() == pytest.approx(trade_price.to_numpy())
    assert costs["volume"].sum() == pytest.approx(shown.trades["volume"].sum())

    illiq = result.metric("amihud")
    turnover = (shown.trades["price"] * shown.trades["volume"]).sum()
    assert illiq["turnover"].iloc[0] == pytest.approx(turnover)

    ofi = result.metric("order_flow_imbalance")
    assert ofi["buy_volume"].sum() == pytest.approx(
        shown.trades.loc[shown.trades["direction"] == "buy", "volume"].sum()
    )
