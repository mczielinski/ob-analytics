"""The built-in metrics: the flow-toxicity faces and the L1 quote (issue #118).

Each is a registered metric, so it is listed by ``available_concepts`` and
drawn by ``result.plot(name)`` with the defaults of the function it wraps.
Keyword arguments reach ``compute`` or ``prepare``, whichever names them.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ob_analytics import metrics
from ob_analytics.bitstamp import BitstampSource
from ob_analytics.flow_toxicity import (
    compute_kyle_lambda,
    compute_vpin,
    ofi_by_horizon,
    order_flow_imbalance,
)
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import Level
from ob_analytics.visualization import available_concepts, plot
from ob_analytics.visualization._data import (
    l1_card_texts,
    prepare_l1_ticker_data,
    prepare_ofi_horizon_data,
)
from ob_analytics.visualization.gallery import (
    _metric_payload,
    build_gallery_model,
    display_result,
)

BUILT_IN = ("kyle_lambda", "l1_ticker", "ofi_horizon", "order_flow_imbalance", "vpin")


@pytest.fixture(scope="module")
def result(tiny_bitstamp_orders_csv) -> PipelineResult:
    return Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))


@pytest.fixture(autouse=True)
def _quiet():
    # VPIN warns when the tiny run fills fewer than n_buckets buckets.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Discovery and the one-liner
# ---------------------------------------------------------------------------


def test_built_in_metrics_are_registered() -> None:
    assert set(BUILT_IN) <= set(metrics.list_metrics())


def test_built_in_metrics_are_listed_as_level_less_concepts(result) -> None:
    concepts = available_concepts(result)
    for name in BUILT_IN:
        assert concepts[name] == [], name


def test_built_in_metrics_apply_to_an_l2_run(result) -> None:
    import dataclasses

    l2 = dataclasses.replace(result, level=Level.L2)
    assert set(BUILT_IN) <= set(l2.metrics())


@pytest.mark.parametrize("name", BUILT_IN)
def test_result_plots_each_built_in_metric(result, name) -> None:
    assert isinstance(result.plot(name), Figure)


@pytest.mark.parametrize("name", BUILT_IN)
def test_result_plots_each_built_in_metric_on_plotly(result, name) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    assert isinstance(result.plot(name, backend="plotly"), go.Figure)


def test_building_the_model_computes_no_metric(result, monkeypatch) -> None:
    calls = []
    vpin = metrics.get_metric("vpin")
    monkeypatch.setattr(vpin, "compute", lambda *a, **k: calls.append(1))

    build_gallery_model(result)
    available_concepts(result)

    assert calls == []


# ---------------------------------------------------------------------------
# Settings: compute's keywords to compute, the rest to prepare
# ---------------------------------------------------------------------------


def test_keywords_go_to_the_method_that_names_them(result) -> None:
    # Sized from the data: raw volumes are integer lots, not base units.
    bucket = float(result.trades["volume"].sum()) / 4
    payload = _metric_payload(
        metrics.get_metric("vpin"), result, bucket_volume=bucket, threshold=0.25
    )

    assert payload["vpin_df"].attrs["bucket_volume"] == bucket
    assert payload["threshold"] == 0.25


def test_result_metric_passes_settings_to_compute(result) -> None:
    bucket = float(result.trades["volume"].sum()) / 4
    frame = result.metric("vpin", bucket_volume=bucket, n_buckets=3)

    expected = compute_vpin(result.trades, bucket_volume=bucket, n_buckets=3)
    pd.testing.assert_frame_equal(frame, expected)


def test_an_unknown_keyword_is_an_error(result) -> None:
    with pytest.raises(TypeError):
        result.plot("vpin", not_a_setting=1)


# ---------------------------------------------------------------------------
# Each metric's table is the function it wraps
# ---------------------------------------------------------------------------


def test_kyle_metric_keeps_the_fit_in_attrs(result) -> None:
    frame = result.metric("kyle_lambda", window="5s")
    kyle = compute_kyle_lambda(result.trades, "5s")

    pd.testing.assert_frame_equal(frame, kyle.regression_df, check_like=True)
    assert frame.attrs["lambda_"] == pytest.approx(kyle.lambda_, nan_ok=True)
    assert frame.attrs["n_windows"] == kyle.n_windows
    assert frame.attrs["diagnostics"] == kyle.diagnostics


def test_ofi_metric_adds_the_window_close_price(result) -> None:
    frame = result.metric("order_flow_imbalance", window="10s")
    ofi = order_flow_imbalance(result.trades, window="10s")

    pd.testing.assert_frame_equal(frame[list(ofi.columns)], ofi)
    closes = result.trades.groupby(pd.Grouper(key="timestamp", freq="10s"))[
        "price"
    ].last()
    assert frame["last_price"].tolist() == closes.loc[frame["timestamp"]].tolist()


def test_ofi_metric_draws_the_close_as_its_price_line(result) -> None:
    payload = _metric_payload(
        metrics.get_metric("order_flow_imbalance"), result, window="10s"
    )
    assert list(payload["trades"].columns) == ["timestamp", "price"]


# ---------------------------------------------------------------------------
# ofi_by_horizon
# ---------------------------------------------------------------------------


def _trades(rows) -> pd.DataFrame:
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [t0 + pd.Timedelta(seconds=s) for s, _, _ in rows],
            "price": 100,
            "volume": [v for _, v, _ in rows],
            "direction": [d for _, _, d in rows],
        }
    )


def test_ofi_by_horizon_sums_a_trailing_window_per_horizon() -> None:
    trades = _trades([(0, 3.0, "buy"), (5, 1.0, "sell"), (10, 1.0, "sell")])

    grid = ofi_by_horizon(trades, horizons=("5s", "15s"), grid="5s")

    assert list(grid.columns) == ["timestamp", "5s", "15s"]
    assert grid["timestamp"].dt.tz is not None
    # 5s: one bucket each.  15s: everything so far.
    assert grid["5s"].tolist() == [1.0, -1.0, -1.0]
    assert grid["15s"].tolist() == pytest.approx([1.0, 2 / 4, 1 / 5])


def test_ofi_horizon_face_draws_the_metric_grid(result) -> None:
    grid = result.metric("ofi_horizon")
    payload = prepare_ofi_horizon_data(result.trades)

    assert payload["horizons"] == list(grid.columns[1:])
    np.testing.assert_allclose(payload["ofi"], grid.iloc[:, 1:].to_numpy().T)


# ---------------------------------------------------------------------------
# The L1 quote
# ---------------------------------------------------------------------------


def test_l1_table_carries_the_quote_and_the_last_trade(result) -> None:
    l1 = result.metric("l1_ticker")

    assert list(l1.columns) == [
        "timestamp",
        "best_bid_price",
        "best_bid_vol",
        "best_ask_price",
        "best_ask_vol",
        "last_price",
        "last_volume",
    ]
    assert l1["timestamp"].is_monotonic_increasing
    # The last row is the last trade and the final touch.
    assert l1["last_price"].iloc[-1] == result.trades["price"].iloc[-1]
    summary = result.depth_summary.iloc[-1]
    for side in ("bid", "ask"):
        price = summary[f"best_{side}_price"]
        got = l1[f"best_{side}_price"].iloc[-1]
        assert np.isnan(got) if price == 0 else got == price


def test_l1_table_reads_an_empty_side_as_missing(result) -> None:
    import dataclasses

    # Empty the bid side from the last quote on: the rows after it, trade
    # rows included, carry it forward as empty, not as the bid before it.
    summary = result.depth_summary.copy()
    summary.loc[summary.index[-1], ["best_bid_price", "best_bid_vol"]] = 0
    l1 = metrics.get_metric("l1_ticker").compute(
        dataclasses.replace(result, depth_summary=summary)
    )

    after = l1[l1["timestamp"] >= summary["timestamp"].iloc[-1]]
    assert after["best_bid_price"].isna().all()
    assert after["best_bid_vol"].isna().all()
    assert l1["best_bid_price"].notna().any()


def _l1(rows) -> pd.DataFrame:
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    columns = ["best_bid_price", "best_ask_price", "last_price"]
    frame = pd.DataFrame(rows, columns=["s", *columns])
    frame.insert(0, "timestamp", [t0 + pd.Timedelta(seconds=s) for s in frame["s"]])
    frame = frame.drop(columns="s")
    frame["best_bid_vol"] = 1.0
    frame["best_ask_vol"] = 2.0
    frame["last_volume"] = 0.5
    return frame


def test_l1_card_is_the_quote_at_or_before_the_instant() -> None:
    l1 = _l1([(0, 99, 101, np.nan), (10, 100, 101, 101)])

    early = prepare_l1_ticker_data(l1, at="2026-01-01 00:00:05", symbol="TOY")
    late = prepare_l1_ticker_data(l1, at=pd.Timestamp("2026-01-01 00:00:10", tz="UTC"))

    assert (early["bid"], early["ask"], early["last"]) == (99, 101, None)
    assert early["symbol"] == "TOY"
    assert early["at"].tzinfo is not None  # a naive time is read in UTC
    assert (late["bid"], late["last"], late["last_size"]) == (100, 101, 0.5)


def test_l1_lines_read_a_naive_window_in_the_data_time_zone() -> None:
    l1 = _l1([(0, 99, 101, 100), (10, 100, 102, 101), (20, 101, 103, 102)])

    lines = prepare_l1_ticker_data(
        l1,
        start_time="2026-01-01 00:00:05",
        end_time=pd.Timestamp("2026-01-01 00:00:15"),
    )

    assert lines["best_bid_price"].tolist() == [100]


def test_ofi_close_is_the_last_trade_in_time(result) -> None:
    import dataclasses

    shuffled = dataclasses.replace(result, trades=result.trades.iloc[::-1])
    a = result.metric("order_flow_imbalance", window="10s")
    b = metrics.OrderFlowImbalanceMetric().compute(shuffled, window="10s")
    assert b["last_price"].tolist() == a["last_price"].tolist()


def test_l1_card_before_the_first_row_is_empty() -> None:
    l1 = _l1([(10, 100, 101, 101)])
    card = prepare_l1_ticker_data(l1, at="2026-01-01 00:00:00")
    assert card["bid"] is None and card["ask"] is None and card["last"] is None


def test_l1_lines_keep_only_rows_where_a_price_changes() -> None:
    l1 = _l1(
        [
            (0, 99, 101, np.nan),
            (1, 99, 101, np.nan),  # a size change only
            (2, 99, 101, 100),
            (3, 99, 102, 100),
            (4, 99, 102, 100),
        ]
    )

    lines = prepare_l1_ticker_data(l1)

    assert len(lines["timestamp"]) == 3
    assert lines["best_ask_price"].tolist() == [101, 101, 102]


def test_l1_line_axis_ignores_a_stray_far_quote() -> None:
    rows = [(s, 99, 101, 100) for s in range(200)] + [(200, 99, 1e9, 100)]
    lines = prepare_l1_ticker_data(_l1(rows))
    assert lines["y_range"][1] < 1000


def test_l1_card_draws_from_plain_numbers() -> None:
    fig = plot("l1_ticker", bid=99, ask=101, last=None, symbol="TOY")
    texts = {t.get_text() for t in fig.axes[0].texts}

    assert {"TOY", "99", "101", "—"} <= texts


def test_l1_card_shrinks_a_long_price() -> None:
    short = l1_card_texts({"bid": 99.0, "ask": 101.0, "last": 100.0})
    long = l1_card_texts({"bid": 105234.56, "ask": 105240.5, "last": 105240.5})

    def price_size(texts):
        return next(t.size for t in texts if t.text in ("99", "105,234.56"))

    assert price_size(long) < price_size(short)


def test_result_plots_the_l1_card_at_an_instant(result) -> None:
    at = display_result(result).trades["timestamp"].iloc[1]
    fig = result.plot("l1_ticker", at=at, symbol="BTC/USD")
    assert "BTC/USD" in {t.get_text() for t in fig.axes[0].texts}


def test_ofi_by_horizon_infers_a_missing_side() -> None:
    trades = _trades([(0, 1.0, "buy"), (5, 1.0, "sell")]).drop(columns="direction")
    trades["price"] = [100, 101]
    # The tick rule reads the uptick as a buy.
    assert ofi_by_horizon(trades, horizons=("5s",))["5s"].iloc[1] == 1.0
