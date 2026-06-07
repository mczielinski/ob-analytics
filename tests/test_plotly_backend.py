"""Tests for ob_analytics._plotly – Plotly interactive rendering backend.

Skipped entirely if plotly is not installed.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects", reason="plotly not installed")

from ob_analytics.visualization._data import (  # noqa: E402
    prepare_book_snapshot_data,
    prepare_cancellations_l3_data,
    prepare_event_map_data,
    prepare_events_histogram_data,
    prepare_kyle_lambda_data,
    prepare_liquidity_at_touch_data,
    prepare_ofi_data,
    prepare_order_activity_l3_data,
    prepare_order_outcome_l3_data,
    prepare_price_levels_data,
    prepare_time_series_data,
    prepare_trade_tape_l3_data,
    prepare_trades_data,
    prepare_volume_map_data,
    prepare_volume_percentiles_data,
    prepare_vpin_data,
)
from ob_analytics.visualization._plotly import (  # noqa: E402
    plotly_book_snapshot_aggregate,
    plotly_book_snapshot_per_order,
    plotly_cancellations_per_order,
    plotly_depth_chart_aggregate,
    plotly_depth_chart_per_order,
    plotly_event_map,
    plotly_events_histogram,
    plotly_kyle_lambda,
    plotly_liquidity_at_touch,
    plotly_order_activity_per_order,
    plotly_order_flow_imbalance,
    plotly_order_outcome_per_order,
    plotly_price_levels,
    plotly_time_series,
    plotly_trade_tape_per_order,
    plotly_trades,
    plotly_volume_map,
    plotly_volume_percentiles,
    plotly_vpin,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    return pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(5)],
            "price": [236.50, 236.55, 236.45, 236.60, 236.50],
            "volume": [100, 200, 150, 300, 250],
            "direction": pd.Categorical(
                ["buy", "sell", "buy", "sell", "buy"],
                categories=["buy", "sell"],
            ),
        }
    )


@pytest.fixture
def sample_events() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    n = 20
    return pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)],
            "price": np.linspace(236.0, 237.0, n),
            "volume": np.random.default_rng(42).uniform(100, 1000, n),
            "direction": pd.Categorical(
                ["bid", "ask"] * (n // 2), categories=["bid", "ask"]
            ),
            "action": pd.Categorical(
                ["created", "deleted"] * (n // 2),
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "type": ["flashed-limit"] * 10 + ["resting-limit"] * 10,
        }
    )


@pytest.fixture
def sample_depth_summary() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    n = 30
    rng = np.random.default_rng(42)
    data: dict = {"timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)]}
    for side in ("bid", "ask"):
        for bps in range(25, 501, 25):
            data[f"{side}_vol{bps}bps"] = rng.uniform(10, 500, n)
    data["best_bid_price"] = np.full(n, 236.50)
    data["best_ask_price"] = np.full(n, 237.00)
    data["best_bid_vol"] = rng.uniform(100, 1000, n)
    data["best_ask_vol"] = rng.uniform(100, 1000, n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_order_book() -> dict:
    return {
        "timestamp": 1430445600,
        "bids": np.array([[236.50, 100, 100], [236.00, 200, 300]]),
        "asks": np.array([[237.00, 150, 150], [237.50, 250, 400]]),
    }


# ---------------------------------------------------------------------------
# Tests – each plotly_*() function returns a plotly Figure
# ---------------------------------------------------------------------------


class TestPlotlyTimeSeries:
    def test_returns_plotly_figure(self) -> None:
        ts = pd.Series(pd.date_range("2015-01-01", periods=5, freq="s"))
        vals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        data = prepare_time_series_data(ts, vals)
        fig = plotly_time_series(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotlyTrades:
    def test_returns_plotly_figure(self, sample_trades: pd.DataFrame) -> None:
        data = prepare_trades_data(sample_trades)
        fig = plotly_trades(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_has_price_trace(self, sample_trades: pd.DataFrame) -> None:
        data = prepare_trades_data(sample_trades)
        fig = plotly_trades(data)
        assert fig.data[0].name == "Price"


class TestPlotlyPriceLevels:
    def test_returns_plotly_figure(self, sample_events: pd.DataFrame) -> None:
        # Use a minimal depth-like DataFrame
        depth = sample_events[["timestamp", "price", "volume"]].copy()
        depth["direction"] = "bid"
        data = prepare_price_levels_data(depth)
        fig = plotly_price_levels(data)
        assert isinstance(fig, go.Figure)


class TestPlotlyEventMap:
    def test_returns_plotly_figure(self, sample_events: pd.DataFrame) -> None:
        data = prepare_event_map_data(sample_events)
        fig = plotly_event_map(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPlotlyVolumeMap:
    def test_returns_plotly_figure(self, sample_events: pd.DataFrame) -> None:
        data = prepare_volume_map_data(sample_events)
        fig = plotly_volume_map(data)
        assert isinstance(fig, go.Figure)


class TestPlotlyCancellationsL3:
    def test_returns_plotly_figure(
        self, sample_cancellation_events: pd.DataFrame
    ) -> None:
        data = prepare_cancellations_l3_data(sample_cancellation_events)
        fig = plotly_cancellations_per_order(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_uses_webgl_scattergl(
        self, sample_cancellation_events: pd.DataFrame
    ) -> None:
        # The per-order point cloud must use the WebGL Scattergl path (like the
        # L2 volume map it pairs with), not the SVG Scatter path that fails to
        # scale to one marker per cancelled order.
        data = prepare_cancellations_l3_data(sample_cancellation_events)
        fig = plotly_cancellations_per_order(data)
        assert all(trace.type == "scattergl" for trace in fig.data)


class TestPlotlyOrderActivityL3:
    def test_returns_plotly_figure(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        data = prepare_order_activity_l3_data(sample_order_lifecycle_events)
        fig = plotly_order_activity_per_order(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_uses_webgl_scattergl(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        # The per-order lifecycle cloud must use the WebGL Scattergl path (like
        # the L2 event map it pairs with), not the SVG Scatter path that fails to
        # scale to one segment per limit order.
        data = prepare_order_activity_l3_data(sample_order_lifecycle_events)
        fig = plotly_order_activity_per_order(data)
        assert all(trace.type == "scattergl" for trace in fig.data)


class TestPlotlyLiquidityAtTouch:
    def test_returns_plotly_figure(self, sample_depth_summary: pd.DataFrame) -> None:
        data = prepare_liquidity_at_touch_data(sample_depth_summary)
        fig = plotly_liquidity_at_touch(data)
        assert isinstance(fig, go.Figure)
        # Two aggregate time series: best bid size + best ask size.
        assert len(fig.data) == 2

    def test_uses_svg_scatter(self, sample_depth_summary: pd.DataFrame) -> None:
        # Two small aggregate series -- the SVG Scatter path is appropriate here,
        # unlike the per-order WebGL clouds.
        data = prepare_liquidity_at_touch_data(sample_depth_summary)
        fig = plotly_liquidity_at_touch(data)
        assert all(trace.type == "scatter" for trace in fig.data)


class TestPlotlyOrderOutcomeL3:
    def test_returns_plotly_figure(self, sample_executed_orders) -> None:
        events, trades = sample_executed_orders
        data = prepare_order_outcome_l3_data(events, trades, bps_quantiles=(0.0, 1.0))
        fig = plotly_order_outcome_per_order(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_uses_webgl_scattergl(self, sample_executed_orders) -> None:
        # One marker per order -> the WebGL Scattergl path, like the other
        # per-order faces.
        events, trades = sample_executed_orders
        data = prepare_order_outcome_l3_data(events, trades, bps_quantiles=(0.0, 1.0))
        fig = plotly_order_outcome_per_order(data)
        assert all(trace.type == "scattergl" for trace in fig.data)


class TestPlotlyTradeTapeL3:
    def test_returns_plotly_figure(self, sample_executed_orders) -> None:
        events, trades = sample_executed_orders
        data = prepare_trade_tape_l3_data(events, trades, price_from=0.0, price_to=1e9)
        fig = plotly_trade_tape_per_order(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_uses_webgl_scattergl(self, sample_executed_orders) -> None:
        # Both the maker-bar segment cloud and the execution-marker cloud use the
        # WebGL Scattergl path, like the L3 order-activity face it pairs with.
        events, trades = sample_executed_orders
        data = prepare_trade_tape_l3_data(events, trades, price_from=0.0, price_to=1e9)
        fig = plotly_trade_tape_per_order(data)
        assert all(trace.type == "scattergl" for trace in fig.data)


class TestPlotlyBookSnapshot:
    def test_aggregate_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=False)
        fig = plotly_book_snapshot_aggregate(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # bid + ask bars

    def test_per_order_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=True)
        fig = plotly_book_snapshot_per_order(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestPlotlyDepthChart:
    def test_aggregate_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=False)
        fig = plotly_depth_chart_aggregate(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # bid + ask step curves

    def test_per_order_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=True)
        fig = plotly_depth_chart_per_order(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestPlotlyVolumePercentiles:
    def test_returns_plotly_figure(self, sample_depth_summary: pd.DataFrame) -> None:
        data = prepare_volume_percentiles_data(sample_depth_summary)
        fig = plotly_volume_percentiles(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 20  # 20 asks + 20 bids


class TestPlotlyEventsHistogram:
    def test_returns_plotly_figure(self, sample_events: pd.DataFrame) -> None:
        data = prepare_events_histogram_data(sample_events, val="price")
        fig = plotly_events_histogram(data)
        assert isinstance(fig, go.Figure)


class TestPlotlyVpin:
    def test_returns_plotly_figure(self) -> None:
        ts = pd.date_range("2015-01-01", periods=5, freq="min")
        vpin_df = pd.DataFrame(
            {
                "timestamp_end": ts,
                "vpin": [0.3, 0.5, 0.7, 0.4, 0.6],
                "vpin_avg": [0.3, 0.4, 0.5, 0.45, 0.5],
            }
        )
        data = prepare_vpin_data(vpin_df)
        fig = plotly_vpin(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # bar + rolling avg line


class TestPlotlyOfi:
    def test_returns_plotly_figure(self) -> None:
        ts = pd.date_range("2015-01-01", periods=5, freq="min")
        ofi_df = pd.DataFrame({"timestamp": ts, "ofi": [0.3, -0.5, 0.7, -0.4, 0.6]})
        data = prepare_ofi_data(ofi_df)
        fig = plotly_order_flow_imbalance(data)
        assert isinstance(fig, go.Figure)


class TestPlotlyKyleLambda:
    def test_returns_plotly_figure(self) -> None:
        class FakeResult:
            regression_df = pd.DataFrame(
                {
                    "signed_volume": [1.0, -2.0, 3.0],
                    "delta_price": [0.01, -0.02, 0.03],
                }
            )
            lambda_ = 0.01
            r_squared = 0.5
            t_stat = 2.1

        data = prepare_kyle_lambda_data(FakeResult())
        fig = plotly_kyle_lambda(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # scatter + regression line


# ---------------------------------------------------------------------------
# Backend dispatch via public API
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_plotly_backend_returns_plotly_figure(
        self, sample_trades: pd.DataFrame
    ) -> None:
        from ob_analytics.visualization import Level, _data, plot

        fig = plot(
            "trade_tape",
            Level.L2,
            backend="plotly",
            **_data.prepare_trades_data(sample_trades),
        )
        assert isinstance(fig, go.Figure)

    def test_invalid_backend_raises(self, sample_trades: pd.DataFrame) -> None:
        from ob_analytics.visualization import _data, plot

        with pytest.raises(ValueError, match="Unknown backend"):
            plot(
                "trade_tape",
                backend="nonexistent",
                **_data.prepare_trades_data(sample_trades),
            )
