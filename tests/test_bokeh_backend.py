"""Tests for ob_analytics._bokeh – Bokeh interactive rendering backend.

Skipped entirely if bokeh is not installed.  Covers the core concepts this
backend registers: trade_tape, depth_heatmap, book_snapshot, depth_chart.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

bpl = pytest.importorskip("bokeh.plotting", reason="bokeh not installed")

from ob_analytics.visualization._bokeh import (
    _bokeh_color_field,
    bokeh_book_snapshot_aggregate,
    bokeh_book_snapshot_per_order,
    bokeh_depth_chart_aggregate,
    bokeh_depth_chart_per_order,
    bokeh_price_levels,
    bokeh_trade_tape_per_order,
    bokeh_trades,
)
from ob_analytics.visualization._data import (
    prepare_book_snapshot_data,
    prepare_price_levels_data,
    prepare_trade_tape_l3_data,
    prepare_trades_data,
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
def sample_depth() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    n = 20
    return pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)],
            "price": np.linspace(236.0, 237.0, n),
            "volume": np.random.default_rng(42).uniform(100, 1000, n),
            "direction": "bid",
        }
    )


@pytest.fixture
def sample_order_book() -> dict:
    return {
        "timestamp": 1430445600,
        "bids": np.array([[236.50, 100, 100], [236.00, 200, 300]]),
        "asks": np.array([[237.00, 150, 150], [237.50, 250, 400]]),
    }


# ---------------------------------------------------------------------------
# Tests – each bokeh_*() function returns a Bokeh figure
# ---------------------------------------------------------------------------


class TestBokehTrades:
    def test_returns_bokeh_figure(self, sample_trades: pd.DataFrame) -> None:
        data = prepare_trades_data(sample_trades)
        fig = bokeh_trades(data)
        assert isinstance(fig, bpl.figure)
        assert len(fig.renderers) >= 1

    def test_l2_ylabel_is_price(self, sample_trades: pd.DataFrame) -> None:
        data = prepare_trades_data(sample_trades)
        fig = bokeh_trades(data)
        assert fig.yaxis[0].axis_label == "Price"

    def test_has_lollipop_legend_entries(self, sample_trades: pd.DataFrame) -> None:
        data = prepare_trades_data(sample_trades)
        fig = bokeh_trades(data)
        labels = {item.label["value"] for item in fig.legend[0].items}
        assert "buy (lifts ask)" in labels
        assert "sell (hits bid)" in labels


class TestBiasedColorField:
    """col_bias maps volumes to [0, 1] color positions, mirroring matplotlib."""

    def test_linear_is_proportional(self) -> None:
        t, tickvals, ticktext = _bokeh_color_field(
            np.array([1.0, 50.5, 100.0]), col_bias=1.0
        )
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(1.0)
        assert t[1] == pytest.approx(0.5, abs=1e-6)
        assert len(tickvals) == len(ticktext)

    def test_fractional_bias_brightens_low_volume(self) -> None:
        vol = np.array([1.0, 10.0, 100.0])
        linear, _, _ = _bokeh_color_field(vol, col_bias=1.0)
        biased, _, _ = _bokeh_color_field(vol, col_bias=0.1)
        assert biased[1] > linear[1]

    def test_outputs_within_unit_range(self) -> None:
        vol = np.array([np.nan, 1.0, 10.0, 100.0])
        for bias in (1.0, 0.5, 0.1, 0.0, -1.0):
            t, _, _ = _bokeh_color_field(vol, col_bias=bias)
            assert np.all((t >= 0.0) & (t <= 1.0))

    def test_log_is_monotonic(self) -> None:
        t, _, _ = _bokeh_color_field(np.array([1.0, 10.0, 100.0]), col_bias=0.0)
        assert t[0] < t[1] < t[2]

    def test_empty_finite_is_safe(self) -> None:
        t, tickvals, ticktext = _bokeh_color_field(
            np.array([np.nan, np.nan]), col_bias=0.1
        )
        assert np.all(t == 0.0)
        assert tickvals == []
        assert ticktext == []


class TestBokehPriceLevels:
    def test_returns_bokeh_figure(self, sample_depth: pd.DataFrame) -> None:
        data = prepare_price_levels_data(sample_depth)
        fig = bokeh_price_levels(data)
        assert isinstance(fig, bpl.figure)

    def test_spread_overlays_present(self, sample_depth: pd.DataFrame) -> None:
        spread = pd.DataFrame(
            {
                "timestamp": sample_depth["timestamp"],
                "best_bid_price": sample_depth["price"] - 0.05,
                "best_ask_price": sample_depth["price"] + 0.05,
            }
        )
        fig = bokeh_price_levels(
            prepare_price_levels_data(sample_depth, spread=spread, show_mp=True)
        )
        assert isinstance(fig, bpl.figure)


class TestBokehTradeTapeL3:
    def test_returns_bokeh_figure(self, sample_executed_orders) -> None:
        events, trades = sample_executed_orders
        data = prepare_trade_tape_l3_data(events, trades, price_from=0.0, price_to=1e9)
        fig = bokeh_trade_tape_per_order(data)
        assert isinstance(fig, bpl.figure)
        assert len(fig.renderers) >= 1


class TestBokehBookSnapshot:
    def test_aggregate_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=False)
        fig = bokeh_book_snapshot_aggregate(data)
        assert isinstance(fig, bpl.figure)
        assert len(fig.renderers) >= 2  # bid + ask bars

    def test_per_order_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=True)
        fig = bokeh_book_snapshot_per_order(data)
        assert isinstance(fig, bpl.figure)
        assert len(fig.renderers) >= 2

    @pytest.mark.parametrize(
        ("renderer", "wrong_per_order"),
        [
            (bokeh_book_snapshot_aggregate, True),
            (bokeh_book_snapshot_per_order, False),
        ],
    )
    def test_payload_level_mismatch_raises(
        self, sample_order_book: dict, renderer, wrong_per_order: bool
    ) -> None:
        from ob_analytics.exceptions import ConfigError

        data = prepare_book_snapshot_data(sample_order_book, per_order=wrong_per_order)
        with pytest.raises(ConfigError, match="prepare.book_snapshot"):
            renderer(data)


class TestBokehDepthChart:
    def test_aggregate_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=False)
        fig = bokeh_depth_chart_aggregate(data)
        assert isinstance(fig, bpl.figure)
        assert len(fig.renderers) >= 2  # bid + ask curves (step + varea each)

    def test_per_order_returns_figure(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, per_order=True)
        fig = bokeh_depth_chart_per_order(data)
        assert isinstance(fig, bpl.figure)

    @pytest.mark.parametrize(
        ("renderer", "wrong_per_order"),
        [
            (bokeh_depth_chart_aggregate, True),
            (bokeh_depth_chart_per_order, False),
        ],
    )
    def test_payload_level_mismatch_raises(
        self, sample_order_book: dict, renderer, wrong_per_order: bool
    ) -> None:
        from ob_analytics.exceptions import ConfigError

        data = prepare_book_snapshot_data(sample_order_book, per_order=wrong_per_order)
        with pytest.raises(ConfigError, match="prepare.book_snapshot"):
            renderer(data)


# ---------------------------------------------------------------------------
# Backend dispatch via public API
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_bokeh_backend_returns_bokeh_figure(
        self, sample_trades: pd.DataFrame
    ) -> None:
        from ob_analytics.visualization import Level, _data, plot

        fig = plot(
            "trade_tape",
            Level.L2,
            backend="bokeh",
            **_data.prepare_trades_data(sample_trades),
        )
        assert isinstance(fig, bpl.figure)

    def test_missing_level_for_comparable_concept_raises(
        self, sample_order_book: dict
    ) -> None:
        from ob_analytics.visualization import _data, plot

        data = _data.prepare_book_snapshot_data(sample_order_book, per_order=False)
        with pytest.raises(ValueError, match="comparable"):
            plot("book_snapshot", backend="bokeh", **data)
