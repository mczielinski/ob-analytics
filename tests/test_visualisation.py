"""Tests for ob_analytics.visualisation – Phase 5 visualization improvements."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ob_analytics.visualisation import (
    PlotTheme,
    _create_axes,
    get_plot_theme,
    plot_current_depth,
    plot_event_map,
    plot_events_histogram,
    plot_price_levels,
    plot_time_series,
    plot_trades,
    plot_volume_map,
    plot_volume_percentiles,
    save_figure,
    set_plot_theme,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


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


# ---------------------------------------------------------------------------
# Theme tests
# ---------------------------------------------------------------------------


class TestPlotTheme:
    def test_default_theme(self):
        theme = PlotTheme()
        assert theme.style == "darkgrid"
        assert theme.font_scale == 1.5

    def test_custom_theme(self):
        theme = PlotTheme(style="whitegrid", font_scale=1.0)
        assert theme.style == "whitegrid"
        assert theme.font_scale == 1.0

    def test_frozen(self):
        theme = PlotTheme()
        with pytest.raises(AttributeError):
            theme.style = "whitegrid"  # type: ignore[misc]

    def test_set_get_roundtrip(self):
        original = get_plot_theme()
        custom = PlotTheme(style="white", font_scale=2.0)
        set_plot_theme(custom)
        assert get_plot_theme() is custom
        set_plot_theme(original)
        assert get_plot_theme() is original


# ---------------------------------------------------------------------------
# _create_axes / save_figure tests
# ---------------------------------------------------------------------------


class TestCreateAxes:
    def test_creates_new_figure_when_ax_is_none(self):
        fig, ax = _create_axes(None, figsize=(8, 4))
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_reuses_provided_axes(self):
        fig_orig, ax_orig = plt.subplots()
        fig, ax = _create_axes(ax_orig)
        assert fig is fig_orig
        assert ax is ax_orig


class TestSaveFigure:
    def test_saves_to_disk(self, tmp_path):
        fig, _ = plt.subplots()
        out = tmp_path / "test.png"
        save_figure(fig, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Plot function return-type / ax-param contract
# ---------------------------------------------------------------------------


class TestPlotTimeSeries:
    def test_returns_figure(self):
        ts = pd.Series(pd.date_range("2020-01-01", periods=5, freq="s"))
        vals = pd.Series([1, 2, 3, 2, 1])
        fig = plot_time_series(ts, vals)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self):
        fig_orig, ax_orig = plt.subplots()
        ts = pd.Series(pd.date_range("2020-01-01", periods=5, freq="s"))
        vals = pd.Series([1, 2, 3, 2, 1])
        fig = plot_time_series(ts, vals, ax=ax_orig)
        assert fig is fig_orig


class TestPlotTrades:
    def test_returns_figure(self, sample_trades):
        fig = plot_trades(sample_trades)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_trades):
        fig_orig, ax_orig = plt.subplots()
        fig = plot_trades(sample_trades, ax=ax_orig)
        assert fig is fig_orig


class TestPlotEventMap:
    def test_returns_figure(self, sample_events):
        fig = plot_event_map(sample_events)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot_event_map(sample_events, ax=ax_orig)
        assert fig is fig_orig


class TestPlotVolumeMap:
    def test_returns_figure(self, sample_events):
        fig = plot_volume_map(sample_events)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot_volume_map(sample_events, ax=ax_orig)
        assert fig is fig_orig


class TestPlotCurrentDepth:
    def test_returns_figure(self):
        bids = pd.DataFrame(
            {"price": [236.50, 236.00], "volume": [100, 200], "liquidity": [100, 300]}
        )
        asks = pd.DataFrame(
            {"price": [237.00, 237.50], "volume": [150, 250], "liquidity": [150, 400]}
        )
        ob = {"bids": bids, "asks": asks, "timestamp": 1430438400}
        fig = plot_current_depth(ob)
        assert isinstance(fig, Figure)


class TestPlotVolumePercentiles:
    def test_returns_figure(self, sample_depth_summary):
        fig = plot_volume_percentiles(sample_depth_summary)
        assert isinstance(fig, Figure)


class TestPlotEventsHistogram:
    def test_returns_figure(self, sample_events):
        fig = plot_events_histogram(sample_events, val="price", bw=0.25)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot_events_histogram(sample_events, val="price", bw=0.25, ax=ax_orig)
        assert fig is fig_orig


# ---------------------------------------------------------------------------
# Subplot composition (key use-case for ax parameter)
# ---------------------------------------------------------------------------


class TestSubplotComposition:
    def test_two_plots_on_shared_figure(self, sample_trades):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig1 = plot_trades(sample_trades, ax=ax1)
        fig2 = plot_trades(sample_trades, ax=ax2)
        assert fig1 is fig
        assert fig2 is fig
        assert len(fig.axes) >= 2
