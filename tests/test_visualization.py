"""Tests for ob_analytics.visualization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ob_analytics.visualization import (
    Level,
    PlotTheme,
    plot,
    save_figure,
)
from ob_analytics.visualization import _data
from ob_analytics.visualization._matplotlib import _create_axes


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
        # The reference (bundle) style: light, despined, dotted grid.
        theme = PlotTheme()
        assert theme.style == "white"
        assert theme.font_scale == 1.05
        assert theme.rc["axes.grid"] is True
        assert theme.rc["axes.spines.top"] is False

    def test_custom_theme(self):
        theme = PlotTheme(style="whitegrid", font_scale=1.0)
        assert theme.style == "whitegrid"
        assert theme.font_scale == 1.0

    def test_frozen(self):
        theme = PlotTheme()
        with pytest.raises(AttributeError):
            theme.style = "whitegrid"  # type: ignore[misc]

    def test_theme_kwarg_threads_through_create_axes(self):
        # A per-call theme is applied when _create_axes builds a new figure;
        # there is no global theme to set or restore.
        custom = PlotTheme(style="white", font_scale=2.0)
        fig, _ = _create_axes(None, theme=custom)
        assert isinstance(fig, Figure)

    def test_plot_accepts_theme_kwarg(self, sample_trades):
        # plot() pops theme= from kwargs and forwards it to the renderer.
        custom = PlotTheme(style="whitegrid", font_scale=1.0)
        fig = plot(
            "trade_tape",
            Level.L2,
            theme=custom,
            **_data.prepare_trades_data(sample_trades),
        )
        assert isinstance(fig, Figure)


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
        fig = plot("time_series", **_data.prepare_time_series_data(ts, vals))
        assert isinstance(fig, Figure)

    def test_accepts_ax(self):
        fig_orig, ax_orig = plt.subplots()
        ts = pd.Series(pd.date_range("2020-01-01", periods=5, freq="s"))
        vals = pd.Series([1, 2, 3, 2, 1])
        fig = plot(
            "time_series", ax=ax_orig, **_data.prepare_time_series_data(ts, vals)
        )
        assert fig is fig_orig


class TestPlotTrades:
    def test_returns_figure(self, sample_trades):
        fig = plot("trade_tape", Level.L2, **_data.prepare_trades_data(sample_trades))
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_trades):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "trade_tape",
            Level.L2,
            ax=ax_orig,
            **_data.prepare_trades_data(sample_trades),
        )
        assert fig is fig_orig

    def test_comparable_requires_level(self, sample_trades):
        with pytest.raises(ValueError, match="comparable"):
            plot("trade_tape", **_data.prepare_trades_data(sample_trades))

    def test_l2_ylabel_is_price(self, sample_trades):
        # The L2 tape plots execution prices, not limit prices.
        fig = plot("trade_tape", Level.L2, **_data.prepare_trades_data(sample_trades))
        assert fig.axes[0].get_ylabel() == "Price"


class TestPlotEventMap:
    def test_returns_figure(self, sample_events):
        fig = plot(
            "order_activity",
            Level.L2,
            **_data.prepare_event_map_data(sample_events),
        )
        assert isinstance(fig, Figure)

    def test_has_legend_disambiguating_marks(self, sample_events):
        # §3.9: the gray create/delete circles and the bid/ask direction dots
        # were unexplained; a legend must label all four.
        fig = plot(
            "order_activity",
            Level.L2,
            **_data.prepare_event_map_data(sample_events),
        )
        legend = fig.axes[0].get_legend()
        assert legend is not None
        labels = {t.get_text() for t in legend.get_texts()}
        assert {"bid", "ask", "created", "deleted"} <= labels

    def test_accepts_ax(self, sample_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "order_activity",
            Level.L2,
            ax=ax_orig,
            **_data.prepare_event_map_data(sample_events),
        )
        assert fig is fig_orig


class TestPlotOrderActivityL3:
    def test_returns_figure(self, sample_order_lifecycle_events):
        data = _data.prepare_order_activity_l3_data(sample_order_lifecycle_events)
        fig = plot("order_activity", Level.L3, **data)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_order_lifecycle_events):
        fig_orig, ax_orig = plt.subplots()
        data = _data.prepare_order_activity_l3_data(sample_order_lifecycle_events)
        fig = plot("order_activity", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_comparable_requires_level(self, sample_order_lifecycle_events):
        data = _data.prepare_order_activity_l3_data(sample_order_lifecycle_events)
        with pytest.raises(ValueError, match="comparable"):
            plot("order_activity", **data)

    def test_rounded_price_ticks_caps_count(self) -> None:
        # The Gantt placed one tick per price_by step, stacking labels into an
        # unreadable smear on a wide book; round-number ticks must be thinned.
        from ob_analytics.visualization._matplotlib import _rounded_price_ticks

        ticks = _rounded_price_ticks(78250.0, 78510.0, 5.0, max_ticks=12)
        assert 0 < len(ticks) <= 12
        for t in ticks:
            assert abs(t / 5.0 - round(t / 5.0)) < 1e-9  # still multiples of price_by

    def test_gantt_thins_y_ticks_on_wide_book(self) -> None:
        # 260-wide window at a 5-unit grid would place 52 ticks; the renderer
        # must thin them to a legible count.
        empty = pd.DataFrame(
            {"price": [], "start_ts": [], "end_ts": [], "linewidth": []}
        )
        one = pd.DataFrame(
            {
                "price": [78300.0],
                "start_ts": pd.to_datetime(["2026-01-01T00:00:00"]),
                "end_ts": pd.to_datetime(["2026-01-01T00:00:10"]),
                "linewidth": [1.5],
            }
        )
        data = {
            "filled": one,
            "cancelled": empty,
            "resting": empty,
            "y_range": (78250.0, 78510.0),
            "price_by": 5.0,
            "shown_of": None,
            "show_markers": False,
        }
        fig = plot("order_activity", Level.L3, **data)
        ax = fig.axes[0]
        assert 0 < len(ax.get_yticks()) <= 13


class TestPlotVolumeMap:
    def test_returns_figure(self, sample_events):
        fig = plot(
            "cancellations",
            Level.L2,
            **_data.prepare_volume_map_data(sample_events),
        )
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "cancellations",
            Level.L2,
            ax=ax_orig,
            **_data.prepare_volume_map_data(sample_events),
        )
        assert fig is fig_orig


class TestPlotCancellationsL3:
    def test_returns_figure(self, sample_cancellation_events):
        data = _data.prepare_cancellations_l3_data(sample_cancellation_events)
        fig = plot("cancellations", Level.L3, **data)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_cancellation_events):
        fig_orig, ax_orig = plt.subplots()
        data = _data.prepare_cancellations_l3_data(sample_cancellation_events)
        fig = plot("cancellations", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_comparable_requires_level(self, sample_cancellation_events):
        data = _data.prepare_cancellations_l3_data(sample_cancellation_events)
        with pytest.raises(ValueError, match="comparable"):
            plot("cancellations", **data)

    def test_hexbin_is_rasterized(self, sample_cancellation_events):
        # The density hexbin is a vector PolyCollection; rasterize it so a dense
        # book does not bloat the saved figure.
        data = _data.prepare_cancellations_l3_data(sample_cancellation_events)
        fig = plot("cancellations", Level.L3, **data)
        ax = fig.axes[0]
        assert ax.collections
        assert all(c.get_rasterized() for c in ax.collections)

    def test_axes_are_log_log(self, sample_cancellation_events):
        # §3.3: the latent populations only separate on log-log axes.
        data = _data.prepare_cancellations_l3_data(sample_cancellation_events)
        fig = plot("cancellations", Level.L3, **data)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    def test_distance_is_positive_and_floored(self, sample_cancellation_events):
        # Distance becomes |bps| floored at 0.05 so log axes can show it.
        data = _data.prepare_cancellations_l3_data(sample_cancellation_events)
        for side in (data["bids"], data["asks"]):
            assert (side["distance_from_touch"] >= 0.05).all()
            assert (side["age_s"] >= 1e-3).all()


class TestPlotTradeTapeL3:
    def test_returns_figure(self, sample_executed_orders):
        events, trades = sample_executed_orders
        data = _data.prepare_trade_tape_l3_data(
            events, trades, price_from=0.0, price_to=1e9
        )
        fig = plot("trade_tape", Level.L3, **data)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_executed_orders):
        events, trades = sample_executed_orders
        fig_orig, ax_orig = plt.subplots()
        data = _data.prepare_trade_tape_l3_data(
            events, trades, price_from=0.0, price_to=1e9
        )
        fig = plot("trade_tape", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig


class TestPlotLiquidityAtTouch:
    # Now a comparable concept (L2 step + L3 composition), so level is explicit.
    def test_returns_figure(self, sample_depth_summary):
        data = _data.prepare_liquidity_at_touch_data(sample_depth_summary)
        fig = plot("liquidity_at_touch", Level.L2, **data)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_depth_summary):
        fig_orig, ax_orig = plt.subplots()
        data = _data.prepare_liquidity_at_touch_data(sample_depth_summary)
        fig = plot("liquidity_at_touch", Level.L2, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_lines_drawn_with_transparency(self, sample_depth_summary):
        # The bid/ask step series overplot heavily in the dense band; drawing
        # them with alpha < 1 keeps both legible where they overlap.
        data = _data.prepare_liquidity_at_touch_data(sample_depth_summary)
        fig = plot("liquidity_at_touch", Level.L2, **data)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert lines  # both series drawn
        for ln in lines:
            alpha = ln.get_alpha()
            assert alpha is not None and alpha < 1.0


class TestPlotLiquidityAtTouchL3:
    """§4.1c queue-composition strip (age x rank pcolormesh)."""

    @staticmethod
    def _grid_data() -> dict:
        ages = np.array([[1.0, 2.0, np.nan], [np.nan, 5.0, 6.0]])  # 2 ranks x 3 cols
        times = pd.date_range("2015-05-01", periods=3, freq="s").to_numpy()
        return {"ages": ages, "times": times, "max_rank": 2, "side": "bid"}

    def test_returns_figure_with_mesh(self) -> None:
        fig = plot("liquidity_at_touch", Level.L3, **self._grid_data())
        ax = fig.axes[0]
        assert isinstance(fig, Figure)
        assert ax.collections  # the pcolormesh QuadMesh
        assert "rank" in ax.get_ylabel().lower()

    def test_empty_grid_is_safe(self) -> None:
        data = {
            "ages": np.empty((0, 0)),
            "times": np.array([], dtype="datetime64[ns]"),
            "max_rank": 0,
            "side": "bid",
        }
        assert isinstance(plot("liquidity_at_touch", Level.L3, **data), Figure)


class TestPlotOrderOutcomeL3:
    def test_returns_figure(self, sample_executed_orders):
        events, trades = sample_executed_orders
        data = _data.prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", Level.L3, **data)
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_executed_orders):
        events, trades = sample_executed_orders
        fig_orig, ax_orig = plt.subplots()
        data = _data.prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_resolves_single_level(self, sample_executed_orders):
        # L3-only: registered at exactly one level, so it resolves without level=.
        events, trades = sample_executed_orders
        data = _data.prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", **data)
        assert isinstance(fig, Figure)

    def test_draws_cancelled_underneath(self, sample_executed_orders):
        import matplotlib.colors as mcolors

        from ob_analytics.visualization._matplotlib import _CANCELLED_COLOR

        events, trades = sample_executed_orders
        data = _data.prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", Level.L3, **data)
        ax = fig.axes[0]
        # The dominant cancelled class must be drawn first (underneath) so the
        # rarer fills/partials are not buried; first collection == cancelled.
        first_rgb = ax.collections[0].get_facecolor()[0][:3]
        assert np.allclose(first_rgb, mcolors.to_rgba(_CANCELLED_COLOR)[:3], atol=0.01)


class TestPlotBookSnapshot:
    @staticmethod
    def _order_book() -> dict:
        bids = pd.DataFrame(
            {"price": [236.50, 236.00], "volume": [100, 200], "liquidity": [100, 300]}
        )
        asks = pd.DataFrame(
            {"price": [237.00, 237.50], "volume": [150, 250], "liquidity": [150, 400]}
        )
        return {"bids": bids, "asks": asks, "timestamp": 1430438400}

    @pytest.mark.parametrize("level", [Level.L2, Level.L3])
    @pytest.mark.parametrize("concept", ["book_snapshot", "depth_chart"])
    def test_returns_figure(self, concept: str, level: Level) -> None:
        per_order = level is Level.L3
        data = _data.prepare_book_snapshot_data(self._order_book(), per_order=per_order)
        fig = plot(concept, level, **data)
        assert isinstance(fig, Figure)

    def test_comparable_requires_level(self) -> None:
        data = _data.prepare_book_snapshot_data(self._order_book())
        with pytest.raises(ValueError, match="comparable"):
            plot("book_snapshot", **data)

    def test_horizontal_ladder_orientation(self) -> None:
        # §3.1: price on y, size on x (was the reverse).
        data = _data.prepare_book_snapshot_data(self._order_book(), per_order=True)
        fig = plot("book_snapshot", Level.L3, **data)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Price"
        assert "Size" in ax.get_xlabel()

    def test_per_order_uses_white_separators(self) -> None:
        # Windowing keeps bars tall, so L3 separators are always on: white edges
        # segment each level into its orders and bid/ask hue survives.
        data = _data.prepare_book_snapshot_data(self._order_book(), per_order=True)
        fig = plot("book_snapshot", Level.L3, **data)
        ax = fig.axes[0]
        edges = [c.patches[0].get_edgecolor() for c in ax.containers if len(c.patches)]
        assert edges  # at least one side drawn
        for rgba in edges:
            assert np.allclose(rgba[:3], 1.0, atol=0.01)

    def test_aggregate_has_no_separators(self) -> None:
        # L2 bars never get a per-order edge (edgecolor "none" -> alpha 0).
        data = _data.prepare_book_snapshot_data(self._order_book(), per_order=False)
        fig = plot("book_snapshot", Level.L2, **data)
        ax = fig.axes[0]
        for container in ax.containers:
            if len(container.patches):
                assert container.patches[0].get_edgecolor()[3] == 0.0

    def test_l2_and_l3_render_differently(self) -> None:
        # The flagship acceptance: equal-total levels with different
        # composition must not render identically (they were md5-identical
        # before §3.1). One level holds three orders; L3 segments it, L2 does
        # not.
        book = {
            "bids": pd.DataFrame(
                {
                    "price": [236.5, 236.5, 236.5, 236.0],
                    "volume": [40.0, 30.0, 30.0, 100.0],
                    "liquidity": [40.0, 70.0, 100.0, 200.0],
                }
            ),
            "asks": pd.DataFrame(
                {
                    "price": [237.0, 237.5],
                    "volume": [100.0, 100.0],
                    "liquidity": [100.0, 200.0],
                }
            ),
            "timestamp": 1430438400,
        }
        buffers = {}
        for level, per_order in ((Level.L2, False), (Level.L3, True)):
            data = _data.prepare_book_snapshot_data(book, per_order=per_order)
            fig = plot("book_snapshot", level, **data)
            fig.canvas.draw()
            buffers[level] = np.asarray(fig.canvas.buffer_rgba()).copy()
            plt.close(fig)
        assert not np.array_equal(buffers[Level.L2], buffers[Level.L3])

    def test_windowing_keeps_top_n_levels(self) -> None:
        prices = np.arange(100.0, 80.0, -1.0)  # 20 levels per side
        book = {
            "bids": pd.DataFrame(
                {
                    "price": prices,
                    "volume": np.ones_like(prices),
                    "liquidity": np.arange(1.0, len(prices) + 1.0),
                }
            ),
            "asks": pd.DataFrame(
                {
                    "price": prices + 25.0,
                    "volume": np.ones_like(prices),
                    "liquidity": np.arange(1.0, len(prices) + 1.0),
                }
            ),
            "timestamp": 1430438400,
        }
        data = _data.prepare_book_snapshot_data(book, top_n=5)
        assert data["bids"]["price"].nunique() == 5
        assert data["asks"]["price"].nunique() == 5
        # Best-first: bids keep the five highest prices, asks the five lowest.
        assert data["bids"]["price"].min() == 96.0
        assert data["asks"]["price"].max() == 110.0

    def test_quantiles_off_by_default(self) -> None:
        import inspect

        sig = inspect.signature(_data.prepare_book_snapshot_data)
        assert sig.parameters["show_quantiles"].default is False
        data = _data.prepare_book_snapshot_data(self._order_book())
        assert len(data["bid_quantiles"]) == 0
        assert len(data["ask_quantiles"]) == 0

    def test_quantiles_capped_at_three_per_side(self) -> None:
        # Many heavy levels (per-order rows) must still yield <= 3 guides/side.
        prices = np.arange(100.0, 80.0, -1.0)
        book = {
            "bids": pd.DataFrame(
                {
                    "price": prices,
                    "volume": np.arange(1.0, len(prices) + 1.0),
                    "liquidity": np.arange(1.0, len(prices) + 1.0),
                }
            ),
            "asks": pd.DataFrame(
                {
                    "price": prices + 25.0,
                    "volume": np.arange(1.0, len(prices) + 1.0),
                    "liquidity": np.arange(1.0, len(prices) + 1.0),
                }
            ),
            "timestamp": 1430438400,
        }
        data = _data.prepare_book_snapshot_data(book, show_quantiles=True, top_n=None)
        assert len(data["bid_quantiles"]) <= 3
        assert len(data["ask_quantiles"]) <= 3


class TestPlotVolumePercentiles:
    def test_returns_figure(self, sample_depth_summary):
        fig = plot(
            "volume_percentiles",
            **_data.prepare_volume_percentiles_data(sample_depth_summary),
        )
        assert isinstance(fig, Figure)


class TestPlotEventsHistogram:
    def test_returns_figure(self, sample_events):
        fig = plot(
            "events_histogram",
            **_data.prepare_events_histogram_data(sample_events, val="price", bw=0.25),
        )
        assert isinstance(fig, Figure)

    def test_accepts_ax(self, sample_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "events_histogram",
            ax=ax_orig,
            **_data.prepare_events_histogram_data(sample_events, val="price", bw=0.25),
        )
        assert fig is fig_orig


class TestVolumeNorm:
    """col_bias selects the depth-heatmap color normalization."""

    def test_unit_bias_is_linear(self):
        import matplotlib.colors as mcolors

        from ob_analytics.visualization._matplotlib import _volume_norm

        norm = _volume_norm(pd.Series([1.0, 10.0, 100.0]), col_bias=1.0)
        # Exactly linear, not a PowerNorm/LogNorm subclass.
        assert type(norm) is mcolors.Normalize

    def test_fractional_bias_is_powernorm(self):
        import matplotlib.colors as mcolors

        from ob_analytics.visualization._matplotlib import _volume_norm

        norm = _volume_norm(pd.Series([1.0, 10.0, 100.0]), col_bias=0.1)
        assert isinstance(norm, mcolors.PowerNorm)
        assert norm.gamma == 0.1

    def test_fractional_bias_brightens_low_volume(self):
        from ob_analytics.visualization._matplotlib import _volume_norm

        vol = pd.Series([1.0, 10.0, 100.0])
        linear = _volume_norm(vol, col_bias=1.0)
        biased = _volume_norm(vol, col_bias=0.1)
        # A low-volume level lands higher on the ramp under the bias.
        assert biased(10.0) > linear(10.0)

    def test_nonpositive_bias_is_log(self):
        import matplotlib.colors as mcolors

        from ob_analytics.visualization._matplotlib import _volume_norm

        # Zeros (which become NaN upstream) must not poison the log vmin.
        norm = _volume_norm(pd.Series([0.0, 1.0, 100.0]), col_bias=0.0)
        assert isinstance(norm, mcolors.LogNorm)
        assert norm.vmin == 1.0


class TestPriceLevelsColBias:
    """The depth heatmap renders under every col_bias regime."""

    def _depth(self, sample_events):
        depth = sample_events[["timestamp", "price", "volume"]].copy()
        depth["direction"] = "bid"
        return depth

    @pytest.mark.parametrize("col_bias", [1.0, 0.5, 0.1, 0.0, -1.0])
    def test_renders_for_each_bias(self, sample_events, col_bias):
        depth = self._depth(sample_events)
        fig = plot(
            "depth_heatmap",
            **_data.prepare_price_levels_data(depth, col_bias=col_bias),
        )
        assert isinstance(fig, Figure)

    def test_default_bias_is_linear(self):
        # The library default is the hero linear look (col_bias=1.0).
        import inspect

        sig = inspect.signature(_data.prepare_price_levels_data)
        assert sig.parameters["col_bias"].default == 1.0


# ---------------------------------------------------------------------------
# Subplot composition (key use-case for ax parameter)
# ---------------------------------------------------------------------------


class TestSubplotComposition:
    def test_two_plots_on_shared_figure(self, sample_trades):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig1 = plot(
            "trade_tape", Level.L2, ax=ax1, **_data.prepare_trades_data(sample_trades)
        )
        fig2 = plot(
            "trade_tape", Level.L2, ax=ax2, **_data.prepare_trades_data(sample_trades)
        )
        assert fig1 is fig
        assert fig2 is fig
        assert len(fig.axes) >= 2


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_default_backend_returns_matplotlib_figure(self, sample_trades):
        fig = plot("trade_tape", Level.L2, **_data.prepare_trades_data(sample_trades))
        assert isinstance(fig, Figure)

    def test_explicit_matplotlib_returns_figure(self, sample_trades):
        fig = plot(
            "trade_tape",
            Level.L2,
            backend="matplotlib",
            **_data.prepare_trades_data(sample_trades),
        )
        assert isinstance(fig, Figure)

    def test_invalid_backend_raises_value_error(self, sample_trades):
        with pytest.raises(ValueError, match="Unknown backend"):
            plot(
                "trade_tape",
                backend="nonexistent",
                **_data.prepare_trades_data(sample_trades),
            )

    def test_plot_dispatch_matplotlib(self, sample_trades):
        """The unified plot() dispatcher renders from already-prepared data."""
        from ob_analytics.visualization import _data, plot

        fig = plot(
            "trade_tape",
            Level.L2,
            backend="matplotlib",
            **_data.prepare_trades_data(sample_trades),
        )
        assert isinstance(fig, Figure)

    def test_plot_unknown_backend_raises(self):
        from ob_analytics.visualization import plot

        with pytest.raises(ValueError, match="Unknown backend"):
            plot("trade_tape", backend="nope")


class TestRegisterBackend:
    def test_register_and_dispatch(self, sample_trades, tmp_path):
        """A backend module self-registers its renderers; plot() dispatches."""
        from ob_analytics.visualization import (
            RENDERERS,
            Level,
            _BACKEND_MODULES,
            plot,
            register_plot_backend,
        )

        # The new extension contract: the backend module registers each
        # renderer into RENDERERS under (concept, level, backend) at import time.
        dummy_module = tmp_path / "dummy_backend.py"
        dummy_module.write_text(
            "from ob_analytics.visualization import RENDERERS, Level\n"
            "class _Sentinel:\n"
            "    pass\n"
            "def dummy_trades(data, *a, **kw):\n"
            "    return _Sentinel()\n"
            "RENDERERS.register(('trade_tape', Level.L2, 'dummy'), dummy_trades)\n"
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            register_plot_backend("dummy", "dummy_backend")
            assert "dummy" in _BACKEND_MODULES

            fig = plot("trade_tape", backend="dummy")
            # Should return the sentinel, not a matplotlib Figure.
            assert type(fig).__name__ == "_Sentinel"
        finally:
            sys.path.pop(0)
            _BACKEND_MODULES.pop("dummy", None)
            sys.modules.pop("dummy_backend", None)
            # Registry has no public removal; the inert (trade_tape, L2, dummy)
            # entry is dropped here so the dummy module's renderer doesn't linger.
            RENDERERS._items.pop(("trade_tape", Level.L2, "dummy"), None)


class TestPlotPriceView:
    """§4.2 spread ribbon + microprice (L2)."""

    def test_returns_figure_with_ribbon(self, sample_depth_summary):
        data = _data.prepare_price_view_data(sample_depth_summary)
        fig = plot("price_view", Level.L2, **data)
        ax = fig.axes[0]
        assert isinstance(fig, Figure)
        assert ax.collections  # the fill_between ribbon (PolyCollection)
        labels = {ln.get_label() for ln in ax.get_lines()}
        assert "microprice" in labels
        assert ax.get_ylabel() == "Price"


class TestPlotTradeSize:
    """§4.3 trade-size strip (log-x jittered dots by side)."""

    def test_returns_figure_log_x(self, sample_trades):
        data = _data.prepare_trade_size_data(sample_trades)
        fig = plot("trade_size", Level.L2, **data)
        ax = fig.axes[0]
        assert isinstance(fig, Figure)
        assert ax.get_xscale() == "log"
        assert [t.get_text() for t in ax.get_yticklabels()] == ["sell", "buy"]
