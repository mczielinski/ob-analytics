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

from ob_analytics.exceptions import ConfigError
from ob_analytics.visualization import prepare
from ob_analytics.visualization._bokeh import (
    _bokeh_color_field,
    bokeh_book_snapshot_aggregate,
    bokeh_book_snapshot_per_order,
    bokeh_depth_chart_aggregate,
    bokeh_depth_chart_per_order,
    bokeh_depth_heatmap,
    bokeh_trade_tape_aggregate,
    bokeh_trade_tape_per_order,
)
from tests._drawn import drawn

# ---------------------------------------------------------------------------
# Tests – each bokeh_*() function returns a Bokeh figure
# ---------------------------------------------------------------------------


class TestBokehTrades:
    def test_returns_bokeh_figure(self, plot_trades: pd.DataFrame) -> None:
        data = prepare.trade_tape(plot_trades, level="L2")
        fig = bokeh_trade_tape_aggregate(data)
        # The price step, and a volume stem and a marker for each trade.
        n = len(plot_trades)
        assert drawn(fig) == {"step": n, "segment": n, "scatter": n}

    def test_l2_ylabel_is_price(self, plot_trades: pd.DataFrame) -> None:
        data = prepare.trade_tape(plot_trades, level="L2")
        fig = bokeh_trade_tape_aggregate(data)
        assert fig.yaxis[0].axis_label == "Price"

    def test_has_lollipop_legend_entries(self, plot_trades: pd.DataFrame) -> None:
        data = prepare.trade_tape(plot_trades, level="L2")
        fig = bokeh_trade_tape_aggregate(data)
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
    def test_returns_bokeh_figure(self, plot_depth: pd.DataFrame) -> None:
        data = prepare.depth_heatmap(plot_depth)
        fig = bokeh_depth_heatmap(data)
        assert drawn(fig)["scatter"] > 0

    def test_spread_overlays_present(self, plot_depth: pd.DataFrame) -> None:
        spread = pd.DataFrame(
            {
                "timestamp": plot_depth["timestamp"],
                "best_bid_price": plot_depth["price"] - 0.05,
                "best_ask_price": plot_depth["price"] + 0.05,
            }
        )
        fig = bokeh_depth_heatmap(
            prepare.depth_heatmap(plot_depth, spread=spread, show_mp=True)
        )
        marks = drawn(fig)
        assert marks["scatter"] > 0
        # With show_mp the spread is drawn as the midprice, one step.
        assert marks["step"] == len(spread)

    def test_hidden_liquidity_overlay_draws_glyphs(
        self, plot_depth: pd.DataFrame, hidden_liquidity_overlay: dict
    ) -> None:
        overlay = hidden_liquidity_overlay
        fig = bokeh_depth_heatmap(prepare.depth_heatmap(plot_depth, **overlay))
        labels = {item.label["value"] for item in fig.legend[0].items}
        assert {"Iceberg chain", "Iceberg refill", "Hidden-order trade"} <= labels
        chains = next(r for r in fig.renderers if type(r.glyph).__name__ == "MultiLine")
        assert (
            len(chains.data_source.data["xs"])
            == overlay["iceberg_lines"]["iceberg"].nunique()
        )
        stems = next(r for r in fig.renderers if type(r.glyph).__name__ == "Segment")
        assert len(stems.data_source.data["x0"]) == len(overlay["hidden_trades"])

    def test_no_overlay_draws_no_hidden_liquidity_glyphs(
        self, plot_depth: pd.DataFrame
    ) -> None:
        fig = bokeh_depth_heatmap(prepare.depth_heatmap(plot_depth))
        glyphs = {type(r.glyph).__name__ for r in fig.renderers}
        assert not glyphs & {"MultiLine", "Segment"}


class TestBokehTradeTapeL3:
    def test_returns_bokeh_figure(self, sample_executed_orders) -> None:
        events, trades = sample_executed_orders
        data = prepare.trade_tape(
            events, trades, price_from=0.0, price_to=1e9, level="L3"
        )
        fig = bokeh_trade_tape_per_order(data)
        # A marker for each trade whose maker is a measured order: three of the
        # four (the third trade's maker is not one).
        assert drawn(fig)["scatter"] == 3


class TestBokehBookSnapshot:
    def test_aggregate_returns_figure(self, plot_order_book: dict) -> None:
        data = prepare.book_snapshot(plot_order_book, level="L2")
        fig = bokeh_book_snapshot_aggregate(data)
        assert isinstance(fig, bpl.figure)
        assert len(fig.renderers) >= 2  # bid + ask bars

    def test_per_order_returns_figure(self, plot_order_book: dict) -> None:
        data = prepare.book_snapshot(plot_order_book, level="L3")
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
        self, plot_order_book: dict, renderer, wrong_per_order: bool
    ) -> None:
        data = prepare.book_snapshot(
            plot_order_book, level="L3" if wrong_per_order else "L2"
        )
        with pytest.raises(ConfigError, match="prepare.book_snapshot"):
            renderer(data)


class TestBokehDepthChart:
    def test_aggregate_returns_figure(self, plot_order_book: dict) -> None:
        data = prepare.book_snapshot(plot_order_book, level="L2")
        fig = bokeh_depth_chart_aggregate(data)
        assert isinstance(fig, bpl.figure)
        assert len(fig.renderers) >= 2  # bid + ask curves (step + varea each)

    def test_per_order_returns_figure(self, plot_order_book: dict) -> None:
        data = prepare.book_snapshot(plot_order_book, level="L3")
        fig = bokeh_depth_chart_per_order(data)
        # Two levels a side: each side's step and fill, and a marker per level.
        assert drawn(fig) == {"step": 4, "varea": 4, "scatter": 4}

    @pytest.mark.parametrize(
        ("renderer", "wrong_per_order"),
        [
            (bokeh_depth_chart_aggregate, True),
            (bokeh_depth_chart_per_order, False),
        ],
    )
    def test_payload_level_mismatch_raises(
        self, plot_order_book: dict, renderer, wrong_per_order: bool
    ) -> None:
        data = prepare.book_snapshot(
            plot_order_book, level="L3" if wrong_per_order else "L2"
        )
        with pytest.raises(ConfigError, match="prepare.book_snapshot"):
            renderer(data)


# ---------------------------------------------------------------------------
# Backend dispatch via public API
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_bokeh_backend_returns_bokeh_figure(
        self, plot_trades: pd.DataFrame
    ) -> None:
        from ob_analytics.visualization import Level, plot, prepare

        fig = plot(
            "trade_tape",
            Level.L2,
            backend="bokeh",
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        assert drawn(fig)["scatter"] == len(plot_trades)

    def test_missing_level_for_comparable_concept_raises(
        self, plot_order_book: dict
    ) -> None:
        from ob_analytics.visualization import plot, prepare

        data = prepare.book_snapshot(plot_order_book, level="L2")
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
            plot("book_snapshot", backend="bokeh", **data)

    def test_ax_is_refused(self, plot_order_book: dict) -> None:
        import matplotlib.pyplot as plt

        from ob_analytics.visualization import plot, prepare

        data = prepare.book_snapshot(plot_order_book, level="L2")
        _, ax = plt.subplots()
        with pytest.raises(ConfigError, match="ax= is a Matplotlib axes"):
            plot("book_snapshot", "L2", backend="bokeh", ax=ax, **data)
        plt.close("all")
