"""Tests for ob_analytics.visualization."""

import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.text import Text

from ob_analytics.exceptions import ConfigError, UnknownNameError
from ob_analytics.visualization import (
    Level,
    PlotTheme,
    _data,
    plot,
    prepare,
    save_figure,
)
from ob_analytics.visualization._matplotlib import _create_axes, _step_envelope
from tests._drawn import drawn

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
            theme.style = "whitegrid"  # ty: ignore[invalid-assignment]

    def test_theme_does_not_leak_into_rcparams(self, plot_trades):
        # The theme is scoped to the figure plot() creates: matplotlib's
        # global rcParams are the same before and after the call.
        before = dict(matplotlib.rcParams)
        plot(
            "trade_tape",
            Level.L2,
            theme=PlotTheme(style="darkgrid", context="talk"),
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        assert dict(matplotlib.rcParams) == before

    def test_themed_figure_keeps_theme_after_call(self, plot_trades):
        # The returned figure still shows the theme when drawn after the
        # call, when tick artists are built outside the theme's rc_context.
        fig = plot(
            "trade_tape",
            Level.L2,
            theme=PlotTheme(style="darkgrid", context="talk", font_scale=1.0),
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        fig.canvas.draw()
        ax = fig.axes[0]
        # darkgrid background; talk context ticks (11 * 1.5 = 16.5 pt).
        assert matplotlib.colors.to_hex(ax.get_facecolor()) == "#eaeaf2"
        for tick in ax.xaxis.get_major_ticks():
            assert tick.label1.get_fontsize() == pytest.approx(16.5)
        # A plain figure made afterwards gets matplotlib's own settings.
        plain_fig, plain_ax = plt.subplots()
        plain_fig.canvas.draw()
        assert matplotlib.colors.to_hex(plain_ax.get_facecolor()) != "#eaeaf2"
        plain_tick = plain_ax.xaxis.get_major_ticks()[0]
        assert plain_tick.label1.get_fontsize() != pytest.approx(16.5)

    def test_plot_accepts_theme_kwarg(self, plot_trades):
        # plot() pops theme= from kwargs and forwards it to the renderer.
        custom = PlotTheme(style="whitegrid", font_scale=1.0)
        fig = plot(
            "trade_tape",
            Level.L2,
            theme=custom,
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        assert drawn(fig)["pathcollection"] == len(plot_trades)


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


class TestFigureLayout:
    """A face lays out a figure it creates and leaves a caller's figure alone."""

    @staticmethod
    def _time_series() -> dict:
        ts = pd.Series(pd.date_range("2020-01-01", periods=5, freq="s", tz="UTC"))
        return prepare.time_series(ts, pd.Series([1, 2, 3, 2, 1]))

    def test_own_figure_is_laid_out(self):
        fig = plot("time_series", **self._time_series())
        default_left = plt.rcParams["figure.subplot.left"]
        assert fig.axes[0].get_position().x0 != pytest.approx(default_left)

    def test_caller_figure_layout_is_untouched(self):
        # An axes placed by hand is one tight_layout cannot handle: laying out
        # this figure warns, and would move the caller's panel.
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        fig.add_axes((0.1, 0.05, 0.8, 0.1))
        before = ax.get_position().bounds
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot("time_series", ax=ax, **self._time_series())
        assert ax.get_position().bounds == pytest.approx(before)

    def test_no_face_lays_out_a_caller_figure(self, tiny_bitstamp_orders_csv):
        # Every matplotlib face the gallery draws, each on a caller's figure
        # that tight_layout cannot handle.  A face that lays the figure out
        # warns and changes its subplot parameters.  (A face's colorbar may
        # still narrow the caller's panel; that is not a layout.)
        from ob_analytics.bitstamp import BitstampSource
        from ob_analytics.pipeline import Pipeline
        from ob_analytics.visualization import RENDERERS
        from ob_analytics.visualization.gallery import (
            _metric_cards,
            build_gallery_model,
            display_result,
        )

        result = Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))
        model = build_gallery_model(result)
        faces: list[tuple[Any, Level | None]] = [
            (spec, level) for c in model.concepts for level, spec in c.variants.items()
        ]
        faces += [(spec, None) for spec in _metric_cards(display_result(result))]
        # Only matplotlib lays out a figure; a face drawn by other backends
        # alone (such as the plotly book replay) has nothing to check here.
        faces = [
            (spec, level)
            for spec, level in faces
            if (spec.plot_name, level, "matplotlib") in RENDERERS
        ]

        def subplot_params(fig: Figure) -> tuple[float, ...]:
            p = fig.subplotpars
            return (p.left, p.right, p.bottom, p.top, p.wspace, p.hspace)

        drawn, unfed = [], []
        for spec, level in faces:
            try:
                payload = spec.prepare(**spec.prep_kwargs)
            except Exception as e:  # noqa: BLE001 -- a face the tiny run cannot feed
                unfed.append((spec.plot_name, e))
                continue
            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1)
            fig.add_axes((0.1, 0.05, 0.8, 0.1))
            before = subplot_params(fig)
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*tight_layout")
                plot(spec.plot_name, level, ax=ax, **payload)
            assert subplot_params(fig) == pytest.approx(before), spec.plot_name
            drawn.append(spec.plot_name)
            plt.close(fig)
        # The check means little unless most faces were actually drawn.
        assert len(drawn) >= 15, (drawn, unfed)

    def test_caller_can_lay_out_a_cancellations_panel(self, sample_cancellation_events):
        # The colorbar must share the caller's gridspec, or the caller's own
        # tight_layout cannot place it and warns.
        fig, axes = plt.subplots(1, 2)
        data = prepare.cancellations(sample_cancellation_events, level="L3")
        plot("cancellations", Level.L3, ax=axes[0], **data)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig.tight_layout()

    def test_empty_book_draws_no_legend(self):
        empty = pd.DataFrame({"price": [], "volume": [], "liquidity": []})
        book = {"bids": empty, "asks": empty, "timestamp": 1430438400}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot(
                "book_snapshot", Level.L2, **prepare.book_snapshot(book, level="L2")
            )
        assert fig.axes[0].get_legend() is None


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
        fig = plot("time_series", **prepare.time_series(ts, vals))
        assert drawn(fig) == {"line": len(vals)}

    def test_accepts_ax(self):
        fig_orig, ax_orig = plt.subplots()
        ts = pd.Series(pd.date_range("2020-01-01", periods=5, freq="s"))
        vals = pd.Series([1, 2, 3, 2, 1])
        fig = plot("time_series", ax=ax_orig, **prepare.time_series(ts, vals))
        assert fig is fig_orig


class TestPlotTrades:
    def test_returns_figure(self, plot_trades):
        fig = plot(
            "trade_tape", Level.L2, **prepare.trade_tape(plot_trades, level="L2")
        )
        # The price step, and a volume stem and a marker for each trade.
        n = len(plot_trades)
        assert drawn(fig) == {"line": n, "linecollection": n, "pathcollection": n}

    def test_accepts_ax(self, plot_trades):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "trade_tape",
            Level.L2,
            ax=ax_orig,
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        assert fig is fig_orig

    def test_comparable_requires_level(self, plot_trades):
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
            plot("trade_tape", **prepare.trade_tape(plot_trades, level="L2"))

    def test_l2_ylabel_is_price(self, plot_trades):
        # The L2 tape plots execution prices, not limit prices.
        fig = plot(
            "trade_tape", Level.L2, **prepare.trade_tape(plot_trades, level="L2")
        )
        assert fig.axes[0].get_ylabel() == "Price"

    def test_mid_line_is_step(self, plot_trades):
        # The mid holds until the book changes; linear interpolation would
        # paint a diagonal ramp between sparse samples that never existed.
        fig = plot(
            "trade_tape", Level.L2, **prepare.trade_tape(plot_trades, level="L2")
        )
        (mid,) = fig.axes[0].get_lines()
        assert mid.get_drawstyle() == "steps-post"


class TestPlotEventMap:
    def test_returns_figure(self, plot_events):
        fig = plot(
            "order_activity",
            Level.L2,
            **prepare.order_activity(plot_events, level="L2"),
        )
        assert drawn(fig)["pathcollection"] > 0

    def test_has_legend_disambiguating_marks(self, plot_events):
        # §3.9: the gray create/delete circles and the bid/ask direction dots
        # were unexplained; a legend must label all four.
        fig = plot(
            "order_activity",
            Level.L2,
            **prepare.order_activity(plot_events, level="L2"),
        )
        legend = fig.axes[0].get_legend()
        assert legend is not None
        labels = {t.get_text() for t in legend.get_texts()}
        assert {"bid", "ask", "created", "deleted"} <= labels

    def test_accepts_ax(self, plot_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "order_activity",
            Level.L2,
            ax=ax_orig,
            **prepare.order_activity(plot_events, level="L2"),
        )
        assert fig is fig_orig

    def test_empty_window_draws_no_data(self, plot_events):
        # A window before the data: the price-axis step used to raise
        # ValueError on the NaN price range of zero rows.
        data = prepare.order_activity(
            plot_events,
            start_time=pd.Timestamp("2015-05-01 00:00:00"),
            end_time=pd.Timestamp("2015-05-01 00:30:00"),
            level="L2",
        )
        ax = plot("order_activity", Level.L2, **data).axes[0]
        # The theme places titles on the left; read every slot.
        title = " ".join(ax.get_title(loc=s) for s in ("left", "center", "right"))
        assert "no data" in title


class TestPlotOrderActivityL3:
    def test_returns_figure(self, sample_order_lifecycle_events):
        data = prepare.order_activity(sample_order_lifecycle_events, level="L3")
        fig = plot("order_activity", Level.L3, **data)
        marks = drawn(fig)
        assert marks["linecollection"] > 0  # the order spans
        assert marks["pathcollection"] > 0

    def test_accepts_ax(self, sample_order_lifecycle_events):
        fig_orig, ax_orig = plt.subplots()
        data = prepare.order_activity(sample_order_lifecycle_events, level="L3")
        fig = plot("order_activity", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_comparable_requires_level(self, sample_order_lifecycle_events):
        data = prepare.order_activity(sample_order_lifecycle_events, level="L3")
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
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
        data: dict[str, Any] = {
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

    def test_hidden_liquidity_overlay_draws_and_legends(
        self, sample_order_lifecycle_events, hidden_liquidity_overlay
    ) -> None:
        overlay = hidden_liquidity_overlay

        data = prepare.order_activity(
            sample_order_lifecycle_events,
            iceberg_lines=overlay["iceberg_lines"],
            iceberg_refills=overlay["iceberg_refills"],
            hidden_trades=overlay["hidden_trades"],
            level="L3",
        )
        fig = plot("order_activity", Level.L3, **data)
        labels = {h.get_label() for h in fig.axes[0].get_legend().legend_handles}
        assert {
            "Iceberg chain",
            "Iceberg refill",
            "Hidden-order trade",
        } <= labels

    def test_no_overlay_draws_no_extra_legend_entries(
        self, sample_order_lifecycle_events
    ) -> None:
        data = prepare.order_activity(sample_order_lifecycle_events, level="L3")
        fig = plot("order_activity", Level.L3, **data)
        legend = fig.axes[0].get_legend()
        labels = {h.get_label() for h in legend.legend_handles} if legend else set()
        assert "Iceberg refill" not in labels
        assert "Hidden-order trade" not in labels


class TestPlotVolumeMap:
    def test_returns_figure(self, plot_events):
        fig = plot(
            "cancellations",
            Level.L2,
            **prepare.cancellations(plot_events, level="L2"),
        )
        assert drawn(fig)["pathcollection"] > 0

    def test_accepts_ax(self, plot_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "cancellations",
            Level.L2,
            ax=ax_orig,
            **prepare.cancellations(plot_events, level="L2"),
        )
        assert fig is fig_orig


class TestPlotCancellationsL3:
    def test_returns_figure(self, sample_cancellation_events):
        data = prepare.cancellations(sample_cancellation_events, level="L3")
        fig = plot("cancellations", Level.L3, **data)
        assert drawn(fig)["polycollection"] > 0

    def test_accepts_ax(self, sample_cancellation_events):
        fig_orig, ax_orig = plt.subplots()
        data = prepare.cancellations(sample_cancellation_events, level="L3")
        fig = plot("cancellations", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_comparable_requires_level(self, sample_cancellation_events):
        data = prepare.cancellations(sample_cancellation_events, level="L3")
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
            plot("cancellations", **data)

    def test_hexbin_is_rasterized(self, sample_cancellation_events):
        # The density hexbin is a vector PolyCollection; rasterize it so a dense
        # book does not bloat the saved figure.
        data = prepare.cancellations(sample_cancellation_events, level="L3")
        fig = plot("cancellations", Level.L3, **data)
        ax = fig.axes[0]
        assert ax.collections
        assert all(c.get_rasterized() for c in ax.collections)

    def test_axes_are_log_log(self, sample_cancellation_events):
        # §3.3: the latent populations only separate on log-log axes.
        data = prepare.cancellations(sample_cancellation_events, level="L3")
        fig = plot("cancellations", Level.L3, **data)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    def test_distance_is_positive_and_floored(self, sample_cancellation_events):
        # Distance becomes |bps| floored at 0.05 so log axes can show it.
        data = prepare.cancellations(sample_cancellation_events, level="L3")
        for side in (data["bids"], data["asks"]):
            assert (side["distance_from_touch"] >= 0.05).all()
            assert (side["age_s"] >= 1e-3).all()


class TestPlotTradeTapeL3:
    def test_returns_figure(self, sample_executed_orders):
        events, trades = sample_executed_orders
        data = prepare.trade_tape(
            events, trades, price_from=0.0, price_to=1e9, level="L3"
        )
        fig = plot("trade_tape", Level.L3, **data)
        # A marker for each trade whose maker is a measured order: three of
        # the four (the third trade's maker is not one).
        assert drawn(fig)["pathcollection"] == 3

    def test_accepts_ax(self, sample_executed_orders):
        events, trades = sample_executed_orders
        fig_orig, ax_orig = plt.subplots()
        data = prepare.trade_tape(
            events, trades, price_from=0.0, price_to=1e9, level="L3"
        )
        fig = plot("trade_tape", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_mid_line_is_step(self, sample_executed_orders):
        # Same step semantics as the L2 tape: the mid holds between samples.
        events, trades = sample_executed_orders
        data = prepare.trade_tape(
            events, trades, price_from=0.0, price_to=1e9, level="L3"
        )
        fig = plot("trade_tape", Level.L3, **data)
        (mid,) = fig.axes[0].get_lines()
        assert mid.get_drawstyle() == "steps-post"


class TestPlotLiquidityAtTouch:
    # Now a comparable concept (L2 step + L3 composition), so level is explicit.
    def test_returns_figure(self, plot_depth_summary):
        data = prepare.liquidity_at_touch(plot_depth_summary, level="L2")
        fig = plot("liquidity_at_touch", Level.L2, **data)
        # A step through every row, for the bid and for the ask.
        assert drawn(fig) == {"line": 2 * len(plot_depth_summary)}

    def test_accepts_ax(self, plot_depth_summary):
        fig_orig, ax_orig = plt.subplots()
        data = prepare.liquidity_at_touch(plot_depth_summary, level="L2")
        fig = plot("liquidity_at_touch", Level.L2, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_lines_drawn_with_transparency(self, plot_depth_summary):
        # The bid/ask step series overplot heavily in the dense band; drawing
        # them with alpha < 1 keeps both legible where they overlap.
        data = prepare.liquidity_at_touch(plot_depth_summary, level="L2")
        fig = plot("liquidity_at_touch", Level.L2, **data)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert lines  # both series drawn
        for ln in lines:
            alpha = ln.get_alpha()
            assert alpha is not None and alpha < 1.0

    def test_event_rug_drawn_from_events(self, plot_depth_summary, plot_events):
        # Passing events adds a rug (created/cancelled vlines) below the series;
        # without events there is no rug.
        bare = prepare.liquidity_at_touch(plot_depth_summary, level="L2")
        with_rug = prepare.liquidity_at_touch(
            plot_depth_summary, events=plot_events, level="L2"
        )
        assert bare["rug"] is None
        n_bare = len(plot("liquidity_at_touch", Level.L2, **bare).axes[0].collections)
        rugged = plot("liquidity_at_touch", Level.L2, **with_rug).axes[0]
        assert len(rugged.collections) > n_bare  # vlines LineCollections added


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
        data: dict[str, Any] = {
            "ages": np.empty((0, 0)),
            "times": np.array([], dtype="datetime64[ns]"),
            "max_rank": 0,
            "side": "bid",
        }
        assert not drawn(plot("liquidity_at_touch", Level.L3, **data))


class TestPlotOfiByWindow:
    """OFI horizon graph (level-less analytic): stacked banded rows."""

    @staticmethod
    def _trades() -> pd.DataFrame:
        ts = pd.Timestamp("2015-05-01 01:00:00")
        n = 120
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "timestamp": [ts + pd.Timedelta(seconds=5 * i) for i in range(n)],
                "price": 236.0 + rng.normal(0, 0.05, n),
                "volume": rng.uniform(100, 1000, n),
                "direction": pd.Categorical(
                    rng.choice(["buy", "sell"], n), categories=["buy", "sell"]
                ),
            }
        )

    def test_returns_banded_figure(self) -> None:
        data = prepare.ofi_by_window(self._trades())
        fig = plot("ofi_by_window", **data)
        ax = fig.axes[0]
        assert isinstance(fig, Figure)
        assert ax.collections  # fill_between PolyCollections (the bands)
        # Each window is labelled as a text artist beside its row.
        labels = {t.get_text().strip() for t in ax.texts}
        assert set(data["windows"]) <= labels

    def test_empty_is_safe(self) -> None:
        data: dict[str, Any] = {
            "ofi": np.empty((0, 0)),
            "times": np.array([], dtype="datetime64[ns]"),
            "windows": ["5s", "60s"],
        }
        assert not drawn(plot("ofi_by_window", **data))

    @staticmethod
    def _grid(ofi: np.ndarray) -> dict[str, Any]:
        """A 45-second, 5-second-step payload with the given OFI rows."""
        times = pd.date_range("2015-05-01 01:00:00", periods=10, freq="5s", tz="UTC")
        return {
            "ofi": ofi,
            "times": times.to_numpy(),
            "windows": ["5s", "60s"],
            **_data._horizon_bands(ofi),
        }

    @pytest.mark.parametrize(
        "row",
        [np.zeros(10), np.where(np.arange(10) == 4, 0.5, 0.0)],
        ids=["all-zero", "one-value"],
    )
    def test_time_axis_spans_the_data(self, row: np.ndarray, tmp_path) -> None:
        """Rows with few or no bands still save, on the data's own time range."""
        data = self._grid(np.vstack([row, row]))
        fig = plot("ofi_by_window", **data)
        save_figure(fig, tmp_path / "ofi.png")

        x = mdates.date2num(pd.to_datetime(data["times"]))
        assert fig.axes[0].get_xlim() == pytest.approx((x[0], x[-1]))

    def test_one_time_step_spans_seconds(self, tmp_path) -> None:
        times = pd.DatetimeIndex(["2015-05-01 01:00:00"], tz="UTC").to_numpy()
        ofi = np.array([[0.3], [0.1]])
        data: dict[str, Any] = {
            "ofi": ofi,
            "times": times,
            "windows": ["5s", "60s"],
            **_data._horizon_bands(ofi),
        }
        fig = plot("ofi_by_window", **data)
        save_figure(fig, tmp_path / "ofi.png")

        lo, hi = fig.axes[0].get_xlim()
        assert 0 < (hi - lo) * 86_400 <= 10

    def test_all_missing_says_no_data(self, tmp_path) -> None:
        data = self._grid(np.full((2, 10), np.nan))
        fig = plot("ofi_by_window", **data)
        save_figure(fig, tmp_path / "ofi.png")

        assert "no data" in fig.axes[0].get_title("left")


class TestPlotOrderOutcomeL3:
    def test_returns_figure(self, sample_executed_orders):
        events, _trades = sample_executed_orders
        data = prepare.order_outcome(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", Level.L3, **data)
        # Every order but the one still resting has an outcome.
        assert drawn(fig)["pathcollection"] == 5

    def test_accepts_ax(self, sample_executed_orders):
        events, _trades = sample_executed_orders
        fig_orig, ax_orig = plt.subplots()
        data = prepare.order_outcome(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", Level.L3, ax=ax_orig, **data)
        assert fig is fig_orig

    def test_resolves_single_level(self, sample_executed_orders):
        # L3-only: registered at exactly one level, so it resolves without level=.
        events, _trades = sample_executed_orders
        data = prepare.order_outcome(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", **data)
        assert drawn(fig)["pathcollection"] == 5

    def test_draws_cancelled_underneath(self, sample_executed_orders):
        import matplotlib.colors as mcolors

        from ob_analytics.visualization import DEFAULT_PALETTE

        events, _trades = sample_executed_orders
        data = prepare.order_outcome(events, bps_quantiles=(0.0, 1.0))
        fig = plot("order_outcome", Level.L3, **data)
        ax = fig.axes[0]
        # The dominant cancelled class must be drawn first (underneath) so the
        # rarer fills/partials are not buried; first collection == cancelled.
        first_rgb = ax.collections[0].get_facecolor()[0][:3]
        assert np.allclose(
            first_rgb, mcolors.to_rgba(DEFAULT_PALETTE.cancelled)[:3], atol=0.01
        )


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
        data = prepare.book_snapshot(self._order_book(), level=level)
        fig = plot(concept, level, **data)
        # Two levels a side: a bar for each level, or a step through both.
        marks = drawn(fig)
        assert marks["rectangle" if concept == "book_snapshot" else "line"] == 4

    def test_comparable_requires_level(self) -> None:
        data = prepare.book_snapshot(self._order_book(), level="L2")
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
            plot("book_snapshot", **data)

    @pytest.mark.parametrize("level", [Level.L2, Level.L3])
    @pytest.mark.parametrize("concept", ["book_snapshot", "depth_chart"])
    def test_payload_level_mismatch_raises(self, concept: str, level: Level) -> None:
        # An L2 payload rendered at L3 previously drew aggregate bars dressed
        # as per-order (and vice versa); the payload's per_order stamp must
        # match the requested level.
        from ob_analytics.exceptions import ConfigError

        wrong_per_order = level is Level.L2  # contradict the requested level
        data = prepare.book_snapshot(
            self._order_book(), level="L3" if wrong_per_order else "L2"
        )
        with pytest.raises(ConfigError, match="prepare.book_snapshot"):
            plot(concept, level, **data)

    def test_horizontal_ladder_orientation(self) -> None:
        # §3.1: price on y, size on x (was the reverse).
        data = prepare.book_snapshot(self._order_book(), level="L3")
        fig = plot("book_snapshot", Level.L3, **data)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Price"
        assert "Size" in ax.get_xlabel()

    def test_per_order_uses_white_separators(self) -> None:
        # Windowing keeps bars tall, so L3 separators are always on: white edges
        # segment each level into its orders and bid/ask hue survives.
        data = prepare.book_snapshot(self._order_book(), level="L3")
        fig = plot("book_snapshot", Level.L3, **data)
        ax = fig.axes[0]
        edges = [c.patches[0].get_edgecolor() for c in ax.containers if len(c.patches)]
        assert edges  # at least one side drawn
        for rgba in edges:
            assert np.allclose(rgba[:3], 1.0, atol=0.01)

    def test_aggregate_has_no_separators(self) -> None:
        # L2 bars never get a per-order edge (edgecolor "none" -> alpha 0).
        data = prepare.book_snapshot(self._order_book(), level="L2")
        fig = plot("book_snapshot", Level.L2, **data)
        ax = fig.axes[0]
        drawn_sides = [c for c in ax.containers if len(c.patches)]
        assert drawn_sides  # at least one side drawn
        for container in drawn_sides:
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
        for level in (Level.L2, Level.L3):
            data = prepare.book_snapshot(book, level=level)
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
        data = prepare.book_snapshot(book, top_n=5, level="L2")
        assert data["bids"]["price"].nunique() == 5
        assert data["asks"]["price"].nunique() == 5
        # Best-first: bids keep the five highest prices, asks the five lowest.
        assert data["bids"]["price"].min() == 96.0
        assert data["asks"]["price"].max() == 110.0

    def test_quantiles_off_by_default(self) -> None:
        import inspect

        sig = inspect.signature(prepare.book_snapshot)
        assert sig.parameters["show_quantiles"].default is False
        data = prepare.book_snapshot(self._order_book(), level="L2")
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
        data = prepare.book_snapshot(book, show_quantiles=True, top_n=None, level="L2")
        assert len(data["bid_quantiles"]) <= 3
        assert len(data["ask_quantiles"]) <= 3


class TestPlotVolumePercentiles:
    def test_returns_figure(self, plot_depth_summary):
        fig = plot(
            "volume_percentiles",
            **prepare.volume_percentiles(plot_depth_summary),
        )
        # Twenty depth bands a side.
        assert drawn(fig)["fillbetweenpolycollection"] == 40


# The kinds of mark each backend draws a line and markers with.
_LINE = {"matplotlib": "line", "plotly": "scatter lines"}
_MARKERS = {"matplotlib": "pathcollection", "plotly": "scatter markers"}


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
class TestPlotHiddenExecutions:
    def test_draws_each_hidden_execution_over_the_trade_price(
        self, backend, plot_trades, plot_hidden_executions
    ):
        pytest.importorskip(backend)
        data = prepare.hidden_executions(plot_hidden_executions, plot_trades)
        fig = plot("hidden_executions", backend=backend, **data)
        assert drawn(fig) == {
            _LINE[backend]: len(plot_trades),
            _MARKERS[backend]: len(plot_hidden_executions),
        }

    def test_without_hidden_executions_draws_the_trade_price_only(
        self, backend, plot_trades, plot_hidden_executions
    ):
        pytest.importorskip(backend)
        events = plot_hidden_executions.assign(raw_event_type=1)
        data = prepare.hidden_executions(events, plot_trades)
        fig = plot("hidden_executions", backend=backend, **data)
        assert drawn(fig) == {_LINE[backend]: len(plot_trades)}


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
class TestPlotTradingHalts:
    def test_shades_each_halt_over_the_trade_price(self, backend, plot_trades):
        pytest.importorskip(backend)
        halts = pd.DataFrame({"timestamp": plot_trades["timestamp"].iloc[[2]]})
        fig = plot(
            "trading_halts",
            backend=backend,
            **prepare.trading_halts(plot_trades, halts),
        )
        shade = "rectangle" if backend == "matplotlib" else "shape"
        assert drawn(fig) == {_LINE[backend]: len(plot_trades), shade: 1}

    def test_without_halts_draws_the_trade_price_only(self, backend, plot_trades):
        pytest.importorskip(backend)
        fig = plot(
            "trading_halts", backend=backend, **prepare.trading_halts(plot_trades)
        )
        assert drawn(fig) == {_LINE[backend]: len(plot_trades)}


class TestPlotEventsHistogram:
    def test_returns_figure(self, plot_events):
        fig = plot(
            "events_histogram",
            **prepare.events_histogram(plot_events, val="price", bw=0.25),
        )
        # One histogram for the bids and one for the asks.
        assert drawn(fig) == {"fillbetweenpolycollection": 2}

    def test_accepts_ax(self, plot_events):
        fig_orig, ax_orig = plt.subplots()
        fig = plot(
            "events_histogram",
            ax=ax_orig,
            **prepare.events_histogram(plot_events, val="price", bw=0.25),
        )
        assert fig is fig_orig

    def test_empty_window_draws_no_data(self, plot_events):
        # A window before the data: seaborn's histplot used to raise
        # ValueError while working out bin edges for zero rows.
        data = prepare.events_histogram(
            plot_events,
            start_time=pd.Timestamp("2015-05-01 00:00:00"),
            end_time=pd.Timestamp("2015-05-01 00:30:00"),
            val="price",
            bw=0.25,
        )
        ax = plot("events_histogram", **data).axes[0]
        # The theme places titles on the left; read every slot.
        title = " ".join(ax.get_title(loc=s) for s in ("left", "center", "right"))
        assert "no data" in title


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

    def _depth(self, plot_events):
        depth = plot_events[["timestamp", "price", "volume"]].copy()
        depth["direction"] = "bid"
        return depth

    @pytest.mark.parametrize("col_bias", [1.0, 0.5, 0.1, 0.0, -1.0])
    def test_renders_for_each_bias(self, plot_events, col_bias):
        depth = self._depth(plot_events)
        fig = plot(
            "depth_heatmap",
            **prepare.depth_heatmap(depth, col_bias=col_bias),
        )
        assert drawn(fig)["linecollection"] > 0

    def test_default_bias_is_linear(self):
        # The library default is the hero linear look (col_bias=1.0).
        import inspect

        sig = inspect.signature(prepare.depth_heatmap)
        assert sig.parameters["col_bias"].default == 1.0

    def test_midprice_overlay_is_step(self, plot_events):
        # The midprice holds until the book changes, like the best bid/ask
        # step lines it replaces under show_mp; no ramps between samples.
        depth = self._depth(plot_events)
        spread = pd.DataFrame(
            {
                "timestamp": depth["timestamp"],
                "best_bid_price": depth["price"] - 0.05,
                "best_ask_price": depth["price"] + 0.05,
            }
        )
        fig = plot(
            "depth_heatmap",
            **prepare.depth_heatmap(depth, spread=spread, show_mp=True),
        )
        mid = next(ln for ln in fig.axes[0].get_lines() if ln.get_label() == "Midprice")
        assert mid.get_drawstyle() == "steps-post"


# ---------------------------------------------------------------------------
# Hidden-liquidity overlay (#272): iceberg refills + trades against hidden
# orders, on the depth heatmap and the L3 order-activity map.
# ---------------------------------------------------------------------------


class TestPrepareHiddenLiquidityOverlay:
    @staticmethod
    def _overlay(iceberg_events_and_trades, hidden_trade_rows, **window):
        from ob_analytics.hidden_liquidity import iceberg_slices

        slices = iceberg_slices(*iceberg_events_and_trades)
        return prepare.hidden_liquidity_overlay(slices, hidden_trade_rows, **window)

    def test_lines_need_two_surviving_slices(
        self, iceberg_events_and_trades, hidden_trade_rows
    ):
        out = self._overlay(iceberg_events_and_trades, hidden_trade_rows)
        assert len(out["iceberg_lines"]) == 2
        assert sorted(out["iceberg_refills"]["slice"]) == [2]

    def test_price_window_drops_icebergs_outside_it(
        self, iceberg_events_and_trades, hidden_trade_rows
    ):
        out = self._overlay(
            iceberg_events_and_trades,
            hidden_trade_rows,
            price_from=300.0,
            price_to=400.0,
        )
        assert out["iceberg_lines"].empty
        assert out["iceberg_refills"].empty

    def test_time_window_clips_slices(
        self, iceberg_events_and_trades, hidden_trade_rows
    ):
        events, _ = iceberg_events_and_trades
        ts = events["timestamp"].iloc[0]

        out = self._overlay(
            iceberg_events_and_trades,
            hidden_trade_rows,
            start_time=ts,
            end_time=ts + pd.Timedelta(milliseconds=50),
        )
        # Only the first slice survives -- no line, no refill.
        assert out["iceberg_lines"].empty
        assert out["iceberg_refills"].empty

    def test_no_icebergs_or_hidden_trades_is_safe(self, hidden_trade_rows):
        empty_slices = pd.DataFrame(
            columns=[
                "iceberg",
                "slice",
                "id",
                "timestamp",
                "volume",
                "fill",
                "delay_s",
                "direction",
                "price",
                "confidence",
            ]
        )
        out = prepare.hidden_liquidity_overlay(
            empty_slices, hidden_trade_rows.iloc[0:0]
        )
        assert out["iceberg_lines"].empty
        assert out["iceberg_refills"].empty
        assert out["hidden_trades"].empty


class TestPriceLevelsHiddenLiquidityOverlay:
    """Depth heatmap draws the #272 overlay when given, and skips it when not."""

    def _depth(self, plot_events):
        depth = plot_events[["timestamp", "price", "volume"]].copy()
        depth["direction"] = "bid"
        return depth

    def test_no_overlay_draws_no_extra_legend_entries(self, plot_events):
        depth = self._depth(plot_events)
        fig = plot("depth_heatmap", **prepare.depth_heatmap(depth))
        labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
        assert "Iceberg refill" not in labels
        assert "Hidden-order trade" not in labels

    def test_iceberg_and_hidden_trade_overlay_draws_and_legends(
        self, plot_events, hidden_liquidity_overlay
    ):
        overlay = hidden_liquidity_overlay

        depth = self._depth(plot_events)
        fig = plot(
            "depth_heatmap",
            **prepare.depth_heatmap(
                depth,
                iceberg_lines=overlay["iceberg_lines"],
                iceberg_refills=overlay["iceberg_refills"],
                hidden_trades=overlay["hidden_trades"],
            ),
        )
        labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
        assert {
            "Iceberg chain",
            "Iceberg refill",
            "Hidden-order trade",
        } <= labels


class TestPriceLevelsNotice:
    """The depth heatmap says why it is empty, or that no level changed (#303).

    Unchanged levels are left out by default, so a quiet book used to leave
    every level out and draw blank axes with no reason given.
    """

    T0 = pd.Timestamp("2026-09-01 12:00", tz="UTC")

    @classmethod
    def _quiet_depth(cls) -> pd.DataFrame:
        # One row per level at the opening book and nothing after it: the
        # shape of a short capture of a market whose book never moved.
        return pd.DataFrame(
            {
                "timestamp": [cls.T0] * 3,
                "price": [0.04, 0.05, 0.09],
                "volume": [30.0, 12.0, 306.0],
                "direction": ["bid", "bid", "ask"],
            }
        )

    @classmethod
    def _one_change(cls) -> pd.DataFrame:
        # Level 0.04 grows after 30s; the other two never change.
        moved = pd.DataFrame(
            {
                "timestamp": [cls.T0 + pd.Timedelta(seconds=30)],
                "price": [0.04],
                "volume": [40.0],
                "direction": ["bid"],
            }
        )
        return pd.concat([cls._quiet_depth(), moved], ignore_index=True)

    @classmethod
    def _spread(cls) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [cls.T0, cls.T0 + pd.Timedelta(seconds=60)],
                "best_bid_price": [0.05, 0.05],
                "best_bid_vol": [12.0, 12.0],
                "best_ask_price": [0.09, 0.09],
                "best_ask_vol": [306.0, 306.0],
            }
        )

    def _prepare(self, depth: pd.DataFrame, **kwargs: Any) -> dict:
        kwargs.setdefault("end_time", self.T0 + pd.Timedelta(seconds=60))
        return prepare.depth_heatmap(depth, **kwargs)

    def test_no_level_changed_draws_every_level(self):
        data = self._prepare(self._quiet_depth())
        assert set(data["depth"]["price"]) == {0.04, 0.05, 0.09}
        assert data["notice"] == (
            "No price level changed in this window, so every level is drawn."
        )

    def test_unchanged_levels_left_out_when_one_changes(self):
        data = self._prepare(self._one_change())
        assert set(data["depth"]["price"]) == {0.04}
        assert data["notice"] is None

    def test_show_all_depth_keeps_every_level(self):
        data = self._prepare(self._one_change(), show_all_depth=True)
        assert set(data["depth"]["price"]) == {0.04, 0.05, 0.09}
        assert data["notice"] is None

    def test_empty_depth_frame_says_so(self):
        data = prepare.depth_heatmap(self._quiet_depth().iloc[0:0])
        assert data["depth"].empty
        assert data["notice"] == _data.NO_DEPTH_NOTICE

    def test_price_range_with_no_level_names_the_range(self):
        data = self._prepare(self._quiet_depth(), price_from=0.5)
        assert data["notice"] == "No price level at or above 0.5."
        data = self._prepare(self._quiet_depth(), price_from=0.1, price_to=0.2)
        assert data["notice"] == "No price level between 0.1 and 0.2."

    def test_spread_outside_window_sets_no_price_range(self):
        # The spread's only row is before the window, so it gives no price
        # bounds; NaN bounds used to drop every level.
        spread = self._spread()
        spread["timestamp"] = self.T0 - pd.Timedelta(hours=1)
        data = self._prepare(self._quiet_depth(), spread=spread.iloc[:1])
        assert set(data["depth"]["price"]) == {0.04, 0.05, 0.09}
        assert data["notice"] == (
            "No price level changed in this window, so every level is drawn."
        )

    def test_volume_range_with_no_level_says_so(self):
        data = self._prepare(self._quiet_depth(), volume_from=1000)
        assert data["notice"] == "No price level in the chosen volume range."

    def test_time_window_with_no_orders_says_so(self):
        data = self._prepare(
            self._quiet_depth(),
            start_time=self.T0 - pd.Timedelta(hours=2),
            end_time=self.T0 - pd.Timedelta(hours=1),
        )
        assert data["notice"] == "No resting orders in this time window."

    def test_notice_is_logged(self):
        from loguru import logger

        messages: list[str] = []
        logger.enable("ob_analytics")
        sink = logger.add(lambda m: messages.append(m.record["message"]))
        try:
            self._prepare(self._quiet_depth(), price_from=0.5)
        finally:
            logger.remove(sink)
            logger.disable("ob_analytics")
        assert messages == ["Depth heatmap: No price level at or above 0.5."]

    def test_empty_window_frames_the_spread(self):
        data = self._prepare(self._quiet_depth(), spread=self._spread(), price_from=0.1)
        assert data["depth"].empty
        assert data["y_range"] == (0.05, 0.09)

    @staticmethod
    def _title_and_notice(ax: Axes) -> tuple[Text, Text]:
        (notice,) = ax.texts
        title = next(
            t
            for t in ax.get_children()
            if isinstance(t, Text) and t.get_text() == "Price Levels Over Time"
        )
        return title, notice

    def test_matplotlib_draws_flat_levels_with_notice(self):
        data = self._prepare(self._quiet_depth())
        ax = plot("depth_heatmap", **data).axes[0]
        assert ax.collections
        title, notice = self._title_and_notice(ax)
        assert title.get_text() == "Price Levels Over Time"
        assert notice.get_text() == data["notice"]

    @pytest.mark.parametrize("width", [12, 6, 4])
    def test_matplotlib_notice_sits_under_the_title(self, width):
        # Between the title and the plot, even in a narrow subplot, where a
        # notice beside the title used to print over it.
        data = self._prepare(self._quiet_depth())
        fig, axes = plt.subplots(2, 2, figsize=(width, 8))
        ax = axes[0, 0]
        plot("depth_heatmap", ax=ax, **data)
        fig.draw_without_rendering()
        title, notice = self._title_and_notice(ax)
        title_box = title.get_window_extent()
        notice_box = notice.get_window_extent()
        plot_box = ax.get_window_extent()
        assert notice_box.y1 <= title_box.y0
        assert notice_box.y0 >= plot_box.y1

    def test_matplotlib_empty_still_draws_midprice(self):
        data = self._prepare(self._quiet_depth(), spread=self._spread(), price_from=0.1)
        ax = plot("depth_heatmap", **data).axes[0]
        _, notice = self._title_and_notice(ax)
        assert notice.get_text() == data["notice"]
        assert [line.get_label() for line in ax.get_lines()] == ["Midprice"]

    def test_matplotlib_hand_built_empty_payload_says_no_depth(self):
        data = self._prepare(self._quiet_depth(), price_from=0.5)
        data["notice"] = None
        ax = plot("depth_heatmap", **data).axes[0]
        _, notice = self._title_and_notice(ax)
        assert notice.get_text() == _data.NO_DEPTH_NOTICE

    def test_matplotlib_draws_a_level_with_one_row(self):
        # A level whose only row in the window is a removal draws no segment,
        # but it no longer blanks the whole heatmap.
        removed = pd.DataFrame(
            {
                "timestamp": [self.T0 + pd.Timedelta(seconds=30)],
                "price": [0.07],
                "volume": [0.0],
                "direction": ["ask"],
            }
        )
        data = self._prepare(pd.concat([self._one_change(), removed]))
        assert data["depth"].groupby("price").size().min() == 1
        ax = plot("depth_heatmap", **data).axes[0]
        assert ax.collections
        assert not ax.texts

    @pytest.mark.parametrize("price_from", [None, 0.5])
    def test_plotly_puts_the_notice_under_the_title(self, price_from):
        # A second title line, on top of the plot area: clear of the data and
        # of the hover toolbar, whether the chart has levels or not.
        pytest.importorskip("plotly")
        data = self._prepare(self._quiet_depth(), price_from=price_from)
        fig = plot("depth_heatmap", backend="plotly", **data)
        title = fig.layout.title
        first, second = title.text.split("<br>")
        assert first == "Price Levels Over Time"
        assert data["notice"] in second
        assert (title.yref, title.y, title.yanchor) == ("paper", 1.0, "bottom")
        assert title.automargin
        assert not fig.layout.annotations

    def test_bokeh_draws_the_notice(self):
        pytest.importorskip("bokeh")
        data = self._prepare(self._quiet_depth())
        fig = plot("depth_heatmap", backend="bokeh", **data)
        assert [getattr(obj, "text", "") for obj in fig.above] == [data["notice"]]


# ---------------------------------------------------------------------------
# Subplot composition (key use-case for ax parameter)
# ---------------------------------------------------------------------------


class TestSubplotComposition:
    def test_two_plots_on_shared_figure(self, plot_trades):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig1 = plot(
            "trade_tape",
            Level.L2,
            ax=ax1,
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        fig2 = plot(
            "trade_tape",
            Level.L2,
            ax=ax2,
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        assert fig1 is fig
        assert fig2 is fig
        assert len(fig.axes) >= 2


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_default_backend_returns_matplotlib_figure(self, plot_trades):
        fig = plot(
            "trade_tape", Level.L2, **prepare.trade_tape(plot_trades, level="L2")
        )
        assert drawn(fig)["pathcollection"] == len(plot_trades)

    def test_explicit_matplotlib_returns_figure(self, plot_trades):
        fig = plot(
            "trade_tape",
            Level.L2,
            backend="matplotlib",
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        assert drawn(fig)["pathcollection"] == len(plot_trades)

    def test_invalid_backend_raises_value_error(self, plot_trades):
        with pytest.raises(UnknownNameError, match="Unknown plot backend"):
            plot(
                "trade_tape",
                backend="nonexistent",
                **prepare.trade_tape(plot_trades, level="L2"),
            )

    def test_plot_dispatch_matplotlib(self, plot_trades):
        """The unified plot() dispatcher renders from already-prepared data."""
        from ob_analytics.visualization import plot, prepare

        fig = plot(
            "trade_tape",
            Level.L2,
            backend="matplotlib",
            **prepare.trade_tape(plot_trades, level="L2"),
        )
        assert drawn(fig)["pathcollection"] == len(plot_trades)

    def test_plot_unknown_backend_raises(self):
        from ob_analytics.visualization import plot

        with pytest.raises(UnknownNameError, match="Unknown plot backend"):
            plot("trade_tape", backend="nope")


class TestRegisterBackend:
    def test_register_and_dispatch(self, plot_trades, tmp_path, restore_registries):
        """A backend module self-registers its renderers; plot() dispatches."""
        from ob_analytics.visualization import (
            plot,
            register_plot_backend,
        )
        from ob_analytics.visualization._renderers import _BACKEND_MODULES

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
            sys.modules.pop("dummy_backend", None)


@pytest.mark.usefixtures("restore_registries")
class TestRendererKinds:
    """A concept is level-less or leveled on every backend, never both (#302)."""

    def test_kind_text(self) -> None:
        from ob_analytics.visualization._renderers import _kind_text

        assert _kind_text([None]) == "level-less"
        assert _kind_text([Level.L3, Level.L2, Level.L3]) == "at L2 and L3"
        with pytest.raises(ValueError, match="No levels"):
            _kind_text([])

    @staticmethod
    def _renderer(data, ax=None):
        return _create_axes(ax)[0]

    @pytest.mark.parametrize(
        ("first", "second"),
        [(Level.L2, None), (None, Level.L2), (Level.L3, None), (None, Level.L3)],
    )
    def test_mixing_kinds_raises(self, first, second):
        from ob_analytics.visualization import RENDERERS

        RENDERERS.register(("kind_probe", first, "matplotlib"), self._renderer)
        with pytest.raises(ValueError, match="already registered"):
            RENDERERS.register(("kind_probe", second, "matplotlib"), self._renderer)
        assert ("kind_probe", second, "matplotlib") not in RENDERERS

    def test_mixing_kinds_across_backends_raises(self):
        from ob_analytics.visualization import RENDERERS

        RENDERERS.register(("kind_probe", Level.L2, "matplotlib"), self._renderer)
        with pytest.raises(ValueError, match="at L2 on 'matplotlib'"):
            RENDERERS.register(("kind_probe", None, "plotly"), self._renderer)
        assert ("kind_probe", None, "plotly") not in RENDERERS

    def test_same_key_and_both_levels_still_register(self):
        from ob_analytics.visualization import RENDERERS

        RENDERERS.register(("kind_probe", Level.L2, "matplotlib"), self._renderer)
        RENDERERS.register(("kind_probe", Level.L2, "matplotlib"), self._renderer)
        RENDERERS.register(("kind_probe", Level.L3, "matplotlib"), self._renderer)
        RENDERERS.register(("kind_probe", Level.L2, "plotly"), self._renderer)
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
            plot("kind_probe")

    def test_placements_can_be_looped_over_while_registering(self):
        from ob_analytics.visualization import RENDERERS

        RENDERERS.register(("kind_probe", Level.L2, "matplotlib"), self._renderer)
        RENDERERS.register(("kind_probe", Level.L3, "matplotlib"), self._renderer)
        for level, _ in RENDERERS.placements("kind_probe"):
            RENDERERS.register(("kind_probe", level, "plotly"), self._renderer)
        assert RENDERERS.placements("kind_probe") == (
            (Level.L2, "matplotlib"),
            (Level.L3, "matplotlib"),
            (Level.L2, "plotly"),
            (Level.L3, "plotly"),
        )

    def test_text_level_is_stored_as_a_level(self):
        from ob_analytics.visualization import RENDERERS

        key = ("kind_probe", "L2", "matplotlib")  # text, not Level.L2
        RENDERERS.register(key, self._renderer)  # ty: ignore[invalid-argument-type]
        ((level, _),) = RENDERERS.placements("kind_probe")
        assert level is Level.L2
        assert isinstance(plot("kind_probe"), Figure)

    @pytest.mark.parametrize(
        "key",
        [
            ("kind_probe", "matplotlib"),  # no level
            ("kind_probe", "matplotlib", None),  # level and backend swapped
            ("kind_probe", "L4", "matplotlib"),  # not a level
            (7, Level.L2, "matplotlib"),  # concept not text
        ],
    )
    def test_malformed_key_raises(self, key):
        from ob_analytics.visualization import RENDERERS

        with pytest.raises(ValueError, match=r"\(concept, level, backend\)"):
            RENDERERS.register(key, self._renderer)
        assert not [k for k in RENDERERS._items if k[0] in ("kind_probe", 7)]

    def test_the_guide_plot_section_runs_as_written(
        self, tiny_bitstamp_orders_csv, tmp_path
    ):
        """Section 3 of extending.md: register, plot, gallery, plot again."""
        text = (Path(__file__).parents[1] / "docs" / "extending.md").read_text()
        section = text[
            text.index("## 3. A new plot") : text.index("## 4. A new metric")
        ]
        blocks = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
        gallery = tmp_path / "gallery"
        replaced = {'"orders.csv"': 0, '"output/gallery/"': 0}
        namespace: dict[str, Any] = {}
        for block in blocks:
            # The custom backend and the leveled face name modules and
            # functions the guide leaves to the reader.
            if "my_pkg" in block or "my_face" in block:
                continue
            for old in replaced:
                replaced[old] += block.count(old)
            block = block.replace('"orders.csv"', repr(str(tiny_bitstamp_orders_csv)))
            block = block.replace('"output/gallery/"', repr(str(gallery)))
            exec(compile(block, "extending.md", "exec"), namespace)  # noqa: S102 - the guide's own code
        assert all(replaced.values()), replaced

        assert (gallery / "matplotlib" / "cumvol.png").exists()
        html = (gallery / "gallery.html").read_text()
        start = html.index('<div class="card-title">Cumulative Volume<')
        end = html.find('<div class="card">', start)
        assert "Not available" not in html[start : end if end != -1 else None]
        trades = namespace["result"].trades
        fig = plot("cumvol", **namespace["prepare_cumvol_data"](trades))
        assert isinstance(fig, Figure)


class TestPlotPriceView:
    """§4.2 spread ribbon + microprice (L2)."""

    def test_returns_figure_with_ribbon(self, plot_depth_summary):
        data = prepare.price_view(plot_depth_summary)
        fig = plot("price_view", Level.L2, **data)
        ax = fig.axes[0]
        assert isinstance(fig, Figure)
        assert ax.collections  # the fill_between ribbon (PolyCollection)
        labels = {ln.get_label() for ln in ax.get_lines()}
        assert "microprice" in labels
        assert ax.get_ylabel() == "Price"


class TestPlotTradeSize:
    """§4.3 trade-size strip (log-x jittered dots by side)."""

    def test_returns_figure_log_x(self, plot_trades):
        data = prepare.trade_size(plot_trades)
        fig = plot("trade_size", Level.L2, **data)
        ax = fig.axes[0]
        assert isinstance(fig, Figure)
        assert ax.get_xscale() == "log"
        assert [t.get_text() for t in ax.get_yticklabels()] == ["sell", "buy"]


def _book_signals_payload(n: int) -> dict[str, Any]:
    """A book-signals payload of *n* rows whose touch OBI flips sign each row."""
    obi = np.where(np.arange(n) % 2 == 0, 0.5, -0.25)
    obi[1] = np.nan
    mid = np.full(n, 100.5)
    return {
        "timestamp": pd.Series(pd.date_range("2020-01-01", periods=n, freq="s")),
        "best_bid_price": mid - 0.5,
        "best_ask_price": mid + 0.5,
        "mid": mid,
        "microprice": mid + 0.1,
        "obi": obi,
        "obi_depth": obi / 2,
        "levels": 3,
    }


def _artist_count(fig: Figure) -> int:
    return sum(len(ax.patches) + len(ax.collections) + len(ax.lines) for ax in fig.axes)


class TestPlotBookSignals:
    """Book signals: the OBI strip costs a fixed number of artists."""

    def test_artist_count_does_not_grow_with_rows(self):
        small = plot("book_signals", backend="matplotlib", **_book_signals_payload(10))
        large = plot(
            "book_signals", backend="matplotlib", **_book_signals_payload(5_000)
        )
        assert _artist_count(large) == _artist_count(small)

    def test_obi_strip_is_buy_above_and_sell_below_zero(self):
        from ob_analytics.visualization._theme import DEFAULT_THEME

        pal = DEFAULT_THEME.palette
        fig = plot("book_signals", backend="matplotlib", **_book_signals_payload(10))
        price_ax, obi_ax = fig.axes
        assert price_ax.get_zorder() > obi_ax.get_zorder()
        fills = obi_ax.collections
        assert len(fills) == 2
        buy, sell = (f.get_paths()[0].vertices[:, 1] for f in fills)
        assert np.allclose(fills[0].get_facecolor()[0][:3], to_rgb(pal.buy))
        assert np.allclose(fills[1].get_facecolor()[0][:3], to_rgb(pal.sell))
        assert buy.max() == pytest.approx(0.5) and buy.min() == pytest.approx(0.0)
        assert sell.min() == pytest.approx(-0.25) and sell.max() == pytest.approx(0.0)


class TestStepEnvelope:
    """The OBI strip keeps each time column's extremes when it has many steps."""

    def test_short_strip_comes_back_unchanged(self):
        x = np.arange(5.0)
        above = np.array([0.1, 0.0, 0.3, 0.0, 0.2])
        below = -above[::-1]
        xs, a, b = _step_envelope(x, above, below, columns=10)
        assert xs is x and a is above and b is below

    def test_columns_hold_the_extremes_of_their_steps(self):
        # Ten steps into two columns: [0, 4.5) shows steps 0-4, [4.5, 9]
        # shows step 4 (still showing at 4.5) through step 9.
        x = np.arange(10.0)
        above = np.array([0.1, 0.9, 0.0, 0.2, 0.4, 0.0, 0.3, 0.0, 0.0, 0.1])
        below = -above
        xs, a, b = _step_envelope(x, above, below, columns=2)
        np.testing.assert_allclose(xs, [0.0, 4.5, 9.0])
        np.testing.assert_allclose(a, [0.9, 0.4, 0.4])
        np.testing.assert_allclose(b, [-0.9, -0.4, -0.4])

    def test_long_strip_reaches_the_same_extremes_with_few_vertices(self):
        fig = plot(
            "book_signals", backend="matplotlib", **_book_signals_payload(20_000)
        )
        buy, sell = fig.axes[1].collections
        buy_y = buy.get_paths()[0].vertices[:, 1]
        sell_y = sell.get_paths()[0].vertices[:, 1]
        assert len(buy_y) < 20_000
        assert buy_y.max() == pytest.approx(0.5)
        assert sell_y.min() == pytest.approx(-0.25)
