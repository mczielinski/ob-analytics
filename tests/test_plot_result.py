"""Tests for the result-level plotting API (WS-5.1 / 5.2)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from ob_analytics.bitstamp import BitstampSource
from ob_analytics.exceptions import ConfigError, UnknownNameError
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.visualization import available_concepts, plot_result, prepare


@pytest.fixture(scope="module")
def result(tiny_bitstamp_orders_csv) -> PipelineResult:
    """One tiny pipeline run shared across the module."""
    return Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotResult:
    def test_returns_matplotlib_figure(self, result: PipelineResult) -> None:
        assert isinstance(plot_result(result, "depth_heatmap"), Figure)

    def test_explicit_level(self, result: PipelineResult) -> None:
        # A comparable concept renders at either resolution.
        assert isinstance(plot_result(result, "trade_tape", "L2"), Figure)
        assert isinstance(plot_result(result, "trade_tape", "L3"), Figure)

    def test_one_level_resolves(self, result: PipelineResult) -> None:
        # An L2-only and an L3-only concept need no level.
        assert isinstance(plot_result(result, "depth_heatmap"), Figure)
        assert isinstance(plot_result(result, "order_outcome"), Figure)

    def test_two_levels_require_level(self, result: PipelineResult) -> None:
        # This L3 run has the trade tape at L2 and at L3: no default.
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
            plot_result(result, "trade_tape")

    def test_draws_on_ax(self, result: PipelineResult) -> None:
        fig, ax = plt.subplots()
        assert plot_result(result, "trade_tape", "L2", ax=ax) is fig

    def test_ax_needs_matplotlib(self, result: PipelineResult) -> None:
        _, ax = plt.subplots()
        with pytest.raises(ConfigError, match="ax= is a Matplotlib axes"):
            plot_result(result, "depth_heatmap", backend="plotly", ax=ax)

    def test_a_metric_takes_no_level(self, result: PipelineResult) -> None:
        with pytest.raises(UnknownNameError, match="level-less"):
            plot_result(result, "roll_spread", "L2")

    def test_override_flows_to_prepare(self, result: PipelineResult) -> None:
        # A prepare kwarg passes through (col_bias is real since §3.5).
        assert isinstance(plot_result(result, "depth_heatmap", col_bias=0.1), Figure)

    def test_plotly_backend(self, result: PipelineResult) -> None:
        pytest.importorskip("plotly.graph_objects")
        import plotly.graph_objects as go

        fig = plot_result(result, "cancellations", "L3", backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_unknown_concept_raises(self, result: PipelineResult) -> None:
        with pytest.raises(UnknownNameError, match="Unknown concept") as info:
            plot_result(result, "does_not_exist")
        assert "not built for this result" not in str(info.value)

    def test_a_registered_concept_the_result_lacks_says_so(
        self, result: PipelineResult, monkeypatch
    ) -> None:
        from ob_analytics.visualization import RENDERERS

        key = ("probe_not_built", None, "matplotlib")
        monkeypatch.setitem(RENDERERS._items, key, lambda data, ax=None: None)
        with pytest.raises(UnknownNameError, match="It is not built for this result.$"):
            plot_result(result, "probe_not_built")

    def test_missing_level_raises(self, result: PipelineResult) -> None:
        # order_outcome is L3-only; asking for L2 lists what is available.
        with pytest.raises(
            UnknownNameError, match="Unknown L2 concept 'order_outcome'.*drawn at L3"
        ):
            plot_result(result, "order_outcome", "L2")


class TestPipelineResultPlot:
    def test_method_delegates(self, result: PipelineResult) -> None:
        assert isinstance(result.plot("depth_heatmap"), Figure)

    def test_method_passes_overrides(self, result: PipelineResult) -> None:
        assert isinstance(result.plot("trade_tape", "L3", volume_scale=1.0), Figure)

    def test_method_requires_level_like_plot_result(
        self, result: PipelineResult
    ) -> None:
        with pytest.raises(ConfigError, match='pass level="L2" or level="L3"'):
            result.plot("book_snapshot")

    def test_method_draws_on_ax(self, result: PipelineResult) -> None:
        fig, ax = plt.subplots()
        assert result.plot("trade_tape", "L3", ax=ax) is fig


class TestQueuePositionFace:
    """The §4.1b queue_position.L3 face (FIFO rank-vs-time at the touch)."""

    def test_renders_both_backends(self, result: PipelineResult) -> None:
        pytest.importorskip("plotly.graph_objects")
        import plotly.graph_objects as go

        assert isinstance(plot_result(result, "queue_position"), Figure)
        assert isinstance(
            plot_result(result, "queue_position", "L3", backend="plotly"), go.Figure
        )

    def test_l3_only(self, result: PipelineResult) -> None:
        assert available_concepts(result)["queue_position"] == ["L3"]

    def test_rank_axis_inverted_front_at_top(self, result: PipelineResult) -> None:
        # rank 1 (front of queue) must sit at the TOP -> inverted y-axis.
        ax = plot_result(result, "queue_position").axes[0]
        lo, hi = ax.get_ylim()
        assert lo > hi  # inverted
        assert "rank" in ax.get_ylabel().lower()

    def test_prepare_schema(self, result: PipelineResult) -> None:
        data = prepare.queue_position(result.events)
        assert set(data) >= {
            "filled",
            "cancelled",
            "resting",
            "max_rank",
            "show_markers",
        }
        for fate in (data["filled"], data["cancelled"], data["resting"]):
            assert {"timestamp", "id", "rank", "age_s"} <= set(fate.columns)


class TestAvailableConcepts:
    def test_lists_concepts_and_levels(self, result: PipelineResult) -> None:
        ac = available_concepts(result)
        assert isinstance(ac, dict)
        # Unconditional order-book concepts are always present.
        assert "depth_heatmap" in ac
        assert ac["trade_tape"] == ["L2", "L3"]  # comparable
        assert ac["order_outcome"] == ["L3"]  # L3-only
        assert ac["queue_position"] == ["L3"]  # L3-only (FIFO queue engine)


class TestPrepareNamespace:
    def test_a_concept_at_both_levels_needs_level(self, result: PipelineResult) -> None:
        with pytest.raises(ConfigError, match='level="L2" or level="L3"'):
            prepare.trade_tape(result.trades)

    def test_a_concept_at_one_level_resolves_it(self, result: PipelineResult) -> None:
        data = prepare.trade_size(result.trades)
        assert set(data) == set(prepare.trade_size(result.trades, level="L2"))

    def test_a_level_the_concept_lacks_is_refused(self, result: PipelineResult) -> None:
        with pytest.raises(ConfigError, match="drawn at L2"):
            prepare.trade_size(result.trades, level="L3")
        with pytest.raises(ConfigError, match="level-less"):
            prepare.amihud(result.trades, level="L2")

    def test_trades_returns_payload(self, result: PipelineResult) -> None:
        data = prepare.trade_tape(result.trades, level="L2")
        assert {"buys", "sells", "mid_line"} <= set(data)

    def test_every_concept_has_a_prepare_function_of_its_name(self) -> None:
        from ob_analytics.visualization import _concepts

        missing = [c.key for c in _concepts.CONCEPTS if not hasattr(prepare, c.key)]
        assert missing == []
