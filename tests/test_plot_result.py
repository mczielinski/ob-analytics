"""Tests for the result-level plotting API (WS-5.1 / 5.2)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from ob_analytics.bitstamp import BitstampFormat
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.visualization import available_concepts, plot_result, prepare


@pytest.fixture(scope="module")
def result(tiny_bitstamp_orders_csv) -> PipelineResult:
    """One tiny pipeline run shared across the module."""
    return Pipeline(format=BitstampFormat()).run(str(tiny_bitstamp_orders_csv))


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

    def test_level_none_resolves(self, result: PipelineResult) -> None:
        # Comparable concept (both levels) and an L3-only concept both resolve.
        assert isinstance(plot_result(result, "trade_tape"), Figure)
        assert isinstance(plot_result(result, "order_outcome"), Figure)

    def test_override_flows_to_prepare(self, result: PipelineResult) -> None:
        # A prepare kwarg passes through (col_bias is real since §3.5).
        assert isinstance(plot_result(result, "depth_heatmap", col_bias=0.1), Figure)

    def test_plotly_backend(self, result: PipelineResult) -> None:
        pytest.importorskip("plotly.graph_objects")
        import plotly.graph_objects as go

        fig = plot_result(result, "cancellations", "L3", backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_unknown_concept_raises(self, result: PipelineResult) -> None:
        with pytest.raises(KeyError, match="Unknown concept"):
            plot_result(result, "does_not_exist")

    def test_missing_level_raises(self, result: PipelineResult) -> None:
        # order_outcome is L3-only; asking for L2 lists what is available.
        with pytest.raises(KeyError, match="no L2 variant"):
            plot_result(result, "order_outcome", "L2")


class TestPipelineResultPlot:
    def test_method_delegates(self, result: PipelineResult) -> None:
        assert isinstance(result.plot("depth_heatmap"), Figure)

    def test_method_passes_overrides(self, result: PipelineResult) -> None:
        assert isinstance(result.plot("trade_tape", "L3", volume_scale=1.0), Figure)


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
        data = prepare.queue_position_l3(result.events)
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
    def test_short_names_are_callable(self) -> None:
        # The public namespace drops the prepare_/_data ceremony.
        for name in ("trades", "price_levels", "trade_tape_l3", "order_outcome_l3"):
            assert callable(getattr(prepare, name))

    def test_trades_returns_payload(self, result: PipelineResult) -> None:
        data = prepare.trades(result.trades)
        assert {"buys", "sells", "mid_line"} <= set(data)

    def test_is_thin_reexport_of_impl(self) -> None:
        # Same function object as the private impl, just a friendlier name.
        from ob_analytics.visualization import _data

        assert prepare.trades is _data.prepare_trades_data
