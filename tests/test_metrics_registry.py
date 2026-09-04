"""The metric registry and its plug-in discovery (issue #140).

The headline acceptance: a metric written outside the core is registered under
a name, runs over a run's tables, and shows as a plot -- with no edit to
ob-analytics.  The "third-party package" is stood up here by monkeypatching
``importlib.metadata.entry_points`` so the test stays offline.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from ob_analytics import metrics
from ob_analytics.protocols import Level


class SpreadWidthMetric:
    """A stand-in third-party metric, defined entirely outside the core."""

    name = "spread_width"
    title = "Spread Width"
    levels = (Level.L2, Level.L3)

    def compute(self, result) -> pd.DataFrame:
        summary = result.depth_summary
        return pd.DataFrame(
            {
                "timestamp": summary["timestamp"],
                "spread_width": summary["best_ask_price"] - summary["best_bid_price"],
            }
        )

    def prepare(self, frame: pd.DataFrame) -> dict:
        return {"series": frame}


@pytest.fixture
def _restore_registry():
    """Snapshot / restore the global metric registry around the test."""
    before = dict(metrics.METRICS._items)
    yield
    metrics.METRICS._items.clear()
    metrics.METRICS._items.update(before)


def test_registered_metric_is_listed_and_retrievable(_restore_registry):
    metric = SpreadWidthMetric()
    metrics.register_metric(metric)

    assert metrics.get_metric("spread_width") is metric
    assert "spread_width" in metrics.list_metrics()


class _FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    name = "spread_width"
    value = "tests.test_metrics_registry:SpreadWidthMetric"

    def load(self):
        return SpreadWidthMetric


def _fake_entry_points(*, group: str | None = None):
    if group == metrics.ENTRY_POINT_GROUP:
        return [_FakeEntryPoint()]
    return []


def test_plugin_metric_is_discovered_through_entry_points(
    monkeypatch, _restore_registry
):
    monkeypatch.setattr(metrics, "entry_points", _fake_entry_points)

    registered = metrics.load_metric_plugins(force=True)

    assert "spread_width" in registered
    assert isinstance(metrics.get_metric("spread_width"), SpreadWidthMetric)


# ---------------------------------------------------------------------------
# Running a registered metric over a run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_result(tiny_bitstamp_orders_csv):
    """One tiny L3 run, shared across the run-level tests."""
    from ob_analytics.bitstamp import BitstampSource
    from ob_analytics.pipeline import Pipeline

    return Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))


def test_result_computes_one_registered_metric(tiny_result, _restore_registry):
    metrics.register_metric(SpreadWidthMetric())

    frame = tiny_result.metric("spread_width")

    expected = (
        tiny_result.depth_summary["best_ask_price"]
        - tiny_result.depth_summary["best_bid_price"]
    )
    assert list(frame.columns) == ["timestamp", "spread_width"]
    pd.testing.assert_series_equal(frame["spread_width"], expected, check_names=False)


class HiddenRefillMetric:
    """An L3-only metric: it reads per-order events, absent on an L2 run."""

    name = "hidden_refill"
    title = "Hidden Refill"
    levels = (Level.L3,)

    def compute(self, result) -> pd.DataFrame:
        return result.events[["timestamp", "id"]].head(1)

    def prepare(self, frame: pd.DataFrame) -> dict:
        return {"series": frame}


def test_result_metrics_runs_every_metric_registered_for_the_run_level(
    tiny_result, _restore_registry
):
    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    metrics.register_metric(HiddenRefillMetric())

    computed = tiny_result.metrics()

    assert sorted(computed) == ["hidden_refill", "spread_width"]


def test_result_metrics_skips_a_metric_that_does_not_apply_to_the_level(
    tiny_result, _restore_registry
):
    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    metrics.register_metric(HiddenRefillMetric())
    l2_result = dataclasses.replace(tiny_result, level=Level.L2)

    computed = l2_result.metrics()

    assert sorted(computed) == ["spread_width"]


# ---------------------------------------------------------------------------
# The metric as a plot: gallery panel + concept listing
# ---------------------------------------------------------------------------


class BrokenMetric:
    """A metric that raises — the third-party plug-in that has a bad day."""

    name = "broken"
    title = "Broken"
    levels = (Level.L2, Level.L3)

    def compute(self, result) -> pd.DataFrame:
        raise RuntimeError("no data for this run")

    def prepare(self, frame: pd.DataFrame) -> dict:
        return {"series": frame}


def test_gallery_model_carries_a_panel_per_registered_metric(
    tiny_result, _restore_registry
):
    from ob_analytics.visualization.gallery import build_gallery_model

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())

    model = build_gallery_model(tiny_result)

    (panel,) = model.analytics
    assert panel.name == "spread_width"
    assert panel.title == "Spread Width"
    assert panel.plot_name == "spread_width"
    payload = panel.prepare(**panel.prep_kwargs)
    assert list(payload["series"].columns) == ["timestamp", "spread_width"]


def test_gallery_model_skips_a_metric_that_fails(tiny_result, _restore_registry):
    from ob_analytics.visualization.gallery import build_gallery_model

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    metrics.register_metric(BrokenMetric())

    model = build_gallery_model(tiny_result)

    assert [panel.name for panel in model.analytics] == ["spread_width"]


@pytest.fixture
def _restore_renderers():
    """Snapshot / restore the renderer registry around the test."""
    from ob_analytics.visualization import RENDERERS

    before = dict(RENDERERS._items)
    yield
    RENDERERS._items.clear()
    RENDERERS._items.update(before)


def _mpl_spread_width(data, ax=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    frame = data["series"]
    ax.plot(frame["timestamp"], frame["spread_width"])
    return fig


def test_registered_metric_is_listed_by_available_concepts(
    tiny_result, _restore_registry
):
    from ob_analytics.visualization import available_concepts

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())

    concepts = available_concepts(tiny_result)

    # A metric is level-less: it has no L2/L3 variants to choose between.
    assert concepts["spread_width"] == []


def test_result_plots_a_registered_metric(
    tiny_result, _restore_registry, _restore_renderers
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from ob_analytics.visualization import RENDERERS

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    RENDERERS.register(("spread_width", None, "matplotlib"), _mpl_spread_width)

    fig = tiny_result.plot("spread_width")

    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# The headline acceptance: a plug-in metric runs and plots, no core edit
# ---------------------------------------------------------------------------


def test_plugin_metric_runs_and_plots_from_the_public_api(
    monkeypatch, tiny_result, _restore_registry, _restore_renderers
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    import ob_analytics as ob
    from ob_analytics.visualization import RENDERERS

    metrics.METRICS._items.clear()
    monkeypatch.setattr(metrics, "entry_points", _fake_entry_points)
    ob.load_metric_plugins(force=True)
    RENDERERS.register(("spread_width", None, "matplotlib"), _mpl_spread_width)

    assert "spread_width" in ob.list_metrics()
    assert "spread_width" in ob.visualization.available_concepts(tiny_result)
    assert not tiny_result.metric("spread_width").empty
    assert isinstance(tiny_result.plot("spread_width"), Figure)
