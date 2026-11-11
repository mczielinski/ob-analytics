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

from ob_analytics import _registry, metrics
from ob_analytics.protocols import Level
from tests._drawn import drawn


class SpreadWidthMetric:
    """A stand-in third-party metric, defined entirely outside the core."""

    name = "spread_width"
    title = "Spread Width"
    levels: tuple[Level, ...] = (Level.L2, Level.L3)
    needs: tuple[str, ...] = ()

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


def test_registered_metric_is_listed_and_retrievable(restore_registries):
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
    monkeypatch, restore_registries
):
    monkeypatch.setattr(_registry, "entry_points", _fake_entry_points)

    registered = metrics.load_metric_plugins(force=True)

    assert "spread_width" in registered
    assert isinstance(metrics.get_metric("spread_width"), SpreadWidthMetric)


# ---------------------------------------------------------------------------
# Running a registered metric over a run
# ---------------------------------------------------------------------------


def test_result_computes_one_registered_metric(result, restore_registries):
    metrics.register_metric(SpreadWidthMetric())

    frame = result.metric("spread_width")

    expected = (
        result.depth_summary["best_ask_price"] - result.depth_summary["best_bid_price"]
    )
    assert list(frame.columns) == ["timestamp", "spread_width"]
    pd.testing.assert_series_equal(frame["spread_width"], expected, check_names=False)


class HiddenRefillMetric:
    """An L3-only metric: it reads per-order events, absent on an L2 run."""

    name = "hidden_refill"
    title = "Hidden Refill"
    levels: tuple[Level, ...] = (Level.L3,)
    needs: tuple[str, ...] = ()

    def compute(self, result) -> pd.DataFrame:
        return result.events[["timestamp", "id"]].head(1)

    def prepare(self, frame: pd.DataFrame) -> dict:
        return {"series": frame}


def test_result_metrics_runs_every_metric_registered_for_the_run_level(
    result, restore_registries
):
    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    metrics.register_metric(HiddenRefillMetric())

    computed = result.metrics()

    assert sorted(computed) == ["hidden_refill", "spread_width"]


def test_result_metrics_skips_a_metric_that_does_not_apply_to_the_level(
    result, restore_registries
):
    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    metrics.register_metric(HiddenRefillMetric())
    l2_result = dataclasses.replace(result, level=Level.L2)

    computed = l2_result.metrics()

    assert sorted(computed) == ["spread_width"]


# ---------------------------------------------------------------------------
# The metric as a plot: gallery panel + concept listing
# ---------------------------------------------------------------------------


class BrokenMetric:
    """A metric that raises — the third-party plug-in that has a bad day."""

    name = "broken"
    title = "Broken"
    levels: tuple[Level, ...] = (Level.L2, Level.L3)
    needs: tuple[str, ...] = ()

    def compute(self, result) -> pd.DataFrame:
        raise RuntimeError("no data for this run")

    def prepare(self, frame: pd.DataFrame) -> dict:
        return {"series": frame}


def test_gallery_model_carries_a_panel_per_registered_metric(
    result, restore_registries
):
    from ob_analytics.visualization.gallery import build_gallery_model

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())

    model = build_gallery_model(result)

    (panel,) = model.analytics
    assert panel.name == "spread_width"
    assert panel.title == "Spread Width"
    assert panel.plot_name == "spread_width"
    payload = panel.prepare(**panel.prep_kwargs)
    assert list(payload["series"].columns) == ["timestamp", "spread_width"]


def test_a_metric_that_fails_fails_only_its_own_panel(result, restore_registries):
    from ob_analytics.visualization.gallery import build_gallery_model

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    metrics.register_metric(BrokenMetric())

    # Metrics are computed when a panel is prepared, not when the model is
    # built, so the broken one is listed and raises only when drawn.
    model = build_gallery_model(result)
    panels = {panel.name: panel for panel in model.analytics}

    assert sorted(panels) == ["broken", "spread_width"]
    assert panels["spread_width"].prepare(**panels["spread_width"].prep_kwargs)
    with pytest.raises(RuntimeError, match="no data for this run"):
        panels["broken"].prepare(**panels["broken"].prep_kwargs)


def test_a_failing_metric_card_says_why(
    result, restore_registries, tmp_path, monkeypatch
):
    import matplotlib

    matplotlib.use("Agg")
    from ob_analytics.visualization.gallery import generate_gallery

    metrics.METRICS._items.clear()
    metrics.register_metric(BrokenMetric())

    calls = []
    original = BrokenMetric.compute

    def counted(self, result):
        calls.append(1)
        return original(self, result)

    monkeypatch.setattr(BrokenMetric, "compute", counted)
    html = generate_gallery(
        result, tmp_path, view="l2", backends=["matplotlib", "plotly"]
    ).read_text()

    assert html.count("Preparing the data failed: no data for this run") == 2
    assert calls == [1]  # one card, two backends, one attempt


def _mpl_spread_width(data, ax=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    frame = data["series"]
    ax.plot(frame["timestamp"], frame["spread_width"])
    return fig


def test_registered_metric_is_listed_by_available_concepts(result, restore_registries):
    from ob_analytics.visualization import available_concepts

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())

    concepts = available_concepts(result)

    # A metric is level-less: it has no L2/L3 variants to choose between.
    assert concepts["spread_width"] == []


def test_result_plots_a_registered_metric(result, restore_registries):
    import matplotlib

    matplotlib.use("Agg")

    from ob_analytics.visualization import RENDERERS

    metrics.METRICS._items.clear()
    metrics.register_metric(SpreadWidthMetric())
    RENDERERS.register(("spread_width", None, "matplotlib"), _mpl_spread_width)

    fig = result.plot("spread_width")

    assert drawn(fig) == {"line": len(result.depth_summary)}


# ---------------------------------------------------------------------------
# The headline acceptance: a plug-in metric runs and plots, no core edit
# ---------------------------------------------------------------------------


def test_plugin_metric_runs_and_plots_from_the_public_api(
    monkeypatch, result, restore_registries
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    import ob_analytics as ob
    import ob_analytics.visualization  # ob.visualization, below, needs it bound
    from ob_analytics.visualization import RENDERERS

    metrics.METRICS._items.clear()
    monkeypatch.setattr(_registry, "entry_points", _fake_entry_points)
    ob.load_metric_plugins(force=True)
    RENDERERS.register(("spread_width", None, "matplotlib"), _mpl_spread_width)

    assert "spread_width" in ob.list_metrics()
    assert "spread_width" in ob.visualization.available_concepts(result)
    assert not result.metric("spread_width").empty
    assert isinstance(result.plot("spread_width"), Figure)


def test_a_metric_without_needs_is_refused_when_registered(restore_registries):
    # Refused here, so it cannot break every metric a run computes later.
    class OldMetric:
        name = "old"
        title = "Old"
        levels: tuple[Level, ...] = (Level.L2, Level.L3)

        def compute(self, result) -> pd.DataFrame:
            return pd.DataFrame()

        def prepare(self, frame: pd.DataFrame) -> dict:
            return {}

    with pytest.raises(TypeError, match="lacks needs"):
        metrics.register_metric(OldMetric())  # ty: ignore[invalid-argument-type]
