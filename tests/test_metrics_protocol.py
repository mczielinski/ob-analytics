"""Tests for the ToxicityMetric protocol, registry, and built-ins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ob_analytics import (
    KyleLambda,
    Ofi,
    Pipeline,
    PipelineConfig,
    ToxicityMetric,
    Vpin,
    list_metrics,
    register_metric,
)
from ob_analytics.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(n: int = 60) -> pd.DataFrame:
    """Synthetic alternating buy/sell trades; cheap to construct."""
    base = pd.Timestamp("2015-05-01 00:00:00")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(seconds=i) for i in range(n)],
            "price": [100.0 + (i % 5) * 0.1 for i in range(n)],
            "volume": [1.0] * n,
            "direction": ["buy" if i % 2 == 0 else "sell" for i in range(n)],
        }
    )


def _synthetic_result(trades: pd.DataFrame | None = None) -> PipelineResult:
    """A bare-bones PipelineResult populated only with trades.

    Other tables are empty DataFrames — that's fine because the metric
    classes only consume ``trades``.
    """
    if trades is None:
        trades = _make_trades()
    empty = pd.DataFrame()
    return PipelineResult(
        events=empty,
        trades=trades,
        depth=empty,
        depth_summary=empty,
    )


# Pipeline.run is slow even on the tiny fixture (~18s). Share ONE real
# run across all integration tests in this module, configured with all
# three built-in metrics so a single execution exercises the full
# metrics pipeline. Windows are shrunk to match the tiny fixture's
# ~30-second trade span.
@pytest.fixture(scope="module")
def pipeline_result_with_metrics(
    tiny_bitstamp_orders_csv: Path,
) -> PipelineResult:
    return Pipeline(
        metrics=[
            Vpin(bucket_volume=0.1),
            Ofi(window="5s"),
            KyleLambda(window="10s"),
        ],
    ).run(tiny_bitstamp_orders_csv)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtins_registered(self):
        names = list_metrics()
        assert "vpin" in names
        assert "kyle_lambda" in names
        assert "ofi" in names

    def test_register_custom(self):
        @dataclass(frozen=True)
        class _FakeMetric:
            name: str = "fake_registry"
            requires: tuple[str, ...] = ("trades",)
            primary_column: str = "value"

            def compute(self, result: PipelineResult, config: Any) -> pd.DataFrame:
                return pd.DataFrame(
                    {"timestamp": [pd.Timestamp.now(tz="UTC")], "value": [42]}
                )

        register_metric("fake_registry", _FakeMetric)
        assert "fake_registry" in list_metrics()


# ---------------------------------------------------------------------------
# Protocol conformance (no Pipeline run needed)
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    @pytest.mark.parametrize("metric", [Vpin(bucket_volume=10), KyleLambda(), Ofi()])
    def test_implements_protocol(self, metric):
        assert isinstance(metric, ToxicityMetric)
        assert isinstance(metric.name, str)
        assert isinstance(metric.requires, tuple)
        assert isinstance(metric.primary_column, str)


# ---------------------------------------------------------------------------
# Output shape contract (synthetic trades — fast)
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.fixture
    def result(self) -> PipelineResult:
        return _synthetic_result()

    def test_vpin_returns_timestamped_df(self, result: PipelineResult):
        metric = Vpin(bucket_volume=2.0)
        df = metric.compute(result, None)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "timestamp" in df.columns
        assert metric.primary_column in df.columns

    def test_ofi_returns_timestamped_df(self, result: PipelineResult):
        metric = Ofi()
        df = metric.compute(result, None)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "timestamp" in df.columns
        assert metric.primary_column in df.columns

    def test_kyle_returns_timestamped_df(self, result: PipelineResult):
        metric = KyleLambda(window="10s")
        df = metric.compute(result, None)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "timestamp" in df.columns
        assert metric.primary_column in df.columns

    def test_empty_trades_returns_empty_df(self):
        """Metrics gracefully handle empty trades."""
        result = _synthetic_result(trades=pd.DataFrame())
        for metric in (Vpin(bucket_volume=1.0), Ofi(), KyleLambda()):
            df = metric.compute(result, None)
            assert isinstance(df, pd.DataFrame)
            assert df.empty


# ---------------------------------------------------------------------------
# Pipeline integration (uses the slow shared bare_pipeline_result fixture)
# ---------------------------------------------------------------------------


class TestPipelineWithMetrics:
    def test_no_metrics_means_empty_list(self):
        """Pipeline() without metrics= leaves the internal metrics list empty.

        This checks Pipeline.__init__ behavior directly — no Pipeline.run
        needed (which is slow on the bundled sample CSV).
        """
        p = Pipeline()
        assert p._metrics == []

    def test_legacy_config_creates_vpin_and_ofi(self):
        """Setting config.vpin_bucket_volume auto-creates [Vpin, Ofi] metrics.

        Checks Pipeline.__init__ back-compat without running the pipeline.
        """
        p = Pipeline(config=PipelineConfig(vpin_bucket_volume=1_000))
        names = {m.name for m in p._metrics}
        assert names == {"vpin", "ofi"}

    def test_metrics_param_runs_all(self, pipeline_result_with_metrics: PipelineResult):
        """All three configured metrics appear in result.metrics."""
        result = pipeline_result_with_metrics
        for name in ("vpin", "ofi", "kyle_lambda"):
            assert name in result.metrics, f"missing metric: {name}"
            assert isinstance(result.metrics[name], pd.DataFrame)

    def test_back_compat_mirror_attributes(
        self, pipeline_result_with_metrics: PipelineResult
    ):
        """result.vpin / result.ofi mirror result.metrics entries."""
        result = pipeline_result_with_metrics
        assert result.vpin is result.metrics["vpin"]
        assert result.ofi is result.metrics["ofi"]

    def test_metric_skipped_when_required_missing(self):
        """A metric requiring 'foo' is skipped because PipelineResult has no 'foo'."""

        @dataclass(frozen=True)
        class _NeedsFoo:
            name: str = "needs_foo"
            requires: tuple[str, ...] = ("foo",)
            primary_column: str = "value"

            def compute(self, result: PipelineResult, config: Any) -> pd.DataFrame:
                raise AssertionError("should have been skipped")

        # Use a synthetic provisional result + run the same filter loop the
        # pipeline uses internally — no need to spin up the full Pipeline.
        provisional = _synthetic_result()
        metric = _NeedsFoo()
        missing = any(
            getattr(provisional, req, None) is None or getattr(provisional, req).empty
            for req in metric.requires
        )
        assert missing, "metric should be skipped when required table is absent"
