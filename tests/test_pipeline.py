"""Tests for the Pipeline orchestrator."""


from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ob_analytics.config import PipelineConfig
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import EventLoader, MatchingEngine, TradeInferrer


def _make_full_events() -> pd.DataFrame:
    """Build a realistic events DataFrame that survives the full pipeline."""
    ts = pd.Timestamp("2015-05-01 00:00:00")
    n = 8
    return pd.DataFrame(
        {
            "event_id": list(range(1, n + 1)),
            "id": [100, 200, 100, 200, 300, 400, 300, 400],
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)],
            "exchange_timestamp": [ts + pd.Timedelta(seconds=i - 1) for i in range(n)],
            "price": [236.50, 237.00, 236.50, 237.00, 236.60, 237.10, 236.60, 237.10],
            "volume": [10000, 5000, 8000, 3000, 7000, 4000, 5000, 2000],
            "action": pd.Categorical(
                ["created", "created", "changed", "changed", "created", "created", "changed", "changed"],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "direction": pd.Categorical(
                ["bid", "ask", "bid", "ask", "bid", "ask", "bid", "ask"],
                categories=["bid", "ask"],
                ordered=True,
            ),
            "fill": [0.0, 0.0, 2000.0, 2000.0, 0.0, 0.0, 2000.0, 2000.0],
            "original_number": list(range(1, n + 1)),
            "matching_event": [np.nan, np.nan, 4.0, 3.0, np.nan, np.nan, 8.0, 7.0],
            "type": pd.Categorical(
                ["resting-limit", "resting-limit", "resting-limit", "resting-limit",
                 "resting-limit", "resting-limit", "resting-limit", "resting-limit"],
                categories=["unknown", "flashed-limit", "resting-limit", "market-limit", "pacman", "market"],
                ordered=True,
            ),
        }
    )


def _make_trades() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 00:00:02")
    return pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(seconds=4)],
            "price": [236.50, 236.60],
            "volume": [2000.0, 2000.0],
            "direction": ["sell", "sell"],
            "maker_event_id": [3, 7],
            "taker_event_id": [4, 8],
            "maker": [100, 300],
            "taker": [200, 400],
            "maker_og": [3, 7],
            "taker_og": [4, 8],
        }
    )


class TestPipelineResult:
    def test_frozen(self):
        result = PipelineResult(
            events=pd.DataFrame(),
            trades=pd.DataFrame(),
            depth=pd.DataFrame(),
            depth_summary=pd.DataFrame(),
        )
        with pytest.raises(AttributeError):
            result.events = pd.DataFrame({"new": [1]})


class TestPipelineWithMocks:
    def test_run_calls_all_components(self):
        events = _make_full_events()
        trades = _make_trades()

        mock_loader = MagicMock(spec=EventLoader)
        mock_loader.load.return_value = events

        mock_matcher = MagicMock(spec=MatchingEngine)
        mock_matcher.match.return_value = events

        mock_inferrer = MagicMock(spec=TradeInferrer)
        mock_inferrer.infer_trades.return_value = trades

        pipeline = Pipeline(
            loader=mock_loader,
            matcher=mock_matcher,
            trade_inferrer=mock_inferrer,
        )
        result = pipeline.run("dummy.csv")

        mock_loader.load.assert_called_once_with("dummy.csv")
        mock_matcher.match.assert_called_once()
        mock_inferrer.infer_trades.assert_called_once()

        assert isinstance(result, PipelineResult)
        assert isinstance(result.depth, pd.DataFrame)
        assert isinstance(result.depth_summary, pd.DataFrame)

    def test_default_components_created(self):
        pipeline = Pipeline()
        assert pipeline.config is not None
        assert pipeline.loader is not None
        assert pipeline.matcher is not None
        assert pipeline.trade_inferrer is not None

    def test_custom_config_propagated(self):
        config = PipelineConfig(match_cutoff_ms=100)
        pipeline = Pipeline(config=config)
        assert pipeline.config.match_cutoff_ms == 100


class TestPipelineEndToEnd:
    def test_full_pipeline_with_sample_data(self, sample_csv_path: Path):
        """Smoke test: run the real pipeline on sample data."""
        result = Pipeline().run(sample_csv_path)
        assert isinstance(result, PipelineResult)
        assert len(result.events) > 0
        assert len(result.trades) > 0
        assert len(result.depth) > 0
        assert len(result.depth_summary) > 0
