"""Tests for the Pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ob_analytics.config import PipelineConfig
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import EventLoader, TradeSource


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
                [
                    "created",
                    "created",
                    "changed",
                    "changed",
                    "created",
                    "created",
                    "changed",
                    "changed",
                ],
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
            "type": pd.Categorical(
                np.repeat("unknown", n),
                categories=[
                    "unknown",
                    "flashed-limit",
                    "resting-limit",
                    "market-limit",
                    "pacman",
                    "market",
                ],
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
            "direction": pd.Categorical(
                ["sell", "sell"], categories=["buy", "sell"], ordered=True
            ),
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

        mock_trade_source = MagicMock(spec=TradeSource)
        mock_trade_source.load.return_value = trades

        pipeline = Pipeline(
            loader=mock_loader,
            trade_source=mock_trade_source,
        )
        result = pipeline.run("dummy.csv")

        mock_loader.load.assert_called_once_with("dummy.csv")
        mock_trade_source.load.assert_called_once()
        call_events, call_source = mock_trade_source.load.call_args[0]
        assert len(call_events) == len(events)
        assert call_source == "dummy.csv"

        assert isinstance(result, PipelineResult)
        assert isinstance(result.depth, pd.DataFrame)
        assert isinstance(result.depth_summary, pd.DataFrame)

    def test_default_components_created(self):
        pipeline = Pipeline()
        assert pipeline.config is not None
        assert pipeline.loader is not None
        assert pipeline.trade_source is not None

    def test_custom_config_propagated(self):
        config = PipelineConfig(depth_bps=50)
        pipeline = Pipeline(config=config)
        assert pipeline.config.depth_bps == 50


class TestPipelineFlow:
    def test_pipeline_uses_trade_source(self):
        import pandas as pd
        from ob_analytics import BitstampLoader, Pipeline, sample_csv_path

        captured = {}

        class StubTradeSource:
            def load(self, events, source):
                captured["called"] = True
                captured["n_events"] = len(events)
                return pd.DataFrame(
                    columns=[
                        "timestamp",
                        "price",
                        "volume",
                        "direction",
                        "maker_event_id",
                        "taker_event_id",
                        "maker",
                        "taker",
                        "maker_og",
                        "taker_og",
                    ]
                )

        result = Pipeline(
            loader=BitstampLoader(),
            trade_source=StubTradeSource(),
        ).run(sample_csv_path())

        assert captured["called"]
        assert captured["n_events"] > 0
        assert len(result.trades) == 0

    def test_pipeline_no_longer_accepts_matcher_or_inferrer(self):
        import pytest
        from ob_analytics import Pipeline

        with pytest.raises(TypeError):
            Pipeline(matcher=object())  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            Pipeline(trade_inferrer=object())  # type: ignore[call-arg]


class TestPipelineEndToEnd:
    def test_full_pipeline_tiny_bitstamp_dir(self, tmp_path: Path):
        """Fast smoke: real loader + trade reader on a minimal directory."""
        orders = pd.DataFrame(
            [
                dict(
                    id=10,
                    timestamp=100,
                    exchange_timestamp=100,
                    price=100.0,
                    volume=1.0,
                    action="created",
                    direction="ask",
                ),
                dict(
                    id=10,
                    timestamp=200,
                    exchange_timestamp=200,
                    price=100.0,
                    volume=0.5,
                    action="changed",
                    direction="ask",
                ),
                dict(
                    id=11,
                    timestamp=200,
                    exchange_timestamp=200,
                    price=200.0,
                    volume=0.5,
                    action="created",
                    direction="bid",
                ),
                dict(
                    id=11,
                    timestamp=200,
                    exchange_timestamp=200,
                    price=100.0,
                    volume=0.0,
                    action="deleted",
                    direction="bid",
                ),
            ]
        )
        orders.to_csv(tmp_path / "orders.csv", index=False)
        trades = pd.DataFrame(
            [
                dict(
                    trade_id=999,
                    timestamp=200,
                    exchange_timestamp=200,
                    price=100.0,
                    amount=0.5,
                    buy_order_id=11,
                    sell_order_id=10,
                    side="buy",
                ),
            ]
        )
        trades.to_csv(tmp_path / "trades.csv", index=False)

        result = Pipeline().run(tmp_path / "orders.csv")
        assert len(result.events) == 4
        assert len(result.trades) == 1
        assert len(result.depth) > 0

    @pytest.mark.slow
    def test_full_pipeline_with_sample_data(self, sample_csv_path: Path):
        """Smoke test: run the real pipeline on the bundled capture."""
        result = Pipeline().run(sample_csv_path)
        assert isinstance(result, PipelineResult)
        assert len(result.events) > 0
        assert len(result.trades) > 0
        assert len(result.depth) > 0
        assert len(result.depth_summary) > 0
