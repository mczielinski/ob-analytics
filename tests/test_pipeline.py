"""Tests for the Pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ob_analytics._utils import empty_trades
from ob_analytics.config import PipelineConfig
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import EventLoader, TradeReader
from ob_analytics.schemas import TRADE_DTYPES, cast_columns, cast_events


def _make_full_events() -> pd.DataFrame:
    """Build a realistic events DataFrame that survives the full pipeline."""
    ts = pd.Timestamp("2015-05-01 00:00:00", tz="UTC")
    n = 8
    events = pd.DataFrame(
        {
            "event_id": list(range(1, n + 1)),
            "id": [100, 200, 100, 200, 300, 400, 300, 400],
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)],
            "exchange_timestamp": [ts + pd.Timedelta(seconds=i - 1) for i in range(n)],
            "price": [23650, 23700, 23650, 23700, 23660, 23710, 23660, 23710],
            "volume": [10000, 5000, 8000, 3000, 7000, 4000, 5000, 2000],
            "action": [
                "created",
                "created",
                "changed",
                "changed",
                "created",
                "created",
                "changed",
                "changed",
            ],
            "direction": ["bid", "ask", "bid", "ask", "bid", "ask", "bid", "ask"],
            "fill": [0, 0, 2000, 2000, 0, 0, 2000, 2000],
            "original_number": list(range(1, n + 1)),
            "type": np.repeat("unknown", n),
        }
    )
    return cast_events(events)


def _make_trades() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 00:00:02", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": [ts, ts + pd.Timedelta(seconds=4)],
            "price": [23650, 23660],
            "volume": [2000, 2000],
            "direction": ["sell", "sell"],
            "maker_event_id": [3, 7],
            "taker_event_id": [4, 8],
            "maker": [100, 300],
            "taker": [200, 400],
            "maker_og": [3, 7],
            "taker_og": [4, 8],
        }
    )
    return cast_columns(trades, TRADE_DTYPES)


class TestPipelineResult:
    def test_frozen(self):
        result = PipelineResult(
            events=pd.DataFrame(),
            trades=pd.DataFrame(),
            depth=pd.DataFrame(),
            depth_summary=pd.DataFrame(),
            config=PipelineConfig(),
        )
        with pytest.raises(AttributeError):
            result.events = pd.DataFrame({"new": [1]})  # ty: ignore[invalid-assignment]


class TestPipelineWithMocks:
    def test_run_calls_all_components(self):
        events = _make_full_events()
        trades = _make_trades()

        mock_loader = MagicMock(spec=EventLoader)
        mock_loader.load.return_value = events

        mock_trade_reader = MagicMock(spec=TradeReader)
        mock_trade_reader.load.return_value = trades

        pipeline = Pipeline(
            loader=mock_loader,
            trade_reader=mock_trade_reader,
        )
        result = pipeline.run("dummy.csv")

        mock_loader.load.assert_called_once_with("dummy.csv")
        mock_trade_reader.load.assert_called_once()
        call_events, call_source = mock_trade_reader.load.call_args[0]
        assert len(call_events) == len(events)
        assert call_source == "dummy.csv"

        assert isinstance(result, PipelineResult)
        assert isinstance(result.depth, pd.DataFrame)
        assert isinstance(result.depth_summary, pd.DataFrame)

    def test_default_config_created(self):
        pipeline = Pipeline()
        assert pipeline.config == PipelineConfig()

    def test_custom_config_propagated(self):
        config = PipelineConfig(depth_bps=50)
        pipeline = Pipeline(config=config)
        assert pipeline.config.depth_bps == 50


class TestPipelineFlow:
    def test_pipeline_uses_trade_reader(self, tmp_path: Path):
        """Verify Pipeline.run() invokes the injected TradeReader.

        Uses a 4-event minimal orders.csv rather than the bundled 22 MB
        capture: this test only checks dependency-injection wiring, not
        real data flow, so there's no value in paying the full-pipeline
        cost (~5 min on the bundled sample).
        """
        from ob_analytics import Pipeline
        from ob_analytics.bitstamp import BitstampLoader

        orders = pd.DataFrame(
            [
                {
                    "id": 10,
                    "timestamp": 100,
                    "exchange_timestamp": 100,
                    "price": 100.0,
                    "volume": 1.0,
                    "action": "created",
                    "direction": "ask",
                },
                {
                    "id": 10,
                    "timestamp": 200,
                    "exchange_timestamp": 200,
                    "price": 100.0,
                    "volume": 0.0,
                    "action": "deleted",
                    "direction": "ask",
                },
                {
                    "id": 11,
                    "timestamp": 200,
                    "exchange_timestamp": 200,
                    "price": 99.0,
                    "volume": 0.5,
                    "action": "created",
                    "direction": "bid",
                },
                {
                    "id": 11,
                    "timestamp": 300,
                    "exchange_timestamp": 300,
                    "price": 99.0,
                    "volume": 0.0,
                    "action": "deleted",
                    "direction": "bid",
                },
            ]
        )
        orders.to_csv(tmp_path / "orders.csv", index=False)

        captured = {}

        class StubTradeReader:
            def load(self, events, path):
                captured["called"] = True
                captured["n_events"] = len(events)
                return empty_trades()

        result = Pipeline(
            loader=BitstampLoader(),
            trade_reader=StubTradeReader(),
        ).run(tmp_path / "orders.csv")

        assert captured["called"]
        assert captured["n_events"] > 0
        assert len(result.trades) == 0

    def test_pipeline_no_longer_accepts_matcher_or_inferrer(self):
        import pytest

        from ob_analytics import Pipeline

        with pytest.raises(TypeError):
            Pipeline(matcher=object())  # ty: ignore[unknown-argument]
        with pytest.raises(TypeError):
            Pipeline(trade_inferrer=object())  # ty: ignore[unknown-argument]


class TestPipelineEndToEnd:
    def test_full_pipeline_tiny_bitstamp_dir(self, tmp_path: Path):
        """Fast smoke: real loader + trade reader on a minimal directory."""
        orders = pd.DataFrame(
            [
                {
                    "id": 10,
                    "timestamp": 100,
                    "exchange_timestamp": 100,
                    "price": 100.0,
                    "volume": 1.0,
                    "action": "created",
                    "direction": "ask",
                },
                {
                    "id": 10,
                    "timestamp": 200,
                    "exchange_timestamp": 200,
                    "price": 100.0,
                    "volume": 0.5,
                    "action": "changed",
                    "direction": "ask",
                },
                {
                    "id": 11,
                    "timestamp": 200,
                    "exchange_timestamp": 200,
                    "price": 200.0,
                    "volume": 0.5,
                    "action": "created",
                    "direction": "bid",
                },
                {
                    "id": 11,
                    "timestamp": 200,
                    "exchange_timestamp": 200,
                    "price": 100.0,
                    "volume": 0.0,
                    "action": "deleted",
                    "direction": "bid",
                },
            ]
        )
        orders.to_csv(tmp_path / "orders.csv", index=False)
        trades = pd.DataFrame(
            [
                {
                    "trade_id": 999,
                    "timestamp": 200,
                    "exchange_timestamp": 200,
                    "price": 100.0,
                    "amount": 0.5,
                    "buy_order_id": 11,
                    "sell_order_id": 10,
                    "side": "buy",
                },
            ]
        )
        trades.to_csv(tmp_path / "trades.csv", index=False)

        result = Pipeline().run(tmp_path / "orders.csv")
        assert len(result.events) == 4
        assert len(result.trades) == 1
        assert len(result.depth) > 0


def test_pipeline_result_is_slim(tmp_path: Path):
    """PipelineResult carries only the core tables plus config, level and
    declarations.

    Analytic outputs (VPIN/OFI/Kyle's λ) and the old metadata/extras/metrics
    bags are gone — compute them post-pipeline from ``result.trades``.
    ``level`` records which order-book level the run was produced at (L2 vs L3),
    and ``declarations`` what the run's data is declared to show.
    """
    assert set(PipelineResult.__dataclass_fields__) == {
        "events",
        "trades",
        "depth",
        "depth_summary",
        "config",
        "level",
        "declarations",
    }

    orders = pd.DataFrame(
        [
            {
                "id": 10,
                "timestamp": 100,
                "exchange_timestamp": 100,
                "price": 100.0,
                "volume": 1.0,
                "action": "created",
                "direction": "ask",
            },
            {
                "id": 10,
                "timestamp": 200,
                "exchange_timestamp": 200,
                "price": 100.0,
                "volume": 0.0,
                "action": "deleted",
                "direction": "ask",
            },
            {
                "id": 11,
                "timestamp": 200,
                "exchange_timestamp": 200,
                "price": 99.0,
                "volume": 0.5,
                "action": "created",
                "direction": "bid",
            },
            {
                "id": 11,
                "timestamp": 300,
                "exchange_timestamp": 300,
                "price": 99.0,
                "volume": 0.0,
                "action": "deleted",
                "direction": "bid",
            },
        ]
    )
    orders.to_csv(tmp_path / "orders.csv", index=False)
    # BitstampTradeReader requires a companion trades.csv; a header-only file
    # exercises the no-trades path (load() returns empty_trades()), which is
    # all this shape test needs.
    pd.DataFrame(
        columns=[
            "trade_id",
            "timestamp",
            "exchange_timestamp",
            "price",
            "amount",
            "buy_order_id",
            "sell_order_id",
            "side",
        ]
    ).to_csv(tmp_path / "trades.csv", index=False)

    res = Pipeline().run(tmp_path / "orders.csv")
    assert isinstance(res, PipelineResult)
    assert res.config is not None
