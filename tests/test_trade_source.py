"""Tests for the TradeSource protocol and built-in implementations."""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics.protocols import TradeSource


class TestTradeSourceProtocol:
    def test_protocol_is_runtime_checkable(self):
        class Stub:
            def load(self, events, source):
                return pd.DataFrame()

        assert isinstance(Stub(), TradeSource)

    def test_protocol_rejects_missing_load(self):
        class NoLoad:
            pass

        assert not isinstance(NoLoad(), TradeSource)


class TestBitstampTradeReader:
    @pytest.fixture
    def run_dir(self, tmp_path):
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
        return tmp_path

    def test_reader_emits_one_trade_per_csv_row(self, run_dir):
        from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader

        events = BitstampLoader().load(run_dir / "orders.csv")
        trades = BitstampTradeReader().load(events, run_dir)
        assert len(trades) == 1

    def test_reader_resolves_maker_and_taker_event_ids(self, run_dir):
        from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader

        events = BitstampLoader().load(run_dir / "orders.csv")
        trades = BitstampTradeReader().load(events, run_dir)
        assert trades.loc[0, "maker"] == 10
        assert trades.loc[0, "taker"] == 11
        assert trades.loc[0, "maker_event_id"] in events["event_id"].values
        assert trades.loc[0, "taker_event_id"] in events["event_id"].values

    def test_reader_returns_empty_with_correct_schema_when_no_trades(
        self, tmp_path
    ):
        from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader

        orders = pd.DataFrame(
            [
                dict(
                    id=1,
                    timestamp=100,
                    exchange_timestamp=100,
                    price=10.0,
                    volume=1.0,
                    action="created",
                    direction="bid",
                ),
            ]
        )
        orders.to_csv(tmp_path / "orders.csv", index=False)
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

        events = BitstampLoader().load(tmp_path / "orders.csv")
        trades = BitstampTradeReader().load(events, tmp_path)
        assert len(trades) == 0
        for col in [
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
        ]:
            assert col in trades.columns

    def test_reader_raises_when_trades_csv_missing(self, tmp_path):
        from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader

        orders = pd.DataFrame(
            [
                dict(
                    id=1,
                    timestamp=100,
                    exchange_timestamp=100,
                    price=10.0,
                    volume=1.0,
                    action="created",
                    direction="bid",
                ),
            ]
        )
        orders.to_csv(tmp_path / "orders.csv", index=False)
        events = BitstampLoader().load(tmp_path / "orders.csv")
        with pytest.raises(FileNotFoundError, match="trades.csv"):
            BitstampTradeReader().load(events, tmp_path)


class TestFormatCreateTradeSource:
    def test_bitstamp_format_creates_trade_reader(self):
        from ob_analytics import BitstampFormat, PipelineConfig
        from ob_analytics.bitstamp import BitstampTradeReader

        ts = BitstampFormat().create_trade_source(PipelineConfig())
        assert isinstance(ts, BitstampTradeReader)

    def test_lobster_format_creates_trade_source(self):
        from ob_analytics import LobsterFormat, PipelineConfig

        ts = LobsterFormat(trading_date="2025-01-01").create_trade_source(
            PipelineConfig()
        )
        assert isinstance(ts, TradeSource)
