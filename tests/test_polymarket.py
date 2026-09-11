"""Polymarket through the ccxt source -- no network.

Polymarket publishes one order book per outcome token, and prices each token's
trades in that token, so a capture of one outcome is consistent on its own.
These tests pin that ccxt passes a token's book and trades through unchanged:
no ``1 - p`` conversion, unlike ccxt's Kalshi class.  They also check that a
capture whose tick becomes finer part-way through replays every price exactly.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

from ob_analytics.live._base import CaptureConfig
from ob_analytics.live._runner import run_capturer
from ob_analytics.live.ccxt_source import CcxtSettings, CcxtSource
from tests.test_ccxt_source import _FakeCcxtExchange

_PREDICTION_INSTALLED = (
    importlib.util.find_spec("ccxt") is not None
    and importlib.util.find_spec("ccxt.prediction") is not None
)

# ccxt's outcome for one Polymarket token.
_OUTCOME = {
    "outcomeId": "123",
    "outcome": "FED_HIKE:YES",
    "label": "Yes",
    "market": "FED_HIKE",
}


@pytest.mark.skipif(
    not _PREDICTION_INSTALLED, reason="ccxt release without ccxt.prediction"
)
class TestTokenBook:
    def test_plain_id_reaches_polymarket(self):
        import ccxt.prediction

        from ob_analytics.live.ccxt_source import _make_exchange

        assert isinstance(_make_exchange("polymarket"), ccxt.prediction.polymarket)

    def test_book_keeps_the_token_prices(self, monkeypatch):
        from ccxt.prediction.polymarket import polymarket

        ex = polymarket()

        async def load_outcome(*_args, **_kwargs):
            return _OUTCOME

        async def get_book(*_args, **_kwargs):
            return {
                "asset_id": "123",
                "timestamp": "1789086598130",
                "bids": [
                    {"price": "0.62", "size": "24160.34"},
                    {"price": "0.63", "size": "7083.72"},
                ],
                "asks": [{"price": "0.64", "size": "16150.19"}],
                "tick_size": "0.01",
            }

        monkeypatch.setattr(ex, "load_outcome", load_outcome)
        monkeypatch.setattr(ex, "clobPublicGetBook", get_book)

        async def fetch():
            try:
                return await ex.fetch_order_book("FED_HIKE:YES")
            finally:
                await ex.close()

        book = asyncio.run(fetch())
        # Best first on each side, and no conversion of the prices.
        assert book["bids"] == [[0.63, 7083.72], [0.62, 24160.34]]
        assert book["asks"] == [[0.64, 16150.19]]
        assert book["timestamp"] == 1_789_086_598_130

    def test_trade_keeps_its_price_and_side(self):
        from ccxt.prediction.polymarket import polymarket

        trade = polymarket().parse_prediction_trade(
            {
                "transactionHash": "0xabc",
                "timestamp": 1789086057,
                "price": 0.36,
                "size": 71.39,
                "side": "SELL",
                "asset": "123",
            },
            _OUTCOME,
        )
        assert trade["price"] == 0.36
        assert trade["amount"] == 71.39
        assert trade["side"] == "sell"
        assert trade["timestamp"] == 1_789_086_057_000


class _FakePolymarket(_FakeCcxtExchange):
    """A ccxt Polymarket stand-in: streamed, with a 0.01 tick in its metadata."""

    id = "polymarket"
    precisionMode = 4  # ccxt.TICK_SIZE

    def outcome(self, symbol):
        return {**_OUTCOME, "precision": {"amount": 0.01, "price": 0.01}}


class TestTickBecomesFiner:
    """The price nears 0, Polymarket's tick drops to 0.001, and the file keeps up."""

    @pytest.fixture
    def capture(self, tmp_path):
        snapshot = {
            "bids": [[0.04, 100.0]],
            "asks": [[0.05, 80.0]],
            "timestamp": 1_000,
        }
        streamed = [
            {"bids": [[0.039, 20.0], [0.04, 100.0]], "asks": [[0.05, 80.0]]},
        ]
        trades = [[{"id": "t1", "timestamp": 2_000, "price": 0.039, "amount": 5.0}]]
        ex = _FakePolymarket(snapshot, streamed, trades, ws=True)
        source = CcxtSource(settings=CcxtSettings(exchange=ex))
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="123", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(source, cfg))
        return out

    def test_meta_records_the_finer_tick(self, capture):
        meta = json.loads((capture / "meta.json").read_text())
        assert meta["exchange"] == "polymarket"
        assert meta["tick_size"] == 0.001
        assert meta["tick_size_changes"] == 1

    def test_cli_process_replays_every_price(self, capture, cli_runner, tmp_path):
        from ob_analytics.data import load_data

        out = tmp_path / "out"
        r = cli_runner(
            "process", str(capture), "--source", "depth_csv", "--output", str(out)
        )
        assert r.returncode == 0, r.stderr
        data = load_data(out)
        assert data["depth"].attrs["tick_size"] == 0.001
        assert sorted(set(data["depth"]["price"])) == [39, 40, 50]
        assert data["trades"]["price"].tolist() == [39]
