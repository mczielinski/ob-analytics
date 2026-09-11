"""Kalshi through the ccxt source -- no network.

Two things are pinned here.  First, the convention the docs promise: ccxt
reads a Kalshi market as its Yes book, so a No bid at ``p`` is a Yes ask at
``1 - p``, and a taker who bought No sold Yes.  That conversion is ccxt's, so
these tests run the real ccxt class on canned responses and fail if a ccxt
release changes it.  Second, a Kalshi-shaped capture replays at the market's
own tick size.  The most traded Kalshi markets quote in tenths of a cent, which
the default 0.01 grid would round away.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

from ob_analytics import DepthCsvSource, Pipeline
from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import ConfigError
from ob_analytics.live._base import CaptureConfig
from ob_analytics.live._runner import run_capturer
from ob_analytics.live.ccxt_source import CcxtSettings, CcxtSource
from ob_analytics.schemas import validate_depth_df, validate_trades_df
from tests.test_ccxt_source import _FakeCcxtExchange

_PREDICTION_INSTALLED = (
    importlib.util.find_spec("ccxt") is not None
    and importlib.util.find_spec("ccxt.prediction") is not None
)

# ccxt's outcome for a Kalshi market's Yes side.
_YES_OUTCOME = {
    "info": {"ticker": "KXTEST"},
    "label": "YES",
    "outcome": "KXTEST:YES",
    "outcomeId": "KXTEST",
    "market": "KXTEST",
}


@pytest.mark.skipif(
    not _PREDICTION_INSTALLED, reason="ccxt release without ccxt.prediction"
)
class TestYesBook:
    def _kalshi(self, monkeypatch, orderbook_response):
        from ccxt.prediction.kalshi import kalshi

        ex = kalshi()

        async def load_outcome(*_args, **_kwargs):
            return _YES_OUTCOME

        async def get_orderbook(*_args, **_kwargs):
            return orderbook_response

        monkeypatch.setattr(ex, "load_outcome", load_outcome)
        monkeypatch.setattr(ex, "outcome", lambda *_: _YES_OUTCOME)
        monkeypatch.setattr(ex, "kalshiPublicGetMarketsTickerOrderbook", get_orderbook)
        return ex

    def test_no_bids_become_yes_asks(self, monkeypatch):
        ex = self._kalshi(
            monkeypatch,
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.0500", "30.00"], ["0.0400", "12.50"]],
                    "no_dollars": [["0.9100", "306.00"]],
                }
            },
        )

        async def fetch():
            try:
                return await ex.fetch_order_book("KXTEST")
            finally:
                await ex.close()

        book = asyncio.run(fetch())
        assert book["bids"] == [[0.05, 30.0], [0.04, 12.5]]
        assert book["asks"] == [[0.09, 306.0]]  # 1 - 0.91

    def test_taker_buying_no_sold_yes(self):
        from ccxt.prediction.kalshi import kalshi

        trade = kalshi().parse_prediction_trade(
            {
                "trade_id": "t1",
                "created_time": "2026-09-10T23:20:23.313911Z",
                "yes_price_dollars": "0.1200",
                "no_price_dollars": "0.8800",
                "count_fp": "2.00",
                "taker_side": "no",
            },
            _YES_OUTCOME,
        )
        assert trade["price"] == 0.12  # the Yes price
        assert trade["amount"] == 2.0
        assert trade["side"] == "sell"


class _FakeKalshi(_FakeCcxtExchange):
    """A ccxt Kalshi stand-in: polled, with a tenth-of-a-cent tick."""

    id = "kalshi"
    precisionMode = 4  # ccxt.TICK_SIZE

    def outcome(self, symbol):
        return {**_YES_OUTCOME, "precision": {"amount": 1, "price": 0.001}}


class TestDeciCentCapture:
    """A capture from a market quoted in tenths of a cent replays exactly."""

    @pytest.fixture
    def capture(self, tmp_path):
        snapshot = {
            "bids": [[0.036, 28.0], [0.035, 30000.0]],
            "asks": [[0.039, 215.86]],
            "timestamp": 1_000,
        }
        polled = [
            # bid 0.036 grows, bid 0.035 empties, ask 0.040 appears
            {
                "bids": [[0.036, 30.5]],
                "asks": [[0.039, 215.86], [0.04, 9399.51]],
                "timestamp": 2_000,
            }
        ]
        trades = [[{"id": "t1", "timestamp": 2_000, "price": 0.037, "amount": 2.0}]]
        ex = _FakeKalshi(snapshot, polled, trades, ws=False)
        source = CcxtSource(settings=CcxtSettings(exchange=ex, poll_interval=0.0))
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="KXTEST", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(source, cfg))
        return out

    def test_meta_records_the_tick(self, capture):
        meta = json.loads((capture / "meta.json").read_text())
        assert meta["exchange"] == "kalshi"
        assert meta["tick_size"] == 0.001

    def test_default_tick_refuses_instead_of_rounding(self, capture):
        with pytest.raises(ConfigError, match="tick_size"):
            Pipeline.from_source("depth_csv").run(capture)

    def test_recorded_tick_replays_every_price(self, capture):
        config = PipelineConfig(tick_size=0.001, price_decimals=3)
        result = Pipeline(config, source=DepthCsvSource()).run(capture)
        validate_depth_df(result.depth)
        validate_trades_df(result.trades)
        assert sorted(set(result.depth["price"])) == [35, 36, 39, 40]
        assert result.trades["price"].tolist() == [37]

    def test_cli_process_needs_no_flags(self, capture, cli_runner, tmp_path):
        from ob_analytics.data import load_data

        out = tmp_path / "out"
        r = cli_runner(
            "process", str(capture), "--source", "depth_csv", "--output", str(out)
        )
        assert r.returncode == 0, r.stderr
        assert load_data(out)["depth"].attrs["tick_size"] == 0.001
