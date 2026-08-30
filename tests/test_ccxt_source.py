"""Tests for the CCXT L2 capturer (issue #106) -- no network.

A fake CCXT exchange feeds scripted books/trades so the capturer's
translation (snapshot, book diffing, trade mapping) and the L2 capture ->
replay round-trip are exercised deterministically, without ccxt or a socket.
"""

from __future__ import annotations

import asyncio
import importlib.util

import pandas as pd
import pytest

from ob_analytics.live._base import CaptureConfig
from ob_analytics.live._runner import run_capturer
from ob_analytics.live.ccxt_source import CcxtSettings, CcxtSource, _epoch_ms_to_ts
from ob_analytics.protocols import Level

_CCXT_INSTALLED = importlib.util.find_spec("ccxt") is not None


class _FakeCcxtExchange:
    """Duck-typed stand-in for a ccxt.pro exchange.

    ``watch_*`` / ``fetch_*`` return scripted payloads and raise
    ``StopAsyncIteration`` once exhausted (the capturer treats that as a
    closed feed; real feeds block instead).
    """

    id = "fake"

    def __init__(
        self,
        snapshot_book: dict,
        ws_books: list[dict] | None = None,
        ws_trades: list[list[dict]] | None = None,
        *,
        ws: bool = True,
    ) -> None:
        self._snapshot = snapshot_book
        self._books = list(ws_books or [])
        self._trades = list(ws_trades or [])
        self._snapshot_served = False
        self.has = {
            "watchOrderBook": ws,
            "watchTrades": ws,
            "fetchOrderBook": True,
            "fetchTrades": True,
        }
        self.closed = False

    async def fetch_order_book(self, symbol, limit=None):
        # First call is the snapshot seed; later calls (the REST poll loop)
        # serve subsequent books, then signal a closed feed.
        if not self._snapshot_served:
            self._snapshot_served = True
            return self._snapshot
        if not self._books:
            raise StopAsyncIteration
        return self._books.pop(0)

    async def watch_order_book(self, symbol, limit=None):
        if not self._books:
            raise StopAsyncIteration
        return self._books.pop(0)

    async def watch_trades(self, symbol):
        if not self._trades:
            raise StopAsyncIteration
        return self._trades.pop(0)

    async def fetch_trades(self, symbol, since=None):
        if not self._trades:
            raise StopAsyncIteration
        return self._trades.pop(0)

    async def close(self):
        self.closed = True


def _cfg(tmp_path) -> CaptureConfig:
    return CaptureConfig(pair="BTC/USDT", out_dir=tmp_path / "cap", minutes=0.05)


def _source(exchange, **knobs) -> CcxtSource:
    """A CcxtSource with the venue + knobs as typed settings (was extras dict)."""
    return CcxtSource(settings=CcxtSettings(exchange=exchange, **knobs))


async def _collect_snapshot(cap, cfg):
    return [ev async for ev in cap.snapshot(cfg)]


# ---------------------------------------------------------------------------
# Protocol + config
# ---------------------------------------------------------------------------


class TestConformance:
    def test_is_l2_livesource(self):
        from ob_analytics.live import LiveSource

        cap = CcxtSource()
        assert isinstance(cap, LiveSource)
        assert cap.level is Level.L2

    def test_missing_exchange_errors(self, tmp_path):
        cap = CcxtSource()  # empty settings -> no exchange
        cfg = CaptureConfig(pair="X", out_dir=tmp_path)
        with pytest.raises(ValueError, match="exchange"):
            asyncio.run(_collect_snapshot(cap, cfg))

    def test_rest_only_venue_selects_poll(self, tmp_path):
        ex = _FakeCcxtExchange({"bids": [], "asks": [], "timestamp": 0}, ws=False)
        cap = _source(ex)
        asyncio.run(_collect_snapshot(cap, _cfg(tmp_path)))
        assert cap._use_ws_book is False
        assert cap._use_ws_trades is False

    def test_ws_venue_selects_websocket(self, tmp_path):
        ex = _FakeCcxtExchange({"bids": [], "asks": [], "timestamp": 0}, ws=True)
        cap = _source(ex)
        asyncio.run(_collect_snapshot(cap, _cfg(tmp_path)))
        assert cap._use_ws_book is True
        assert cap._use_ws_trades is True


# ---------------------------------------------------------------------------
# Pure translation
# ---------------------------------------------------------------------------


class TestDiffBook:
    def test_changes_additions_and_removals(self):
        cap = CcxtSource()
        cap._last = {"bid": {100.0: 5.0, 99.0: 3.0}, "ask": {101.0: 4.0}}
        book = {
            "bids": [[100.0, 7.0], [98.0, 2.0]],  # 100 changed, 98 new, 99 gone
            "asks": [[101.0, 4.0]],  # unchanged
            "timestamp": 0,
        }
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        rows = [r for r, _raw in cap._diff_book(book, ts)]
        got = {(r["side"], r["price"], r["volume"]) for r in rows}
        assert got == {
            ("bid", 100.0, 7.0),  # size change
            ("bid", 98.0, 2.0),  # new level
            ("bid", 99.0, 0.0),  # removal
        }
        # Baseline advanced to the new book.
        assert cap._last["bid"] == {100.0: 7.0, 98.0: 2.0}
        assert cap._last["ask"] == {101.0: 4.0}

    def test_unchanged_book_emits_nothing(self):
        cap = CcxtSource()
        cap._last = {"bid": {100.0: 5.0}, "ask": {101.0: 4.0}}
        book = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 0}
        rows = list(cap._diff_book(book, pd.Timestamp("2025-01-01", tz="UTC")))
        assert rows == []

    def test_raw_frame_attached_once(self):
        cap = CcxtSource()
        cap._last = {"bid": {}, "ask": {}}
        book = {"bids": [[100.0, 1.0], [99.0, 1.0]], "asks": [[101.0, 1.0]], "ts": 0}
        raws = [raw for _r, raw in cap._diff_book(book, pd.Timestamp.now(tz="UTC"))]
        # Exactly one row carries the raw book; the rest are None.
        assert sum(raw is not None for raw in raws) == 1


class TestMapTrade:
    def test_maps_ccxt_trade(self):
        cap = CcxtSource()
        ev = cap._map_trade(
            {
                "id": "abc",
                "timestamp": 1_700_000_000_000,
                "price": 100.5,
                "amount": 0.5,
                "side": "buy",
            }
        )
        assert ev["trade_id"] == "abc"
        assert ev["price"] == 100.5
        assert ev["amount"] == 0.5
        assert ev["side"] == "buy"
        assert ev["buy_order_id"] == ""  # public trades carry no order IDs
        assert ev["exchange_timestamp"] == _epoch_ms_to_ts(1_700_000_000_000)


# ---------------------------------------------------------------------------
# Snapshot + full capture
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_yields_absolute_levels_and_seeds_baseline(self, tmp_path):
        snap = {
            "bids": [[100.0, 5.0], [99.0, 3.0]],
            "asks": [[101.0, 4.0]],
            "timestamp": 1_700_000_000_000,
        }
        ex = _FakeCcxtExchange(snap)
        cap = _source(ex)
        rows = asyncio.run(_collect_snapshot(cap, _cfg(tmp_path)))
        assert len(rows) == 3
        assert cap._last["bid"] == {100.0: 5.0, 99.0: 3.0}
        assert cap._last["ask"] == {101.0: 4.0}


class TestFullCapture:
    def test_capture_writes_depth_and_replays_through_l2(self, tmp_path):
        from ob_analytics.depth_l2 import L2DepthLoader

        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        ws_books = [
            # bid 100 grows 5 -> 7; ask 101 unchanged -> 1 depth row
            {"bids": [[100.0, 7.0]], "asks": [[101.0, 4.0]], "timestamp": 2_000},
        ]
        ws_trades = [
            [
                {
                    "id": "t1",
                    "timestamp": 2_000,
                    "price": 100.5,
                    "amount": 0.5,
                    "side": "buy",
                }
            ]
        ]
        ex = _FakeCcxtExchange(snap, ws_books, ws_trades)
        cap = _source(ex)
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="BTC/USDT", out_dir=out, minutes=0.05)
        result = asyncio.run(run_capturer(cap, cfg))

        assert (out / "depth.csv").exists()
        assert (out / "trades.csv").exists()
        assert not (out / "orders.csv").exists()
        # snapshot (2 levels) + stream (1 changed level) = 3 depth rows
        assert result.n_depth_events == 3
        assert result.n_trade_events == 1
        assert result.n_order_events == 0
        assert ex.closed is True

        depth = L2DepthLoader().load(out / "depth.csv")
        assert len(depth) == 3
        assert set(depth["direction"].dropna().unique()) <= {"bid", "ask"}

    def test_diagnostics_land_in_meta(self, tmp_path):
        import json

        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        ex = _FakeCcxtExchange(snap, [], [])
        cap = _source(ex)
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="BTC/USDT", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(cap, cfg))
        meta = json.loads((out / "meta.json").read_text())
        assert meta["exchange"] == "fake"
        assert meta["n_depth_events"] == 2


class TestRestPoll:
    """REST-poll transport -- venues without CCXT Pro websockets (e.g. Kalshi /
    Polymarket) fetch_* on a poll loop, de-duped by a `since` cursor."""

    def test_rest_poll_capture_replays_through_l2(self, tmp_path):
        from ob_analytics.depth_l2 import L2DepthLoader

        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        poll_books = [
            # bid 100 grows 5 -> 6 on the next poll -> 1 depth row
            {"bids": [[100.0, 6.0]], "asks": [[101.0, 4.0]], "timestamp": 2_000},
        ]
        poll_trades = [
            [
                {
                    "id": "t1",
                    "timestamp": 2_000,
                    "price": 100.5,
                    "amount": 0.3,
                    "side": "sell",
                }
            ]
        ]
        ex = _FakeCcxtExchange(snap, poll_books, poll_trades, ws=False)
        cap = _source(ex, poll_interval=0.0)
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="BTC/USDT", out_dir=out, minutes=0.05)
        result = asyncio.run(run_capturer(cap, cfg))

        assert cap._use_ws_book is False  # the REST poll path was exercised
        assert (out / "depth.csv").exists()
        assert not (out / "orders.csv").exists()
        # snapshot (2 levels) + poll (1 changed level) = 3 depth rows
        assert result.n_depth_events == 3
        assert result.n_trade_events == 1

        depth = L2DepthLoader().load(out / "depth.csv")
        assert len(depth) == 3

    def test_rest_poll_dedupes_trades_via_since(self, tmp_path):
        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        # The second poll re-serves t1 (as REST tape endpoints do) and adds t2.
        t1 = {
            "id": "t1",
            "timestamp": 2_000,
            "price": 100.0,
            "amount": 1.0,
            "side": "buy",
        }
        t2 = {
            "id": "t2",
            "timestamp": 3_000,
            "price": 100.0,
            "amount": 1.0,
            "side": "buy",
        }
        ex = _FakeCcxtExchange(snap, [], [[t1], [t1, t2]], ws=False)
        cap = _source(ex, poll_interval=0.0)
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="X", out_dir=out, minutes=0.05)
        result = asyncio.run(run_capturer(cap, cfg))
        # t1 counted once (not re-emitted on the second poll) + t2 = 2.
        assert result.n_trade_events == 2


# ---------------------------------------------------------------------------
# ccxt-dependent (skipped without the extra)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CCXT_INSTALLED, reason="ccxt extra not installed")
class TestCcxtInstalled:
    def test_unknown_exchange_raises(self):
        from ob_analytics.live.ccxt_source import _make_exchange

        with pytest.raises(ValueError, match="Unknown CCXT exchange"):
            _make_exchange("not_a_real_exchange_xyz")

    def test_registered_when_ccxt_present(self):
        from ob_analytics.sources import list_sources

        assert "ccxt" in list_sources()
