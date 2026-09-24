"""Tests for the CCXT L2 capturer (issue #106) -- no network.

A fake CCXT exchange feeds scripted books/trades so the capturer's
translation (snapshot, book diffing, trade mapping) and the L2 capture ->
replay round-trip are exercised deterministically, without ccxt or a socket.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from typing import Any

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
        self.options: dict = {}
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

    def test_failed_snapshot_closes_the_exchange(self, tmp_path):
        class _Unresolvable(_FakeCcxtExchange):
            async def fetch_order_book(self, symbol, limit=None):
                raise ValueError("could not resolve outcome")

        ex = _Unresolvable({"bids": [], "asks": [], "timestamp": 0})
        with pytest.raises(ValueError, match="resolve"):
            asyncio.run(_collect_snapshot(_source(ex), _cfg(tmp_path)))
        assert ex.closed is True


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
        # Exactly one row carries the raw frame; the rest are None.
        assert sum(raw is not None for raw in raws) == 1

    def test_raw_frame_holds_only_the_changed_levels(self):
        # A deep book (Coinbase BTC/USD holds ~40k levels) with one level
        # changed and one removed: the raw frame follows the update, not the
        # book, so raw.jsonl stays near the size of depth.csv.
        cap = CcxtSource()
        cap._full_book_written = True  # past the first frame
        bids = [[50_000.0 - i * 0.01, 1.0] for i in range(20_000)]
        asks = [[50_001.0 + i * 0.01, 1.0] for i in range(20_000)]
        cap._last = {
            "bid": {p: s for p, s in bids},
            "ask": {p: s for p, s in asks},
        }
        book = {
            "symbol": "BTC/USD",
            "bids": [[bids[0][0], 2.5], *bids[1:]],  # top bid resized
            "asks": asks[1:],  # top ask removed
            "timestamp": 1_700_000_000_000,
            "datetime": "2023-11-14T22:13:20.000Z",
            "nonce": 42,
        }
        raws = [raw for _r, raw in cap._diff_book(book, pd.Timestamp.now(tz="UTC"))]
        frame = next(raw for raw in raws if raw is not None)
        assert frame == {
            "symbol": "BTC/USD",
            "frame": "changes",
            "timestamp": 1_700_000_000_000,
            "datetime": "2023-11-14T22:13:20.000Z",
            "nonce": 42,
            "bids": [[50_000.0, 2.5]],
            "asks": [[50_001.0, 0.0]],
        }
        assert len(json.dumps(frame)) < 500

    def test_first_frame_is_the_whole_book_and_changes_rebuild_the_rest(self):
        cap = CcxtSource()
        cap._last = {"bid": {100.0: 1.0}, "ask": {101.0: 1.0}}
        books = [
            {"bids": [[100.0, 1.0]], "asks": [[101.0, 1.0]], "nonce": 1},  # no change
            {"bids": [[100.0, 2.0], [99.0, 1.0]], "asks": [[101.0, 1.0]], "nonce": 2},
            {"bids": [[99.0, 1.0]], "asks": [[101.0, 3.0]], "nonce": 3},
        ]
        ts = pd.Timestamp.now(tz="UTC")
        frames = [raw for book in books for _r, raw in cap._diff_book(book, ts) if raw]
        # An unchanged book writes nothing; the next one writes the whole book.
        assert frames[0] == {"frame": "book", **books[1]}
        assert [f["frame"] for f in frames[1:]] == ["changes"]
        # Replaying raw.jsonl alone gives the venue's last book.
        rebuilt = {
            side: {p: s for p, s in frames[0][side]} for side in ("bids", "asks")
        }
        for f in frames[1:]:
            for side in ("bids", "asks"):
                for price, size in f[side]:
                    if size:
                        rebuilt[side][price] = size
                    else:
                        rebuilt[side].pop(price, None)
        assert rebuilt == {"bids": {99.0: 1.0}, "asks": {101.0: 3.0}}


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

    def test_rest_poll_drops_trades_before_the_opening_book(self, tmp_path):
        # The first poll of a REST tape returns the venue's recent history.
        # A trade older than the opening book would otherwise be stamped with
        # the capture's receive time, as if it had just happened.
        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 5_000}
        old = {"id": "old", "timestamp": 1_000, "price": 100.0, "amount": 1.0}
        new = {"id": "new", "timestamp": 6_000, "price": 100.0, "amount": 1.0}
        ex = _FakeCcxtExchange(snap, [], [[old, new]], ws=False)
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="X", out_dir=out, minutes=0.05)
        result = asyncio.run(run_capturer(_source(ex, poll_interval=0.0), cfg))
        assert result.n_trade_events == 1
        assert pd.read_csv(out / "trades.csv")["trade_id"].tolist() == ["new"]


_WS_SNAPSHOT = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}


class TestWebsocketTrades:
    def _capture(self, trades, tmp_path):
        import json

        ex = _FakeCcxtExchange(_WS_SNAPSHOT, [], trades, ws=True)
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="X", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(_source(ex), cfg))
        meta = json.loads((out / "meta.json").read_text())
        return pd.read_csv(out / "trades.csv"), meta

    def test_a_trade_delivered_twice_is_written_once(self, tmp_path):
        # ccxt's Polymarket websocket handed back an earlier trade alongside
        # the next new one.
        t1 = {"id": "a", "timestamp": 2_000, "price": 100.0, "amount": 1.0}
        t2 = {"id": "b", "timestamp": 3_000, "price": 101.0, "amount": 2.0}
        trades, meta = self._capture([[t1], [t1, t2]], tmp_path)
        assert trades["trade_id"].tolist() == ["a", "b"]
        assert meta["duplicate_trades"] == 1

    def test_two_fills_sharing_an_id_are_both_written(self, tmp_path):
        # A Polymarket trade id is the settling transaction, which can carry
        # more than one fill; only an exact repeat is a duplicate.
        f1 = {"id": "0xtx", "timestamp": 2_000, "price": 100.0, "amount": 1.0}
        f2 = {"id": "0xtx", "timestamp": 2_000, "price": 100.0, "amount": 3.0}
        trades, meta = self._capture([[f1, f2]], tmp_path)
        assert trades["amount"].tolist() == [1.0, 3.0]
        assert meta["duplicate_trades"] == 0


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


_PREDICTION_INSTALLED = (
    _CCXT_INSTALLED and importlib.util.find_spec("ccxt.prediction") is not None
)


@pytest.mark.skipif(
    not _PREDICTION_INSTALLED, reason="ccxt release without ccxt.prediction"
)
class TestPredictionVenues:
    """CCXT's prediction markets live in ``ccxt.prediction``, not ``ccxt.pro``."""

    def test_plain_id_reaches_a_prediction_market(self):
        import ccxt.prediction

        from ob_analytics.live.ccxt_source import _make_exchange

        assert isinstance(_make_exchange("kalshi"), ccxt.prediction.kalshi)

    def test_plain_id_prefers_the_crypto_exchange(self):
        import ccxt.prediction
        import ccxt.pro

        from ob_analytics.live.ccxt_source import _make_exchange

        # binance is both; the plain id keeps meaning the crypto exchange.
        assert isinstance(_make_exchange("binance"), ccxt.pro.binance)
        assert isinstance(_make_exchange("prediction/binance"), ccxt.prediction.binance)

    def test_prefix_only_searches_prediction_markets(self):
        from ob_analytics.live.ccxt_source import _make_exchange

        with pytest.raises(ValueError, match="Unknown CCXT exchange"):
            _make_exchange("prediction/kraken")

    def test_venue_column_drops_the_prefix(self, tmp_path):
        cap = _source("prediction/kalshi")
        cap._configure(CaptureConfig(pair="KXTEST", out_dir=tmp_path))
        assert cap._identity()["venue"] == "kalshi"


# ---------------------------------------------------------------------------
# Tick size, recorded for the replay
# ---------------------------------------------------------------------------


class _FakeWithMetadata(_FakeCcxtExchange):
    """A fake exchange that also answers CCXT's market-metadata lookups."""

    def __init__(self, *args, precision, mode, lookup="market", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precisionMode = mode
        setattr(self, lookup, lambda symbol: {"precision": {"price": precision}})


_TICK_SNAPSHOT = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}


class TestTickSize:
    def _meta_tick(self, exchange, tmp_path):
        import json

        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="X", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(_source(exchange), cfg))
        return json.loads((out / "meta.json").read_text())["tick_size"]

    def test_tick_size_mode_is_the_tick(self, tmp_path):
        ex = _FakeWithMetadata(_TICK_SNAPSHOT, precision=0.001, mode=4)
        assert self._meta_tick(ex, tmp_path) == 0.001

    def test_prediction_outcome_is_looked_up(self, tmp_path):
        ex = _FakeWithMetadata(
            _TICK_SNAPSHOT, precision=0.001, mode=4, lookup="outcome"
        )
        assert self._meta_tick(ex, tmp_path) == 0.001

    def test_decimal_places_mode_is_converted(self, tmp_path):
        ex = _FakeWithMetadata(_TICK_SNAPSHOT, precision=2, mode=2)
        assert self._meta_tick(ex, tmp_path) == 0.01

    def _meta(self, exchange, tmp_path):
        import json

        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="X", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(_source(exchange, poll_interval=0.0), cfg))
        return json.loads((out / "meta.json").read_text())

    def test_a_finer_book_price_makes_the_tick_finer(self, tmp_path):
        # Polymarket makes a market's tick finer as a price nears 0 or 1; the
        # capture records a grid every written price sits on.
        snap = {"bids": [[0.05, 5.0]], "asks": [[0.06, 4.0]], "timestamp": 1_000}
        later = [{"bids": [[0.035, 5.0]], "asks": [[0.06, 4.0]], "timestamp": 2_000}]
        ex = _FakeWithMetadata(snap, later, precision=0.01, mode=4)
        meta = self._meta(ex, tmp_path)
        assert meta["tick_size"] == 0.001
        assert meta["tick_size_changes"] == 1

    def test_a_finer_trade_price_makes_the_tick_finer(self, tmp_path):
        snap = {"bids": [[0.05, 5.0]], "asks": [[0.06, 4.0]], "timestamp": 1_000}
        trade = {"id": "t1", "timestamp": 2_000, "price": 0.0525, "amount": 1.0}
        ex = _FakeWithMetadata(snap, [], [[trade]], precision=0.01, mode=4)
        assert self._meta(ex, tmp_path)["tick_size"] == 0.0001

    def test_no_recorded_tick_is_left_alone(self, tmp_path):
        snap = {"bids": [[0.035, 5.0]], "asks": [[0.06, 4.0]], "timestamp": 1_000}
        meta = self._meta(_FakeCcxtExchange(snap), tmp_path)
        assert meta["tick_size"] is None
        assert meta["tick_size_changes"] == 0

    def test_significant_digits_has_no_grid(self, tmp_path):
        ex = _FakeWithMetadata(_TICK_SNAPSHOT, precision=5, mode=3)
        assert self._meta_tick(ex, tmp_path) is None

    def test_no_metadata_records_none(self, tmp_path):
        assert self._meta_tick(_FakeCcxtExchange(_TICK_SNAPSHOT), tmp_path) is None


# ---------------------------------------------------------------------------
# Depth window, whole-book venues, market-data mirror, refused location (#101)
# ---------------------------------------------------------------------------


class TestDepthRecording:
    """The capture no longer crops what CCXT reports to ``depth_limit``
    (issue #275): a level that is only outside that number, but still in
    CCXT's own book, used to be recorded as a synthetic cancel (``0``) when
    the price moved and a real cancel when it moved back, indistinguishable
    from an honest one.

    ``_diff_book``/``_side`` no longer branch on the venue -- what differs
    per venue is only the *book CCXT hands them*, which mirrors the three
    rows of the table in the issue.  ``exchange_id`` is set in each test
    below purely to document which venue's behaviour is being simulated.
    """

    def test_binance_family_keeps_levels_past_depth_limit(self):
        # Binance is asked for the whole book (TestWholeBookVenues), so
        # CCXT's own maintained book reaches past depth_limit. A deep level
        # must stay recorded, not read as gone just because it is outside
        # depth_limit.
        cap = CcxtSource()
        cap.exchange_id = "binance"
        cap._depth_limit = 2
        cap._last = {"bid": {}, "ask": {}}
        book = {"bids": [[100.0, 1.0], [99.0, 1.0], [98.0, 1.0]], "asks": []}
        rows = [r for r, _ in cap._diff_book(book, pd.Timestamp.now(tz="UTC"))]
        assert {r["price"] for r in rows} == {100.0, 99.0, 98.0}
        # A later update that only touches the deep level does not zero it
        # out first -- it was never treated as removed to begin with.
        cap._last = {"bid": {100.0: 1.0, 99.0: 1.0, 98.0: 1.0}, "ask": {}}
        moved = {"bids": [[100.0, 1.0], [99.0, 1.0], [98.0, 2.0]], "asks": []}
        rows = [
            (r["price"], r["volume"])
            for r, _ in cap._diff_book(moved, pd.Timestamp.now(tz="UTC"))
        ]
        assert rows == [(98.0, 2.0)]

    def test_ignoring_venues_keep_levels_past_depth_limit(self):
        # Coinbase, Bitstamp and OKX hand ccxt the whole book regardless of
        # the limit ccxt is asked for -- same fix, and no whole-book-venue
        # special-casing (_WHOLE_BOOK_VENUES) is needed to get it.
        cap = CcxtSource()
        cap.exchange_id = "coinbase"
        cap._depth_limit = 1
        cap._last = {"bid": {}, "ask": {}}
        book = {"bids": [[100.0, 1.0], [99.0, 1.0], [98.0, 1.0]], "asks": []}
        rows = [r for r, _ in cap._diff_book(book, pd.Timestamp.now(tz="UTC"))]
        assert {r["price"] for r in rows} == {100.0, 99.0, 98.0}

    def test_kraken_style_venue_still_reads_its_own_window_exit_as_removed(self):
        # Kraken subscribes at depth_limit and its own book drops a level
        # once it leaves that window, so ccxt hands the capture a book that
        # already lacks it -- correctly a removal. This is the one case
        # #275 leaves as-is (the issue: "Kraken's own window exits look like
        # cancels whichever option is chosen").
        cap = CcxtSource()
        cap.exchange_id = "kraken"
        cap._depth_limit = 2
        cap._last = {"bid": {100.0: 1.0, 99.0: 1.0}, "ask": {}}
        ts = pd.Timestamp.now(tz="UTC")
        # A better bid arrives; Kraken's own window drops 99 from what it sends.
        pushed = {"bids": [[101.0, 1.0], [100.0, 1.0]], "asks": []}
        rows = [(r["price"], r["volume"]) for r, _ in cap._diff_book(pushed, ts)]
        assert sorted(rows) == [(99.0, 0.0), (101.0, 1.0)]
        # 99 comes back once Kraken's own window reports it again.
        back = {"bids": [[100.0, 1.0], [99.0, 1.0]], "asks": []}
        rows = [(r["price"], r["volume"]) for r, _ in cap._diff_book(back, ts)]
        assert sorted(rows) == [(99.0, 1.0), (101.0, 0.0)]

    def test_the_archived_frame_holds_whatever_ccxt_reported(self):
        cap = CcxtSource()
        cap._depth_limit = 1
        cap._last = {"bid": {}, "ask": {}}
        book = {"bids": [[100.0, 1.0], [99.0, 1.0]], "asks": [], "nonce": 7}
        raws = [raw for _, raw in cap._diff_book(book, pd.Timestamp.now(tz="UTC"))]
        archived = next(raw for raw in raws if raw is not None)
        assert archived["frame"] == "book"
        assert archived["bids"] == [[100.0, 1.0], [99.0, 1.0]]
        assert archived["nonce"] == 7


class _LimitRecorder(_FakeCcxtExchange):
    """Records the ``limit`` each websocket book call asked for."""

    def __init__(self, exchange_id: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.id = exchange_id
        self.limits: list[int | None] = []

    async def watch_order_book(self, symbol, limit=None):
        self.limits.append(limit)
        return await super().watch_order_book(symbol, limit)


class TestWholeBookVenues:
    @staticmethod
    def _limits(exchange_id: str, tmp_path) -> list[int | None]:
        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        later = [{"bids": [[100.0, 6.0]], "asks": [[101.0, 4.0]], "timestamp": 2_000}]
        ex = _LimitRecorder(exchange_id, snap, later, [])
        asyncio.run(run_capturer(_source(ex, depth_limit=50), _cfg(tmp_path)))
        return ex.limits

    def test_binance_is_asked_for_the_whole_book(self, tmp_path):
        # A Binance book cut at the limit forgets the levels it drops.
        assert set(self._limits("binance", tmp_path)) == {None}

    def test_other_venues_are_asked_for_the_depth_limit(self, tmp_path):
        # Kraken subscribes at the limit it is given; None would mean 10.
        assert set(self._limits("kraken", tmp_path)) == {50}


class TestWholeBookSnapshot:
    """ccxt's opening Binance snapshot must reach the recorded window."""

    @staticmethod
    def _snapshot_size(exchange_id: str, depth_limit: int, tmp_path) -> object:
        ex = _LimitRecorder(exchange_id, {"bids": [], "asks": [], "timestamp": 0})
        cap = _source(ex, depth_limit=depth_limit)
        asyncio.run(_collect_snapshot(cap, _cfg(tmp_path)))
        return ex.options.get("watchOrderBookLimit")

    def test_a_deep_window_gets_a_snapshot_as_deep(self, tmp_path):
        assert self._snapshot_size("binance", 5000, tmp_path) == 5000

    def test_a_shallow_window_keeps_ccxts_default(self, tmp_path):
        # 1,000 levels leave room past a 100-level window.
        assert self._snapshot_size("binance", 100, tmp_path) == 1000

    def test_other_venues_are_left_alone(self, tmp_path):
        assert self._snapshot_size("kraken", 5000, tmp_path) is None

    def test_more_than_the_venue_returns_is_refused(self, tmp_path):
        from ob_analytics.exceptions import ConfigError

        with pytest.raises(ConfigError, match="at most 5000"):
            self._snapshot_size("binance", 5001, tmp_path)
        with pytest.raises(ConfigError, match="at most 1000"):
            self._snapshot_size("binanceusdm", 5000, tmp_path)


class _FakeWithUrls(_FakeCcxtExchange):
    def __init__(self, exchange_id: str) -> None:
        super().__init__({"bids": [], "asks": [], "timestamp": 0})
        self.id = exchange_id
        self.urls: dict[str, Any] = {
            "api": {"public": "main", "ws": {"spot": "main-ws"}}
        }
        self.options = {"fetchMarkets": {"types": ["spot", "linear"], "keep": 1}}


class TestMarketDataMirror:
    def test_points_binance_at_the_mirror(self, tmp_path):
        from ob_analytics.live.ccxt_source import MARKET_DATA_MIRRORS

        ex = _FakeWithUrls("binance")
        cap = _source(ex, market_data_mirror=True)
        asyncio.run(_collect_snapshot(cap, _cfg(tmp_path)))
        assert ex.urls["api"]["public"] == MARKET_DATA_MIRRORS["binance"]["rest"]
        assert ex.urls["api"]["ws"]["spot"] == MARKET_DATA_MIRRORS["binance"]["ws"]
        # Spot only, and ccxt's other market-loading options are kept.
        assert ex.options["fetchMarkets"] == {"types": ["spot"], "keep": 1}

    def test_an_older_ccxt_with_a_list_of_types(self, tmp_path):
        # Earlier ccxt 4.x (4.1.100, for one) made the option the list itself.
        ex = _FakeWithUrls("binance")
        ex.options = {"fetchMarkets": ["spot", "linear", "inverse"]}
        cap = _source(ex, market_data_mirror=True)
        asyncio.run(_collect_snapshot(cap, _cfg(tmp_path)))
        assert ex.options["fetchMarkets"] == ["spot"]

    def test_off_by_default(self, tmp_path):
        ex = _FakeWithUrls("binance")
        asyncio.run(_collect_snapshot(_source(ex), _cfg(tmp_path)))
        assert ex.urls["api"]["public"] == "main"

    def test_a_venue_with_no_mirror_is_refused(self, tmp_path):
        from ob_analytics.exceptions import ConfigError

        cap = _source(_FakeWithUrls("kraken"), market_data_mirror=True)
        with pytest.raises(ConfigError, match="No market-data mirror"):
            asyncio.run(_collect_snapshot(cap, _cfg(tmp_path)))


class _Refusing(_FakeCcxtExchange):
    def __init__(self, exchange_id: str, message: str) -> None:
        super().__init__({"bids": [], "asks": [], "timestamp": 0})
        self.id = exchange_id
        self._message = message

    async def fetch_order_book(self, symbol, limit=None):
        raise RuntimeError(self._message)


class TestRefusedLocation:
    _451 = "binance GET https://api.binance.com/api/v3/exchangeInfo 451  {}"

    def test_a_451_becomes_a_readable_error(self, tmp_path):
        from ob_analytics.exceptions import ConfigError

        ex = _Refusing("binance", self._451)
        with pytest.raises(ConfigError, match="binanceus") as info:
            asyncio.run(_collect_snapshot(_source(ex), _cfg(tmp_path)))
        assert "market-data-mirror" in str(info.value)
        assert ex.closed is True

    def test_other_venues_get_no_binance_hint(self, tmp_path):
        from ob_analytics.exceptions import ConfigError

        ex = _Refusing("bybit", "bybit GET https://api.bybit.com 451  {}")
        with pytest.raises(ConfigError, match="HTTP 451") as info:
            asyncio.run(_collect_snapshot(_source(ex), _cfg(tmp_path)))
        assert "binanceus" not in str(info.value)

    def test_other_failures_are_left_alone(self, tmp_path):
        ex = _Refusing("binance", "binance GET https://x 503 Service Unavailable")
        with pytest.raises(RuntimeError, match="503"):
            asyncio.run(_collect_snapshot(_source(ex), _cfg(tmp_path)))


class TestSequenceKind:
    def test_the_nonce_is_declared_monotonic_in_meta(self, tmp_path):
        import json

        from ob_analytics.depth_l2 import recorded_sequence_kind
        from ob_analytics.protocols import SequenceKind

        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="BTC/USDT", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(_source(_FakeCcxtExchange(snap)), cfg))
        assert json.loads((out / "meta.json").read_text())["sequence_kind"] == (
            "monotonic"
        )
        assert recorded_sequence_kind(out) is SequenceKind.MONOTONIC
        assert recorded_sequence_kind(out / "depth.csv") is SequenceKind.MONOTONIC


class TestClocks:
    def test_replay_follows_arrival_when_the_venue_clock_steps_back(self, tmp_path):
        # ccxt stamps its first Binance book with its own snapshot's time,
        # then applies older buffered diffs: the venue clock steps back.
        from ob_analytics.depth_l2 import L2DepthLoader

        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        ws_books = [
            {"bids": [[100.0, 6.0]], "asks": [[101.0, 4.0]], "timestamp": 3_000},
            {"bids": [[100.0, 7.0]], "asks": [[101.0, 4.0]], "timestamp": 2_000},
        ]
        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="BTC/USDT", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(_source(_FakeCcxtExchange(snap, ws_books)), cfg))

        written = pd.read_csv(out / "depth.csv")
        assert written["timestamp"].is_monotonic_increasing
        assert list(written["exchange_timestamp"]) == [1_000, 1_000, 3_000, 2_000]
        # Replay applies the bid sizes 5, 6, 7 in the order they arrived.
        depth = L2DepthLoader().load(out / "depth.csv")
        bids = depth.loc[depth["direction"] == "bid", "volume"].tolist()
        assert len(bids) == 3
        assert bids == sorted(bids)


class TestBookLoopYield:
    """A large book update yields to the event loop without losing rows."""

    @staticmethod
    def _loop_parts(tmp_path, n_levels: int):
        book = {
            "bids": [[100.0 - i * 0.01, 1.0] for i in range(n_levels)],
            "asks": [],
            "timestamp": 1_000,
        }
        ex = _FakeCcxtExchange({"bids": [], "asks": [], "timestamp": 0}, [book])
        cap = _source(ex)
        cap._configure(_cfg(tmp_path))
        return cap, asyncio.Queue(), asyncio.Event()

    def test_a_large_update_yields_to_the_event_loop(self, tmp_path, monkeypatch):
        from ob_analytics.live import ccxt_source

        cap, queue, stop = self._loop_parts(tmp_path, 600)
        real_sleep = asyncio.sleep
        yields: list[float] = []

        async def counting_sleep(delay, *args, **kwargs):
            yields.append(delay)
            await real_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(ccxt_source.asyncio, "sleep", counting_sleep)
        asyncio.run(cap._book_loop(queue, stop, deadline=float("inf")))
        assert queue.qsize() == 600
        # 600 rows with a yield every 256: after rows 256 and 512.
        assert yields == [0, 0]

    def test_a_cancel_at_the_yield_still_enqueues_the_whole_update(
        self, tmp_path, monkeypatch
    ):
        # _diff_book has already applied the whole update to _last by the
        # time the first row is enqueued, so stopping part way would leave
        # depth.csv missing rows the book state already includes.
        from ob_analytics.live import ccxt_source

        cap, queue, stop = self._loop_parts(tmp_path, 300)
        real_sleep = asyncio.sleep

        async def cancelling_sleep(delay, *args, **kwargs):
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await real_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(ccxt_source.asyncio, "sleep", cancelling_sleep)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(cap._book_loop(queue, stop, deadline=float("inf")))
        assert queue.qsize() == 300
        assert len(cap._last["bid"]) == 300


@pytest.mark.skipif(not _CCXT_INSTALLED, reason="needs the ccxt extra")
class TestLostSync:
    class _Gappy(_FakeCcxtExchange):
        """Raises ccxt's out-of-sync error once, as on a missing Binance diff."""

        def __init__(self, *args, failures: int = 1, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._failures = failures

        async def watch_order_book(self, symbol, limit=None):
            if self._failures:
                self._failures -= 1
                from ccxt.base.errors import ChecksumError

                raise ChecksumError("binance BTC/USDT out of sync")
            return await super().watch_order_book(symbol, limit)

    def _meta(self, ex, tmp_path) -> dict:
        import json

        out = tmp_path / "cap"
        cfg = CaptureConfig(pair="BTC/USDT", out_dir=out, minutes=0.05)
        asyncio.run(run_capturer(_source(ex), cfg))
        return json.loads((out / "meta.json").read_text())

    def test_the_book_is_fetched_again_and_the_capture_goes_on(self, tmp_path):
        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        later = [{"bids": [[100.0, 6.0]], "asks": [[101.0, 4.0]], "timestamp": 2_000}]
        meta = self._meta(self._Gappy(snap, later), tmp_path)
        assert meta["book_resyncs"] == 1
        assert meta["book_updates"] == 1
        assert meta["errors"] == 0

    def test_a_feed_that_keeps_losing_sync_is_given_up(self, tmp_path):
        from ob_analytics.live.ccxt_source import _MAX_BOOK_RESYNCS

        snap = {"bids": [[100.0, 5.0]], "asks": [[101.0, 4.0]], "timestamp": 1_000}
        ex = self._Gappy(snap, [], failures=_MAX_BOOK_RESYNCS + 1)
        meta = self._meta(ex, tmp_path)
        assert meta["book_resyncs"] == _MAX_BOOK_RESYNCS
        assert meta["errors"] == 1
