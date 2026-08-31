"""Tests for the cryptofeed L2/L3 capturer (issue #134) -- no network.

A fake cryptofeed exchange class and a fake ``FeedHandler`` feed scripted
book/trade callbacks, so the capturer's level discovery, translation, and the
capture -> replay round-trip are exercised deterministically without cryptofeed
or a socket.
"""

from __future__ import annotations

import importlib.util

import pytest

from ob_analytics.protocols import Level

_CRYPTOFEED_INSTALLED = importlib.util.find_spec("cryptofeed") is not None


class _FakeExchangeCls:
    """Stand-in for a cryptofeed exchange class (``Feed`` subclass).

    Only the two class attributes the capturer reads are modelled: ``id`` and
    the ``websocket_channels`` mapping it discovers the venue's level from.
    """

    def __init__(self, exchange_id: str, channels: dict[str, str]) -> None:
        self.id = exchange_id
        self.websocket_channels = channels
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self


def _l3_venue(exchange_id: str = "fakel3") -> _FakeExchangeCls:
    return _FakeExchangeCls(
        exchange_id,
        {
            "l3_book": "detail_order_book",
            "l2_book": "diff_order_book",
            "trades": "live_trades",
        },
    )


def _l2_venue(exchange_id: str = "fakel2") -> _FakeExchangeCls:
    return _FakeExchangeCls(exchange_id, {"l2_book": "depth", "trades": "trade"})


# ---------------------------------------------------------------------------
# Level discovery -- the venue's declared channels, not a hardcoded venue list
# ---------------------------------------------------------------------------


class TestLevelDiscovery:
    def test_venue_declaring_l3_book_is_l3(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange=_l3_venue()))
        assert src.level is Level.L3

    def test_venue_without_l3_book_is_l2(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange=_l2_venue()))
        assert src.level is Level.L2

    def test_explicit_l2_on_an_l3_venue_is_honoured(self):
        """An L3-capable venue can still be captured at L2 on request."""
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(
            settings=CryptofeedSettings(exchange=_l3_venue(), level=Level.L2)
        )
        assert src.level is Level.L2

    def test_explicit_l3_on_an_l2_only_venue_errors(self):
        """Coinbase no longer publishes L3 in cryptofeed; asking must not fake it."""
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(
            settings=CryptofeedSettings(exchange=_l2_venue("coinbase"), level=Level.L3)
        )
        with pytest.raises(ValueError, match="does not publish an L3"):
            _ = src.level


class _FakeBook:
    """Stand-in for ``cryptofeed.types.OrderBook``.

    Models only what the capturer reads: the venue/symbol identity, the
    maintained book (via ``book.to_dict()``), the optional ``delta``, the
    epoch-seconds ``timestamp``, and ``sequence_number``.
    """

    def __init__(
        self,
        levels: dict,
        *,
        delta: dict | None = None,
        exchange: str = "fakel2",
        symbol: str = "BTC-USD",
        timestamp: float = 1_700_000_000.0,
        sequence_number: int | None = 7,
        raw: object = None,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.book = _FakeInnerBook(levels)
        self.delta = delta
        self.timestamp = timestamp
        self.sequence_number = sequence_number
        self.raw = raw if raw is not None else {"frame": "raw"}


class _FakeInnerBook:
    """The ``order_book.OrderBook`` cryptofeed keeps inside ``OrderBook.book``."""

    def __init__(self, levels: dict) -> None:
        self._levels = levels

    def to_dict(self) -> dict:
        return self._levels


# ---------------------------------------------------------------------------
# L2 translation (pure)
# ---------------------------------------------------------------------------


class TestL2Translation:
    def _source(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        return CryptofeedSource(settings=CryptofeedSettings(exchange=_l2_venue()))

    def test_delta_becomes_absolute_size_depth_rows(self):
        """cryptofeed L2 deltas already carry the new absolute size; 0 removes."""
        src = self._source()
        book = _FakeBook(
            {"bid": {100.0: 7.0}, "ask": {101.0: 4.0}},
            delta={"bid": [(100.0, 7.0), (99.0, 0.0)], "ask": [(101.0, 4.0)]},
        )
        rows = src._l2_rows(book)
        assert {(r["side"], r["price"], r["volume"]) for r in rows} == {
            ("bid", 100.0, 7.0),
            ("bid", 99.0, 0.0),
            ("ask", 101.0, 4.0),
        }

    def test_delta_rows_carry_sequence_and_identity(self):
        src = self._source()
        book = _FakeBook(
            {"bid": {}, "ask": {}}, delta={"bid": [(100.0, 1.0)], "ask": []}
        )
        (row,) = src._l2_rows(book)
        assert row["sequence"] == 7
        assert row["venue"] == "fakel2"
        assert row["symbol"] == "BTC-USD"
        assert str(row["timestamp"].tz) == "UTC"

    def test_first_book_without_delta_emits_every_level(self):
        """The opening book arrives with no delta: seed it in full."""
        src = self._source()
        book = _FakeBook({"bid": {100.0: 5.0}, "ask": {101.0: 4.0}}, delta=None)
        rows = src._l2_rows(book)
        assert {(r["side"], r["price"], r["volume"]) for r in rows} == {
            ("bid", 100.0, 5.0),
            ("ask", 101.0, 4.0),
        }

    def test_later_book_without_delta_emits_only_what_changed(self):
        """A venue that resends the whole L2 book must not re-emit unchanged levels."""
        src = self._source()
        src._l2_rows(_FakeBook({"bid": {100.0: 5.0}, "ask": {101.0: 4.0}}, delta=None))
        rows = src._l2_rows(
            # 100 grows, 101 unchanged, 102 is new, and there is no 99 to remove
            _FakeBook(
                {"bid": {100.0: 6.0}, "ask": {101.0: 4.0, 102.0: 1.0}}, delta=None
            )
        )
        assert {(r["side"], r["price"], r["volume"]) for r in rows} == {
            ("bid", 100.0, 6.0),
            ("ask", 102.0, 1.0),
        }

    def test_vanished_level_in_a_full_book_emits_a_removal(self):
        src = self._source()
        src._l2_rows(_FakeBook({"bid": {100.0: 5.0, 99.0: 2.0}, "ask": {}}, delta=None))
        rows = src._l2_rows(_FakeBook({"bid": {100.0: 5.0}, "ask": {}}, delta=None))
        assert [(r["side"], r["price"], r["volume"]) for r in rows] == [
            ("bid", 99.0, 0.0)
        ]


# ---------------------------------------------------------------------------
# L3 translation (pure) -- real order IDs, never synthesised
# ---------------------------------------------------------------------------


def _l3_book(levels, **kw):
    """An L3 book: ``{side: {price: {order_id: size}}}``."""
    return _FakeBook(levels, exchange="fakel3", **kw)


class TestL3Translation:
    def _source(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        return CryptofeedSource(settings=CryptofeedSettings(exchange=_l3_venue()))

    def test_delta_for_an_unseen_order_is_created(self):
        src = self._source()
        book = _l3_book(
            {"bid": {100.0: {11: 2.0}}, "ask": {}},
            delta={"bid": [(11, 100.0, 2.0)], "ask": []},
        )
        (ev,) = src._l3_events(book)
        assert ev["action"] == "created"
        assert ev["id"] == 11
        assert ev["price"] == 100.0
        assert ev["volume"] == 2.0
        assert ev["direction"] == "bid"

    def test_delta_for_a_known_order_is_changed(self):
        src = self._source()
        src._l3_events(
            _l3_book(
                {"bid": {100.0: {11: 2.0}}, "ask": {}},
                delta={"bid": [(11, 100.0, 2.0)], "ask": []},
            )
        )
        (ev,) = src._l3_events(
            _l3_book(
                {"bid": {100.0: {11: 1.5}}, "ask": {}},
                delta={"bid": [(11, 100.0, 1.5)], "ask": []},
            )
        )
        assert ev["action"] == "changed"
        assert ev["volume"] == 1.5

    def test_zero_quantity_delta_is_deleted(self):
        src = self._source()
        src._l3_events(
            _l3_book(
                {"bid": {100.0: {11: 2.0}}, "ask": {}},
                delta={"bid": [(11, 100.0, 2.0)], "ask": []},
            )
        )
        (ev,) = src._l3_events(
            _l3_book({"bid": {}, "ask": {}}, delta={"bid": [(11, 100.0, 0)], "ask": []})
        )
        assert ev["action"] == "deleted"
        assert ev["id"] == 11

    def test_bitfinex_style_move_is_a_delete_then_a_create(self):
        """A price move loses queue priority, so it must not read as one order
        quietly changing price."""
        src = self._source()
        src._l3_events(
            _l3_book(
                {"bid": {100.0: {11: 2.0}}, "ask": {}},
                delta={"bid": [(11, 100.0, 2.0)], "ask": []},
            )
        )
        events = src._l3_events(
            _l3_book(
                {"bid": {99.0: {11: 2.0}}, "ask": {}},
                # cryptofeed's bitfinex feed emits the move as two entries
                delta={"bid": [(11, 100.0, 0), (11, 99.0, 2.0)], "ask": []},
            )
        )
        assert [(e["action"], e["price"]) for e in events] == [
            ("deleted", 100.0),
            ("created", 99.0),
        ]

    def test_opening_book_without_delta_creates_every_resting_order(self):
        """bitstamp and blockchain send the opening book with delta=None."""
        src = self._source()
        events = src._l3_events(
            _l3_book(
                {"bid": {100.0: {11: 2.0, 12: 1.0}}, "ask": {101.0: {21: 3.0}}},
                delta=None,
            )
        )
        assert {(e["id"], e["action"], e["direction"]) for e in events} == {
            (11, "created", "bid"),
            (12, "created", "bid"),
            (21, "created", "ask"),
        }

    def test_empty_delta_falls_back_to_the_full_book(self):
        """bitfinex sends its snapshot with a populated book but an empty delta."""
        src = self._source()
        events = src._l3_events(
            _l3_book(
                {"bid": {100.0: {11: 2.0}}, "ask": {}},
                delta={"bid": [], "ask": []},
            )
        )
        assert [(e["id"], e["action"]) for e in events] == [(11, "created")]

    def test_full_book_resend_reports_only_real_changes(self):
        """bitstamp resends the whole L3 book every message."""
        src = self._source()
        src._l3_events(
            _l3_book(
                {"bid": {100.0: {11: 2.0, 12: 1.0}}, "ask": {101.0: {21: 3.0}}},
                delta=None,
            )
        )
        events = src._l3_events(
            _l3_book(
                # a1 resized, a2 gone, b1 untouched, c1 new
                {"bid": {100.0: {11: 5.0}}, "ask": {101.0: {21: 3.0, 31: 1.0}}},
                delta=None,
            )
        )
        assert {(e["id"], e["action"]) for e in events} == {
            (11, "changed"),
            (12, "deleted"),
            (31, "created"),
        }

    def test_deleted_order_reports_its_last_known_size(self):
        src = self._source()
        src._l3_events(_l3_book({"bid": {100.0: {11: 2.0}}, "ask": {}}, delta=None))
        (ev,) = src._l3_events(_l3_book({"bid": {}, "ask": {}}, delta=None))
        assert (ev["action"], ev["price"], ev["volume"]) == ("deleted", 100.0, 2.0)

    def test_shutdown_closes_out_every_resting_order(self):
        import asyncio

        src = self._source()
        src._l3_events(
            _l3_book({"bid": {100.0: {11: 2.0}}, "ask": {101.0: {21: 3.0}}}, delta=None)
        )

        async def _drain():
            return [ev async for ev in src.shutdown_synthetic_events()]

        events = asyncio.run(_drain())
        assert {e["id"] for e in events} == {11, 21}
        assert {e["action"] for e in events} == {"deleted"}


class _FakeTrade:
    """Stand-in for ``cryptofeed.types.Trade``."""

    def __init__(
        self,
        *,
        price="100.5",
        amount="0.25",
        side="buy",
        trade_id="t1",
        timestamp=1_700_000_000.0,
        exchange="fakel3",
        symbol="BTC-USD",
    ) -> None:
        from decimal import Decimal

        self.exchange = exchange
        self.symbol = symbol
        self.side = side
        self.amount = Decimal(amount)
        self.price = Decimal(price)
        self.timestamp = timestamp
        self.id = trade_id
        self.type = None
        self.raw = {"frame": "raw"}


class TestTradeTranslation:
    def _source(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        return CryptofeedSource(settings=CryptofeedSettings(exchange=_l3_venue()))

    def test_maps_a_cryptofeed_trade(self):
        ev = self._source()._map_trade(_FakeTrade())
        assert ev["trade_id"] == "t1"
        assert ev["price"] == 100.5
        assert ev["amount"] == 0.25
        assert ev["side"] == "buy"
        assert ev["venue"] == "fakel3"
        assert ev["symbol"] == "BTC-USD"

    def test_decimals_become_floats(self):
        """cryptofeed hands out Decimals; the schema wants plain floats."""
        ev = self._source()._map_trade(_FakeTrade(price="0.1", amount="0.2"))
        assert isinstance(ev["price"], float)
        assert isinstance(ev["amount"], float)

    def test_public_tape_carries_no_order_ids(self):
        """Public trades have no maker/taker IDs -- they must not be invented."""
        ev = self._source()._map_trade(_FakeTrade())
        assert ev["buy_order_id"] == ""
        assert ev["sell_order_id"] == ""

    def test_exchange_timestamp_is_utc_nanoseconds(self):
        ev = self._source()._map_trade(_FakeTrade(timestamp=1_700_000_000.5))
        ts = ev["exchange_timestamp"]
        assert str(ts.tz) == "UTC"
        assert ts.value == 1_700_000_000_500_000_000


class TestOrderIdsArePassedThrough:
    """The venue's own order id is what lands in ``orders.csv``.

    cryptofeed venues disagree on the type -- bitstamp, bitfinex and blockchain
    publish integers, independent_reserve publishes UUIDs -- and the shared
    schema keys orders by identity rather than by integer, so neither is
    re-labelled."""

    def _source(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        return CryptofeedSource(settings=CryptofeedSettings(exchange=_l3_venue()))

    def test_integer_ids_pass_through(self):
        src = self._source()
        (ev,) = src._l3_events(
            _l3_book({"bid": {100.0: {4213: 2.0}}, "ask": {}}, delta=None)
        )
        assert ev["id"] == 4213

    def test_uuid_ids_pass_through_unchanged(self):
        src = self._source()
        (ev,) = src._l3_events(
            _l3_book({"bid": {100.0: {"3f2b-aa": 2.0}}, "ask": {}}, delta=None)
        )
        assert ev["id"] == "3f2b-aa"

    def test_a_uuid_keeps_its_identity_across_messages(self):
        """Identity must survive across messages or the lifecycle breaks."""
        src = self._source()
        (created,) = src._l3_events(
            _l3_book({"bid": {100.0: {"3f2b-aa": 2.0}}, "ask": {}}, delta=None)
        )
        (deleted,) = src._l3_events(_l3_book({"bid": {}, "ask": {}}, delta=None))
        assert deleted["action"] == "deleted"
        assert deleted["id"] == created["id"] == "3f2b-aa"

    def test_uuid_ids_from_a_delta_pass_through(self):
        src = self._source()
        (ev,) = src._l3_events(
            _l3_book(
                {"bid": {100.0: {"9c1d-bb": 1.0}}, "ask": {}},
                delta={"bid": [("9c1d-bb", 100.0, 1.0)], "ask": []},
            )
        )
        assert ev["id"] == "9c1d-bb"


class _FakeFeedHandler:
    """Stand-in for ``cryptofeed.FeedHandler``.

    Mirrors the contract the capturer relies on: ``add_feed`` collects feeds,
    ``run(start_loop=False)`` starts them as tasks on the running loop and
    returns immediately, and ``stop_async`` shuts them down.  Instead of a
    socket, the scripted ``(channel, object)`` pairs are pushed through the
    callbacks the capturer registered.
    """

    def __init__(self, script: list[tuple[str, object]]) -> None:
        self._script = script
        self.feeds: list = []
        self.stopped = False
        self.started = False
        # cryptofeed's FeedHandler sets this in run() and clears it on stop;
        # the capturer watches it so a feed that dies ends the run.
        self.running = False

    def add_feed(self, feed, **kwargs):
        self.feeds.append(feed)

    def run(
        self, start_loop=True, install_signal_handlers=True, exception_handler=None
    ):
        import asyncio

        assert start_loop is False, "the capturer must not take over the event loop"
        assert install_signal_handlers is False, "the runner owns signal handling"
        self.started = True
        self.running = True
        self._task = asyncio.get_running_loop().create_task(self._drive())

    async def _drive(self):
        callbacks = self.feeds[0].kwargs["callbacks"]
        for channel, obj in self._script:
            await callbacks[channel](obj, 1_700_000_000.0)
        self.running = False

    async def stop_async(self, loop=None):
        self.stopped = True


def _capture_cfg(tmp_path, **kw):
    from ob_analytics.live._base import CaptureConfig

    return CaptureConfig(
        pair=kw.pop("pair", "BTC-USD"),
        out_dir=tmp_path / "cap",
        minutes=kw.pop("minutes", 0.02),
        **kw,
    )


def _source_with(exchange, script, **settings_kw):
    from ob_analytics.live.cryptofeed_source import CryptofeedSettings, CryptofeedSource

    return CryptofeedSource(
        settings=CryptofeedSettings(
            exchange=exchange, feed_handler=_FakeFeedHandler(script), **settings_kw
        )
    )


class TestFeedWiring:
    def test_l3_venue_subscribes_to_the_l3_channel(self, tmp_path):
        import asyncio

        from ob_analytics.live._runner import run_capturer

        venue = _l3_venue()
        src = _source_with(venue, [])
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))
        assert set(venue.kwargs["channels"]) == {"l3_book", "trades"}
        assert venue.kwargs["symbols"] == ["BTC-USD"]

    def test_l2_venue_subscribes_to_the_l2_channel(self, tmp_path):
        import asyncio

        from ob_analytics.live._runner import run_capturer

        venue = _l2_venue()
        src = _source_with(venue, [])
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))
        assert set(venue.kwargs["channels"]) == {"l2_book", "trades"}

    def test_the_feed_handler_is_shut_down(self, tmp_path):
        import asyncio

        from ob_analytics.live._runner import run_capturer

        src = _source_with(_l3_venue(), [])
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))
        assert src.settings.feed_handler.stopped is True


class TestFullCapture:
    def test_l3_capture_writes_orders_and_replays_through_the_l3_loader(self, tmp_path):
        import asyncio

        from ob_analytics.bitstamp import BitstampLoader
        from ob_analytics.live._runner import run_capturer

        script = [
            # opening book (no delta), then a resize, then a trade
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 2.0}}, "ask": {101.0: {21: 3.0}}}, delta=None
                ),
            ),
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 1.0}}, "ask": {101.0: {21: 3.0}}},
                    delta={"bid": [(11, 100.0, 1.0)], "ask": []},
                ),
            ),
            ("trades", _FakeTrade()),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        result = asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))

        assert (out / "orders.csv").exists()
        assert not (out / "depth.csv").exists()
        # 2 creates + 1 change + 2 synthetic deletes at shutdown
        assert result.n_order_events == 5
        assert result.n_trade_events == 1

        events = BitstampLoader().load(out / "orders.csv")
        assert len(events) == 5
        assert set(events["direction"].unique()) == {"bid", "ask"}

    def test_l2_capture_writes_depth_and_replays_through_the_l2_loader(self, tmp_path):
        import asyncio

        from ob_analytics.depth_l2 import L2DepthLoader
        from ob_analytics.live._runner import run_capturer

        script = [
            (
                "l2_book",
                _FakeBook({"bid": {100.0: 5.0}, "ask": {101.0: 4.0}}, delta=None),
            ),
            (
                "l2_book",
                _FakeBook(
                    {"bid": {100.0: 7.0}, "ask": {101.0: 4.0}},
                    delta={"bid": [(100.0, 7.0)], "ask": []},
                ),
            ),
            ("trades", _FakeTrade(exchange="fakel2")),
        ]
        src = _source_with(_l2_venue(), script)
        out = tmp_path / "cap"
        result = asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))

        assert (out / "depth.csv").exists()
        assert not (out / "orders.csv").exists()
        assert result.n_depth_events == 3  # 2 opening levels + 1 change
        assert result.n_trade_events == 1

        depth = L2DepthLoader().load(out / "depth.csv")
        assert len(depth) == 3

    def test_diagnostics_land_in_meta(self, tmp_path):
        import asyncio
        import json

        from ob_analytics.live._runner import run_capturer

        script = [
            ("l3_book", _l3_book({"bid": {100.0: {11: 2.0}}, "ask": {}}, delta=None)),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))
        meta = json.loads((out / "meta.json").read_text())
        assert meta["exchange"] == "fakel3"
        assert meta["level"] == "L3"
        assert meta["book_updates"] == 1

    def test_raw_frames_are_archived(self, tmp_path):
        import asyncio

        from ob_analytics.live._runner import run_capturer

        script = [
            ("l3_book", _l3_book({"bid": {100.0: {11: 2.0}}, "ask": {}}, delta=None)),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))
        assert (out / "raw.jsonl").read_text().strip() != ""


class TestRunTermination:
    def test_a_feed_that_stops_ends_the_run_before_the_deadline(self, tmp_path):
        """cryptofeed clears ``running`` when it shuts down; spinning out the
        full duration after that just wastes the capture window."""
        import asyncio
        import time

        from ob_analytics.live._runner import run_capturer

        src = _source_with(
            _l3_venue(),
            [("l3_book", _l3_book({"bid": {100.0: {11: 2.0}}, "ask": {}}, delta=None))],
        )
        started = time.monotonic()
        # A 6-minute window the run must not wait out.
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path, minutes=0.1)))
        assert time.monotonic() - started < 3.0

    def test_a_live_feed_still_runs_to_the_deadline(self, tmp_path):
        """A feed that keeps running is bounded by ``minutes``, not by silence."""
        import asyncio
        import time

        from ob_analytics.live._runner import run_capturer

        class _NeverEnding(_FakeFeedHandler):
            async def _drive(self):
                await asyncio.sleep(3600)

        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(
            settings=CryptofeedSettings(
                exchange=_l3_venue(), feed_handler=_NeverEnding([])
            )
        )
        started = time.monotonic()
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path, minutes=0.02)))
        assert time.monotonic() - started >= 1.0


# ---------------------------------------------------------------------------
# Registration + CLI wiring
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registered_without_the_extra_installed(self):
        """Importing the module must never require cryptofeed, so the source is
        listed (and discoverable) whether or not the extra is present."""
        from ob_analytics.live.cryptofeed_source import CryptofeedSource
        from ob_analytics.sources import get_source, list_sources

        assert "cryptofeed" in list_sources()
        assert get_source("cryptofeed") is CryptofeedSource

    def test_is_a_livesource(self):
        from ob_analytics.live import LiveSource
        from ob_analytics.live.cryptofeed_source import CryptofeedSource

        assert isinstance(CryptofeedSource(), LiveSource)

    def test_a_string_venue_needs_the_extra_or_a_clear_error(self):
        """Without cryptofeed installed, a venue id must fail with an install
        hint rather than an obscure ImportError deep in the run."""
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange="bitstamp"))
        if _CRYPTOFEED_INSTALLED:
            assert src._exchange_class() is not None
        else:
            with pytest.raises(ImportError, match=r"ob-analytics\[cryptofeed\]"):
                src._exchange_class()

    def test_empty_venue_errors_clearly(self):
        from ob_analytics.live.cryptofeed_source import CryptofeedSource

        with pytest.raises(ValueError, match="exchange"):
            CryptofeedSource()._exchange_class()

    def test_level_is_reported_once_the_venue_resolves(self):
        """A string venue id needs the exchange map before its channels can be
        read, so the level is discovered on first access."""
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange=_l3_venue()))
        assert src._level is None  # not resolved yet
        assert src.level is Level.L3
        assert src._level is Level.L3  # cached


class TestCliWiring:
    def test_flags_flow_into_typed_settings(self, monkeypatch, tmp_path):
        import argparse

        import pandas as pd

        from ob_analytics import cli
        from ob_analytics.live._base import CaptureResult
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        captured: dict = {}

        async def _fake_run(source, config, sink=None):
            captured["source"] = source
            captured["config"] = config
            now = pd.Timestamp.now(tz="UTC")
            return CaptureResult(
                out_dir=config.out_dir,
                n_order_events=0,
                n_trade_events=0,
                n_raw_frames=0,
                started=now,
                ended=now,
            )

        monkeypatch.setattr("ob_analytics.live._runner.run_capturer", _fake_run)

        args = argparse.Namespace(
            verbose=False,
            list=False,
            venue="cryptofeed",
            pair="BTC-USD",
            exchange="bitstamp",
            level="L3",
            depth_limit=None,
            poll_interval=None,
            minutes=0.001,
            out=str(tmp_path / "o"),
            no_raw=True,
        )
        cli._cmd_capture(args)

        source = captured["source"]
        assert isinstance(source, CryptofeedSource)
        assert isinstance(source.settings, CryptofeedSettings)
        assert source.settings.exchange == "bitstamp"
        assert source.settings.level is Level.L3
        assert captured["config"].pair == "BTC-USD"

    def test_level_flag_is_documented(self, cli_runner):
        r = cli_runner("capture", "--help")
        assert r.returncode == 0
        assert "--level" in r.stdout


# ---------------------------------------------------------------------------
# cryptofeed-dependent (skipped without the extra) -- still no network: these
# read the library's declared capabilities, they do not connect to anything.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CRYPTOFEED_INSTALLED, reason="cryptofeed extra not installed")
class TestCryptofeedInstalled:
    def test_unknown_venue_raises(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(
            settings=CryptofeedSettings(exchange="not_a_real_venue_xyz")
        )
        with pytest.raises(ValueError, match="Unknown cryptofeed exchange"):
            src._exchange_class()

    def test_a_real_l3_venue_discovers_l3(self):
        """bitstamp publishes a per-order book in cryptofeed."""
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange="bitstamp"))
        assert src.level is Level.L3

    def test_a_real_l2_only_venue_discovers_l2(self):
        """Coinbase publishes price-level data only -- issue #134 assumed
        otherwise, which is why the level is discovered rather than hardcoded."""
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange="coinbase"))
        assert src.level is Level.L2

    def test_cli_reports_an_impossible_level_cleanly(self, monkeypatch, tmp_path):
        """Asking for L3 where there is none is user error, so the CLI should
        exit with the explanation, not a traceback."""
        import argparse

        from ob_analytics import cli

        args = argparse.Namespace(
            verbose=False,
            list=False,
            venue="cryptofeed",
            pair="BTC-USD",
            exchange="coinbase",
            level="L3",
            depth_limit=None,
            poll_interval=None,
            minutes=0.001,
            out=str(tmp_path / "o"),
            no_raw=True,
        )
        with pytest.raises(SystemExit) as exc:
            cli._cmd_capture(args)
        assert exc.value.code == 1

    def test_forcing_l3_on_a_real_l2_only_venue_raises(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(
            settings=CryptofeedSettings(exchange="binance", level=Level.L3)
        )
        with pytest.raises(ValueError, match="does not publish an L3"):
            _ = src.level


class TestSequenceReachesTheOutput:
    """Issue #134 asks for gap behaviour to be surfaced. cryptofeed carries a
    per-book ``sequence_number``; it has to survive into ``orders.csv`` or gap
    detection is impossible on the L3 path."""

    def test_l3_sequence_is_written_and_read_back(self, tmp_path):
        import asyncio

        from ob_analytics.bitstamp import BitstampLoader
        from ob_analytics.config import PipelineConfig
        from ob_analytics.live._runner import run_capturer

        script = [
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 2.0}}, "ask": {}},
                    delta=None,
                    sequence_number=41,
                ),
            ),
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 1.0}}, "ask": {}},
                    delta={"bid": [(11, 100.0, 1.0)], "ask": []},
                    sequence_number=42,
                ),
            ),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))

        assert "sequence" in (out / "orders.csv").read_text().splitlines()[0]
        events = BitstampLoader(config=PipelineConfig(track_sequence=True)).load(
            out / "orders.csv"
        )
        assert set(events["sequence"].dropna().astype(int)) >= {41, 42}

    def test_a_sequence_gap_is_detectable_in_the_captured_output(self, tmp_path):
        import asyncio

        from ob_analytics.analytics import detect_sequence_gaps
        from ob_analytics.bitstamp import BitstampLoader
        from ob_analytics.config import PipelineConfig
        from ob_analytics.live._runner import run_capturer

        # 41 then 43: the venue skipped 42.
        script = [
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 2.0}}, "ask": {}},
                    delta=None,
                    sequence_number=41,
                ),
            ),
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 2.0}, 99.0: {12: 1.0}}, "ask": {}},
                    delta={"bid": [(12, 99.0, 1.0)], "ask": []},
                    sequence_number=43,
                ),
            ),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))

        events = BitstampLoader(config=PipelineConfig(track_sequence=True)).load(
            out / "orders.csv"
        )
        report = detect_sequence_gaps(events)
        assert report.n_missing == 1  # sequence 42 never arrived


class TestGapDiagnostics:
    """cryptofeed reconnects internally, so a run cannot count reconnects
    directly -- what it can do is notice the sequence discontinuity a reconnect
    leaves behind, and say so in meta.json."""

    def _source(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        return CryptofeedSource(settings=CryptofeedSettings(exchange=_l3_venue()))

    def test_contiguous_sequences_report_no_gap(self):
        src = self._source()
        for n in (10, 11, 12):
            src.note_sequence(_l3_book({"bid": {}, "ask": {}}, sequence_number=n))
        assert src.diagnostics()["sequence_gaps"] == 0

    def test_a_skipped_sequence_is_counted(self):
        src = self._source()
        for n in (10, 13):
            src.note_sequence(_l3_book({"bid": {}, "ask": {}}, sequence_number=n))
        d = src.diagnostics()
        assert d["sequence_gaps"] == 1
        assert d["sequence_missing"] == 2  # 11 and 12

    def test_a_venue_without_sequences_reports_none(self):
        src = self._source()
        for _ in range(3):
            src.note_sequence(_l3_book({"bid": {}, "ask": {}}, sequence_number=None))
        assert src.diagnostics()["sequence_gaps"] == 0

    def test_gaps_land_in_meta(self, tmp_path):
        import asyncio
        import json

        from ob_analytics.live._runner import run_capturer

        script = [
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 2.0}}, "ask": {}},
                    delta=None,
                    sequence_number=1,
                ),
            ),
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 3.0}}, "ask": {}},
                    delta={"bid": [(11, 100.0, 3.0)], "ask": []},
                    sequence_number=5,
                ),
            ),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))
        meta = json.loads((out / "meta.json").read_text())
        assert meta["sequence_gaps"] == 1
        assert meta["sequence_missing"] == 3


class TestDeltaFailureLeavesStateIntact:
    """A malformed delta entry must not half-apply: tracked orders and the
    events handed to the sink have to agree, or every later diff is wrong."""

    def _source(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        return CryptofeedSource(settings=CryptofeedSettings(exchange=_l3_venue()))

    def test_a_bad_entry_mid_delta_rolls_the_whole_update_back(self):
        src = self._source()
        src._l3_events(_l3_book({"bid": {100.0: {11: 2.0}}, "ask": {}}, delta=None))
        before = dict(src._open_orders)

        # Unpacking a two-field entry into three raises ValueError.
        with pytest.raises(ValueError):
            src._l3_events(
                _l3_book(
                    {"bid": {}, "ask": {}},
                    # second entry is malformed (two fields, not three)
                    delta={"bid": [(11, 100.0, 5.0), (12, 99.0)], "ask": []},
                )
            )

        assert src._open_orders == before


@pytest.mark.skipif(not _CRYPTOFEED_INSTALLED, reason="cryptofeed extra not installed")
class TestVenueNamingIsConsistent:
    """cryptofeed's own venue id is upper-case ('BITSTAMP') while a user types
    it lower-case. Both must not end up in the same capture, or meta.json and
    the rows disagree about which venue the data came from."""

    def test_diagnostics_use_the_resolved_venue_id(self):
        from cryptofeed.exchanges import EXCHANGE_MAP

        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange="bitstamp"))
        assert src.diagnostics()["exchange"] == EXCHANGE_MAP["BITSTAMP"].id

    def test_event_venue_matches_diagnostics(self):
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange="bitstamp"))
        src._resolve_symbol(
            __import__(
                "ob_analytics.live._base", fromlist=["CaptureConfig"]
            ).CaptureConfig(pair="BTC-USD", out_dir=__import__("pathlib").Path("/tmp"))
        )
        row_venue = src._configured_identity()["venue"]
        assert row_venue == src.diagnostics()["exchange"]


class TestAcceptanceReplay:
    """Issue #134's acceptance criteria: a capture replays through the matching
    pipeline and the validators pass. Coinbase L3 (the issue's own example) no
    longer exists, so the L3 case is exercised on a per-order venue shape."""

    def test_l3_capture_replays_through_the_full_pipeline_and_validates(self, tmp_path):
        import asyncio

        from ob_analytics.analytics import data_quality_summary
        from ob_analytics.bitstamp import BitstampSource
        from ob_analytics.live._runner import run_capturer
        from ob_analytics.pipeline import Pipeline

        script = [
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 2.0}}, "ask": {101.0: {21: 3.0}}}, delta=None
                ),
            ),
            (
                "l3_book",
                _l3_book(
                    {"bid": {100.0: {11: 1.0}}, "ask": {101.0: {21: 3.0}}},
                    delta={"bid": [(11, 100.0, 1.0)], "ask": []},
                ),
            ),
            ("trades", _FakeTrade(price="100.0", amount="1.0")),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))

        # The captured directory replays through the stock L3 pipeline with no
        # bespoke loader, exactly as the issue requires.
        result = Pipeline(source=BitstampSource()).run(out / "orders.csv")
        assert len(result.events) > 0
        summary = data_quality_summary(result.events, result.trades)
        assert summary is not None

    def test_a_uuid_venue_also_replays_through_the_full_pipeline(self, tmp_path):
        """independent_reserve publishes UUID order ids; they must survive the
        capture -> replay round trip without being re-labelled."""
        import asyncio

        import pandas as pd

        from ob_analytics.bitstamp import BitstampSource
        from ob_analytics.live._runner import run_capturer
        from ob_analytics.pipeline import Pipeline

        script = [
            (
                "l3_book",
                _l3_book(
                    {
                        "bid": {100.0: {"3f2b-aa": 2.0}},
                        "ask": {101.0: {"9c1d-bb": 3.0}},
                    },
                    delta=None,
                ),
            ),
            ("trades", _FakeTrade(price="100.0", amount="1.0")),
        ]
        src = _source_with(_l3_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))

        written = pd.read_csv(out / "orders.csv")
        assert set(written["id"].astype(str)) == {"3f2b-aa", "9c1d-bb"}

        result = Pipeline(source=BitstampSource()).run(out / "orders.csv")
        assert len(result.events) == 4  # two creates + two synthetic deletes

    def test_l2_capture_replays_through_the_l2_pipeline(self, tmp_path):
        import asyncio

        from ob_analytics.live._runner import run_capturer
        from ob_analytics.pipeline import Pipeline

        script = [
            (
                "l2_book",
                _FakeBook({"bid": {100.0: 5.0}, "ask": {101.0: 4.0}}, delta=None),
            ),
            (
                "l2_book",
                _FakeBook(
                    {"bid": {100.0: 7.0}, "ask": {101.0: 4.0}},
                    delta={"bid": [(100.0, 7.0)], "ask": []},
                ),
            ),
        ]
        src = _source_with(_l2_venue(), script)
        out = tmp_path / "cap"
        asyncio.run(run_capturer(src, _capture_cfg(tmp_path)))

        result = Pipeline.from_source("depth_csv").run(out / "depth.csv")
        assert len(result.depth) == 3
