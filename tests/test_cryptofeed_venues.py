"""Tests for the corrected cryptofeed venue feeds -- no network.

The feed is built without cryptofeed's constructor, which fetches the venue's
symbols, and given a fake REST connection that hands out scripted books.  Its
own message handling then runs as it would on a live connection.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections import defaultdict
from decimal import Decimal
from typing import Any, cast

import pandas as pd
import pytest

from tests._logging import warnings_logged

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cryptofeed") is None,
    reason="cryptofeed extra not installed",
)

#: Venue time of the first stream message, in epoch seconds.
T0 = 1_791_000_000.0

#: SHA-256 of cryptofeed's ``IndependentReserve._snapshot`` source, the same in
#: cryptofeed 2.4.1 and 2.5.0.
SNAPSHOT_SHA256 = "bcb3407779fd5a2d13471c66abe168f706a96ed34569324da547c660915e0eea"


def _iso(seconds: float) -> str:
    return pd.Timestamp(seconds, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _rest_book(created: float, *guids: str) -> str:
    """A REST ``GetAllOrders`` reply holding a 0.3 bid at 100 + i per guid."""
    return json.dumps(
        {
            "BuyOrders": [
                {"Guid": g, "Price": 100.0 + i, "Volume": 0.3}
                for i, g in enumerate(guids)
            ],
            "SellOrders": [{"Guid": "ask-1", "Price": 200.0, "Volume": 0.2}],
            "CreatedTimestampUtc": _iso(created),
        }
    )


def _new_order(guid: str, *, at: float, nonce: int = 1, price: int = 150) -> dict:
    """A stream ``NewOrder`` for a 0.3 bid, numbers parsed as cryptofeed does."""
    return {
        "Channel": "orderbook-xbt",
        "Nonce": nonce,
        "Data": {
            "OrderType": "LimitBid",
            "OrderGuid": guid,
            "Price": {"aud": Decimal(price), "usd": Decimal(price - 50)},
            "Volume": Decimal("0.3"),
        },
        "Time": int(at * 1000),
        "Event": "NewOrder",
    }


class _FakeRest:
    """Hands out scripted REST replies, the last one again once they run out."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.reads = 0

    async def read(self, url: str) -> str:
        self.reads += 1
        return self.replies[min(self.reads, len(self.replies)) - 1]


def _feed(rest: _FakeRest):
    """An Independent Reserve feed subscribed to BTC-AUD, recording its books."""
    from cryptofeed.defines import L3_BOOK

    from ob_analytics.live._cryptofeed_venues import IndependentReserveFeed

    class RecordingFeed(IndependentReserveFeed):
        # cryptofeed waits 1 / request_limit seconds after each REST request.
        request_limit = 1_000_000
        books: list[tuple[set[str], Any]]
        raws: list[Any]

        def exchange_symbol_to_std_symbol(self, symbol: str) -> str:
            return "BTC-AUD"

        async def callback(self, data_type, obj, receipt_timestamp):
            bids = obj.book.bid.to_dict()
            self.books.append(
                ({oid for orders in bids.values() for oid in orders}, obj.delta)
            )
            self.raws.append(obj.raw)

    feed = object.__new__(RecordingFeed)
    feed.books = []
    feed.raws = []
    feed.cross_check = False
    feed.http_conn = cast(Any, rest)
    feed.sandbox = False
    feed.max_depth = 0
    feed.subscription = {feed.std_channel_to_exchange(L3_BOOK): ["xbt-aud"]}
    feed._l3_book = {}
    feed._order_ids = defaultdict(dict)
    feed._sequence_no = {}
    return feed


class TestIndependentReserveOpeningBook:
    """The opening REST book must be newer than the first stream message."""

    def test_a_book_older_than_the_stream_is_fetched_again(self):
        from ob_analytics.live._cryptofeed_venues import REST_BOOK_LAG_S

        # "gone" was cancelled before the stream started; the first two books,
        # from the venue's cache, still hold it.
        rest = _FakeRest(
            _rest_book(T0 - 1.0, "kept", "gone"),
            _rest_book(T0 + REST_BOOK_LAG_S - 0.5, "kept", "gone"),
            _rest_book(T0 + REST_BOOK_LAG_S + 0.5, "kept"),
        )
        feed = _feed(rest)

        asyncio.run(feed._book(_new_order("new", at=T0), T0))

        assert rest.reads == 3
        opening, delta = feed.books[0]
        assert delta is None
        assert opening == {"kept"}
        # The stream message is applied on top of the new book.
        after, delta = feed.books[1]
        assert after == {"kept", "new"}
        assert [entry[0] for entry in delta["bid"]] == ["new"]

    def test_a_new_enough_book_is_taken_at_once(self):
        from ob_analytics.live._cryptofeed_venues import REST_BOOK_LAG_S

        rest = _FakeRest(_rest_book(T0 + REST_BOOK_LAG_S, "kept"))
        feed = _feed(rest)

        asyncio.run(feed._book(_new_order("new", at=T0), T0))

        assert rest.reads == 1
        assert feed.books[0][0] == {"kept"}

    def test_a_message_the_book_already_holds_leaves_it_unchanged(self):
        """The stream waits for the book, so it replays messages the book has."""
        from ob_analytics.live._cryptofeed_venues import REST_BOOK_LAG_S

        rest = _FakeRest(_rest_book(T0 + REST_BOOK_LAG_S, "kept", "new"))
        feed = _feed(rest)

        # "new" is the book's second bid, at 101.
        asyncio.run(feed._book(_new_order("new", at=T0, price=101), T0))

        assert feed.books[0][0] == {"kept", "new"}
        assert feed.books[1][0] == {"kept", "new"}
        bids = feed._l3_book["BTC-AUD"].book.bid.to_dict()
        assert bids == {
            Decimal(100): {"kept": Decimal("0.3")},
            Decimal(101): {"new": Decimal("0.3")},
        }

    def test_after_the_last_try_the_newest_book_is_used(self):
        from ob_analytics.live._cryptofeed_venues import MAX_BOOK_FETCHES

        rest = _FakeRest(_rest_book(T0 - 5.0, "kept", "gone"))
        feed = _feed(rest)

        with warnings_logged() as warnings:
            asyncio.run(feed._book(_new_order("new", at=T0), T0))

        assert rest.reads == MAX_BOOK_FETCHES
        assert feed.books[0][0] == {"kept", "gone"}
        assert any("no REST book newer than the stream" in w for w in warnings)

    def test_a_book_time_that_does_not_parse_is_taken_at_once(self):
        """A change to the venue's format must not stop every connection."""
        reply = json.loads(_rest_book(T0, "kept"))
        reply["CreatedTimestampUtc"] = "not a time"
        rest = _FakeRest(json.dumps(reply))
        feed = _feed(rest)

        asyncio.run(feed._book(_new_order("new", at=T0), T0))

        assert rest.reads == 1
        assert feed.books[0][0] == {"kept"}

    def test_a_new_order_the_book_holds_keeps_the_book_s_size(self):
        """A replayed new order must not undo a fill the book already shows."""
        from ob_analytics.live._cryptofeed_venues import REST_BOOK_LAG_S

        reply = json.loads(_rest_book(T0 + REST_BOOK_LAG_S, "kept", "filled"))
        reply["BuyOrders"][1]["Volume"] = 0.1  # 0.3 when placed, then a fill
        feed = _feed(_FakeRest(json.dumps(reply)))

        asyncio.run(feed._book(_new_order("filled", at=T0, price=101), T0))

        assert feed.books[1][1]["bid"] == [("filled", Decimal(101), Decimal("0.1"))]
        bids = feed._l3_book["BTC-AUD"].book.bid.to_dict()
        assert bids[Decimal(101)] == {"filled": Decimal("0.1")}
        # The archived frame is the venue's own, with the size it sent.
        assert feed.raws[1]["Data"]["Volume"] == Decimal("0.3")

    def test_a_reconnect_waits_for_a_book_newer_than_its_own_first_message(self):
        from ob_analytics.live._cryptofeed_venues import REST_BOOK_LAG_S

        later = T0 + 60.0
        rest = _FakeRest(
            _rest_book(T0 + REST_BOOK_LAG_S, "kept", "gone"),
            _rest_book(later - 1.0, "kept", "gone"),
            _rest_book(later + REST_BOOK_LAG_S, "kept"),
        )
        feed = _feed(rest)
        asyncio.run(feed._book(_new_order("new", at=T0), T0))
        # cryptofeed clears its book and sequence when it subscribes again.
        feed._l3_book = {}
        feed._order_ids = defaultdict(dict)
        feed._sequence_no = {}

        asyncio.run(feed._book(_new_order("other", at=later, nonce=50), later))

        assert rest.reads == 3
        reopened = [book for book, delta in feed.books if delta is None][-1]
        assert reopened == {"kept"}


class TestTheSourceUsesTheCorrectedFeed:
    def test_independent_reserve_resolves_to_the_corrected_feed(self):
        from ob_analytics.live._cryptofeed_venues import IndependentReserveFeed
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(
            settings=CryptofeedSettings(exchange="independent_reserve")
        )
        assert issubclass(src._exchange_class(), IndependentReserveFeed)
        assert src.diagnostics()["exchange"] == "INDEPENDENT_RESERVE"

    def test_a_caller_s_own_class_gets_the_corrected_feed_too(self):
        from cryptofeed.exchanges import IndependentReserve

        from ob_analytics.live._cryptofeed_venues import IndependentReserveFeed
        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        class Mine(IndependentReserve):
            pass

        feed_cls = CryptofeedSource(
            settings=CryptofeedSettings(exchange=Mine)
        )._exchange_class()
        assert issubclass(feed_cls, IndependentReserveFeed)
        assert issubclass(feed_cls, Mine)
        # This feed's _book runs before the caller's class and cryptofeed's.
        mro = feed_cls.__mro__
        assert mro.index(IndependentReserveFeed) < mro.index(Mine)

    def test_other_venues_keep_cryptofeed_s_feed(self):
        from cryptofeed.exchanges import EXCHANGE_MAP

        from ob_analytics.live.cryptofeed_source import (
            CryptofeedSettings,
            CryptofeedSource,
        )

        src = CryptofeedSource(settings=CryptofeedSettings(exchange="bitstamp"))
        assert src._exchange_class() is EXCHANGE_MAP["BITSTAMP"]


class TestTheCopyOfCryptofeedsCode:
    def test_cryptofeed_s_snapshot_is_the_one_copied(self):
        """``IndependentReserveFeed`` copies cryptofeed's ``_snapshot``.

        When this fails, cryptofeed has changed it: compare the new version
        with ``_snapshot`` and ``_load_book``, bring them into line, and
        update the hash.
        """
        import hashlib
        import inspect

        from cryptofeed.exchanges import IndependentReserve

        source = inspect.getsource(IndependentReserve._snapshot)
        assert hashlib.sha256(source.encode()).hexdigest() == SNAPSHOT_SHA256
