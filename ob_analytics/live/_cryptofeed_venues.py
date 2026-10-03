"""Corrections to cryptofeed's venue feeds, used in place of cryptofeed's own.

Imported only when a capture resolves its venue, since it needs cryptofeed.
:data:`FEEDS` maps a cryptofeed exchange id to the class that replaces it.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from time import time
from typing import Any

import pandas as pd
from cryptofeed.defines import ASK, BID, L3_BOOK
from cryptofeed.exchanges import IndependentReserve

# cryptofeed.types is compiled, so type checkers cannot read it.  cryptofeed's
# independent_reserve module imports OrderBook from there, so take it from that.
from cryptofeed.exchanges.independent_reserve import OrderBook
from loguru import logger

#: How far, in seconds, Independent Reserve's REST book can be older than its
#: ``CreatedTimestampUtc``.  Measured against the websocket stream over 48
#: books: the book held orders the stream had removed 0.5 to 0.75 s earlier,
#: and once 2.0 s earlier.
REST_BOOK_LAG_S = 3.0

#: How many times to fetch the REST book before taking one that is not new
#: enough.  The venue's cache serves books up to 3 s old, and cryptofeed allows
#: one request a second, so a new enough book usually takes 3 to 6 fetches.
MAX_BOOK_FETCHES = 15


def _created_at(book: dict[str, Any]) -> float | None:
    """The REST book's ``CreatedTimestampUtc`` in epoch seconds, if it has one.

    ``None`` also for a time that does not parse, so a change to the venue's
    format makes the feed take the first book, as cryptofeed does, rather than
    fail every connection.
    """
    created = book.get("CreatedTimestampUtc")
    if not created:
        return None
    try:
        return pd.Timestamp(created).timestamp()
    except (TypeError, ValueError):
        return None


class IndependentReserveFeed(IndependentReserve):
    """Independent Reserve, with an opening book no older than the stream.

    cryptofeed takes the opening book from REST when the first stream message
    arrives, and then applies that message and every one after it.  The REST
    book can be older than the first message: the venue serves it from a cache,
    and its contents are older than its ``CreatedTimestampUtc``.  An order
    cancelled between the two is in the book, but its cancel came before the
    stream started, so nothing ever removes it.  It stays in the book, and a
    trade at a price past it makes the book crossed.

    This feed fetches the REST book again until it was created at least
    :data:`REST_BOOK_LAG_S` after the first stream message.  The stream waits
    meanwhile: cryptofeed handles one message at a time, so the messages that
    arrive are kept and applied after the book.  Messages that the book
    already includes leave it as it was: a change or cancel to an order the
    book does not hold is skipped, and a new order the book holds is added
    again.  So a newer book loses nothing, and an older one does.

    The new order needs help to leave the book as it was.  cryptofeed applies
    it at the size the message gives, but the book may hold the order at a
    smaller size, after a fill.  Applied as sent, the message would raise
    the order's size until the fill's own message lowers it again, and the
    capture would record a size that went up.  So a new order the book already
    holds keeps the book's size.  The size is changed in a copy of the message;
    the raw frame the capture archives is the venue's own.
    """

    # The message being handled, as applied, and as the venue sent it.  For the
    # first one, the opening book is fetched inside cryptofeed's ``_book``,
    # before the message is applied.
    _message: dict | None = None
    _sent: dict | None = None

    async def _book(self, msg: dict, timestamp: float) -> None:
        applied = {**msg, "Data": dict(msg["Data"])}
        self._message, self._sent = applied, msg
        self._keep_held_size(applied)
        await super()._book(applied, timestamp)

    async def book_callback(
        self,
        book_type: str,
        book: Any,
        receipt_timestamp: float,
        timestamp=None,
        raw=None,
        sequence_number=None,
        checksum=None,
        delta=None,
    ):
        if raw is not None and raw is self._message:
            raw = self._sent
        await super().book_callback(
            book_type,
            book,
            receipt_timestamp,
            timestamp=timestamp,
            raw=raw,
            sequence_number=sequence_number,
            checksum=checksum,
            delta=delta,
        )

    def _keep_held_size(self, msg: dict) -> None:
        """Give a new order the book already holds the size the book holds."""
        if msg.get("Event") != "NewOrder":
            return
        data = msg["Data"]
        uuid = data.get("OrderGuid")
        for instrument, order_ids in self._order_ids.items():
            if uuid not in order_ids or instrument not in self._l3_book:
                continue
            price, side = order_ids[uuid]
            levels = self._l3_book[instrument].book[side]
            size = levels[price].get(uuid) if price in levels else None
            if size is not None:
                data["Volume"] = size
                return

    async def _snapshot(self, base: str, quote: str) -> None:
        url = self.rest_endpoints[0].route("l3book", self.sandbox).format(base, quote)
        msg = self._message
        not_before = None if msg is None else msg["Time"] / 1000 + REST_BOOK_LAG_S
        for _ in range(MAX_BOOK_FETCHES):
            timestamp = time()
            ret = json.loads(await self.http_conn.read(url), parse_float=Decimal)
            created = _created_at(ret)
            await asyncio.sleep(1 / self.request_limit)
            if not_before is None or created is None or created >= not_before:
                break
        else:
            logger.warning(
                "[cryptofeed] Independent Reserve sent no REST book newer than "
                "the stream in {} tries; orders cancelled just before the "
                "stream started may stay in the book",
                MAX_BOOK_FETCHES,
            )
        book = self._load_book(base, quote, ret)
        if msg is not None:
            self._keep_held_size(msg)
        await self.book_callback(L3_BOOK, book, timestamp, raw=ret)

    def _load_book(self, base: str, quote: str, ret: dict) -> OrderBook:
        """Replace the held book with a REST book, as cryptofeed does.

        This and :meth:`_snapshot` copy cryptofeed's own ``_snapshot``; a test
        fails when cryptofeed's version changes, so the copy is checked again.
        """
        normalized = self.exchange_symbol_to_std_symbol(f"{base}-{quote}")
        book = OrderBook(self.id, normalized, max_depth=self.max_depth)
        self._l3_book[normalized] = book
        for side, key in ((BID, "BuyOrders"), (ASK, "SellOrders")):
            for order in ret[key]:
                price = Decimal(order["Price"])
                size = Decimal(order["Volume"])
                uuid = order["Guid"]
                self._order_ids[normalized][uuid] = (price, side)
                if price in book.book[side]:
                    book.book[side][price][uuid] = size
                else:
                    book.book[side][price] = {uuid: size}
        return book


FEEDS: dict[str, Any] = {IndependentReserve.id: IndependentReserveFeed}
