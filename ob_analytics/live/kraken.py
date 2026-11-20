"""Kraken per-order (L3) live capture from Kraken's WebSocket v2 feed.

``ob-analytics capture kraken --pair BTC/USD`` runs it.  The ``level3``
channel needs an API key with the permission "WebSocket interface - On".  A
normal Kraken account can make one at no cost.  Set the key in
``KRAKEN_API_KEY`` and its private key (the "secret") in
``KRAKEN_API_SECRET``.

What the capture does:

* Asks Kraken's REST API for a WebSocket token (``GetWebSocketsToken``),
  signed with the key.  The token must be used within 15 minutes, so the
  capture gets a new one for each segment.
* Opens two connections.  The ``level3`` channel on ``ws-l3.kraken.com``
  sends the book, one row per order: an opening book (``snapshot``), then
  ``add``, ``modify`` and ``delete`` events.  The public ``trade`` channel on
  ``ws.kraken.com`` sends the trades.
* Writes ``add`` as ``created``, ``modify`` as ``changed`` and ``delete`` as
  ``deleted``.  ``volume`` is the order's remaining visible size.
* Writes an amend as one row where it can.  Kraken shows a change of price
  as a ``delete`` and, in the next message, an ``add`` with the same id, so
  each ``delete`` is held for one message.  An amend that keeps or raises the
  size is written as one ``changed`` row at the new price.  One that lowers
  the size is written as ``deleted`` and a ``created`` under a new id
  (``<id>#1``), since a replay would read the smaller size as a fill at the
  old price.
* Keeps each price's queue as Kraken does: an order whose size goes down
  keeps its place, and one whose size goes up or whose price changes goes to
  the back.
* Checks Kraken's CRC32 checksum after every book message.  The feed has no
  sequence number, so the checksum is the only way to see a lost or
  misapplied message.  A mismatch is counted in ``checksum_failures``; by
  default it also ends the segment, and
  :func:`~ob_analytics.live.run_capture` starts the next one from a new
  opening book.
* Keeps only the levels Kraken sends: ``depth`` levels a side (1000 by
  default).  Kraken sends no ``delete`` for an order that leaves this window.
  The capture keeps such an order open, writes nothing, and waits.  When the
  window reaches its price again, Kraken sends every order at that price as
  an ``add``.  An order that comes back is written as ``changed`` if its
  price or size moved, and not written at all if not.  An order that does
  not come back left the book while it was outside the window, and is
  written as ``deleted`` then.  So inside the window the capture's book
  matches Kraken's after every message, and a window exit is never written
  as a cancel.
* Writes each trade with the taker's side.  The trade channel names no
  orders, so ``buy_order_id`` and ``sell_order_id`` are empty.  Trades
  received before the opening book are left out.
* Records the ``tick_size`` and ``lot_size`` every price and size it wrote
  sits on, in ``meta.json``.

Requires the ``websockets`` package (install via the ``[live]`` extra).  It
is imported when a capture starts, so importing this module never needs it.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import bisect
import hashlib
import heapq
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

import pandas as pd
from loguru import logger
from pydantic import SecretStr

from ob_analytics._secrets import remember
from ob_analytics.capture_files import ORIGIN_STREAM
from ob_analytics.config import Credential, InstrumentGrid, SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.live._base import (
    CaptureConfig,
    EventDict,
    OpenOrders,
    RecordedGrid,
    UnmappedMessageError,
    epoch_ms_to_ts,
)
from ob_analytics.protocols import Clocks, FeedType, Level, TradeAttribution
from ob_analytics.schemas import ORIGIN_COLUMN
from ob_analytics.sources import CliOption

__all__ = [
    "KrakenSettings",
    "KrakenSource",
]

L3_WS_URL = "wss://ws-l3.kraken.com/v2"
TRADE_WS_URL = "wss://ws.kraken.com/v2"
REST_URL = "https://api.kraken.com"
TOKEN_PATH = "/0/private/GetWebSocketsToken"
KEYS_URL = "https://pro.kraken.com/app/settings/api"

_INSTALL_HINT = (
    'The kraken source needs the live extra: pip install "ob-analytics[live]"'
)

# How long to wait for the opening book after subscribing.
SNAPSHOT_TIMEOUT_SECONDS = 30.0

# Kraken's checksum covers this many price levels a side, whatever the depth.
CHECKSUM_LEVELS = 10

# How many frames may wait to be parsed before the readers wait too.
QUEUE_FRAMES = 100_000

# How many deleted order ids the capture remembers, to give an order Kraken
# adds again a new recorded id.  Kraken's amends re-add an id in the very next
# message, which needs one; the rest is a margin.
ENDED_IDS_KEPT = 100_000

_ACTIONS = {"add": "created", "modify": "changed", "delete": "deleted"}
_SIDES = {"bids": "bid", "asks": "ask"}
_TAKER_SIDES = {"buy", "sell"}

# What Kraken's REST errors mean for a user, by the error's prefix.
_TOKEN_ERRORS = {
    "EAPI:Invalid key": (
        "Kraken refused the API key. Check that KRAKEN_API_KEY is the key and "
        "KRAKEN_API_SECRET its private key, and that the key has not been "
        "deleted."
    ),
    "EAPI:Invalid signature": (
        "Kraken refused the signature. Check that KRAKEN_API_SECRET is the "
        "private key of the key in KRAKEN_API_KEY."
    ),
    "EGeneral:Permission denied": (
        'The Kraken API key lacks the permission "WebSocket interface - On". '
        "Turn it on for the key in Kraken's API settings."
    ),
    "EAPI:Invalid nonce": (
        "Kraken refused the request's nonce. Another program that uses the "
        "same key may have sent a larger one; give this capture its own key."
    ),
}


class KrakenSettings(SourceSettings):
    """The API key and book depth of a Kraken capture.

    Both key fields are read from the environment when not given: the key
    from ``KRAKEN_API_KEY``, and its private key (the "secret") from
    ``KRAKEN_API_SECRET``.  Kraken shows both when the key is made, in the
    API settings of a Kraken account.  The key needs the permission
    "WebSocket interface - On".  Kraken also asks for at least one
    permission: one that only reads, such as "Query Open Orders & Trades",
    is enough.  The key needs no permission to trade or to move funds.
    """

    api_key: Annotated[
        SecretStr | None,
        Credential(env="KRAKEN_API_KEY", issued_at=KEYS_URL),
    ] = None
    api_secret: Annotated[
        SecretStr | None,
        Credential(env="KRAKEN_API_SECRET", issued_at=KEYS_URL),
    ] = None
    depth: Annotated[
        Literal[10, 100, 1000],
        CliOption(help="Price levels a side Kraken sends"),
    ] = 1000
    on_checksum_failure: Annotated[
        Literal["restart", "count"],
        CliOption(
            help=(
                "What a checksum failure does: restart starts a new segment "
                "from a new opening book; count only counts it"
            )
        ),
    ] = "restart"


# -- signing ------------------------------------------------------------------


def _decoded_secret(secret: str) -> bytes:
    """The private key as bytes; Kraken gives it in base64.

    Raises :class:`~ob_analytics.exceptions.ConfigError` when the text is not
    base64.  The message never quotes the text.
    """
    try:
        return base64.b64decode(secret.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ConfigError(
            "KRAKEN_API_SECRET is not base64 text. Use the private key Kraken "
            "showed when the key was made."
        ) from exc


def _sign(path: str, nonce: str, post_data: str, secret: str) -> str:
    """The ``API-Sign`` header for one private REST request to Kraken.

    The HMAC-SHA512, keyed with the decoded private key, of the URI path
    followed by the SHA-256 of the nonce and the encoded POST data; in
    base64.
    """
    digest = hashlib.sha256((nonce + post_data).encode()).digest()
    mac = hmac.new(_decoded_secret(secret), path.encode() + digest, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


_last_nonce = 0


def _nonce() -> str:
    """A nonce larger than any this process sent: the time in milliseconds."""
    global _last_nonce
    _last_nonce = max(_last_nonce + 1, int(time.time() * 1000))
    return str(_last_nonce)


def _fetch_token(api_key: str, api_secret: str) -> str:
    """Get a WebSocket token from Kraken's REST API.

    Raises :class:`~ob_analytics.exceptions.ConfigError` when Kraken refuses
    the key, with what to change.
    """
    nonce = _nonce()
    post_data = urllib.parse.urlencode({"nonce": nonce})
    req = urllib.request.Request(
        REST_URL + TOKEN_PATH,
        data=post_data.encode(),
        headers={
            "API-Key": api_key,
            "API-Sign": _sign(TOKEN_PATH, nonce, post_data, api_secret),
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "ob-analytics-collector/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise ConfigError(
            f"Kraken refused the request for a WebSocket token (HTTP {exc.code})."
        ) from exc
    return _token_from(body)


def _token_from(body: Any) -> str:
    """The token in a ``GetWebSocketsToken`` reply, or the reason there is none."""
    errors = body.get("error") if isinstance(body, dict) else None
    if errors:
        text = str(errors[0])
        for prefix, message in _TOKEN_ERRORS.items():
            if text.startswith(prefix):
                raise ConfigError(message)
        raise ConfigError(f"Kraken gave no WebSocket token: {text}")
    try:
        token = str(body["result"]["token"])
    except (KeyError, TypeError) as exc:
        raise ConfigError("Kraken's reply holds no WebSocket token.") from exc
    # The token is as good as the key for an hour of reading the account's
    # private channels, so keep it out of every file the capture writes.
    remember(token)
    return token


# -- the book and its checksum ------------------------------------------------


def _number(value: Any) -> Decimal:
    """A price or size from a frame, exact as Kraken sent it.

    Frames are parsed with ``parse_float=Decimal``, so a number keeps the
    digits Kraken wrote, trailing zeros included; the checksum needs them.
    Raises :class:`~decimal.InvalidOperation` for anything that is not a
    finite number.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise InvalidOperation(f"not a number: {value!r}")
    number = Decimal(value)
    if not number.is_finite():
        raise InvalidOperation(f"not a finite number: {value!r}")
    return number


def _checksum_text(number: Decimal) -> str:
    """One number as the checksum reads it: no point and no leading zeros."""
    return format(number, "f").replace(".", "").lstrip("0")


class _Book:
    """Kraken's book of one symbol, as the ``level3`` channel shows it.

    Each side maps a price to the orders resting there, in queue order: order
    id to remaining size.  Only the levels inside the window are held;
    :meth:`truncate` removes the rest, as Kraken's guide says a client must.
    """

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.sides: dict[str, dict[Decimal, dict[str, Decimal]]] = {
            "bid": {},
            "ask": {},
        }
        # order id -> (side, price), for the orders held.
        self.where: dict[str, tuple[str, Decimal]] = {}

    def __contains__(self, order_id: str) -> bool:
        return order_id in self.where

    def put(self, order_id: str, side: str, price: Decimal, qty: Decimal) -> None:
        """Rest an order at the back of its level, or change it in place.

        An order that keeps its price and does not grow keeps its place in
        the queue.  One that moves to another price, or grows, joins the back
        of its level, as Kraken's checksum shows it does.
        """
        old = self.where.get(order_id)
        if old is not None and (
            old != (side, price) or qty > self.sides[side][price][order_id]
        ):
            self.remove(order_id)
        self.sides[side].setdefault(price, {})[order_id] = qty
        self.where[order_id] = (side, price)

    def remove(self, order_id: str) -> None:
        """Take an order off the book; drop its level if it was the last."""
        side, price = self.where.pop(order_id)
        level = self.sides[side][price]
        del level[order_id]
        if not level:
            del self.sides[side][price]

    def top(self, side: str, n: int) -> list[Decimal]:
        """The best *n* prices of *side*, best first, without sorting the side."""
        pick = heapq.nlargest if side == "bid" else heapq.nsmallest
        return pick(n, self.sides[side])

    def truncate(self) -> list[tuple[str, str, Decimal, Decimal]]:
        """Cut each side to the window; return what was cut.

        Each order cut comes back as ``(order_id, side, price, qty)``.
        """
        cut: list[tuple[str, str, Decimal, Decimal]] = []
        for side, levels in self.sides.items():
            if len(levels) <= self.depth:
                continue
            # The worst levels past the window: usually one, so no full sort.
            pick = heapq.nsmallest if side == "bid" else heapq.nlargest
            for price in pick(len(levels) - self.depth, levels):
                for order_id, qty in levels.pop(price).items():
                    del self.where[order_id]
                    cut.append((order_id, side, price, qty))
        return cut

    def worst_held(self, side: str) -> Decimal | None:
        """The worst price held on *side* if the side is full, else ``None``.

        A side with fewer levels than the window shows all of Kraken's book
        on that side, so no price lies beyond it.
        """
        levels = self.sides[side]
        if len(levels) < self.depth:
            return None
        return min(levels) if side == "bid" else max(levels)

    def checksum(self) -> int:
        """Kraken's CRC32 of the top ten levels a side, order by order.

        The asks from the lowest price, then the bids from the highest; each
        order adds its price and then its size, with the point and leading
        zeros taken out.
        """
        parts: list[str] = []
        for side in ("ask", "bid"):
            for price in self.top(side, CHECKSUM_LEVELS):
                price_text = _checksum_text(price)
                for qty in self.sides[side][price].values():
                    parts.append(price_text)
                    parts.append(_checksum_text(qty))
        return binascii.crc32("".join(parts).encode()) & 0xFFFFFFFF


class _Parked:
    """The orders outside the window, still open in the capture.

    Indexed by price, so after each message the capture can find the parked
    orders the window has reached again without reading them all.
    """

    def __init__(self) -> None:
        # order id -> (side, price).
        self.where: dict[str, tuple[str, Decimal]] = {}
        self.ids: dict[str, dict[Decimal, set[str]]] = {"bid": {}, "ask": {}}
        # The parked prices of each side, ascending.
        self.prices: dict[str, list[Decimal]] = {"bid": [], "ask": []}

    def __contains__(self, order_id: str) -> bool:
        return order_id in self.where

    def __len__(self) -> int:
        return len(self.where)

    def add(self, order_id: str, side: str, price: Decimal) -> None:
        at_price = self.ids[side].get(price)
        if at_price is None:
            at_price = self.ids[side][price] = set()
            bisect.insort(self.prices[side], price)
        at_price.add(order_id)
        self.where[order_id] = (side, price)

    def discard(self, order_id: str) -> None:
        found = self.where.pop(order_id, None)
        if found is None:
            return
        side, price = found
        at_price = self.ids[side][price]
        at_price.discard(order_id)
        if not at_price:
            del self.ids[side][price]
            prices = self.prices[side]
            del prices[bisect.bisect_left(prices, price)]

    def reached(self, side: str, worst: Decimal | None) -> list[str]:
        """The parked ids on *side* at *worst* or better (all when ``None``)."""
        prices = self.prices[side]
        if worst is None:
            inside: Iterable[Decimal] = list(prices)
        elif side == "bid":
            inside = prices[bisect.bisect_left(prices, worst) :]
        else:
            inside = prices[: bisect.bisect_right(prices, worst)]
        return [order_id for price in inside for order_id in self.ids[side][price]]


def _parse_time(text: Any) -> pd.Timestamp | None:
    """An RFC 3339 time from Kraken, on the canonical clock; ``None`` if absent."""
    if not isinstance(text, str) or not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").as_unit("ns")


def _now() -> pd.Timestamp:
    return epoch_ms_to_ts(int(time.time() * 1000))


class _BookMismatch(UnmappedMessageError):
    """Kraken's checksum disagrees with the capture's book.

    Carries the rows of the message that showed it: they were applied to the
    capture's state, so they must still be written.
    """

    def __init__(self, message: str, rows: list[EventDict]) -> None:
        super().__init__(message)
        self.rows = rows


# -- the source ---------------------------------------------------------------


class KrakenSource:
    """Live-capture one Kraken pair, order by order, from the ``level3`` channel.

    Satisfies :class:`~ob_analytics.live.LiveSource`.  Its ``orders.csv`` and
    ``trades.csv`` replay offline through the ``bitstamp`` file format.
    Instances are not reusable across runs: construct a fresh one for each
    capture, as :func:`~ob_analytics.live.run_capture` does for each segment.
    """

    name = "kraken"
    level = Level.L3
    # Kraken's own book of resting orders: bids never rest above asks.
    feed_type = FeedType.MATCHED_BOOK
    # The book holds resting orders only, and trades name no orders, so a
    # trade can be matched to its maker only, by price and size.
    trade_attribution = TradeAttribution.MAKER_ONLY
    # Every book message carries Kraken's time; the capture adds its own.
    clocks = Clocks.BOTH
    # Kraken quotes BTC/USD in tenths of a dollar and sizes in 1e-8 BTC; the
    # capture records the grid it wrote on in meta.json.
    grid = InstrumentGrid()

    def __init__(self, settings: KrakenSettings | None = None) -> None:
        # Typed as the protocol's SourceSettings so the source satisfies it;
        # _key() requires KrakenSettings by the time a capture connects.
        self.settings: SourceSettings = settings or KrakenSettings()
        self._symbol = ""
        self._depth = getattr(self.settings, "depth", 1000)
        self._book = _Book(self._depth)
        self._parked = _Parked()
        # Every order with a written lifecycle that has not ended: the ones
        # in the book and the parked ones.  Read for the shutdown deletes.
        self._open = OpenOrders()
        # Orders the capture ended because they did not come back to the
        # window; a later ``add`` of one gets a new recorded id so its two
        # lifecycles stay apart.
        self._ended_unseen: set[str] = set()
        # The last orders Kraken deleted, oldest first, so one added again
        # more than a message later gets a new recorded id too.
        self._ended_deleted: OrderedDict[str, None] = OrderedDict()
        # Kraken id -> how many new recorded ids it has had.
        self._reuses: dict[str, int] = {}
        # ``deleted`` rows held for one message: Kraken amends an order's
        # price as a ``delete`` and, in the next message, an ``add`` with the
        # same id, which the capture writes as one ``changed``.
        self._held_deletes: dict[str, EventDict] = {}
        # Kraken id -> the id written to orders.csv, where they differ.
        self._recorded_id: dict[str, str] = {}
        self._snapshot_ts: pd.Timestamp | None = None
        self._stack: AsyncExitStack | None = None
        self._sockets: dict[str, Any] = {}
        # Bounded, so a parser that falls behind slows the readers instead of
        # holding an ever longer backlog in memory.
        # Each item is ("frame", text, receive time) or ("closed", error, None).
        self._queue: asyncio.Queue[tuple[str, Any, pd.Timestamp | None]] = (
            asyncio.Queue(QUEUE_FRAMES)
        )
        self._readers: list[asyncio.Task[None]] = []
        # Frames read while waiting for the opening book, the book's own
        # included; stream() hands them to the runner first, for raw.jsonl.
        self._early_frames: list[dict[str, Any]] = []

        # Diagnostics, written to meta.json.
        self.grid_seen = RecordedGrid()
        self.snapshot_orders = 0
        self.adds = 0
        self.modifies = 0
        self.deletes = 0
        self.checksum_checks = 0
        self.checksum_failures = 0
        self.window_exits = 0
        self.window_returns = 0
        self.window_unseen_deletes = 0
        self.readded_after_unseen_delete = 0
        self.readded_after_delete = 0
        self.amends = 0
        self.amends_split = 0
        self.messages_without_time = 0
        self.trade_events = 0
        self.pre_snapshot_trades_skipped = 0
        self.synthetic_deleted = 0
        # Trades received that could not be parsed, and were left out.
        self.dropped = 0

    # -- capability checks ---------------------------------------------------

    def preflight(self) -> None:
        """Raise now if the capture cannot run (SupportsPreflight).

        Raises :class:`ImportError` with the install hint when ``websockets``
        is missing, and :class:`~ob_analytics.exceptions.ConfigError` when the
        private key given is not base64.  A key that is not set is left to
        the capture's own check, which names the variable to set.
        """
        try:
            import websockets  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        secret = getattr(self.settings, "api_secret", None)
        if secret is not None:
            _decoded_secret(secret.get_secret_value())

    # -- snapshot -------------------------------------------------------------

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        """Get a token, subscribe, and yield the opening book as ``created`` rows."""
        self._symbol = config.pair
        api_key, api_secret = self._key()
        token = await asyncio.to_thread(_fetch_token, api_key, api_secret)
        await self._connect(token)
        mismatch: _BookMismatch | None = None
        try:
            entry, received = await self._await_snapshot()
            try:
                rows = self._apply_snapshot(entry, received)
            except _BookMismatch as exc:
                # The opening book is written all the same, so each of its
                # orders has the ``created`` row its shutdown ``deleted`` needs.
                rows, mismatch = exc.rows, exc
        except BaseException:
            # Nothing was written, so no order is open; and the stream will
            # not run, so its close never comes.
            self._open.clear()
            await self._close()
            raise
        for row in rows:
            yield row
        if mismatch is not None:
            await self._close()
            raise mismatch
        logger.info(
            "[kraken] {} opening book: {} orders on {} bid / {} ask levels",
            self._symbol,
            self.snapshot_orders,
            len(self._book.sides["bid"]),
            len(self._book.sides["ask"]),
        )

    async def _await_snapshot(self) -> tuple[dict[str, Any], pd.Timestamp]:
        """Read frames until the opening book arrives.

        Returns its data entry and the time it was received.
        """
        try:
            async with asyncio.timeout(SNAPSHOT_TIMEOUT_SECONDS):
                while True:
                    frame, received = await self._next_frame()
                    self._early_frames.append(frame)
                    entry = self._opening_book_in(frame)
                    if entry is not None:
                        return entry, received
        except TimeoutError as exc:
            raise TimeoutError(
                f"Kraken sent no opening book for {self._symbol!r} within "
                f"{SNAPSHOT_TIMEOUT_SECONDS:.0f} s"
            ) from exc

    def _opening_book_in(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """The opening book's entry in *msg*, if it is one.

        Raises :class:`~ob_analytics.exceptions.ConfigError` when Kraken
        refuses a subscription, and counts a trade that came first.
        """
        self._check_reply(msg)
        channel = msg.get("channel")
        if channel == "trade":
            self.pre_snapshot_trades_skipped += len(msg.get("data") or ())
            return None
        if channel == "level3" and msg.get("type") == "snapshot":
            return self._entry(msg)
        return None

    def _check_reply(self, msg: dict[str, Any]) -> None:
        """Raise if *msg* is Kraken refusing a subscription."""
        if msg.get("method") == "subscribe" and msg.get("success") is False:
            error = str(msg.get("error") or "no reason given")
            hint = ""
            if "token" in error.lower() or "permission" in error.lower():
                hint = (
                    ' The key needs the permission "WebSocket interface - On" '
                    "in Kraken's API settings."
                )
            raise ConfigError(f"Kraken refused the subscription: {error}.{hint}")

    def _entry(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """The data entry of *msg* for this capture's symbol."""
        for entry in msg.get("data") or ():
            if isinstance(entry, dict) and entry.get("symbol") == self._symbol:
                return entry
        return None

    def _apply_snapshot(
        self, entry: dict[str, Any], received: pd.Timestamp | None = None
    ) -> list[EventDict]:
        """Seed the book from the opening book; return its ``created`` rows.

        *received* is when the frame arrived; ``None`` means now.
        """
        received = received or _now()
        venue_ts = _parse_time(entry.get("timestamp"))
        if venue_ts is None:
            self.messages_without_time += 1
            venue_ts = received
        self._snapshot_ts = venue_ts
        rows: list[EventDict] = []
        for key, side in _SIDES.items():
            for order in entry.get(key) or ():
                order_id, price, qty = self._order_fields(order, "the opening book")
                self._book.put(order_id, side, price, qty)
                self._open[order_id] = (float(price), side, float(qty))
                rows.append(
                    self._row(order_id, "created", side, price, qty, received, venue_ts)
                )
        # Kraken sends the window it was asked for; a longer side would be
        # cut by the first update anyway.
        for order_id, side, price, _qty in self._book.truncate():
            self._park(order_id, side, price)
        self.snapshot_orders = len(rows)
        self._check_checksum(entry, rows)
        return rows

    # -- stream ---------------------------------------------------------------

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        """Read both connections until the deadline; yield orders and trades.

        A closed connection before the deadline raises
        :class:`ConnectionError`, so
        :func:`~ob_analytics.live.run_capture` starts a new segment.
        """
        deadline = time.monotonic() + config.minutes * 60.0
        last_progress = time.monotonic()
        try:
            for frame in self._early_frames:
                yield ("raw", {}, frame)
            self._early_frames.clear()
            while self._sockets:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                # asyncio.timeout, not wait_for: on Python 3.11 wait_for can drop the
                # cancel that stops this segment (see _runner._cancel_until_done).
                try:
                    async with asyncio.timeout(min(remaining, 5.0)):
                        frame, received = await self._next_frame()
                except TimeoutError:
                    continue
                except ConnectionError:
                    if time.monotonic() >= deadline:
                        return
                    raise
                try:
                    items = self.parse(frame, received)
                except UnmappedMessageError as exc:
                    # Archive the frame that broke the book before the
                    # segment ends, so raw.jsonl shows what went wrong, and
                    # write the rows it applied, so every order the shutdown
                    # closes has its ``created`` row.
                    yield ("raw", {}, frame)
                    for row in getattr(exc, "rows", ()):
                        yield ("order", row, None)
                    raise
                for item in items:
                    yield item

                now = time.monotonic()
                if now - last_progress >= 15.0:
                    logger.info(
                        "[kraken] open={} parked={} trades={} checksum "
                        "failures={} ~{:.1f} min remaining",
                        len(self._open),
                        len(self._parked),
                        self.trade_events,
                        self.checksum_failures,
                        max(0.0, (deadline - now) / 60.0),
                    )
                    last_progress = now
        finally:
            await self._close()

    def parse(
        self, msg: dict[str, Any], received: pd.Timestamp | None = None
    ) -> list[tuple[str, EventDict, Any]]:
        """Turn one frame after the opening book into ``(kind, event, raw)`` items.

        *msg* is the frame parsed with ``parse_float=Decimal``, and *received*
        the time it arrived (``None`` means now).  The first item
        carries it as ``raw``, for ``raw.jsonl``, and the rest ``None``, so the
        frame is archived once.  ``kind`` is ``"order"`` for a book event,
        ``"trade"`` for a trade, and ``"raw"`` (with an empty event) for a
        frame with neither.
        """
        self._check_reply(msg)
        channel = msg.get("channel")
        items: list[tuple[str, EventDict]] = []
        if channel == "level3":
            kind = msg.get("type")
            if kind == "snapshot":
                # A second opening book replaces the book, and its rows could
                # not be told apart from changes; a new segment starts instead.
                raise UnmappedMessageError(
                    "Kraken sent a new opening book during the stream; the "
                    "next segment starts from a new one"
                )
            entry = self._entry(msg)
            if kind == "update" and entry is not None:
                items = [("order", ev) for ev in self._apply_update(entry, received)]
        elif channel == "trade" and msg.get("type") == "update":
            items = [("trade", ev) for ev in self._trades(msg, received)]
        if not items:
            return [("raw", {}, msg)]
        return [
            (kind, event, msg if i == 0 else None)
            for i, (kind, event) in enumerate(items)
        ]

    def _apply_update(
        self, entry: dict[str, Any], received: pd.Timestamp | None = None
    ) -> list[EventDict]:
        """Apply one ``update`` entry to the book; return its order events."""
        received = received or _now()
        message_ts = _parse_time(entry.get("timestamp"))
        if message_ts is None:
            self.messages_without_time += 1
        # Deletes from the last message: an ``add`` of the same id in this one
        # makes it an amend; the rest are written as deleted, first.
        held, self._held_deletes = self._held_deletes, {}
        events: list[EventDict] = []
        for key, side in _SIDES.items():
            for order in entry.get(key) or ():
                events.extend(
                    self._apply_order(order, side, received, message_ts, held)
                )
        events[:0] = [self._end_held(row) for row in held.values()]
        for order_id, cut_side, price, _qty in self._book.truncate():
            self._park(order_id, cut_side, price)
        events.extend(self._end_unseen(received, message_ts or received))
        self._check_checksum(entry, events)
        return events

    def _apply_order(
        self,
        order: Any,
        side: str,
        received: pd.Timestamp,
        message_ts: pd.Timestamp | None,
        held: dict[str, EventDict],
    ) -> list[EventDict]:
        """Apply one order event to the book; return the rows to write.

        A ``delete`` returns nothing yet: its row is held for one message
        (see :meth:`_apply_update`).  *held* holds the last message's.
        """
        what = order.get("event") if isinstance(order, dict) else None
        action = _ACTIONS.get(what) if isinstance(what, str) else None
        if action is None:
            raise UnmappedMessageError(f"an order event of unknown kind: {order!r}")
        order_id, price, qty = self._order_fields(order, "an update")
        # The message's time is the venue's time for every event; without it,
        # an ``add`` still has its own (the time it was inserted).
        venue_ts = message_ts
        if venue_ts is None:
            own = _parse_time(order.get("timestamp")) if action == "created" else None
            venue_ts = own or received

        if action == "deleted":
            self.deletes += 1
            if order_id in self._book:
                self._book.remove(order_id)
            elif order_id in self._parked:
                self._parked.discard(order_id)
            else:
                raise UnmappedMessageError(
                    f"Kraken deleted order {order_id}, which the capture's book "
                    "does not hold: the book no longer matches Kraken's"
                )
            recorded = self._recorded_id.get(order_id, order_id)
            self._held_deletes[order_id] = self._row(
                recorded, "deleted", side, price, qty, received, venue_ts
            )
            return []

        if action == "changed":
            self.modifies += 1
            if order_id not in self._book and order_id not in self._parked:
                raise UnmappedMessageError(
                    f"Kraken changed order {order_id}, which the capture's book "
                    "does not hold: the book no longer matches Kraken's"
                )
            self._parked.discard(order_id)
            self._book.put(order_id, side, price, qty)
            return [
                self._record(order_id, "changed", side, price, qty, received, venue_ts)
            ]

        # An ``add``.
        self.adds += 1
        deleted = held.pop(order_id, None) or self._held_deletes.pop(order_id, None)
        if deleted is not None:
            # Deleted in the last message (or earlier in this one) and back
            # now, at a new price: an amend.  The order leaves its old level
            # for the back of the new.
            self.amends += 1
            self._book.put(order_id, side, price, qty)
            if float(qty) < deleted["volume"]:
                # An amend that shrinks the order: a replay would read the
                # smaller size as a fill (at the old price, if it moved) and
                # keep the order's place in the queue.  So it is written as a
                # cancel and a new order, under a new id.
                self.amends_split += 1
                ended = self._end_held(deleted)
                self._reuse(order_id)
                return [
                    ended,
                    self._record(
                        order_id, "created", side, price, qty, received, venue_ts
                    ),
                ]
            return [
                self._record(order_id, "changed", side, price, qty, received, venue_ts)
            ]
        if order_id in self._parked:
            # Back inside the window: written only if it moved meanwhile.
            self.window_returns += 1
            self._parked.discard(order_id)
            self._book.put(order_id, side, price, qty)
            recorded = self._recorded_id.get(order_id, order_id)
            if self._open.get(recorded) == (float(price), side, float(qty)):
                return []
            return [
                self._record(order_id, "changed", side, price, qty, received, venue_ts)
            ]
        if order_id in self._book:
            # Kraken added an order it already showed: take it as a change.
            self._book.put(order_id, side, price, qty)
            return [
                self._record(order_id, "changed", side, price, qty, received, venue_ts)
            ]
        if order_id in self._ended_unseen:
            self._ended_unseen.discard(order_id)
            # The capture ended this order when the window reached its price
            # and Kraken did not send it; it was there after all.  A second
            # ``created`` under one id would break the id's lifecycle, so the
            # order is written under a new id from here on.
            self.readded_after_unseen_delete += 1
            self._reuse(order_id)
        elif self._ended_deleted.pop(order_id, False) is None:
            # Kraken deleted this order more than a message ago; it is back.
            self.readded_after_delete += 1
            self._reuse(order_id)
        self._book.put(order_id, side, price, qty)
        return [self._record(order_id, "created", side, price, qty, received, venue_ts)]

    def _reuse(self, order_id: str) -> None:
        """Give an order whose lifecycle ended a new recorded id.

        A second ``created`` under one id would break that id's lifecycle, so
        the order is written as ``<id>#<n>`` from here on.
        """
        reuses = self._reuses[order_id] = self._reuses.get(order_id, 0) + 1
        self._recorded_id[order_id] = f"{order_id}#{reuses}"

    def _end_held(self, row: EventDict) -> EventDict:
        """Close the lifecycle of a held ``deleted`` row, and return the row."""
        order_id = str(row["id"]).split("#", 1)[0]
        self._recorded_id.pop(order_id, None)
        self._open.pop(row["id"], None)
        self._ended_deleted[order_id] = None
        self._ended_deleted.move_to_end(order_id)
        while len(self._ended_deleted) > ENDED_IDS_KEPT:
            self._ended_deleted.popitem(last=False)
        return row

    def _park(self, order_id: str, side: str, price: Decimal) -> None:
        """Keep an order that left the window open, and write nothing."""
        self.window_exits += 1
        self._parked.add(order_id, side, price)

    def _end_unseen(
        self, received: pd.Timestamp, venue_ts: pd.Timestamp
    ) -> list[EventDict]:
        """Delete the parked orders the window reached that Kraken did not send.

        When the window reaches a price again, Kraken sends every order at
        it.  A parked order there that did not come back left the book while
        it was outside the window, at a time the feed does not say; it is
        written as ``deleted`` at this message's time.
        """
        events: list[EventDict] = []
        for side in ("bid", "ask"):
            if not self._parked.prices[side]:
                continue
            for order_id in self._parked.reached(side, self._book.worst_held(side)):
                self._parked.discard(order_id)
                recorded = self._recorded_id.pop(order_id, order_id)
                last = self._open.pop(recorded, None)
                self._ended_unseen.add(order_id)
                self.window_unseen_deletes += 1
                if last is None:
                    continue
                price, direction, volume = last
                event: EventDict = {
                    "id": recorded,
                    "timestamp": received,
                    "exchange_timestamp": venue_ts,
                    "price": price,
                    "volume": volume,
                    "action": "deleted",
                    "direction": direction,
                    "venue": self.name,
                    "symbol": self._symbol,
                }
                events.append(event)
        return events

    def _check_checksum(self, entry: dict[str, Any], rows: list[EventDict]) -> None:
        """Compare the book with Kraken's checksum for this message.

        A mismatch is counted; with ``on_checksum_failure="restart"`` it also
        raises :class:`_BookMismatch`, which carries *rows* (the message's
        rows, already applied to the capture's state) so they are still
        written, and the next segment starts from a new opening book.
        """
        expected = entry.get("checksum")
        if expected is None:
            return
        self.checksum_checks += 1
        actual = self._book.checksum()
        try:
            matches = int(expected) == actual
        except (TypeError, ValueError):
            matches = False
        if matches:
            return
        self.checksum_failures += 1
        message = (
            f"Kraken's checksum for {self._symbol} is {expected!r}, but the "
            f"capture's book gives {actual}: the book no longer matches Kraken's"
        )
        if getattr(self.settings, "on_checksum_failure", "restart") == "restart":
            raise _BookMismatch(message, rows)
        logger.warning("[kraken] {}", message)
        if self.checksum_failures == 1:
            warnings.warn(
                f"{message}. The capture goes on and counts each failure in "
                "meta.json (checksum_failures).",
                stacklevel=2,
            )

    def _trades(
        self, msg: dict[str, Any], received: pd.Timestamp | None = None
    ) -> list[EventDict]:
        """The trade events in one ``trade`` frame; unparseable ones are dropped."""
        received = received or _now()
        events: list[EventDict] = []
        for trade in msg.get("data") or ():
            if not isinstance(trade, dict) or trade.get("symbol") != self._symbol:
                continue
            try:
                price = float(_number(trade["price"]))
                qty = float(_number(trade["qty"]))
            except (KeyError, InvalidOperation):
                self.dropped += 1
                continue
            side = trade.get("side")
            venue_ts = _parse_time(trade.get("timestamp"))
            if side not in _TAKER_SIDES or venue_ts is None:
                self.dropped += 1
                continue
            if self._snapshot_ts is not None and venue_ts < self._snapshot_ts:
                # Before the opening book: an order it filled completely is
                # not in the capture.
                self.pre_snapshot_trades_skipped += 1
                continue
            event: EventDict = {
                "trade_id": trade.get("trade_id", ""),
                "timestamp": received,
                "exchange_timestamp": venue_ts,
                "price": price,
                "volume": qty,
                "buy_order_id": "",
                "sell_order_id": "",
                "direction": side,
                "venue": self.name,
                "symbol": self._symbol,
            }
            self.grid_seen.fit(event)
            self.trade_events += 1
            events.append(event)
        return events

    # -- shutdown -------------------------------------------------------------

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        """Yield a ``deleted`` event for every order still open, parked ones too.

        A delete held for an amend that never came is written first, at its
        own time.
        """
        held, self._held_deletes = self._held_deletes, {}
        for row in held.values():
            row[ORIGIN_COLUMN] = ORIGIN_STREAM
            yield self._end_held(row)
        events = self._open.close(_now(), venue=self.name, symbol=self._symbol)
        for event in events:
            self.synthetic_deleted += 1
            yield event

    # -- internals ------------------------------------------------------------

    def _order_fields(self, order: Any, where: str) -> tuple[str, Decimal, Decimal]:
        """``(order_id, price, qty)`` of one order; raise if it cannot be read."""
        order_id = order.get("order_id") if isinstance(order, dict) else None
        if not isinstance(order_id, str) or not order_id:
            raise UnmappedMessageError(
                f"cannot parse an order in {where}: {order!r} (no order id)"
            )
        try:
            return order_id, _number(order["limit_price"]), _number(order["order_qty"])
        except (KeyError, InvalidOperation) as exc:
            raise UnmappedMessageError(
                f"cannot parse an order in {where}: {order!r} ({exc!r})"
            ) from exc

    def _record(
        self,
        order_id: str,
        action: str,
        side: str,
        price: Decimal,
        qty: Decimal,
        received: pd.Timestamp,
        venue_ts: pd.Timestamp,
    ) -> EventDict:
        """Note an order as open and return its row, under its recorded id."""
        recorded = self._recorded_id.get(order_id, order_id)
        self._open[recorded] = (float(price), side, float(qty))
        return self._row(recorded, action, side, price, qty, received, venue_ts)

    def _row(
        self,
        order_id: str,
        action: str,
        side: str,
        price: Decimal,
        qty: Decimal,
        received: pd.Timestamp,
        venue_ts: pd.Timestamp,
    ) -> EventDict:
        """One order event."""
        event: EventDict = {
            "id": order_id,
            "timestamp": received,
            "exchange_timestamp": venue_ts,
            "price": float(price),
            "volume": float(qty),
            "action": action,
            "direction": side,
            "venue": self.name,
            "symbol": self._symbol,
        }
        self.grid_seen.fit(event)
        return event

    def _key(self) -> tuple[str, str]:
        """The API key and its private key from the settings."""
        settings = self.settings
        if not isinstance(settings, KrakenSettings):
            raise TypeError(
                f"KrakenSource needs KrakenSettings; got {type(settings).__name__}."
            )
        # run_capture checks the credentials before it gets here; a source
        # driven another way gets the same message.
        settings.check_credentials(self.name)
        if settings.api_key is None or settings.api_secret is None:
            raise ConfigError("Kraken needs KRAKEN_API_KEY and KRAKEN_API_SECRET")
        return (
            settings.api_key.get_secret_value(),
            settings.api_secret.get_secret_value(),
        )

    async def _connect(self, token: str) -> None:
        """Open both connections, subscribe, and start reading them."""
        import websockets

        stack = AsyncExitStack()
        try:
            for name, url in (("level3", L3_WS_URL), ("trade", TRADE_WS_URL)):
                self._sockets[name] = await stack.enter_async_context(
                    websockets.connect(
                        url,
                        ping_interval=20,
                        ping_timeout=20,
                        max_size=2**24,
                        close_timeout=2,
                    )
                )
            await self._sockets["level3"].send(
                json.dumps(
                    {
                        "method": "subscribe",
                        "params": {
                            "channel": "level3",
                            "symbol": [self._symbol],
                            "depth": self._depth,
                            "snapshot": True,
                            "token": token,
                        },
                    }
                )
            )
            await self._sockets["trade"].send(
                json.dumps(
                    {
                        "method": "subscribe",
                        "params": {
                            "channel": "trade",
                            "symbol": [self._symbol],
                            "snapshot": False,
                        },
                    }
                )
            )
        except BaseException:
            self._sockets.clear()
            await stack.aclose()
            raise
        self._stack = stack
        self._readers = [
            asyncio.create_task(self._read(name, ws))
            for name, ws in self._sockets.items()
        ]

    async def _read(self, name: str, ws: Any) -> None:
        """Put each frame from one connection on the queue, then its end."""
        try:
            async for raw in ws:
                # Stamped on arrival: the frame may wait in the queue.
                await self._queue.put(("frame", raw, _now()))
        except Exception as exc:  # noqa: BLE001 - handed to the stream
            await self._queue.put(("closed", ConnectionError(f"{name}: {exc!r}"), None))
            return
        await self._queue.put(
            ("closed", ConnectionError(f"{name}: connection closed"), None)
        )

    async def _next_frame(self) -> tuple[dict[str, Any], pd.Timestamp]:
        """The next frame from either connection, and the time it arrived.

        Numbers are read as :class:`~decimal.Decimal`, so they keep every
        digit Kraken sent; ``raw.jsonl`` writes them as strings.  Raises
        :class:`ConnectionError` when a connection closes.  A frame that is
        not a JSON object is skipped.
        """
        while True:
            kind, value, received = await self._queue.get()
            if kind == "closed":
                logger.warning("[kraken] {}", value)
                raise value
            try:
                frame = json.loads(value, parse_float=Decimal)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(frame, dict) and received is not None:
                return frame, received

    async def _close(self) -> None:
        """Stop the readers and close both connections. Idempotent."""
        for task in self._readers:
            task.cancel()
        for task in self._readers:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - closing on the way out
                pass
        self._readers = []
        self._sockets.clear()
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as exc:  # noqa: BLE001 - closing on the way out
                logger.debug("[kraken] ws close raised: {!r}", exc)
            self._stack = None

    # -- diagnostics ----------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Per-run counters for meta.json (SupportsDiagnostics)."""
        return {
            "depth": self._depth,
            "snapshot_orders": self.snapshot_orders,
            "adds": self.adds,
            "modifies": self.modifies,
            "deletes": self.deletes,
            "checksum_checks": self.checksum_checks,
            "checksum_failures": self.checksum_failures,
            "window_exits": self.window_exits,
            "window_returns": self.window_returns,
            "window_unseen_deletes": self.window_unseen_deletes,
            "readded_after_unseen_delete": self.readded_after_unseen_delete,
            "readded_after_delete": self.readded_after_delete,
            "amends": self.amends,
            "amends_split": self.amends_split,
            "parked_at_end": len(self._parked),
            "messages_without_time": self.messages_without_time,
            "trade_events": self.trade_events,
            "pre_snapshot_trades_skipped": self.pre_snapshot_trades_skipped,
            "synthetic_deleted": self.synthetic_deleted,
            "dropped": self.dropped,
            **self.grid_seen.fields(),
        }


# ── Register this source ──────────────────────────────────────────────
# Registered unconditionally (importing this module never imports
# websockets); a capture without the ``[live]`` extra stops at preflight with
# the install hint.
from ob_analytics.sources import register_source

register_source("kraken", KrakenSource)
