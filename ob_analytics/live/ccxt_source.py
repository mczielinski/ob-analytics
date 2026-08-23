"""CCXT L2 live capturer -- one adapter for ~all crypto CEX + prediction markets.

Wraps CCXT / CCXT Pro (issue #106) so any CCXT-supported venue becomes an
ob-analytics source through a single adapter instead of a hand-written
connector per venue.  CCXT's unified order book is **price-level (L2)** for
every venue, so this capturer declares
:attr:`~ob_analytics.protocols.Level.L2` and emits price-level *depth updates*
that replay through the L2 path -- **no faked per-order IDs** (the exact
anti-pattern the L2 path exists to avoid).

Transport is chosen per venue from the exchange's declared capabilities:

* venues with CCXT Pro websockets (``exchange.has['watchOrderBook']``) stream
  via ``watch_order_book`` / ``watch_trades``;
* the rest (e.g. Kalshi / Polymarket, REST-only in CCXT) are polled via
  ``fetch_order_book`` / ``fetch_trades``.

Book updates are turned into depth rows by diffing CCXT's *maintained book*
against the previous state: a level whose absolute size changed emits its new
size; a level that vanished emits ``0`` (a removal).

``ccxt`` is an optional dependency (``pip install "ob-analytics[ccxt]"``; CCXT
Pro ships inside ``ccxt``).  It is imported lazily, so importing this module --
and listing capturers -- never requires ccxt to be installed.

The venue id and per-run knobs travel in :attr:`CaptureConfig.extras`::

    extras = {"exchange": "binance", "depth_limit": 100, "poll_interval": 1.0}

and the symbol is :attr:`CaptureConfig.pair` (e.g. ``"BTC/USDT"``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pandas as pd
from loguru import logger

from ob_analytics.live._base import CaptureConfig, EventDict, LiveCapturer
from ob_analytics.protocols import Level

# Per-venue defaults, overridable via CaptureConfig.extras.
_DEFAULT_DEPTH_LIMIT = 100
_DEFAULT_POLL_INTERVAL = 1.0  # seconds; REST-poll venues only


def _make_exchange(exchange_id: str) -> Any:
    """Instantiate a CCXT Pro exchange by id (lazy import of ``ccxt``).

    Raises :class:`ImportError` with an install hint if ccxt is absent, and
    :class:`ValueError` if *exchange_id* is not a known CCXT venue.
    """
    try:
        import ccxt.pro as ccxtpro
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "The ccxt source requires the 'ccxt' extra: "
            'pip install "ob-analytics[ccxt]"'
        ) from exc
    if exchange_id not in ccxtpro.exchanges:
        raise ValueError(
            f"Unknown CCXT exchange {exchange_id!r}; "
            f"expected one of {len(ccxtpro.exchanges)} ccxt.pro venues."
        )
    return getattr(ccxtpro, exchange_id)({"enableRateLimit": True})


def _epoch_ms_to_ts(ms: Any) -> pd.Timestamp:
    """CCXT timestamps are epoch-ms; ``None`` falls back to receive time."""
    if ms is None:
        return pd.Timestamp.now(tz="UTC")
    return pd.Timestamp(int(ms), unit="ms", tz="UTC")


class CcxtCapturer(LiveCapturer):
    """Live-capture any CCXT venue as an L2 depth + trade stream.

    Conforms to :class:`~ob_analytics.live.LiveCapturer`.  Instances are not
    reusable across runs -- construct a fresh one per capture.
    """

    name = "ccxt"
    resolution = Level.L2

    def __init__(self) -> None:
        self._exchange: Any = None
        self._symbol: str = ""
        self._depth_limit: int = _DEFAULT_DEPTH_LIMIT
        self._poll_interval: float = _DEFAULT_POLL_INTERVAL
        self._use_ws_book = False
        self._use_ws_trades = False
        # Last seen absolute size per price, per side -- the diff baseline.
        self._last: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}

        # Diagnostics (surfaced in meta.json via SupportsDiagnostics).
        self.exchange_id = ""
        self.book_updates = 0
        self.depth_rows = 0
        self.trade_events = 0
        self.errors = 0

    # -- configuration ------------------------------------------------------

    def _configure(self, config: CaptureConfig) -> None:
        """Resolve the exchange, symbol, and per-transport capabilities."""
        extras = config.extras or {}
        exchange = extras.get("exchange")
        if not exchange:
            raise ValueError(
                "ccxt source needs extras={'exchange': '<venue id>'} (e.g. 'binance')."
            )
        self._symbol = config.pair
        self._depth_limit = int(extras.get("depth_limit", _DEFAULT_DEPTH_LIMIT))
        self._poll_interval = float(extras.get("poll_interval", _DEFAULT_POLL_INTERVAL))

        # A string is a venue id (built via ccxt); anything else is treated as
        # a pre-built exchange object (tests / advanced callers).
        if isinstance(exchange, str):
            self.exchange_id = exchange
            self._exchange = _make_exchange(exchange)
        else:
            self.exchange_id = str(getattr(exchange, "id", "custom"))
            self._exchange = exchange

        has = getattr(self._exchange, "has", {}) or {}
        self._use_ws_book = bool(has.get("watchOrderBook"))
        self._use_ws_trades = bool(has.get("watchTrades"))
        logger.info(
            "[ccxt] {} {} book={} trades={}",
            self.exchange_id,
            self._symbol,
            "ws" if self._use_ws_book else "rest-poll",
            "ws" if self._use_ws_trades else "rest-poll",
        )

    # -- snapshot -----------------------------------------------------------

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        """Seed the book from one REST snapshot; yield absolute-size rows.

        A REST snapshot works for both transports and gives a clean starting
        book to diff subsequent updates against.
        """
        self._configure(config)
        book = await self._exchange.fetch_order_book(self._symbol, self._depth_limit)
        ts = _epoch_ms_to_ts(book.get("timestamp"))
        # CCXT's per-book monotonic sequence (``None`` when the venue omits it);
        # carried as the venue ``sequence`` for gap detection on the L2 path.
        nonce = book.get("nonce")
        for side, key in (("bid", "bids"), ("ask", "asks")):
            levels: dict[float, float] = {}
            for row in book.get(key) or ():
                price = float(row[0])
                size = float(row[1])
                levels[price] = size
                if size > 0:
                    yield {
                        "timestamp": ts,
                        "exchange_timestamp": ts,
                        "side": side,
                        "price": price,
                        "volume": size,
                        "sequence": nonce,
                    }
            self._last[side] = levels
        logger.info(
            "[ccxt] snapshot: {} bid / {} ask levels",
            len(self._last["bid"]),
            len(self._last["ask"]),
        )

    # -- stream -------------------------------------------------------------

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        """Interleave book + trade producers through a queue until the deadline.

        Book and trade feeds block independently, so each runs in its own task
        and pushes ``(kind, event, raw)`` onto a queue that this generator
        drains.  The run ends at ``config.minutes`` (production) or when both
        producers finish (finite feeds, i.e. tests).
        """
        deadline = time.monotonic() + config.minutes * 60.0
        queue: asyncio.Queue[tuple[str, EventDict, Any]] = asyncio.Queue()
        stop = asyncio.Event()
        producers = [
            asyncio.create_task(self._book_loop(queue, stop, deadline)),
            asyncio.create_task(self._trades_loop(queue, stop, deadline)),
        ]
        try:
            while True:
                if all(p.done() for p in producers) and queue.empty():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=min(remaining, 0.5)
                    )
                except TimeoutError:
                    continue
                yield item
        finally:
            stop.set()
            for p in producers:
                if not p.done():
                    p.cancel()
            for p in producers:
                try:
                    await p
                except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - draining producers on shutdown
                    pass
            await self._close()

    async def _book_loop(
        self,
        queue: asyncio.Queue[tuple[str, EventDict, Any]],
        stop: asyncio.Event,
        deadline: float,
    ) -> None:
        """Poll/stream the book and push depth diffs. Ends on stop/deadline."""
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                if self._use_ws_book:
                    book = await self._exchange.watch_order_book(
                        self._symbol, self._depth_limit
                    )
                else:
                    book = await self._exchange.fetch_order_book(
                        self._symbol, self._depth_limit
                    )
            except StopAsyncIteration:
                # Finite feed exhausted (tests). Real feeds block instead.
                return
            except Exception as exc:  # noqa: BLE001 - one bad frame ends the loop for v1
                self.errors += 1
                logger.warning("[ccxt] book loop ended on error: {!r}", exc)
                return
            self.book_updates += 1
            ts = _epoch_ms_to_ts(book.get("timestamp"))
            for row, raw in self._diff_book(book, ts):
                self.depth_rows += 1
                await queue.put(("depth", row, raw))
            if not self._use_ws_book:
                await asyncio.sleep(self._poll_interval)

    async def _trades_loop(
        self,
        queue: asyncio.Queue[tuple[str, EventDict, Any]],
        stop: asyncio.Event,
        deadline: float,
    ) -> None:
        """Poll/stream the trade tape and push trade events."""
        since: int | None = None
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                if self._use_ws_trades:
                    trades = await self._exchange.watch_trades(self._symbol)
                else:
                    trades = await self._exchange.fetch_trades(self._symbol, since)
            except StopAsyncIteration:
                return
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                logger.warning("[ccxt] trades loop ended on error: {!r}", exc)
                return
            for t in trades or ():
                ts_ms = t.get("timestamp")
                # REST polling can re-serve trades; drop anything <= the cursor.
                if (
                    not self._use_ws_trades
                    and since is not None
                    and ts_ms is not None
                    and ts_ms < since
                ):
                    continue
                self.trade_events += 1
                await queue.put(("trade", self._map_trade(t), t))
                if ts_ms is not None:
                    since = int(ts_ms) + 1
            if not self._use_ws_trades:
                await asyncio.sleep(self._poll_interval)

    # -- shutdown -----------------------------------------------------------

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        """L2 price levels have no lifecycle to close: emit nothing."""
        for _ in ():
            yield {}

    # -- translation (pure) -------------------------------------------------

    def _diff_book(
        self, book: dict[str, Any], ts: pd.Timestamp
    ) -> Iterator[tuple[EventDict, Any]]:
        """Yield ``(depth_row, raw)`` for levels that changed vs the last book.

        A changed/added level emits its new absolute size; a level present
        before but absent now emits ``0`` (removal).  The raw book frame is
        attached to the first emitted row of the update (for raw.jsonl) and
        ``None`` on the rest, so the full book is archived once, not per level.
        Updates the stored per-side book.
        """
        raw_attached = False
        # CCXT's per-book monotonic sequence (``None`` when the venue omits it);
        # every row from this book update carries it as the venue ``sequence``.
        nonce = book.get("nonce")
        for side, key in (("bid", "bids"), ("ask", "asks")):
            current: dict[float, float] = {}
            for row in book.get(key) or ():
                current[float(row[0])] = float(row[1])
            prev = self._last[side]
            for price, size in current.items():
                if prev.get(price) != size:
                    raw = None if raw_attached else book
                    raw_attached = True
                    yield (
                        {
                            "timestamp": ts,
                            "exchange_timestamp": ts,
                            "side": side,
                            "price": price,
                            "volume": size,
                            "sequence": nonce,
                        },
                        raw,
                    )
            for price in prev:
                if price not in current:
                    raw = None if raw_attached else book
                    raw_attached = True
                    yield (
                        {
                            "timestamp": ts,
                            "exchange_timestamp": ts,
                            "side": side,
                            "price": price,
                            "volume": 0.0,
                            "sequence": nonce,
                        },
                        raw,
                    )
            self._last[side] = current

    def _map_trade(self, t: dict[str, Any]) -> EventDict:
        """Map a CCXT trade to the universal trade-event shape.

        Public trades carry no order IDs, so ``buy_order_id`` /
        ``sell_order_id`` are left empty; ``side`` is CCXT's taker side.
        """
        return {
            "trade_id": t.get("id") or "",
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "exchange_timestamp": _epoch_ms_to_ts(t.get("timestamp")),
            "price": float(t["price"]),
            "amount": float(t["amount"]),
            "buy_order_id": "",
            "sell_order_id": "",
            "side": t.get("side") or "",
        }

    # -- internals ----------------------------------------------------------

    async def _close(self) -> None:
        """Close the exchange connection if open (idempotent)."""
        ex = self._exchange
        if ex is None:
            return
        close = getattr(ex, "close", None)
        if close is not None:
            try:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:  # noqa: BLE001
                logger.debug("[ccxt] exchange close error: {!r}", exc)
        self._exchange = None

    # -- diagnostics --------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Per-run counters for meta.json (SupportsDiagnostics)."""
        return {
            "exchange": self.exchange_id,
            "book_updates": self.book_updates,
            "depth_rows": self.depth_rows,
            "trade_events": self.trade_events,
            "errors": self.errors,
        }
