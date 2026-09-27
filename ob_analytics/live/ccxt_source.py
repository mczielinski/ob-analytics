"""CCXT L2 live capturer -- one adapter for ~all crypto CEX + prediction markets.

Wraps CCXT / CCXT Pro so any CCXT-supported venue becomes an ob-analytics
source through a single adapter instead of a hand-written connector per
venue.  CCXT's unified order book is **price-level (L2)** for
every venue, so this capturer declares
:attr:`~ob_analytics.protocols.Level.L2` and emits price-level *depth updates*
that replay through the L2 path -- **no faked per-order IDs** (the exact
anti-pattern the L2 path exists to avoid).

Transport is chosen per venue from the exchange's declared capabilities:

* venues with CCXT Pro websockets (``exchange.has['watchOrderBook']``) stream
  via ``watch_order_book`` / ``watch_trades``;
* the rest are polled via ``fetch_order_book`` / ``fetch_trades``.

CCXT's prediction markets (``ccxt.prediction``: Kalshi, Polymarket, ...) are
reached by the same venue id.  Kalshi has no websocket there and is polled;
Polymarket streams.

Book updates are turned into depth rows by diffing CCXT's *maintained book*
against the previous state: a level whose absolute size changed emits its new
size; a level that vanished emits ``0`` (a removal).

``ccxt`` is an optional dependency (``pip install "ob-analytics[ccxt]"``; CCXT
Pro ships inside ``ccxt``).  It is imported lazily, so importing this module --
and listing capturers -- never requires ccxt to be installed.

The venue id and per-run knobs are typed :class:`CcxtSettings` carried by the
source::

    CcxtSource(settings=CcxtSettings(exchange="binance", depth_limit=100))

and the symbol is :attr:`CaptureConfig.pair` (e.g. ``"BTC/USDT"``).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pandas as pd
from loguru import logger

from ob_analytics._utils import off_tick_grid
from ob_analytics.config import SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.live._base import CaptureConfig, EventDict
from ob_analytics.protocols import FeedType, Level, SequenceKind, TradeAttribution

# Per-venue defaults, overridable via CcxtSettings.
_DEFAULT_DEPTH_LIMIT = 100
_DEFAULT_POLL_INTERVAL = 1.0  # seconds; REST-poll venues only

# CCXT's precision modes (ccxt.DECIMAL_PLACES / ccxt.TICK_SIZE), written out so
# this module never imports ccxt.  In TICK_SIZE mode a market's price precision
# is the tick itself; in DECIMAL_PLACES mode it is a count of decimals.
_CCXT_DECIMAL_PLACES = 2
_CCXT_TICK_SIZE = 4

# The finest tick a capture will shrink its recorded tick size to: the default
# lot grid, finer than any venue quotes a price.
_FINEST_TICK = 1e-8

# How many recent trades a capture remembers to recognise one delivered twice.
_SEEN_TRADES_LIMIT = 10_000

# How many depth rows _book_loop enqueues before it yields to the event loop.
# A whole-book venue's opening diff (or the one after a resync) can carry
# thousands of rows; queue.put() on an unbounded queue never actually
# suspends, so without a deliberate yield that whole burst would run as one
# uninterrupted stretch and starve _trades_loop and the stop/deadline checks
# in stream() for its entire duration.
_YIELD_EVERY_DEPTH_ROWS = 256


class CcxtSettings(SourceSettings):
    """Typed settings for :class:`CcxtSource` (replaces the former extras dict).

    Attributes
    ----------
    exchange : str or object
        The CCXT venue id (e.g. ``"binance"``, ``"kraken"``), or a pre-built
        exchange object for tests / advanced callers.  Empty by default so a
        source can be constructed before the venue is known; the value is
        required by the time a capture starts (:meth:`CcxtSource._configure`
        raises otherwise).
    depth_limit : int
        Order-book depth (levels per side) to request.  The capture records
        whatever CCXT then hands back, uncropped, so this is a floor rather
        than a cap: Kraken keeps its own subscribed window at exactly this
        many levels, but Coinbase, Bitstamp and OKX ignore it and always hand
        back their whole book, and Binance and its family are asked for at
        least 1,000 levels internally regardless of this value (see
        :data:`_WHOLE_BOOK_VENUES`), so their recorded depth is at least that
        many too.
    poll_interval : float
        Seconds between REST polls, for REST-only venues.
    market_data_mirror : bool
        Read from the venue's market-data-only endpoints instead of its main
        ones (see :data:`MARKET_DATA_MIRRORS`).  Spot markets only.  Binance's
        main endpoints refuse some locations; whether you may use the mirror
        from yours is for you to check against the venue's terms.
    """

    exchange: Any = ""
    depth_limit: int = _DEFAULT_DEPTH_LIMIT
    poll_interval: float = _DEFAULT_POLL_INTERVAL
    market_data_mirror: bool = False


#: Prefix that picks a venue from CCXT's prediction markets.  ``binance`` and
#: ``hyperliquid`` name both a crypto exchange and a prediction market; the
#: plain id means the crypto exchange, ``prediction/binance`` the other one.
PREDICTION_PREFIX = "prediction/"

#: Market-data-only endpoints, by CCXT venue id: the REST base for public
#: spot data and the spot websocket.  They serve public data and no trading.
MARKET_DATA_MIRRORS: dict[str, dict[str, str]] = {
    "binance": {
        "rest": "https://data-api.binance.vision/api/v3",
        "ws": "wss://data-stream.binance.vision/ws",
    },
}

#: Venues whose websocket sends changes at every depth and whose CCXT book
#: deletes the levels past the ``limit`` it is given, with the most levels
#: their REST snapshot returns.  A deleted level is never sent again unless it
#: changes, so after the price moves away and back the book has holes near
#: the top.  These venues are asked for the whole book, which the capture
#: records in full -- cropping it to ``depth_limit`` would turn a level that
#: is still resting, just outside that crop, into what looks like a cancel
#: (issue #275).  (Other venues differ: Kraken subscribes at ``limit`` levels
#: and keeps the window filled, so a level leaving it really is gone from
#: Kraken's own view; Coinbase, Bitstamp and OKX ignore the limit and hand
#: ccxt the whole book too, so nothing needs to crop those either.)
_WHOLE_BOOK_VENUES: dict[str, int] = {
    "binance": 5000,
    "binanceus": 5000,
    "binanceusdm": 1000,
    "binancecoinm": 1000,
}

# The fewest levels ccxt is asked to track for a whole-book venue, whatever
# depth_limit is: ccxt's own default, and deep enough that a shallow
# depth_limit does not make ccxt itself evict (and, per the docstring above,
# potentially never recover) a level near the top. Because the capture now
# records everything ccxt tracks, this floor is also the least a whole-book
# venue's capture records by default.
_MIN_WHOLE_BOOK_SNAPSHOT = 1000


def _size_whole_book_snapshot(
    exchange: Any, exchange_id: str, depth_limit: int
) -> None:
    """Make ccxt's opening snapshot at least as deep as *depth_limit*.

    Past its opening snapshot, ccxt learns a level only when it changes, so a
    depth deeper than the snapshot has gaps at the bottom.  Raises
    :class:`~ob_analytics.exceptions.ConfigError` when *depth_limit* is more
    than the venue's snapshot can return.
    """
    most = _WHOLE_BOOK_VENUES[exchange_id]
    if depth_limit > most:
        raise ConfigError(
            f"{exchange_id} returns at most {most} levels a side in a snapshot; "
            f"--depth-limit {depth_limit} would leave the deeper levels "
            "incomplete."
        )
    exchange.options["watchOrderBookLimit"] = max(depth_limit, _MIN_WHOLE_BOOK_SNAPSHOT)


# How many times one capture restarts a book that lost sync before it gives up.
_MAX_BOOK_RESYNCS = 10


def _lost_sync(exc: Exception) -> bool:
    """Whether *exc* is ccxt reporting a missing book update.

    ccxt raises ``InvalidNonce`` (or its subclass ``ChecksumError``) when a
    diff does not follow on from the one before, after dropping its book.
    """
    try:
        from ccxt.base.errors import InvalidNonce
    except ImportError:  # pragma: no cover - only without the extra
        return False
    return isinstance(exc, InvalidNonce)


# The HTTP status a venue answers with when it does not serve the caller's
# location ("Unavailable For Legal Reasons").
_HTTP_UNAVAILABLE_FOR_LEGAL_REASONS = 451


def _use_market_data_mirror(exchange: Any, exchange_id: str) -> None:
    """Point *exchange* at the venue's market-data-only endpoints.

    Only spot markets are loaded: the mirror serves no derivatives metadata.
    Raises :class:`~ob_analytics.exceptions.ConfigError` for a venue with no
    known mirror.
    """
    mirror = MARKET_DATA_MIRRORS.get(exchange_id)
    if mirror is None:
        raise ConfigError(
            f"No market-data mirror is known for {exchange_id!r}; "
            f"known: {', '.join(MARKET_DATA_MIRRORS)}."
        )
    exchange.urls["api"]["public"] = mirror["rest"]
    exchange.urls["api"]["ws"]["spot"] = mirror["ws"]
    # Recent ccxt keeps the market types under "types"; earlier 4.x
    # releases make the option the list of types itself.
    fetch_markets = exchange.options.get("fetchMarkets")
    if isinstance(fetch_markets, dict):
        fetch_markets["types"] = ["spot"]
    else:
        exchange.options["fetchMarkets"] = ["spot"]


def _refused_location(exc: Exception, exchange_id: str) -> ConfigError | None:
    """A readable error when the venue refused the caller's location.

    CCXT reports an HTTP 451 as ``ExchangeNotAvailable`` with the status in
    the message; ``None`` for any other failure.
    """
    if f" {_HTTP_UNAVAILABLE_FOR_LEGAL_REASONS} " not in str(exc):
        return None
    hint = ""
    if exchange_id == "binance":
        hint = (
            " For Binance US, use --exchange binanceus. Binance's market-data "
            "mirror is --market-data-mirror; check Binance's terms allow you "
            "to use it from your location."
        )
    return ConfigError(
        f"{exchange_id} refused this location (HTTP "
        f"{_HTTP_UNAVAILABLE_FOR_LEGAL_REASONS}).{hint}"
    )


def _import_ccxt_pro() -> Any:
    """Return ``ccxt.pro``, or raise :class:`ImportError` with the install hint."""
    try:
        import ccxt.pro as ccxtpro
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "The ccxt source requires the 'ccxt' extra: "
            'pip install "ob-analytics[ccxt]"'
        ) from exc
    return ccxtpro


def _make_exchange(exchange_id: str) -> Any:
    """Instantiate a CCXT exchange by id (lazy import of ``ccxt``).

    A plain id is looked up among the CCXT Pro venues first, then among CCXT's
    prediction markets (``ccxt.prediction``: Kalshi, Polymarket, ...).  An id
    with :data:`PREDICTION_PREFIX`
    is looked up among the prediction markets only.

    Raises :class:`ImportError` with an install hint if ccxt is absent, and
    :class:`ValueError` if *exchange_id* is not a known CCXT venue.
    """
    return _exchange_class(exchange_id)({"enableRateLimit": True})


def _exchange_class(exchange_id: str) -> Any:
    """Look up the CCXT exchange class for *exchange_id* without building it.

    Raises the same errors as :func:`_make_exchange`.
    """
    ccxtpro = _import_ccxt_pro()
    # Prediction-market classes by id; empty on a ccxt release that predates them.
    prediction: dict[str, Any] = {}
    try:
        import ccxt.prediction as ccxtprediction
    except ImportError:  # pragma: no cover - only on an old ccxt
        pass
    else:
        prediction = {i: getattr(ccxtprediction, i) for i in ccxtprediction.exchanges}

    name = exchange_id.removeprefix(PREDICTION_PREFIX)
    if name == exchange_id and name in ccxtpro.exchanges:
        return getattr(ccxtpro, name)
    if name in prediction:
        return prediction[name]
    raise ValueError(
        f"Unknown CCXT exchange {exchange_id!r}; expected one of "
        f"{len(ccxtpro.exchanges)} ccxt.pro venues or a prediction market "
        f"({', '.join(prediction) or 'none in this ccxt release'})."
    )


def _epoch_ms_to_ts(ms: Any) -> pd.Timestamp:
    """CCXT timestamps are epoch-ms; ``None`` falls back to receive time.

    Both branches land on the canonical tz-aware UTC nanosecond clock.
    """
    if ms is None:
        return pd.Timestamp.now(tz="UTC").as_unit("ns")
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").as_unit("ns")


class CcxtSource:
    """Live-capture any CCXT venue as an L2 depth + trade stream.

    Satisfies :class:`~ob_analytics.live._base.LiveSource` (live only — a CCXT
    capture's ``depth.csv`` replays offline through the generic
    :class:`~ob_analytics.depth_l2.DepthCsvSource`).  Instances are not
    reusable across runs -- construct a fresh one per capture.
    """

    name = "ccxt"
    level = Level.L2
    # CCXT's unified book is the venue's own aggregated view: bids never rest
    # above asks, so the reconstructed book is not crossed.
    feed_type = FeedType.MATCHED_BOOK
    # Price levels carry no order identity, so no order of a trade is named.
    trade_attribution = TradeAttribution.NONE
    # The book nonce only rises: a Binance diff spans a range of update IDs,
    # and watch_order_book can apply several diffs before it returns.
    sequence_kind = SequenceKind.MONOTONIC

    def __init__(self, settings: SourceSettings | None = None) -> None:
        # The venue id and per-run knobs are typed CcxtSettings (the empty
        # default lets the source be constructed before the venue is known;
        # ``_configure`` requires ``exchange`` by the time a capture starts).
        self.settings: SourceSettings = settings or CcxtSettings()
        self._exchange: Any = None
        self._symbol: str = ""
        self._depth_limit: int = _DEFAULT_DEPTH_LIMIT
        self._poll_interval: float = _DEFAULT_POLL_INTERVAL
        self._use_ws_book = False
        self._use_ws_trades = False
        # Last seen absolute size per price, per side -- the diff baseline.
        self._last: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        # raw.jsonl holds the whole book once, on the first book update that
        # changes a level; later frames hold only the changed levels.
        self._full_book_written = False
        # Epoch-ms time of the opening book. A polled trade tape starts with
        # the venue's recent history, which can reach back hours; trades
        # before this are dropped rather than stamped as received now.
        self._opened_ms: int | None = None
        # Trades already written, oldest first, capped at _SEEN_TRADES_LIMIT.
        # A venue can deliver one trade twice: ccxt's Polymarket websocket
        # handed back an earlier trade alongside the next new one.
        self._seen_trades: dict[tuple[Any, ...], None] = {}

        # Diagnostics (surfaced in meta.json via SupportsDiagnostics).
        self.exchange_id = ""
        # The instrument's price increment, when the venue's metadata gives
        # one; meta.json records it so the replay uses the same price grid.
        self.tick_size: float | None = None
        # How many times a price arrived between two ticks and the recorded
        # tick size was made finer to fit it (see _fit_tick).
        self.tick_size_changes = 0
        self.book_updates = 0
        # Times ccxt lost a book update and the book was fetched again.
        self.book_resyncs = 0
        self.depth_rows = 0
        self.trade_events = 0
        self.duplicate_trades = 0
        self.errors = 0

    # -- configuration ------------------------------------------------------

    def preflight(self) -> None:
        """Raise now if the venue cannot be built (SupportsPreflight).

        Raises the install hint when ccxt is missing and :class:`ValueError`
        for no venue or an unknown one. A pre-built exchange object in the
        settings needs no check.
        """
        exchange = getattr(self.settings, "exchange", None)
        if not isinstance(exchange, str):
            return
        if not exchange:
            raise ValueError(
                "ccxt source needs CcxtSettings(exchange='<venue id>') "
                "(e.g. 'binance')."
            )
        _exchange_class(exchange)

    def _configure(self, config: CaptureConfig) -> None:
        """Resolve the exchange, symbol, and per-transport capabilities."""
        settings = self.settings
        if not isinstance(settings, CcxtSettings):
            raise TypeError(
                "CcxtSource needs CcxtSettings; got "
                f"{type(settings).__name__}. Construct it as "
                "CcxtSource(settings=CcxtSettings(exchange='<venue id>'))."
            )
        exchange = settings.exchange
        if not exchange:
            raise ValueError(
                "ccxt source needs CcxtSettings(exchange='<venue id>') "
                "(e.g. 'binance')."
            )
        self._symbol = config.pair
        self._depth_limit = int(settings.depth_limit)
        self._poll_interval = float(settings.poll_interval)

        # A string is a venue id (built via ccxt); anything else is treated as
        # a pre-built exchange object (tests / advanced callers).
        if isinstance(exchange, str):
            # The venue is the same whichever list it came from.
            self.exchange_id = exchange.removeprefix(PREDICTION_PREFIX)
            self._exchange = _make_exchange(exchange)
        else:
            self.exchange_id = str(getattr(exchange, "id", "custom"))
            self._exchange = exchange
        if settings.market_data_mirror:
            _use_market_data_mirror(self._exchange, self.exchange_id)
        if self.exchange_id in _WHOLE_BOOK_VENUES:
            _size_whole_book_snapshot(
                self._exchange, self.exchange_id, self._depth_limit
            )

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
        try:
            book = await self._exchange.fetch_order_book(
                self._symbol, self._depth_limit
            )
        except Exception as exc:
            # stream() closes the connection when it ends; a capture that
            # fails here never reaches it.
            await self._close()
            refused = _refused_location(exc, self.exchange_id)
            if refused is not None:
                raise refused from exc
            raise
        # Fetching the book loads the venue's market metadata, so the tick
        # size can be read from here on.
        self.tick_size = self._tick_size()
        self._opened_ms = book.get("timestamp")
        received = pd.Timestamp.now(tz="UTC").as_unit("ns")
        venue_ts = _epoch_ms_to_ts(book.get("timestamp"))
        # CCXT's per-book monotonic sequence (``None`` when the venue omits it);
        # carried as the venue ``sequence`` for gap detection on the L2 path.
        nonce = book.get("nonce")
        for side, key in (("bid", "bids"), ("ask", "asks")):
            levels: dict[float, float] = {}
            for row in self._side(book, key):
                price = float(row[0])
                size = float(row[1])
                self._fit_tick(price)
                levels[price] = size
                if size > 0:
                    yield {
                        "timestamp": received,
                        "exchange_timestamp": venue_ts,
                        "side": side,
                        "price": price,
                        "volume": size,
                        "sequence": nonce,
                        **self._identity(),
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
                        self._symbol,
                        None
                        if self.exchange_id in _WHOLE_BOOK_VENUES
                        else self._depth_limit,
                    )
                else:
                    book = await self._exchange.fetch_order_book(
                        self._symbol, self._depth_limit
                    )
            except StopAsyncIteration:
                # Finite feed exhausted (tests). Real feeds block instead.
                return
            except Exception as exc:  # noqa: BLE001 - one bad frame ends the loop for v1
                if _lost_sync(exc) and self.book_resyncs < _MAX_BOOK_RESYNCS:
                    # ccxt found a missing update and dropped its book; the
                    # next call fetches a new snapshot, and diffing it against
                    # the last recorded book corrects every level.
                    self.book_resyncs += 1
                    logger.warning("[ccxt] book lost sync, restarting: {!r}", exc)
                    continue
                self.errors += 1
                logger.warning("[ccxt] book loop ended on error: {!r}", exc)
                return
            self.book_updates += 1
            received = pd.Timestamp.now(tz="UTC").as_unit("ns")
            # _diff_book already applied every row of this update to
            # self._last before yielding the first one (it needs the whole
            # picture to build the raw.jsonl frame), so by the time we are
            # enqueueing, that state is committed whether or not we finish
            # enqueueing. A cancellation landing on the periodic yield below
            # must not stop until this generator (pure sync code, no further
            # awaits) is fully drained -- otherwise the tail of an update
            # ccxt already told us happened would be silently missing from
            # depth.csv while self._last (and the next diff) reflects it.
            # queue.put() itself is not a cancellation point: this queue has
            # no maxsize, so put() never actually suspends -- put_nowait()
            # says that plainly instead of relying on that implementation
            # detail.
            cancelled: asyncio.CancelledError | None = None
            for row, raw in self._diff_book(book, received):
                self.depth_rows += 1
                queue.put_nowait(("depth", row, raw))
                if cancelled is None and self.depth_rows % _YIELD_EVERY_DEPTH_ROWS == 0:
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError as exc:
                        cancelled = exc
            if cancelled is not None:
                raise cancelled
            if not self._use_ws_book:
                await asyncio.sleep(self._poll_interval)

    async def _trades_loop(
        self,
        queue: asyncio.Queue[tuple[str, EventDict, Any]],
        stop: asyncio.Event,
        deadline: float,
    ) -> None:
        """Poll/stream the trade tape and push trade events."""
        since = None if self._use_ws_trades else self._opened_ms
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
                # The whole trade, not the id alone: a Polymarket id is the
                # settling transaction, which can carry more than one fill.
                key = tuple(
                    t.get(k) for k in ("id", "timestamp", "price", "amount", "side")
                )
                if key in self._seen_trades:
                    self.duplicate_trades += 1
                    continue
                self._seen_trades[key] = None
                if len(self._seen_trades) > _SEEN_TRADES_LIMIT:
                    del self._seen_trades[next(iter(self._seen_trades))]
                self.trade_events += 1
                self._fit_tick(float(t["price"]))
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

    def _identity(self) -> dict[str, str]:
        """Instrument identity stamped onto every emitted event.

        ``venue`` is the CCXT exchange id and ``symbol`` the traded pair, both
        resolved in :meth:`_configure`.  Carrying them per row lets a combined
        multi-venue frame be split back by ``(venue, symbol)`` downstream.
        """
        return {"venue": self.exchange_id, "symbol": self._symbol}

    def _diff_book(
        self, book: dict[str, Any], received: pd.Timestamp
    ) -> Iterator[tuple[EventDict, Any]]:
        """Yield ``(depth_row, raw)`` for levels that changed vs the last book.

        A changed/added level emits its new absolute size; a level present
        before but absent now emits ``0`` (removal).  Updates the stored
        per-side book.  Every level CCXT's book reports is compared, with no
        crop of its own: for most venues that book already is (or the venue
        makes it) everything CCXT knows about, so ``0`` means the level is
        genuinely gone, not merely outside some smaller recorded window
        (issue #275).  The one exception is Kraken, which keeps CCXT's book
        at exactly its subscribed window; a level leaving that window reads
        as ``0`` too, because it is gone from Kraken's own view as well.

        Rows carry *received*, the time the capture got the book, as
        ``timestamp``, and the venue's book time as ``exchange_timestamp``.
        The venue time can step back: ccxt stamps its first Binance book with
        its own snapshot's time, then applies older buffered diffs.  Replay
        sorts on ``timestamp``, so it must follow arrival order.

        The raw frame for raw.jsonl is attached to the first emitted row of
        the update and ``None`` on the rest.  The first frame is the whole
        book CCXT reported, with ``"frame": "book"``.  Every later frame has
        ``"frame": "changes"`` and holds only the changed levels (as
        ``[price, size]`` pairs, ``0`` for a removal) with the book's
        ``symbol``, ``timestamp``, ``datetime`` and ``nonce``.  Applying the
        changes to the first frame in order gives each recorded book, and the
        file grows with the updates, not with the depth of the book.
        """
        venue_ts = _epoch_ms_to_ts(book.get("timestamp"))
        # CCXT's per-book monotonic sequence (``None`` when the venue omits it);
        # every row from this book update carries it as the venue ``sequence``.
        nonce = book.get("nonce")
        changes: dict[str, list[tuple[float, float]]] = {"bid": [], "ask": []}
        for side, key in (("bid", "bids"), ("ask", "asks")):
            # Diffed by mutating self._last[side] in place, not by building a
            # fresh dict every call: for a deep book (whole-book venues, or
            # Coinbase/Bitstamp/OKX which hand ccxt everything regardless of
            # depth_limit) that dict can hold tens of thousands of entries,
            # and this is the hot path for a capture that can run for hours.
            # Safe because a CCXT book side never repeats a price (ccxt's own
            # OrderBookSide is keyed by price), so one pass both updates
            # changed levels and marks what is still present.
            prev = self._last[side]
            seen: set[float] = set()
            for row in self._side(book, key):
                price = float(row[0])
                size = float(row[1])
                seen.add(price)
                if prev.get(price) != size:
                    self._fit_tick(price)
                    changes[side].append((price, size))
                    prev[price] = size
            gone = [price for price in prev if price not in seen]
            for price in gone:
                changes[side].append((price, 0.0))
                del prev[price]
        if not (changes["bid"] or changes["ask"]):
            return
        raw: dict[str, Any] | None
        if self._full_book_written:
            raw = {
                "frame": "changes",
                "symbol": book.get("symbol"),
                "timestamp": book.get("timestamp"),
                "datetime": book.get("datetime"),
                "nonce": nonce,
                "bids": [list(level) for level in changes["bid"]],
                "asks": [list(level) for level in changes["ask"]],
            }
        else:
            # Copy the levels: ccxt.pro updates its cached book in place, and
            # the frame is serialised later, after the next update may land.
            raw = {
                **book,
                "frame": "book",
                "bids": [list(level) for level in self._side(book, "bids")],
                "asks": [list(level) for level in self._side(book, "asks")],
            }
            self._full_book_written = True
        for side in ("bid", "ask"):
            for price, size in changes[side]:
                yield (
                    {
                        "timestamp": received,
                        "exchange_timestamp": venue_ts,
                        "side": side,
                        "price": price,
                        "volume": size,
                        "sequence": nonce,
                        **self._identity(),
                    },
                    raw,
                )
                raw = None

    def _side(self, book: dict[str, Any], key: str) -> list[Any]:
        """One side of a CCXT book, as CCXT reports it -- no crop of its own.

        CCXT sorts each side best first (bids high to low, asks low to high).
        A shallower depth than CCXT tracks is not cropped back down here: see
        :meth:`_diff_book` for why (issue #275).
        """
        return list(book.get(key) or ())

    def _map_trade(self, t: dict[str, Any]) -> EventDict:
        """Map a CCXT trade to the universal trade-event shape.

        Public trades carry no order IDs, so ``buy_order_id`` /
        ``sell_order_id`` are left empty; ``side`` is CCXT's taker side.
        """
        return {
            "trade_id": t.get("id") or "",
            "timestamp": pd.Timestamp.now(tz="UTC").as_unit("ns"),
            "exchange_timestamp": _epoch_ms_to_ts(t.get("timestamp")),
            "price": float(t["price"]),
            "amount": float(t["amount"]),
            "buy_order_id": "",
            "sell_order_id": "",
            "side": t.get("side") or "",
            **self._identity(),
        }

    # -- internals ----------------------------------------------------------

    def _tick_size(self) -> float | None:
        """The instrument's price increment, from CCXT's market metadata.

        A prediction-market venue describes an instrument as an outcome, and a
        crypto exchange as a market.  ``None`` when the venue has no metadata
        for the symbol, or no fixed price grid (CCXT's significant-digits mode).
        """
        ex = self._exchange
        lookup = getattr(ex, "outcome", None) or getattr(ex, "market", None)
        if lookup is None:
            return None
        try:
            step = (lookup(self._symbol).get("precision") or {}).get("price")
        except Exception as exc:  # noqa: BLE001 - metadata is optional; the capture goes on without it
            logger.debug("[ccxt] no tick size for {}: {!r}", self._symbol, exc)
            return None
        if step is None:
            return None
        mode = getattr(ex, "precisionMode", _CCXT_TICK_SIZE)
        if mode == _CCXT_TICK_SIZE:
            return float(step)
        if mode == _CCXT_DECIMAL_PLACES:
            return 10.0 ** -int(step)
        return None

    def _fit_tick(self, price: float) -> None:
        """Make the recorded tick size fine enough for *price*.

        A venue can make a market's tick finer during a capture: Polymarket
        does as a price nears 0 or 1.  The replay needs a grid every written
        price sits on, so the tick is divided by ten until *price* does.  A
        venue with no tick size in its metadata records none, and is left so.
        """
        tick = self.tick_size
        if tick is None or not off_tick_grid(price, tick):
            return
        while tick > _FINEST_TICK and off_tick_grid(price, tick):
            tick = round(tick / 10, 12)
        logger.info(
            "[ccxt] {}: price {} is between ticks of {}; recording tick size {}",
            self._symbol,
            price,
            self.tick_size,
            tick,
        )
        self.tick_size = tick
        self.tick_size_changes += 1

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
            "sequence_kind": self.sequence_kind.value,
            "tick_size": self.tick_size,
            "tick_size_changes": self.tick_size_changes,
            "book_updates": self.book_updates,
            "book_resyncs": self.book_resyncs,
            "depth_rows": self.depth_rows,
            "trade_events": self.trade_events,
            "duplicate_trades": self.duplicate_trades,
            "errors": self.errors,
        }


# ── Register this source ──────────────────────────────────────────────
# Registered unconditionally (importing this module never imports ccxt — it is
# imported lazily in ``_make_exchange``); a capture without the ``[ccxt]`` extra
# raises a clear install hint at that point.
from ob_analytics.sources import register_source

register_source("ccxt", CcxtSource)
