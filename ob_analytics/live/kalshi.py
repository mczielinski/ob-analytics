"""Kalshi live capture from Kalshi's WebSocket feed.

``ob-analytics capture kalshi --pair <market ticker>`` runs it.  The feed
needs an API key: a Key ID and the private key Kalshi issued with it, set in
``KALSHI_API_KEY_ID`` and ``KALSHI_PRIVATE_KEY_PATH``.  Kalshi's public REST
book needs no key, and the ``ccxt`` source captures it by polling.

What the capture does:

* Subscribes to one market's ``orderbook_delta`` and ``trade`` channels.  The
  book channel sends the whole book first (``orderbook_snapshot``), then every
  change to it (``orderbook_delta``), each numbered by ``seq``.
* Keeps the book as the Yes book, the convention the ccxt capture of Kalshi
  uses: a Yes bid at ``p`` is a bid at ``p``, and a No bid at ``p`` is an ask
  at ``1 - p``.  A delta gives the change in size at a price; the capture
  writes the new absolute size, as ``depth.csv`` needs.
* Carries ``seq`` as the venue ``sequence``.  Every book message adds exactly
  one, so a skipped number is a lost change.  The capture then raises
  :class:`~ob_analytics.live.UnmappedMessageError`, and
  :func:`~ob_analytics.live.run_capture` starts a new segment from a new
  opening book.  So does a change that would leave a negative size.
* Writes each trade at its Yes price.  ``side`` is ``buy`` when the taker
  bought Yes and ``sell`` when the taker bought No.  Trades that arrive
  before the opening book are left out.
* Records the market's tick size (from Kalshi's market data) and lot size in
  ``meta.json``.

Requires the ``websockets`` and ``cryptography`` packages (install via the
``[kalshi]`` extra).  Both are imported when a capture starts, so importing
this module never needs them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

import pandas as pd
from loguru import logger
from pydantic import SecretStr

from ob_analytics.capture_record import LOT_SIZE, TICK_SIZE
from ob_analytics.config import Credential, SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.live._base import (
    CaptureConfig,
    EventDict,
    UnmappedMessageError,
    epoch_ms_to_ts,
    step_for,
)
from ob_analytics.protocols import (
    Clocks,
    FeedType,
    Level,
    SequenceKind,
    TradeAttribution,
)

__all__ = [
    "KalshiSettings",
    "KalshiSource",
]

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
# The path Kalshi's signature covers for the WebSocket handshake.
WS_SIGN_PATH = "/trade-api/ws/v2"
REST_MARKET_URL = "https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
KEYS_URL = "https://kalshi.com/account/profile"

_INSTALL_HINT = (
    'The kalshi source needs the kalshi extra: pip install "ob-analytics[kalshi]"'
)

# Kalshi's sizes are fixed-point with two decimals ("count_fp", "delta_fp"),
# so a contract splits into hundredths.
SIZE_STEP = 0.01

# How long to wait for the opening book after subscribing.
SNAPSHOT_TIMEOUT_SECONDS = 30.0

_ONE = Decimal(1)
_TAKER_SIDE = {"yes": "buy", "no": "sell"}


class KalshiSettings(SourceSettings):
    """The API key a Kalshi capture signs its connection with.

    Both fields are read from the environment when not given: the Key ID
    from ``KALSHI_API_KEY_ID``, and the private key from the file
    ``KALSHI_PRIVATE_KEY_PATH`` names.  Kalshi issues both on the profile
    page of a Kalshi account; the private key is shown once, when the key is
    made.  An Ed25519 or an RSA key works.
    """

    api_key_id: Annotated[
        SecretStr | None,
        Credential(env="KALSHI_API_KEY_ID", issued_at=KEYS_URL),
    ] = None
    private_key: Annotated[
        SecretStr | None,
        Credential(env="KALSHI_PRIVATE_KEY_PATH", issued_at=KEYS_URL, from_file=True),
    ] = None


def _fetch_market(ticker: str) -> dict[str, Any]:
    """GET Kalshi's public data for one market.

    Raises :class:`~ob_analytics.exceptions.ConfigError` when Kalshi has no
    market with that ticker.
    """
    url = REST_MARKET_URL.format(ticker=urllib.parse.quote(ticker, safe=""))
    req = urllib.request.Request(
        url, headers={"User-Agent": "ob-analytics-collector/1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["market"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ConfigError(
                f"Kalshi has no market {ticker!r}. --pair takes a market "
                "ticker, such as KXPRESNOMD-28-MK; see 'Find a market ticker' "
                "in the Kalshi how-to."
            ) from exc
        raise


def _tick_size(market: dict[str, Any]) -> float | None:
    """The finest price step of *market*, from its ``price_ranges``.

    A market can use a finer step near 0 and 1 than in between; the
    recorded tick is the finest, so every price is on it.  ``None`` when
    Kalshi gives no ranges.
    """
    steps = []
    for price_range in market.get("price_ranges") or ():
        try:
            steps.append(float(price_range["step"]))
        except (KeyError, TypeError, ValueError):
            continue
    return min(steps, default=None)


def _load_private_key(pem: str) -> Any:
    """The private key in *pem*, for signing.

    Raises :class:`~ob_analytics.exceptions.ConfigError` when the text is
    not an Ed25519 or RSA private key.  The message never quotes the text.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

    try:
        key = serialization.load_pem_private_key(pem.encode(), password=None)
    except (ValueError, TypeError) as exc:
        raise ConfigError(
            "KALSHI_PRIVATE_KEY_PATH names a file that does not hold a private "
            "key without a password, in PEM form. Use the private key Kalshi "
            "showed when the key was made."
        ) from exc
    if not isinstance(key, (ed25519.Ed25519PrivateKey, rsa.RSAPrivateKey)):
        raise ConfigError(
            f"The Kalshi private key is a {type(key).__name__}; Kalshi keys "
            "are Ed25519 or RSA."
        )
    return key


def _sign_request(
    key: Any, key_id: str, method: str, path: str, now_ms: int
) -> dict[str, str]:
    """The headers that sign one request to Kalshi.

    Kalshi signs the text of the time in milliseconds, the HTTP method and
    the path without its query.  An Ed25519 key signs it directly; an RSA key
    signs it with RSA-PSS and SHA-256.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding

    message = f"{now_ms}{method}{path.split('?')[0]}".encode()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        signature = key.sign(message)
    else:
        signature = key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(now_ms),
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


def _now() -> tuple[int, pd.Timestamp]:
    """The time now, in epoch milliseconds and on the canonical clock."""
    ms = int(time.time() * 1000)
    return ms, epoch_ms_to_ts(ms)


class KalshiSource:
    """Live-capture one Kalshi market from its WebSocket feed, at L2.

    Satisfies :class:`~ob_analytics.live.LiveSource`.  Its ``depth.csv``
    replays offline through :class:`~ob_analytics.depth_csv.DepthCsvSource`.
    Instances are not reusable across runs: construct a fresh one for each
    capture, as :func:`~ob_analytics.live.run_capture` does for each segment.
    """

    name = "kalshi"
    level = Level.L2
    # Kalshi sends the total size at each price.
    feed_type = FeedType.PRICE_LEVELS
    # Price levels carry no order identity, so no order of a trade is named.
    trade_attribution = TradeAttribution.NONE
    # Every book message on a subscription adds exactly one to ``seq``.
    sequence_kind = SequenceKind.CONTIGUOUS
    # Each change carries Kalshi's time, and the opening book the time Kalshi
    # sent it; the capture adds the receive time.
    clocks = Clocks.BOTH

    def __init__(self, settings: KalshiSettings | None = None) -> None:
        # Typed as the protocol's SourceSettings so the source satisfies it;
        # _key() requires KalshiSettings by the time a capture connects.
        self.settings: SourceSettings = settings or KalshiSettings()
        self._ticker = ""
        self._ws: Any = None
        self._ws_stack: AsyncExitStack | None = None
        # The Yes book: side -> Yes price -> size, exact as Kalshi sends them.
        self._book: dict[str, dict[Decimal, Decimal]] = {"bid": {}, "ask": {}}
        self._book_sid: int | None = None
        self._book_seq: int | None = None
        self._trade_seq: int | None = None
        # Frames read while waiting for the opening book, the book's own
        # included; stream() hands them to the runner first, for raw.jsonl.
        self._early_frames: list[dict[str, Any]] = []

        # Diagnostics, written to meta.json.
        self.tick_size: float | None = None
        self.tick_size_changes = 0
        self.lot_size: float | None = SIZE_STEP
        self.lot_size_changes = 0
        self.snapshot_levels = 0
        self.book_updates = 0
        self.trade_events = 0
        # Trades Kalshi numbered but the capture never received.
        self.trades_missed = 0
        self.pre_snapshot_trades_skipped = 0
        # Trades received that could not be parsed, and were left out.
        self.dropped = 0

    # -- capability checks ---------------------------------------------------

    def preflight(self) -> None:
        """Raise now if the capture cannot run (SupportsPreflight).

        Raises :class:`ImportError` with the install hint when ``websockets``
        or ``cryptography`` is missing, and
        :class:`~ob_analytics.exceptions.ConfigError` when the private key
        given is not one.  A key that is not set is left to the capture's own
        check, which names the variable to set.
        """
        try:
            import cryptography  # noqa: F401
            import websockets.asyncio.client  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        private_key = getattr(self.settings, "private_key", None)
        if private_key is not None:
            _load_private_key(private_key.get_secret_value())

    # -- snapshot -------------------------------------------------------------

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        """Subscribe and yield the opening book as absolute-size depth rows.

        Reads the market's tick size from Kalshi's public market data first,
        so an unknown ticker stops the capture before it connects.
        """
        self._ticker = config.pair
        market = await asyncio.to_thread(_fetch_market, self._ticker)
        self.tick_size = _tick_size(market)
        await self._connect()
        try:
            frame = await self._await_snapshot()
            rows = self._apply_snapshot(frame)
        except BaseException:
            # The stream will not run, so its close never comes.
            await self._close()
            raise
        for row in rows:
            yield row
        logger.info(
            "[kalshi] {} opening book: {} bid / {} ask levels (tick {})",
            self._ticker,
            len(self._book["bid"]),
            len(self._book["ask"]),
            self.tick_size,
        )

    async def _await_snapshot(self) -> dict[str, Any]:
        """Read frames until the opening book arrives, and return it."""
        try:
            async with asyncio.timeout(SNAPSHOT_TIMEOUT_SECONDS):
                while True:
                    msg = json.loads(await self._ws.recv())
                    self._early_frames.append(msg)
                    if self._is_opening_book(msg):
                        return msg
        except TimeoutError as exc:
            raise TimeoutError(
                f"Kalshi sent no opening book for {self._ticker!r} within "
                f"{SNAPSHOT_TIMEOUT_SECONDS:.0f} s"
            ) from exc

    def _is_opening_book(self, msg: dict[str, Any]) -> bool:
        """Whether *msg* is the opening book; count a trade that came first."""
        kind = msg.get("type")
        if kind == "error":
            raise ConfigError(f"Kalshi refused the subscription: {msg.get('msg')}")
        if kind == "trade":
            self.pre_snapshot_trades_skipped += 1
            self._note_trade_seq(msg)
        return kind == "orderbook_snapshot"

    def _apply_snapshot(self, frame: dict[str, Any]) -> list[EventDict]:
        """Seed the book from an ``orderbook_snapshot`` frame; return its rows."""
        self._book_sid = frame.get("sid")
        self._book_seq = _int_or_none(frame.get("seq"))
        _, received = _now()
        sent = frame.get("sending_ts_ms")
        venue_ts = epoch_ms_to_ts(sent) if sent is not None else received
        body = frame.get("msg") or {}
        rows: list[EventDict] = []
        for kalshi_side in ("yes", "no"):
            for level in body.get(f"{kalshi_side}_dollars_fp") or ():
                try:
                    price, size = _finite(level[0]), _finite(level[1])
                except (IndexError, TypeError, InvalidOperation) as exc:
                    raise UnmappedMessageError(
                        f"cannot parse a {kalshi_side} level of the opening book: "
                        f"{level!r} ({exc!r})"
                    ) from exc
                side, yes_price = _yes_level(kalshi_side, price)
                if size <= 0:
                    continue
                self._book[side][yes_price] = size
                rows.append(self._row(side, yes_price, size, received, venue_ts))
        self.snapshot_levels = len(rows)
        return rows

    # -- stream ---------------------------------------------------------------

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        """Read frames until the deadline; yield depth rows and trades.

        A closed connection before the deadline raises
        :class:`~websockets.exceptions.ConnectionClosed`, so
        :func:`~ob_analytics.live.run_capture` starts a new segment.
        """
        from websockets.exceptions import ConnectionClosed

        deadline = time.monotonic() + config.minutes * 60.0
        last_progress = time.monotonic()
        try:
            for frame in self._early_frames:
                yield ("raw", {}, frame)
            self._early_frames.clear()
            while self._ws is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                # asyncio.timeout, not wait_for: on Python 3.11 wait_for can drop the
                # cancel that stops this segment (see _runner._cancel_until_done).
                try:
                    async with asyncio.timeout(min(remaining, 5.0)):
                        raw = await self._ws.recv()
                except TimeoutError:
                    continue
                except ConnectionClosed as exc:
                    if time.monotonic() >= deadline:
                        return
                    logger.warning("[kalshi] connection closed: {}", exc)
                    raise
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                yield self.parse(msg)

                now = time.monotonic()
                if now - last_progress >= 15.0:
                    logger.info(
                        "[kalshi] {} book updates, {} trades, ~{:.1f} min remaining",
                        self.book_updates,
                        self.trade_events,
                        max(0.0, (deadline - now) / 60.0),
                    )
                    last_progress = now
        finally:
            await self._close()

    def parse(self, msg: dict[str, Any]) -> tuple[str, EventDict, Any]:
        """Turn one frame after the opening book into ``(kind, event, raw)``.

        ``kind`` is ``"depth"`` for a book change, ``"trade"`` for a trade,
        and ``"raw"`` (with an empty event) for anything else, which is only
        archived.
        """
        kind = msg.get("type")
        if kind == "orderbook_delta":
            return ("depth", self._apply_delta(msg), msg)
        if kind == "trade":
            event = self._trade(msg)
            return ("trade", event, msg) if event is not None else ("raw", {}, msg)
        if kind == "orderbook_snapshot":
            # A second opening book replaces the book, and its rows could not
            # be told apart from changes; a new segment starts from it instead.
            raise UnmappedMessageError(
                "Kalshi sent a new opening book during the stream; the next "
                "segment starts from a new one"
            )
        if kind == "error":
            raise ConnectionError(f"Kalshi sent an error: {msg.get('msg')}")
        return ("raw", {}, msg)

    def _apply_delta(self, frame: dict[str, Any]) -> EventDict:
        """Apply one ``orderbook_delta`` to the book; return its depth row."""
        self._check_book_seq(frame)
        _, received = _now()
        body = frame.get("msg") or {}
        try:
            price = _finite(body["price_dollars"])
            change = _finite(body["delta_fp"])
            kalshi_side = body["side"]
            venue_ms = body.get("ts_ms")
        except (KeyError, TypeError, InvalidOperation) as exc:
            raise UnmappedMessageError(
                f"cannot parse a book change: {body!r} ({exc!r})"
            ) from exc
        side, yes_price = _yes_level(kalshi_side, price)
        size = self._book[side].get(yes_price, Decimal(0)) + change
        if size < 0:
            raise UnmappedMessageError(
                f"a book change leaves a negative size at {kalshi_side} "
                f"{price}: the capture's book no longer matches Kalshi's"
            )
        if size == 0:
            self._book[side].pop(yes_price, None)
        else:
            self._book[side][yes_price] = size
        self.book_updates += 1
        venue_ts = epoch_ms_to_ts(venue_ms) if venue_ms is not None else received
        return self._row(side, yes_price, size, received, venue_ts)

    def _check_book_seq(self, frame: dict[str, Any]) -> None:
        """Raise if this book message is not the one after the last."""
        if frame.get("sid") != self._book_sid:
            raise UnmappedMessageError(
                f"a book change on subscription {frame.get('sid')!r}, not the "
                f"opening book's {self._book_sid!r}"
            )
        seq = _int_or_none(frame.get("seq"))
        if seq is None or self._book_seq is None:
            raise UnmappedMessageError(f"a book message without a sequence: {frame!r}")
        expected = self._book_seq + 1
        if seq != expected:
            what = (
                f"missed {seq - expected} book change(s)"
                if seq > expected
                else "received a book change again or out of order"
            )
            raise UnmappedMessageError(
                f"Kalshi's sequence went from {self._book_seq} to {seq}: the "
                f"capture {what}, so its book no longer matches Kalshi's"
            )
        self._book_seq = seq

    def _trade(self, frame: dict[str, Any]) -> EventDict | None:
        """The trade event for one ``trade`` frame, or ``None`` if unparseable."""
        self._note_trade_seq(frame)
        _, received = _now()
        body = frame.get("msg") or {}
        try:
            price = float(_finite(body["yes_price_dollars"]))
            amount = float(_finite(body["count_fp"]))
            side = _TAKER_SIDE[body["taker_side"]]
            venue_ms = body.get("ts_ms")
            if venue_ms is None:  # a frame with Kalshi's time in seconds only
                venue_ms = int(body["ts"]) * 1000
            venue_ts = epoch_ms_to_ts(venue_ms)
        except (KeyError, TypeError, ValueError, InvalidOperation):
            self.dropped += 1
            return None
        self._fit(price, amount)
        self.trade_events += 1
        return {
            "trade_id": body.get("trade_id") or "",
            "timestamp": received,
            "exchange_timestamp": venue_ts,
            "price": price,
            "amount": amount,
            "buy_order_id": "",
            "sell_order_id": "",
            "side": side,
            "venue": self.name,
            "symbol": self._ticker,
        }

    def _note_trade_seq(self, frame: dict[str, Any]) -> None:
        """Count trades Kalshi numbered that never arrived."""
        seq = _int_or_none(frame.get("seq"))
        if seq is None:
            return
        if self._trade_seq is not None and seq > self._trade_seq + 1:
            self.trades_missed += seq - self._trade_seq - 1
        self._trade_seq = seq if self._trade_seq is None else max(seq, self._trade_seq)

    # -- shutdown -------------------------------------------------------------

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        """L2 price levels have no lifecycle to close: emit nothing."""
        for _ in ():
            yield {}

    # -- internals ------------------------------------------------------------

    def _row(
        self,
        side: str,
        price: Decimal,
        size: Decimal,
        received: pd.Timestamp,
        venue_ts: pd.Timestamp,
    ) -> EventDict:
        """One absolute-size depth row on the Yes book."""
        price_f, size_f = float(price), float(size)
        self._fit(price_f, size_f)
        return {
            "timestamp": received,
            "exchange_timestamp": venue_ts,
            "side": side,
            "price": price_f,
            "volume": size_f,
            "sequence": self._book_seq,
            "venue": self.name,
            "symbol": self._ticker,
        }

    def _fit(self, price: float, size: float) -> None:
        """Make the recorded tick and lot size fine enough for *price* and *size*."""
        tick = step_for(price, self.tick_size)
        if tick != self.tick_size:
            logger.info(
                "[kalshi] price {} is between ticks of {}; recording tick size {}",
                price,
                self.tick_size,
                tick,
            )
            self.tick_size = tick
            self.tick_size_changes += 1
        lot = step_for(size, self.lot_size)
        if lot != self.lot_size:
            self.lot_size = lot
            self.lot_size_changes += 1

    async def _connect(self) -> None:
        """Open the signed WebSocket and subscribe to the market's channels."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus

        key_id, key = self._key()
        now_ms, _ = _now()
        headers = _sign_request(key, key_id, "GET", WS_SIGN_PATH, now_ms)
        stack = AsyncExitStack()
        try:
            try:
                ws = await stack.enter_async_context(
                    connect(
                        WS_URL,
                        additional_headers=headers,
                        ping_interval=20,
                        ping_timeout=20,
                        max_size=2**24,
                        close_timeout=2,
                    )
                )
            except InvalidStatus as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    raise ConfigError(
                        f"Kalshi refused the API key (HTTP {status}). Check "
                        "that KALSHI_API_KEY_ID is the Key ID of the private "
                        "key in KALSHI_PRIVATE_KEY_PATH, and that the key has "
                        "not been deleted."
                    ) from exc
                raise
            await ws.send(
                json.dumps(
                    {
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["orderbook_delta", "trade"],
                            "market_ticker": self._ticker,
                        },
                    }
                )
            )
        except BaseException:
            await stack.aclose()
            raise
        self._ws_stack = stack
        self._ws = ws

    def _key(self) -> tuple[str, Any]:
        """The Key ID and the loaded private key from the settings."""
        settings = self.settings
        if not isinstance(settings, KalshiSettings):
            raise TypeError(
                f"KalshiSource needs KalshiSettings; got {type(settings).__name__}."
            )
        # run_capture checks the credentials before it gets here; a source
        # driven another way gets the same message.
        settings.check_credentials(self.name)
        if settings.api_key_id is None or settings.private_key is None:
            raise ConfigError(
                "Kalshi needs KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH"
            )
        key = _load_private_key(settings.private_key.get_secret_value())
        return settings.api_key_id.get_secret_value(), key

    async def _close(self) -> None:
        """Close the WebSocket if one is open. Idempotent."""
        if self._ws_stack is not None:
            try:
                await self._ws_stack.aclose()
            except Exception as exc:  # noqa: BLE001 - closing on the way out
                logger.debug("[kalshi] ws close raised: {!r}", exc)
            self._ws_stack = None
            self._ws = None

    # -- diagnostics ----------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Per-run counters for meta.json (SupportsDiagnostics)."""
        return {
            "market_ticker": self._ticker,
            "sequence_kind": self.sequence_kind.value,
            TICK_SIZE: self.tick_size,
            "tick_size_changes": self.tick_size_changes,
            LOT_SIZE: self.lot_size,
            "lot_size_changes": self.lot_size_changes,
            "snapshot_levels": self.snapshot_levels,
            "last_book_seq": self._book_seq,
            "book_updates": self.book_updates,
            "trade_events": self.trade_events,
            "trades_missed": self.trades_missed,
            "pre_snapshot_trades_skipped": self.pre_snapshot_trades_skipped,
            "dropped": self.dropped,
        }


def _yes_level(kalshi_side: Any, price: Decimal) -> tuple[str, Decimal]:
    """Where a Kalshi bid sits on the Yes book: ``(side, Yes price)``.

    A Yes bid at ``p`` is a bid at ``p``; a No bid at ``p`` is an offer to
    sell Yes at ``1 - p``, so an ask at ``1 - p``.
    """
    if kalshi_side == "yes":
        return "bid", price
    if kalshi_side == "no":
        return "ask", _ONE - price
    raise UnmappedMessageError(f"a book level on an unknown side {kalshi_side!r}")


def _finite(text: Any) -> Decimal:
    """The number in *text*; raises InvalidOperation for NaN or infinity."""
    value = Decimal(text)
    if not value.is_finite():
        raise InvalidOperation(f"not a finite number: {text!r}")
    return value


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


# ── Register this source ──────────────────────────────────────────────
# Registered unconditionally (importing this module never imports websockets
# or cryptography); a capture without the ``[kalshi]`` extra stops at
# preflight with the install hint.
from ob_analytics.sources import register_source

register_source("kalshi", KalshiSource)
