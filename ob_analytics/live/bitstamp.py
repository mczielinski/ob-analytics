"""Bitstamp BTC/USD live capturer.

Subscribes to ``wss://ws.bitstamp.net`` channels ``live_orders_<pair>``
and ``live_trades_<pair>`` and yields universal-schema events. The
historical ``scripts/collect_bitstamp_btcusd.py`` is now a thin wrapper
around this module.

Requires the ``websockets`` package (install via the ``[live]`` extra).

The capturer preserves the rich behaviour of the historical script:

* Pulls a REST order-book snapshot concurrently with WS reads so no
  pre-snapshot frames are lost during the HTTP round-trip.
* Drops live ``order_*`` events whose ``microtimestamp`` is older than
  the snapshot (already reflected in the synthetic ``created`` rows).
* Emits a synthetic ``deleted`` event at shutdown for every order still
  resting (full ``created -> ... -> deleted`` lifecycle).
* Reconnects on ``ConnectionClosed`` (websockets' built-in retry) and
  re-subscribes without re-snapshotting (would duplicate creates).
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

import pandas as pd
import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosed

from ob_analytics.live._base import CaptureConfig, EventDict, LiveCapturer


WS_URL = "wss://ws.bitstamp.net"
REST_BOOK_URL = "https://www.bitstamp.net/api/v2/order_book/{pair}/?group=2"

# Bitstamp ``live_orders_<pair>`` event names -> ob-analytics action label.
_ACTION_MAP = {
    "order_created": "created",
    "order_changed": "changed",
    "order_deleted": "deleted",
}

# Bitstamp ``order_type``: 0 = buy (bid), 1 = sell (ask).
_DIRECTION_MAP = {0: "bid", 1: "ask"}


def _fetch_book_snapshot(pair: str) -> dict[str, Any]:
    """Synchronously GET the full Bitstamp order book (group=2 -> with IDs).

    Returned dict has ``microtimestamp`` (str, us-since-epoch), ``timestamp``
    (str, s-since-epoch) and ``bids`` / ``asks`` lists of
    ``[price, amount, order_id]`` triples.
    """
    url = REST_BOOK_URL.format(pair=pair)
    req = urllib.request.Request(
        url, headers={"User-Agent": "ob-analytics-collector/1"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        return json.loads(resp.read())


class BitstampCapturer(LiveCapturer):
    """Live-capture Bitstamp BTC/USD (or any spot pair).

    Conforms to :class:`~ob_analytics.live.LiveCapturer`. Instances are
    not reusable across runs -- construct a fresh one per capture.
    """

    name = "bitstamp"

    def __init__(self) -> None:
        # order_id -> (last_price, direction, last_volume). Tracks resting
        # orders so we can emit synthetic deletes at shutdown and recover
        # price/direction when a ``deleted`` event arrives with missing
        # fields.
        self._open_orders: dict[int, tuple[float, str, float]] = {}
        self._snapshot_us: int = 0

        # WebSocket state -- opened in ``snapshot``, used in ``stream``.
        # The live connection is entered through an ``AsyncExitStack`` so it
        # can be opened in one coroutine and closed deterministically from
        # another (``stream``'s ``finally`` / ``_reconnect``).
        self._ws: Any = None
        self._ws_stack: AsyncExitStack | None = None
        # Buffered WS frames received during the REST snapshot fetch.
        self._buffered: list[tuple[dict[str, Any], int]] = []

        # Diagnostic counters (mirror the historical script's meta.json).
        self.dropped = 0
        self.pre_snapshot_skipped = 0
        self.synthetic_created = 0
        self.synthetic_deleted = 0
        self.reconnects = 0

    # ---------------------------------------------------------------
    # snapshot
    # ---------------------------------------------------------------

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        """Open the WS, subscribe, fetch REST snapshot, yield created events.

        While the REST round-trip is in flight, WS frames are buffered so
        no live events are lost. :meth:`stream` drains that buffer before
        entering the long ``recv`` loop.
        """
        orders_channel = f"live_orders_{config.pair}"
        trades_channel = f"live_trades_{config.pair}"

        # Open the websocket connection and subscribe. The connection is held
        # open via ``self._ws_stack`` so it survives across
        # ``snapshot`` -> ``stream`` -> ``shutdown_synthetic_events`` and is
        # closed deterministically in ``stream``'s ``finally``.
        await self._open_ws([orders_channel, trades_channel])

        snap_task = asyncio.create_task(
            asyncio.to_thread(_fetch_book_snapshot, config.pair)
        )

        # While REST is in flight, buffer WS frames.
        while not snap_task.done():
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            recv_ms = int(time.time() * 1000)
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._buffered.append((msg, recv_ms))

        try:
            snap = await snap_task
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning(
                "[bitstamp] snapshot fetch failed ({!r}); continuing without "
                "seed (incomplete order lifecycles likely)",
                exc,
            )
            self._snapshot_us = 0
            return

        self._snapshot_us = int(snap.get("microtimestamp", "0"))
        snap_ms = (
            self._snapshot_us // 1000 if self._snapshot_us else int(time.time() * 1000)
        )

        ts = _epoch_ms_to_ts(snap_ms)
        for direction, side_key in (("bid", "bids"), ("ask", "asks")):
            for row in snap.get(side_key, ()):
                try:
                    price = float(row[0])
                    volume = float(row[1])
                    order_id = int(row[2])
                except (IndexError, TypeError, ValueError):
                    self.dropped += 1
                    continue
                if volume <= 0:
                    continue
                self._open_orders[order_id] = (price, direction, volume)
                self.synthetic_created += 1
                yield {
                    "id": order_id,
                    "timestamp": ts,
                    "exchange_timestamp": ts,
                    "price": price,
                    "volume": volume,
                    "action": "created",
                    "direction": direction,
                }

        logger.info(
            "[bitstamp] snapshot: {} resting orders (microtimestamp={})",
            self.synthetic_created,
            self._snapshot_us,
        )

    # ---------------------------------------------------------------
    # stream
    # ---------------------------------------------------------------

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        """Drain the snapshot buffer, then recv frames until the deadline."""
        orders_channel = f"live_orders_{config.pair}"
        trades_channel = f"live_trades_{config.pair}"
        deadline = time.monotonic() + config.minutes * 60.0

        # 1. Replay buffered frames captured during the REST snapshot fetch.
        for msg, recv_ms in self._buffered:
            parsed = self._parse_buffered(msg, recv_ms, orders_channel, trades_channel)
            if parsed is not None:
                yield parsed
        self._buffered.clear()

        if self._ws is None:
            return

        # 2. Main recv loop with reconnect-on-close.
        last_progress = time.monotonic()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    raw = await asyncio.wait_for(
                        self._ws.recv(), timeout=min(remaining, 5.0)
                    )
                except asyncio.TimeoutError:
                    continue
                except ConnectionClosed as exc:
                    if time.monotonic() >= deadline:
                        return
                    self.reconnects += 1
                    logger.info(
                        "[bitstamp] connection closed ({} {!r}); reconnecting (#{})",
                        exc.code,
                        exc.reason,
                        self.reconnects,
                    )
                    if not await self._reconnect(orders_channel, trades_channel):
                        return
                    continue

                recv_ms = int(time.time() * 1000)
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                parsed = self._parse_buffered(
                    msg, recv_ms, orders_channel, trades_channel
                )
                if parsed is not None:
                    yield parsed

                now = time.monotonic()
                if now - last_progress >= 15.0:
                    mins_left = max(0.0, (deadline - now) / 60.0)
                    logger.info(
                        "[bitstamp] open={} dropped={} reconnects={} "
                        "~{:.1f} min remaining",
                        len(self._open_orders),
                        self.dropped,
                        self.reconnects,
                        mins_left,
                    )
                    last_progress = now
        finally:
            # Close WS cleanly. ``shutdown_synthetic_events`` does not need
            # the websocket -- the synthetic deletes come from local state.
            await self._close_ws()

    # ---------------------------------------------------------------
    # shutdown
    # ---------------------------------------------------------------

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        """Yield synthetic ``deleted`` events for every still-resting order."""
        end_ms = int(time.time() * 1000)
        ts = _epoch_ms_to_ts(end_ms)
        for order_id, (price, direction, volume) in list(self._open_orders.items()):
            self.synthetic_deleted += 1
            yield {
                "id": order_id,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": price,
                "volume": volume,
                "action": "deleted",
                "direction": direction,
            }
        self._open_orders.clear()

    # ---------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------

    @staticmethod
    async def _subscribe(ws: Any, channels: list[str]) -> None:
        for ch in channels:
            await ws.send(
                json.dumps({"event": "bts:subscribe", "data": {"channel": ch}})
            )

    async def _open_ws(self, channels: list[str]) -> None:
        """Open a fresh WS connection and subscribe to *channels*.

        The connection is entered through a per-connection
        :class:`~contextlib.AsyncExitStack` stored on ``self`` so it can be
        closed later from a different coroutine via :meth:`_close_ws`. On any
        failure during connect/subscribe the partially-opened connection is
        closed before the exception propagates (no socket is leaked).
        """
        stack = AsyncExitStack()
        try:
            ws = await stack.enter_async_context(
                websockets.connect(
                    WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=2**22,
                    close_timeout=2,
                )
            )
            await self._subscribe(ws, channels)
        except BaseException:
            await stack.aclose()
            raise
        self._ws_stack = stack
        self._ws = ws

    async def _close_ws(self) -> None:
        """Close the live WS connection if one is open. Idempotent."""
        if self._ws_stack is not None:
            try:
                await self._ws_stack.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[bitstamp] ws close raised: {!r}", exc)
            self._ws_stack = None
            self._ws = None

    async def _reconnect(self, orders_channel: str, trades_channel: str) -> bool:
        """Tear down the dead WS, open a new one, and re-subscribe.

        Returns True on success. The capturer must NOT re-snapshot --
        doing so would emit duplicate ``created`` rows for every resting
        order. Events that occurred during the disconnect window are
        unavoidably lost.
        """
        try:
            await self._close_ws()
            await self._open_ws([orders_channel, trades_channel])
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[bitstamp] reconnect failed: {!r}", exc)
            self._ws = None
            self._ws_stack = None
            return False

    def _parse_buffered(
        self,
        msg: dict[str, Any],
        recv_ms: int,
        orders_channel: str,
        trades_channel: str,
    ) -> tuple[str, EventDict, Any] | None:
        channel = msg.get("channel", "")
        if channel == orders_channel:
            ev = self._normalise_order_event(msg, recv_ms)
            if ev is None:
                return None
            return ("order", ev, msg)
        if channel == trades_channel:
            ev = self._normalise_trade_event(msg, recv_ms)
            if ev is None:
                return None
            return ("trade", ev, msg)
        # Subscription_succeeded / heartbeats / etc. We still want them
        # archived in raw.jsonl, so return them as an ignored "raw" record.
        return ("raw", {}, msg)

    def _normalise_order_event(
        self, msg: dict[str, Any], recv_ms: int
    ) -> EventDict | None:
        action = _ACTION_MAP.get(msg.get("event", ""))
        if action is None:
            return None
        d = msg.get("data") or {}
        try:
            order_id = int(d["id"])
            price = float(d["price"])
            volume = float(d["amount"])
            order_type = int(d["order_type"])
            ex_us = int(d["microtimestamp"])
        except (KeyError, TypeError, ValueError):
            self.dropped += 1
            return None
        direction = _DIRECTION_MAP.get(order_type)
        if direction is None:
            self.dropped += 1
            return None

        # Pre-snapshot frame: already reflected in synthetic creates.
        if self._snapshot_us and ex_us <= self._snapshot_us:
            self.pre_snapshot_skipped += 1
            return None

        ex_ms = ex_us // 1000
        ev: EventDict = {
            "id": order_id,
            "timestamp": _epoch_ms_to_ts(recv_ms),
            "exchange_timestamp": _epoch_ms_to_ts(ex_ms),
            "price": price,
            "volume": volume,
            "action": action,
            "direction": direction,
        }

        if action == "deleted":
            self._open_orders.pop(order_id, None)
        else:
            # ``created`` and ``changed`` both refresh our book state.
            self._open_orders[order_id] = (price, direction, volume)
        return ev

    def _normalise_trade_event(
        self, msg: dict[str, Any], recv_ms: int
    ) -> EventDict | None:
        if msg.get("event") != "trade":
            return None
        d = msg.get("data") or {}
        try:
            trade_id = int(d["id"])
            price = float(d["price"])
            amount = float(d["amount"])
            buy_id = int(d["buy_order_id"])
            sell_id = int(d["sell_order_id"])
            side = "buy" if int(d["type"]) == 0 else "sell"
            ex_ms = int(d["microtimestamp"]) // 1000
        except (KeyError, TypeError, ValueError):
            self.dropped += 1
            return None
        return {
            "trade_id": trade_id,
            "timestamp": _epoch_ms_to_ts(recv_ms),
            "exchange_timestamp": _epoch_ms_to_ts(ex_ms),
            "price": price,
            "amount": amount,
            "buy_order_id": buy_id,
            "sell_order_id": sell_id,
            "side": side,
        }

    # ---------------------------------------------------------------
    # diagnostics
    # ---------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Return per-capturer counters for inclusion in meta.json."""
        return {
            "snapshot_microtimestamp": self._snapshot_us,
            "synthetic_created": self.synthetic_created,
            "synthetic_deleted": self.synthetic_deleted,
            "pre_snapshot_skipped": self.pre_snapshot_skipped,
            "dropped": self.dropped,
            "reconnects": self.reconnects,
        }


def _epoch_ms_to_ts(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")
