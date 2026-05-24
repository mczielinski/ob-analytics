#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "websockets>=12",
# ]
# ///
"""Collect Bitstamp BTC/USD live order events and trades for a fixed window.

Subscribes to the Bitstamp WebSocket v2 API (https://www.bitstamp.net/websocket/v2/)
channels ``live_orders_btcusd`` and ``live_trades_btcusd`` and streams every event
to disk.

To produce a self-contained, ob-analytics-pipeline-ready capture, the script
also:

* Pulls a full order-book snapshot from the REST endpoint
  ``/api/v2/order_book/<pair>/?group=2`` at startup and emits a synthetic
  ``created`` event for every resting order (so any subsequent ``changed`` /
  ``deleted`` events have a matching create in the file).
* Emits a synthetic ``deleted`` event at shutdown for every order still on
  the book (so every ``id`` in the file has a complete create→delete
  lifecycle).
* Drops live ``order_*`` events whose ``microtimestamp`` is older than the
  snapshot, since those are already reflected in it.

Output is written under ``~/Desktop/bitstamp_btcusd_<UTC-stamp>/``:

* ``orders.csv``  — columns ``id, timestamp, exchange_timestamp, price, volume,
  action, direction``.  This is the exact schema consumed by
  :class:`ob_analytics.bitstamp.BitstampLoader`, so the file can be fed straight
  into the ob-analytics pipeline.
* ``trades.csv``  — bitstamp trade ticks (informational; ob-analytics infers
  trades from the order events itself).
* ``raw.jsonl``   — every WebSocket frame, one JSON object per line.  Kept as
  an immutable forensic record in case you want to re-derive anything later.
* ``meta.json``   — run metadata (start/end, counts, snapshot stats).

Usage::

    # Default: 10 minutes of BTC/USD into ~/Desktop
    ./scripts/collect_bitstamp_btcusd.py

    # Custom duration / pair / output dir
    ./scripts/collect_bitstamp_btcusd.py --minutes 30
    ./scripts/collect_bitstamp_btcusd.py --pair ethusd --out ~/data

The script is a uv "PEP 723" inline-metadata script: ``uv run`` will resolve
``websockets`` automatically.  Make it executable once with
``chmod +x scripts/collect_bitstamp_btcusd.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed


WS_URL = "wss://ws.bitstamp.net"
REST_BOOK_URL = "https://www.bitstamp.net/api/v2/order_book/{pair}/?group=2"

# Bitstamp `live_orders_<pair>` event names -> ob-analytics action label.
_ACTION_MAP = {
    "order_created": "created",
    "order_changed": "changed",
    "order_deleted": "deleted",
}

# Bitstamp `order_type`: 0 = buy (bid), 1 = sell (ask).
_DIRECTION_MAP = {0: "bid", 1: "ask"}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--minutes",
        type=float,
        default=10.0,
        help="Capture duration in minutes (default: 10).",
    )
    p.add_argument(
        "--pair",
        default="btcusd",
        help="Bitstamp currency pair, lowercase (default: btcusd).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Desktop",
        help="Parent directory for the run folder (default: ~/Desktop).",
    )
    return p.parse_args()


class Recorder:
    """Owns the three output files and the running counters."""

    _ORDERS_HEADER = (
        "id",
        "timestamp",
        "exchange_timestamp",
        "price",
        "volume",
        "action",
        "direction",
    )
    _TRADES_HEADER = (
        "trade_id",
        "timestamp",
        "exchange_timestamp",
        "price",
        "amount",
        "buy_order_id",
        "sell_order_id",
        "side",
    )

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        self._orders_fp = (run_dir / "orders.csv").open("w", newline="")
        self._trades_fp = (run_dir / "trades.csv").open("w", newline="")
        self._raw_fp = (run_dir / "raw.jsonl").open("w")

        self._orders_w = csv.writer(self._orders_fp)
        self._trades_w = csv.writer(self._trades_fp)
        self._orders_w.writerow(self._ORDERS_HEADER)
        self._trades_w.writerow(self._TRADES_HEADER)

        self.orders_count = 0
        self.trades_count = 0
        self.dropped = 0
        self.synthetic_created = 0
        self.synthetic_deleted = 0
        self.pre_snapshot_skipped = 0

        # order_id -> (last_price, direction, last_volume).  Used to (a) emit
        # synthetic `deleted` events at shutdown for orders still on the book
        # and (b) recover price/direction when a `deleted` event arrives with
        # zero/missing fields.
        self.open_orders: dict[int, tuple[float, str, float]] = {}

    def write_raw(self, payload: dict[str, Any], recv_ms: int) -> None:
        # Stamp every raw frame with our local receive time so the JSONL is
        # self-describing even without the structured CSVs.
        payload["_recv_ms"] = recv_ms
        self._raw_fp.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def write_order(
        self,
        msg: dict[str, Any],
        recv_ms: int,
        snapshot_us: int = 0,
    ) -> None:
        action = _ACTION_MAP.get(msg.get("event", ""))
        if action is None:
            return
        d = msg.get("data") or {}
        try:
            order_id = int(d["id"])
            price = float(d["price"])
            volume = float(d["amount"])
            order_type = int(d["order_type"])
            ex_us = int(d["microtimestamp"])  # microseconds since epoch
        except (KeyError, TypeError, ValueError):
            self.dropped += 1
            return
        direction = _DIRECTION_MAP.get(order_type)
        if direction is None:
            self.dropped += 1
            return

        # Anything older than the snapshot is already reflected in the
        # synthetic `created` rows we emitted from REST, so skip it.
        if snapshot_us and ex_us <= snapshot_us:
            self.pre_snapshot_skipped += 1
            return

        ex_ms = ex_us // 1000
        self._orders_w.writerow(
            (order_id, recv_ms, ex_ms, price, volume, action, direction)
        )
        self.orders_count += 1

        if action == "deleted":
            self.open_orders.pop(order_id, None)
        else:
            # `created` and `changed` both refresh our book state.
            self.open_orders[order_id] = (price, direction, volume)

    def write_synthetic_created(
        self,
        order_id: int,
        recv_ms: int,
        ex_ms: int,
        price: float,
        volume: float,
        direction: str,
    ) -> None:
        self._orders_w.writerow(
            (order_id, recv_ms, ex_ms, price, volume, "created", direction)
        )
        self.synthetic_created += 1
        self.open_orders[order_id] = (price, direction, volume)

    def write_synthetic_deleted(
        self,
        order_id: int,
        recv_ms: int,
        ex_ms: int,
        price: float,
        volume: float,
        direction: str,
    ) -> None:
        self._orders_w.writerow(
            (order_id, recv_ms, ex_ms, price, volume, "deleted", direction)
        )
        self.synthetic_deleted += 1
        self.open_orders.pop(order_id, None)

    def flush_synthetic_deletes(self, recv_ms: int, ex_ms: int) -> None:
        """Emit a synthetic ``deleted`` row for every order still resting."""
        for order_id, (price, direction, volume) in list(self.open_orders.items()):
            self.write_synthetic_deleted(
                order_id, recv_ms, ex_ms, price, volume, direction
            )

    def write_trade(self, msg: dict[str, Any], recv_ms: int) -> None:
        if msg.get("event") != "trade":
            return
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
            return
        self._trades_w.writerow(
            (trade_id, recv_ms, ex_ms, price, amount, buy_id, sell_id, side)
        )
        self.trades_count += 1

    def flush(self) -> None:
        self._orders_fp.flush()
        self._trades_fp.flush()
        self._raw_fp.flush()

    def close(self) -> None:
        self.flush()
        self._orders_fp.close()
        self._trades_fp.close()
        self._raw_fp.close()


async def _subscribe(
    ws: websockets.WebSocketClientProtocol, channels: list[str]
) -> None:
    for ch in channels:
        await ws.send(json.dumps({"event": "bts:subscribe", "data": {"channel": ch}}))


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
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


async def _seed_from_snapshot(
    ws: websockets.WebSocketClientProtocol,
    rec: Recorder,
    pair: str,
    orders_channel: str,
    trades_channel: str,
) -> int:
    """Fetch a REST book snapshot and emit synthetic ``created`` rows.

    Runs concurrently with the WebSocket: we keep ``recv()``-ing into a
    buffer so no live frames are lost while the HTTP round-trip happens.
    Returns the snapshot ``microtimestamp`` (used downstream to drop live
    events that the snapshot already reflects).
    """
    await _subscribe(ws, [orders_channel, trades_channel])

    snap_task = asyncio.create_task(asyncio.to_thread(_fetch_book_snapshot, pair))
    buffered: list[tuple[dict[str, Any], int]] = []

    while not snap_task.done():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        recv_ms = int(time.time() * 1000)
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rec.write_raw(msg, recv_ms)
        buffered.append((msg, recv_ms))

    try:
        snap = await snap_task
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(
            f"[bitstamp] WARNING: snapshot fetch failed ({exc!r}); "
            "continuing without seed (incomplete order lifecycles likely)",
            flush=True,
        )
        # Replay buffer normally with no snapshot filter.
        for msg, recv_ms in buffered:
            ch = msg.get("channel", "")
            if ch == orders_channel:
                rec.write_order(msg, recv_ms, snapshot_us=0)
            elif ch == trades_channel:
                rec.write_trade(msg, recv_ms)
        return 0

    snap_us = int(snap.get("microtimestamp", "0"))
    snap_ms = snap_us // 1000 if snap_us else int(time.time() * 1000)
    recv_ms = int(time.time() * 1000)

    for direction, side_key in (("bid", "bids"), ("ask", "asks")):
        for row in snap.get(side_key, ()):
            try:
                price = float(row[0])
                volume = float(row[1])
                order_id = int(row[2])
            except (IndexError, TypeError, ValueError):
                rec.dropped += 1
                continue
            if volume <= 0:
                continue
            rec.write_synthetic_created(
                order_id, recv_ms, snap_ms, price, volume, direction
            )

    print(
        f"[bitstamp] snapshot: {rec.synthetic_created} resting orders "
        f"(microtimestamp={snap_us})",
        flush=True,
    )

    # Drain the buffer now that we know the snapshot cutoff.
    for msg, msg_recv_ms in buffered:
        ch = msg.get("channel", "")
        if ch == orders_channel:
            rec.write_order(msg, msg_recv_ms, snapshot_us=snap_us)
        elif ch == trades_channel:
            rec.write_trade(msg, msg_recv_ms)

    return snap_us


async def _run(args: argparse.Namespace) -> int:
    run_dir = args.out / f"bitstamp_{args.pair}_{_utc_stamp()}"
    rec = Recorder(run_dir)

    orders_channel = f"live_orders_{args.pair}"
    trades_channel = f"live_trades_{args.pair}"
    channels = [orders_channel, trades_channel]

    deadline = time.monotonic() + args.minutes * 60.0
    started_at = datetime.now(timezone.utc).isoformat()
    reconnects = 0
    last_progress = time.monotonic()
    seeded = False
    snapshot_us = 0

    print(
        f"[bitstamp] writing to {run_dir} for {args.minutes:g} min "
        f"(channels: {', '.join(channels)})",
        flush=True,
    )

    stop = asyncio.Event()

    def _on_signal() -> None:
        if not stop.is_set():
            print("\n[bitstamp] caught signal, finishing up...", flush=True)
            stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows / restricted env — fall back to default handler.
            pass

    try:
        # `websockets.connect` as an async iterator transparently reconnects on
        # ConnectionClosed with exponential backoff, which is exactly what we
        # want for a long-running capture.
        async for ws in websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**22,  # 4 MiB; bitstamp frames are small but be generous
            close_timeout=2,
        ):
            try:
                if not seeded:
                    snapshot_us = await _seed_from_snapshot(
                        ws, rec, args.pair, orders_channel, trades_channel
                    )
                    seeded = True
                else:
                    # Reconnect: just (re)subscribe; do NOT re-snapshot, that
                    # would emit duplicate `created` rows for every resting
                    # order.  Events for orders that churned during the
                    # disconnect window are unavoidably lost.
                    await _subscribe(ws, channels)

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or stop.is_set():
                        return 0
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=min(remaining, 5.0)
                        )
                    except asyncio.TimeoutError:
                        # No frame in 5 s; loop to re-check deadline / stop.
                        continue

                    recv_ms = int(time.time() * 1000)
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    rec.write_raw(msg, recv_ms)

                    channel = msg.get("channel", "")
                    if channel == orders_channel:
                        rec.write_order(msg, recv_ms, snapshot_us=snapshot_us)
                    elif channel == trades_channel:
                        rec.write_trade(msg, recv_ms)
                    # Ignore subscription_succeeded / heartbeats / etc.

                    now = time.monotonic()
                    if now - last_progress >= 15.0:
                        rec.flush()
                        mins_left = max(0.0, (deadline - now) / 60.0)
                        print(
                            f"[bitstamp] orders={rec.orders_count} "
                            f"trades={rec.trades_count} "
                            f"open={len(rec.open_orders)} "
                            f"dropped={rec.dropped} "
                            f"reconnects={reconnects} "
                            f"~{mins_left:.1f} min remaining",
                            flush=True,
                        )
                        last_progress = now
            except ConnectionClosed as exc:
                if stop.is_set() or time.monotonic() >= deadline:
                    return 0
                reconnects += 1
                print(
                    f"[bitstamp] connection closed ({exc.code} {exc.reason!r}); "
                    f"reconnecting (#{reconnects})",
                    flush=True,
                )
                continue
        return 0
    finally:
        # Emit a synthetic `deleted` for every order still on the book so
        # every id in orders.csv has a complete create -> delete lifecycle.
        end_ms = int(time.time() * 1000)
        rec.flush_synthetic_deletes(recv_ms=end_ms, ex_ms=end_ms)
        rec.close()
        meta = {
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_minutes": args.minutes,
            "pair": args.pair,
            "channels": channels,
            "ws_url": WS_URL,
            "rest_book_url": REST_BOOK_URL.format(pair=args.pair),
            "snapshot_microtimestamp": snapshot_us,
            "synthetic_created": rec.synthetic_created,
            "synthetic_deleted": rec.synthetic_deleted,
            "pre_snapshot_skipped": rec.pre_snapshot_skipped,
            "live_orders": rec.orders_count,
            "trades": rec.trades_count,
            "dropped": rec.dropped,
            "reconnects": reconnects,
            "total_order_rows": rec.orders_count
            + rec.synthetic_created
            + rec.synthetic_deleted,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[bitstamp] done. live_orders={rec.orders_count} "
            f"trades={rec.trades_count} "
            f"seeded={rec.synthetic_created} "
            f"final_deletes={rec.synthetic_deleted} "
            f"skipped_pre_snapshot={rec.pre_snapshot_skipped} "
            f"reconnects={reconnects} -> {run_dir}",
            flush=True,
        )


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
