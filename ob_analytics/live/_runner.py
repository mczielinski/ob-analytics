"""Generic asyncio driver that turns any :class:`LiveCapturer` into files."""

from __future__ import annotations

import asyncio
import csv
import json
import signal
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ob_analytics.live._base import (
    CaptureConfig,
    CaptureResult,
    CaptureSink,
    EventDict,
    LiveCapturer,
    SupportsDiagnostics,
)


_ORDER_COLS = [
    "id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "volume",
    "action",
    "direction",
]
_TRADE_COLS = [
    "trade_id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "amount",
    "buy_order_id",
    "sell_order_id",
    "side",
]


def _ts_ms(ts: pd.Timestamp | int | float) -> int:
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 1_000_000)
    return int(ts)


class FileCaptureSink(CaptureSink):
    """Default sink: writes orders.csv, trades.csv, raw.jsonl, meta.json."""

    def __init__(self, out_dir: Path, *, keep_raw: bool) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._keep_raw = keep_raw

        self._orders_fp = (self.out_dir / "orders.csv").open("w", newline="")
        self._orders = csv.DictWriter(
            self._orders_fp, fieldnames=_ORDER_COLS, extrasaction="ignore"
        )
        self._orders.writeheader()

        self._trades_fp = (self.out_dir / "trades.csv").open("w", newline="")
        self._trades = csv.DictWriter(
            self._trades_fp, fieldnames=_TRADE_COLS, extrasaction="ignore"
        )
        self._trades.writeheader()

        self._raw_fp = (self.out_dir / "raw.jsonl").open("w") if keep_raw else None

    def write_order(self, event: EventDict) -> None:
        row = {
            **event,
            "timestamp": _ts_ms(event["timestamp"]),
            "exchange_timestamp": _ts_ms(event["exchange_timestamp"]),
        }
        self._orders.writerow(row)

    def write_trade(self, event: EventDict) -> None:
        row = {
            **event,
            "timestamp": _ts_ms(event["timestamp"]),
            "exchange_timestamp": _ts_ms(event["exchange_timestamp"]),
        }
        self._trades.writerow(row)

    def write_raw(self, frame: Any) -> None:
        if self._raw_fp is None or frame is None:
            return
        self._raw_fp.write(json.dumps(frame, separators=(",", ":")) + "\n")

    def finalize(self, result: CaptureResult) -> None:
        # Flush + close everything.
        for fp in (self._orders_fp, self._trades_fp, self._raw_fp):
            if fp is not None:
                try:
                    fp.flush()
                finally:
                    fp.close()
        self._orders_fp = None  # type: ignore[assignment]
        self._trades_fp = None  # type: ignore[assignment]
        self._raw_fp = None

        meta = {
            "out_dir": str(result.out_dir),
            "started": str(result.started),
            "ended": str(result.ended),
            "duration_seconds": (result.ended - result.started).total_seconds(),
            "n_order_events": result.n_order_events,
            "n_trade_events": result.n_trade_events,
            "n_raw_frames": result.n_raw_frames,
            **result.extras,
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


async def run_capturer(
    capturer: LiveCapturer,
    config: CaptureConfig,
    sink: CaptureSink | None = None,
) -> CaptureResult:
    """Drive a capturer: snapshot, stream, shutdown -- writing through *sink*.

    Handles SIGINT/SIGTERM by cancelling the streaming task; the shutdown
    synthetic events still run so every order id keeps a full lifecycle.
    """
    if sink is None:
        sink = FileCaptureSink(config.out_dir, keep_raw=config.keep_raw)
    started = pd.Timestamp.now(tz="UTC")
    n_order = n_trade = n_raw = 0

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    installed_signals: list[int] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
            installed_signals.append(sig)
        except (NotImplementedError, RuntimeError):
            # Windows / test environments / non-main thread without signal
            # support: fall back to default handling.
            pass

    try:
        logger.info("Capturer '{}': snapshot starting", capturer.name)
        async for ev in capturer.snapshot(config):
            sink.write_order(ev)
            n_order += 1
        logger.info("Capturer '{}': snapshot wrote {} orders", capturer.name, n_order)

        logger.info(
            "Capturer '{}': streaming for {:.1f} min",
            capturer.name,
            config.minutes,
        )
        # _stream updates this mapping in place as it writes, so the counts
        # survive a SIGINT/SIGTERM cancellation: meta.json previously
        # reported only snapshot + shutdown events for interrupted runs
        # even though every streamed row was on disk.
        stream_counts = {"order": 0, "trade": 0, "raw": 0}
        stream_task = asyncio.create_task(
            _stream(capturer, config, sink, stream_counts)
        )
        stop_task = asyncio.create_task(stop.wait())
        try:
            done, pending = await asyncio.wait(
                {stream_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (stream_task, stop_task):
                if not t.done():
                    t.cancel()
            # Drain cancellation cleanly.
            for t in (stream_task, stop_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

        if stream_task in done and not stream_task.cancelled():
            exc = stream_task.exception()
            if exc is not None:
                logger.error("Capturer '{}' stream raised: {!r}", capturer.name, exc)
        n_order += stream_counts["order"]
        n_trade += stream_counts["trade"]
        n_raw += stream_counts["raw"]

        logger.info("Capturer '{}': emitting shutdown synthetic events", capturer.name)
        async for ev in capturer.shutdown_synthetic_events():
            sink.write_order(ev)
            n_order += 1
    finally:
        # Remove signal handlers we installed.
        for sig in installed_signals:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass

        ended = pd.Timestamp.now(tz="UTC")
        extras: dict[str, Any] = {}
        # Capturers may implement the optional SupportsDiagnostics capability
        # to enrich meta.json with per-run counters.
        if isinstance(capturer, SupportsDiagnostics):
            try:
                extras.update(capturer.diagnostics())
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Capturer '{}' diagnostics() raised: {!r}",
                    capturer.name,
                    exc,
                )
        result = CaptureResult(
            out_dir=config.out_dir,
            n_order_events=n_order,
            n_trade_events=n_trade,
            n_raw_frames=n_raw,
            started=started,
            ended=ended,
            extras=extras,
        )
        sink.finalize(result)
        logger.info(
            "Capturer '{}': finished. orders={}, trades={}, raw={}, dur={:.1f}s",
            capturer.name,
            n_order,
            n_trade,
            n_raw,
            (ended - started).total_seconds(),
        )
    return result


async def _stream(
    capturer: LiveCapturer,
    config: CaptureConfig,
    sink: CaptureSink,
    counts: dict[str, int],
) -> None:
    """Pump the capturer's stream into *sink*, updating *counts* in place.

    Counts are incremented per write (not returned) so they remain accurate
    when the task is cancelled mid-stream by a signal.
    """
    async for kind, event, frame in capturer.stream(config):
        if kind == "order":
            sink.write_order(event)
            counts["order"] += 1
        elif kind == "trade":
            sink.write_trade(event)
            counts["trade"] += 1
        # ``raw`` (heartbeats / subscription_succeeded) bypasses CSV writers
        # but still goes to raw.jsonl below for forensic completeness.
        if frame is not None:
            sink.write_raw(frame)
            counts["raw"] += 1
