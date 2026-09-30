"""Generic asyncio driver that turns any :class:`LiveSource` into files."""

from __future__ import annotations

import asyncio
import csv
import json
import signal
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ob_analytics.live._base import (
    CaptureConfig,
    CaptureResult,
    CaptureSink,
    EventDict,
    LiveSource,
    SupportsDiagnostics,
    SupportsPreflight,
)
from ob_analytics.protocols import FeedType, Level, trade_attribution_of

# Which part of the capture wrote a book row: the source's opening book, a live
# message, or a synthetic close-out at the end. The runner stamps it from the
# phase it is in, so no source has to, and a row's origin never has to be
# guessed from its timestamp.
ORIGIN_SNAPSHOT = "snapshot"
ORIGIN_STREAM = "stream"
ORIGIN_SHUTDOWN = "shutdown"

# How often a stopped stream that is still yielding items is cancelled again.
CANCEL_RETRY_SECONDS = 0.5

# L3 (per-order) rows -- the BitstampLoader schema.  ``sequence`` is the
# venue's own per-event number when the source supplies one (blank otherwise);
# BitstampLoader reads it back under ``track_sequence`` for gap detection.
_ORDER_COLS = [
    "id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "volume",
    "action",
    "direction",
    "sequence",
    "origin",
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
# L2 (price-level) depth rows -- the L2DepthLoader schema. ``volume`` is the
# new absolute size at ``price`` (0 removes the level). ``sequence`` is the
# venue's per-book number when the capturer supplies one (blank otherwise);
# L2DepthLoader reads it back for gap detection.  ``timestamp`` is the receive
# time replay sorts on; ``exchange_timestamp`` keeps the venue's time.
_DEPTH_COLS = [
    "timestamp",
    "exchange_timestamp",
    "side",
    "price",
    "volume",
    "sequence",
    "origin",
]


def _ts_ms(ts: pd.Timestamp | float) -> int:
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 1_000_000)
    return int(ts)


class FileCaptureSink(CaptureSink):
    """Default sink: writes the book file, trades.csv, raw.jsonl, meta.json.

    The book file tracks the capturer's :class:`~ob_analytics.protocols.Level`:
    **L3** -> ``orders.csv`` (per-order lifecycle, BitstampLoader schema);
    **L2** -> ``depth.csv`` (price-level updates, L2DepthLoader schema).
    ``trades.csv`` is written for both.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        keep_raw: bool,
        level: Level = Level.L3,
        raw_warned: set[str] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._keep_raw = keep_raw
        self._level = level

        # Exactly one book writer is opened, per resolution. The other stays
        # None so write_order / write_depth are safe no-ops on the wrong side.
        self._orders_fp: Any = None
        self._orders: csv.DictWriter[str] | None = None
        self._depth_fp: Any = None
        self._depth: csv.DictWriter[str] | None = None
        if level is Level.L2:
            self._depth_fp = (self.out_dir / "depth.csv").open("w", newline="")
            self._depth = csv.DictWriter(
                self._depth_fp, fieldnames=_DEPTH_COLS, extrasaction="ignore"
            )
            self._depth.writeheader()
        else:
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
        # raw.jsonl is a record for debugging, so what it cannot hold never
        # stops the capture. The types it wrote as str(value) and the frames it
        # skipped go to meta.json. The warnings already logged are shared by
        # the caller across a capture's segments, so each is logged once.
        self._raw_text_types: set[str] = set()
        self._raw_skipped = 0
        self._raw_warned = set() if raw_warned is None else raw_warned
        # The types the frame being encoded wrote as text. They join
        # _raw_text_types only once the frame is written.
        self._frame_text_types: set[str] = set()
        self._raw_type_names: dict[type, str] = {}
        # Whether the last frame had a dict key JSON cannot hold. A source that
        # sends one usually sends them in every frame, so the next frame goes
        # straight to _encode instead of failing json.dumps first. A frame
        # with no such key turns it off, since _encode is slower.
        self._raw_keys_as_text = False
        self._frame_had_text_key = False

    def write_order(self, event: EventDict) -> None:
        if self._orders is None:
            return
        row = {
            **event,
            "timestamp": _ts_ms(event["timestamp"]),
            "exchange_timestamp": _ts_ms(event["exchange_timestamp"]),
        }
        self._orders.writerow(row)

    def write_depth(self, event: EventDict) -> None:
        if self._depth is None:
            return
        # Only the _DEPTH_COLS are persisted (extras ignored).
        row = {
            **event,
            "timestamp": _ts_ms(event["timestamp"]),
            "exchange_timestamp": _ts_ms(event["exchange_timestamp"]),
        }
        self._depth.writerow(row)

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
        self._frame_text_types.clear()
        self._frame_had_text_key = False
        try:
            if self._raw_keys_as_text:
                line = self._encode(frame, set())
                self._raw_keys_as_text = self._frame_had_text_key
            else:
                try:
                    line = json.dumps(
                        frame, separators=(",", ":"), default=self._raw_default
                    )
                except TypeError:
                    # json.dumps never passes a dict key to default=, so a key
                    # that is not a str, int, float, bool or None raises.
                    self._frame_text_types.clear()
                    line = self._encode(frame, set())
                    self._raw_keys_as_text = self._frame_had_text_key
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
            # A circular reference, or a value whose str() raises: JSON cannot
            # hold the frame at all.
            self._raw_skipped += 1
            kind = type(exc).__name__
            self._warn_once(
                f"skip:{kind}",
                f"raw.jsonl: skipping frames JSON cannot hold ({kind}: {exc})",
            )
            return
        self._raw_fp.write(line + "\n")
        for name in self._frame_text_types:
            self._raw_text_types.add(name)
            self._warn_once(
                f"text:{name}",
                f"raw.jsonl: writing {name} values or keys as text (str(value))",
            )

    def _encode(self, value: Any, path: set[int]) -> str:
        """Encode *value* as ``json.dumps`` would, writing any dict key as text.

        A key is written the way :meth:`_raw_default` writes a value. Keys
        that turn into the same text are all kept, as ``json.dumps`` keeps
        both ``1`` and ``"1"``. *path* holds the containers above *value*, so
        a circular reference raises ``ValueError`` as ``json.dumps`` would.
        """
        if not isinstance(value, dict | list | tuple):
            return json.dumps(value, default=self._raw_default)
        if id(value) in path:
            raise ValueError("Circular reference detected")
        path.add(id(value))
        try:
            if isinstance(value, dict):
                return (
                    "{"
                    + ",".join(
                        f"{json.dumps(self._key_text(k))}:{self._encode(v, path)}"
                        for k, v in value.items()
                    )
                    + "}"
                )
            return "[" + ",".join(self._encode(v, path) for v in value) + "]"
        finally:
            path.discard(id(value))

    def _key_text(self, key: Any) -> str:
        # As json.dumps writes a key: a str as it is, None, a bool or a number
        # as its JSON text.
        if isinstance(key, str):
            return key
        if key is None or isinstance(key, bool | int | float):
            return json.dumps(key)
        self._frame_had_text_key = True
        return str(self._raw_default(key))

    def _raw_default(self, value: Any) -> Any:
        # Some feeds parse prices and sizes as Decimal (cryptofeed does for
        # Bitstamp trades). Write them as strings so raw.jsonl keeps every digit.
        if isinstance(value, Decimal):
            return str(value)
        # cryptofeed parses most venues' frames with yapic json, which turns ISO
        # date and time strings into date, datetime and time objects
        # (independent_reserve, blockchain). Write them back as ISO 8601
        # strings. datetime is a subclass of date.
        if isinstance(value, (date, time)):
            return value.isoformat()
        kind = type(value)
        name = self._raw_type_names.get(kind)
        if name is None:
            name = self._raw_type_names[kind] = f"{kind.__module__}.{kind.__qualname__}"
        self._frame_text_types.add(name)
        return str(value)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._raw_warned:
            self._raw_warned.add(key)
            logger.warning("{}", message)

    @property
    def raw_frames_skipped(self) -> int:
        """Frames raw.jsonl skipped because JSON cannot hold them."""
        return self._raw_skipped

    def raw_diagnostics(self) -> dict[str, Any]:
        """What raw.jsonl wrote as text or skipped, as ``meta.json`` fields.

        Empty when raw.jsonl is off.
        """
        if not self._keep_raw:
            return {}
        return {
            "raw_text_types": sorted(self._raw_text_types),
            "n_raw_frames_skipped": self._raw_skipped,
        }

    def flush(self) -> None:
        """Push buffered rows to disk, so a crash loses as few as possible."""
        for fp in (self._orders_fp, self._depth_fp, self._trades_fp, self._raw_fp):
            if fp is not None:
                fp.flush()

    def bytes_written(self) -> int:
        """Bytes written so far across every open file (what a size roll reads)."""
        return sum(
            fp.tell()
            for fp in (self._orders_fp, self._depth_fp, self._trades_fp, self._raw_fp)
            if fp is not None
        )

    def finalize(self, result: CaptureResult) -> None:
        # Flush + close everything.
        for fp in (self._orders_fp, self._depth_fp, self._trades_fp, self._raw_fp):
            if fp is not None:
                try:
                    fp.flush()
                finally:
                    fp.close()
        self._orders_fp = None
        self._depth_fp = None
        self._trades_fp = None  # type: ignore[assignment]
        self._raw_fp = None

        meta: dict[str, Any] = {
            "out_dir": str(result.out_dir),
            "started": str(result.started),
            "ended": str(result.ended),
            "duration_seconds": (result.ended - result.started).total_seconds(),
            "n_order_events": result.n_order_events,
            "n_depth_events": result.n_depth_events,
            "n_trade_events": result.n_trade_events,
            "n_raw_frames": result.n_raw_frames,
            "n_snapshot_unconfirmed": result.n_snapshot_unconfirmed,
            "stream_started": _iso_or_none(result.stream_started),
            "stream_ended": _iso_or_none(result.stream_ended),
            **result.extras,
            **self.raw_diagnostics(),
        }
        if result.capture_error is not None:
            # A phase that raised is one more error on top of the ones the
            # source counted itself.
            meta["capture_error"] = result.capture_error
            meta["capture_error_phase"] = result.capture_error_phase
            meta["errors"] = int(meta.get("errors") or 0) + 1
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _iso_or_none(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else str(ts)


def _source_declarations(capturer: Any) -> dict[str, Any]:
    """The capturer's declarations about its feed, as ``meta.json`` values.

    ``source`` names the capturer; ``feed_type`` and ``trade_attribution`` are
    what it declares (see :class:`~ob_analytics.protocols.FeedType` and
    :class:`~ob_analytics.protocols.TradeAttribution`).  ``sequence_kind`` is
    written only by a capturer that declares one.  Read back with
    :func:`~ob_analytics.depth_l2.recorded_source` and its siblings.

    It runs while a capture is closing, so a declaration that cannot be read
    (a plug-in's value outside the enum, say) is logged and never stops the
    files being finished: an unreadable ``feed_type`` is recorded as
    ``unknown``, and an unreadable ``trade_attribution`` is left out, so
    ``audit`` falls back to ``--source`` rather than to a guess.
    """
    declared: dict[str, Any] = {"source": capturer.name}
    try:
        declared["feed_type"] = FeedType(
            getattr(capturer, "feed_type", FeedType.UNKNOWN)
        ).value
    except Exception as exc:  # noqa: BLE001 - never block finalize
        logger.warning("Capturer '{}' feed_type unreadable: {!r}", capturer.name, exc)
        declared["feed_type"] = FeedType.UNKNOWN.value
    try:
        declared["trade_attribution"] = trade_attribution_of(capturer).value
    except Exception as exc:  # noqa: BLE001 - never block finalize
        logger.warning(
            "Capturer '{}' trade_attribution unreadable: {!r}", capturer.name, exc
        )
    sequence_kind = getattr(capturer, "sequence_kind", None)
    if sequence_kind is not None:
        declared["sequence_kind"] = str(getattr(sequence_kind, "value", sequence_kind))
    return declared


async def run_capturer(
    capturer: LiveSource,
    config: CaptureConfig,
    sink: CaptureSink | None = None,
    *,
    stop: asyncio.Event | None = None,
    streaming: asyncio.Event | None = None,
) -> CaptureResult:
    """Drive a live source: snapshot, stream, shutdown -- writing through *sink*.

    This is one capture into one directory, and it ends at the first
    disconnect.  :func:`~ob_analytics.live.run_capture` runs it once per
    segment to capture for days.

    Handles SIGINT/SIGTERM by cancelling the streaming task; the shutdown
    synthetic events still run so every order id keeps a full lifecycle.  A
    caller that passes *stop* owns the signals instead: setting it ends the
    stream the same way, and no handler is installed.  *streaming*, when
    given, is set as the first live event arrives.

    A source that implements :class:`SupportsPreflight` is checked before any
    output is created, so a missing optional extra raises :class:`ImportError`
    here and leaves no output directory behind. Once output exists, an
    exception from the snapshot, the stream or the shutdown events does not
    propagate: the rows already written are kept, and the first error is
    recorded in :attr:`CaptureResult.capture_error` (with the phase in
    :attr:`CaptureResult.capture_error_phase`) and in ``meta.json``. A failed
    snapshot skips the stream; the shutdown events still run.
    """
    if isinstance(capturer, SupportsPreflight):
        capturer.preflight()
    # The source declares its granularity; the runner routes book events to
    # the matching writer. Fall back to L3 for sources predating the attr.
    level = getattr(capturer, "level", Level.L3)
    if sink is None:
        sink = FileCaptureSink(config.out_dir, keep_raw=config.keep_raw, level=level)
    started = pd.Timestamp.now(tz="UTC")
    stream_times: dict[str, pd.Timestamp] = {}
    n_order = n_trade = n_depth = n_raw = 0
    capture_error: str | None = None
    capture_error_phase: str | None = None

    def _record_error(phase: str, exc: BaseException) -> None:
        nonlocal capture_error, capture_error_phase
        logger.error("Capturer '{}' {} raised: {!r}", capturer.name, phase, exc)
        # Keep the first error: a later one is usually a consequence of it.
        if capture_error is None:
            capture_error = repr(exc)
            capture_error_phase = phase

    loop = asyncio.get_event_loop()
    installed_signals: list[int] = []
    if stop is None:
        stop = asyncio.Event()
        installed_signals = install_stop_signals(loop, stop)

    # L3 only: ids from the opening book that nothing in the stream has yet
    # mentioned. _stream removes an id when an order event or a trade names
    # it, so what is left at the end is the opening book the stream never
    # confirmed. L2 levels carry no id, so an L2 run leaves this as None.
    unconfirmed: set[Any] | None = None if level is Level.L2 else set()

    try:
        logger.info("Capturer '{}': snapshot starting", capturer.name)
        try:
            async for ev in capturer.snapshot(config):
                ev["origin"] = ORIGIN_SNAPSHOT
                if level is Level.L2:
                    sink.write_depth(ev)
                    n_depth += 1
                else:
                    sink.write_order(ev)
                    n_order += 1
                    if unconfirmed is not None:
                        unconfirmed.add(ev["id"])
        except Exception as exc:  # noqa: BLE001 - recorded in meta.json
            _record_error("snapshot", exc)
        logger.info(
            "Capturer '{}': snapshot wrote {} book events",
            capturer.name,
            n_depth if level is Level.L2 else n_order,
        )

        # _stream updates this mapping in place as it writes, so the counts
        # survive a SIGINT/SIGTERM cancellation: meta.json previously
        # reported only snapshot + shutdown events for interrupted runs
        # even though every streamed row was on disk.
        stream_counts = {"order": 0, "trade": 0, "depth": 0, "raw": 0}
        if capture_error is None:
            await _run_stream(
                capturer,
                config,
                sink,
                _StreamState(stream_counts, unconfirmed, stream_times, streaming),
                stop,
                _record_error,
            )
            stream_times["ended"] = pd.Timestamp.now(tz="UTC")
        n_order += stream_counts["order"]
        n_trade += stream_counts["trade"]
        n_depth += stream_counts["depth"]
        n_raw += stream_counts["raw"]

        logger.info("Capturer '{}': emitting shutdown synthetic events", capturer.name)
        try:
            async for ev in capturer.shutdown_synthetic_events():
                ev["origin"] = ORIGIN_SHUTDOWN
                if level is Level.L2:
                    sink.write_depth(ev)
                    n_depth += 1
                else:
                    sink.write_order(ev)
                    n_order += 1
        except Exception as exc:  # noqa: BLE001 - recorded in meta.json
            _record_error("shutdown", exc)
    finally:
        # Remove signal handlers we installed.
        for sig in installed_signals:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass

        ended = pd.Timestamp.now(tz="UTC")
        # What the source declares about its feed, so a later `audit` can hold
        # the capture to its own source's expectations.  The capture is read
        # back with a file format's source (a cryptofeed L3 capture replays as
        # bitstamp), which would otherwise supply the wrong ones.
        extras: dict[str, Any] = _source_declarations(capturer)
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
            n_depth_events=n_depth,
            n_snapshot_unconfirmed=None if unconfirmed is None else len(unconfirmed),
            capture_error=capture_error,
            capture_error_phase=capture_error_phase,
            stream_started=stream_times.get("started"),
            stream_ended=stream_times.get("ended"),
        )
        sink.finalize(result)
        logger.info(
            "Capturer '{}': finished. orders={}, depth={}, trades={}, raw={}, "
            "dur={:.1f}s",
            capturer.name,
            n_order,
            n_depth,
            n_trade,
            n_raw,
            (ended - started).total_seconds(),
        )
    return result


def install_stop_signals(
    loop: asyncio.AbstractEventLoop, stop: asyncio.Event
) -> list[int]:
    """Make SIGINT/SIGTERM set *stop*; return the signals handled.

    Where signals cannot be handled (Windows, a thread other than the main
    one, some test environments) nothing is installed and default handling
    stays.
    """
    installed: list[int] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
            installed.append(sig)
        except (NotImplementedError, RuntimeError):
            pass
    return installed


class _FirstEvent(asyncio.Event):
    """An event set at a stream's first live event, that also says when.

    :func:`run_capture` passes one as *streaming*, so the manifest records the
    time the runner saw the first event, not the later moment a watcher woke.
    """

    def __init__(self) -> None:
        super().__init__()
        self.at: pd.Timestamp | None = None


@dataclass
class _StreamState:
    """What the stream updates in place, so it survives a cancellation.

    ``counts`` are rows written per kind, and ``items`` everything the stream
    yielded, written or not; ``unconfirmed`` the opening-book
    order ids nothing has named yet (L3 only); ``times["started"]`` is set at
    the first live event, and so is ``streaming``.
    """

    counts: dict[str, int]
    unconfirmed: set[Any] | None
    times: dict[str, pd.Timestamp]
    streaming: asyncio.Event | None = None
    items: int = 0


async def _run_stream(
    capturer: LiveSource,
    config: CaptureConfig,
    sink: CaptureSink,
    state: _StreamState,
    stop: asyncio.Event,
    record_error: Callable[[str, BaseException], None],
) -> None:
    """Stream until the source ends, it raises, or a signal sets *stop*.

    An exception from the stream goes to *record_error*; a stop by signal is
    not an error.
    """
    logger.info(
        "Capturer '{}': streaming for {:.1f} min", capturer.name, config.minutes
    )
    stream_task = asyncio.create_task(_stream(capturer, config, sink, state))
    stop_task = asyncio.create_task(stop.wait())
    try:
        done, _pending = await asyncio.wait(
            {stream_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_task.cancel()
        try:
            await _cancel_until_done(stream_task, state)
        except asyncio.CancelledError:
            # This task was cancelled too (the supervisor stopped waiting for
            # it): leave the stream cancelled and do not wait for it.
            stream_task.cancel()
            stream_task.add_done_callback(_retrieve_outcome)
            raise
        # Drain cancellation cleanly.
        for t in (stream_task, stop_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - draining cancelled tasks; any error here is irrelevant during shutdown
                pass

    if stream_task in done and not stream_task.cancelled():
        exc = stream_task.exception()
        if exc is not None:
            record_error("stream", exc)


async def _cancel_until_done(task: asyncio.Task[Any], state: _StreamState) -> None:
    """Cancel *task* and wait for it to end, cancelling again while it streams.

    A stream that lost the cancel keeps yielding items (``state.items``),
    and only another cancel stops it.  A stream that yields nothing more is
    closing its connection, and is left to finish: another cancel would cut
    that short.
    """
    # On Python 3.11, asyncio.wait_for drops a cancel that arrives in the same
    # loop tick as the result it awaits (CPython gh-86296, fixed in 3.12).  The
    # bundled sources do not wait that way, but a plug-in source may.
    task.cancel()
    while not task.done():
        seen = state.items
        await asyncio.wait({task}, timeout=CANCEL_RETRY_SECONDS)
        if not task.done() and state.items != seen:
            task.cancel()


def _retrieve_outcome(task: asyncio.Task[Any]) -> None:
    """Read how an abandoned task ended, so asyncio does not log it as lost."""
    if not task.cancelled():
        task.exception()


async def _stream(
    capturer: LiveSource,
    config: CaptureConfig,
    sink: CaptureSink,
    state: _StreamState,
) -> None:
    """Pump the capturer's stream into *sink*, updating *state* in place.

    Counts are incremented per write (not returned) so they remain accurate
    when the task is cancelled mid-stream by a signal. For the same reason
    the unconfirmed ids are shrunk in place: an opening-book order id leaves
    them as soon as an order event or a trade names it.
    """
    counts = state.counts
    unconfirmed = state.unconfirmed
    async for kind, event, frame in capturer.stream(config):
        state.items += 1
        if "started" not in state.times:
            state.times["started"] = pd.Timestamp.now(tz="UTC")
            if isinstance(state.streaming, _FirstEvent):
                state.streaming.at = state.times["started"]
            if state.streaming is not None:
                state.streaming.set()
        if kind == "order":
            event["origin"] = ORIGIN_STREAM
            sink.write_order(event)
            counts["order"] += 1
            if unconfirmed:
                unconfirmed.discard(event["id"])
        elif kind == "depth":
            event["origin"] = ORIGIN_STREAM
            sink.write_depth(event)
            counts["depth"] += 1
        elif kind == "trade":
            sink.write_trade(event)
            counts["trade"] += 1
            if unconfirmed:
                unconfirmed.discard(event.get("buy_order_id"))
                unconfirmed.discard(event.get("sell_order_id"))
        # ``raw`` (heartbeats / subscription_succeeded) bypasses CSV writers
        # but still goes to raw.jsonl below for forensic completeness.
        if frame is not None:
            sink.write_raw(frame)
            counts["raw"] += 1
